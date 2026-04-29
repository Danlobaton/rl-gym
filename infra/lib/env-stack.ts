import * as cdk from 'aws-cdk-lib';
import { Duration } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as kms from 'aws-cdk-lib/aws-kms';

export interface EnvStackProps extends cdk.StackProps {
  vpc: ec2.IVpc;
  cluster: ecs.ICluster;
  envWorkerSg: ec2.ISecurityGroup;
  albSg: ec2.ISecurityGroup;
  envEcrRepo: ecr.IRepository;
  kmsKey: kms.IKey;
}

export class EnvStack extends cdk.Stack {
  public readonly alb: elbv2.ApplicationLoadBalancer;

  constructor(scope: Construct, id: string, props: EnvStackProps) {
    super(scope, id, props);

    // Import the KMS key by ARN so log-group encryption doesn't mutate the
    // key policy in DataStack with a reference to this stack's roles (cycle).
    const kmsHandle = kms.Key.fromKeyArn(this, 'EnvKmsHandle', props.kmsKey.keyArn);
    const logGroup = new logs.LogGroup(this, 'EnvWorkerLogs', {
      retention: logs.RetentionDays.THREE_MONTHS,
      encryptionKey: kmsHandle,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const taskDefinition = new ecs.FargateTaskDefinition(this, 'EnvTaskDef', {
      cpu: 512,
      memoryLimitMiB: 1024,
    });

    // Import ECR by ARN to avoid a DataStack ↔ EnvStack dep cycle: a direct
    // grantPull on the cross-stack repo would mutate the repo's resource
    // policy to reference this stack's execution role.
    const ecrHandle = ecr.Repository.fromRepositoryAttributes(this, 'EnvEcrHandle', {
      repositoryArn: props.envEcrRepo.repositoryArn,
      repositoryName: props.envEcrRepo.repositoryName,
    });
    // No env vars / secrets: env workers are sandboxed, have no outbound
    // network path, and don't call any external API. Anything injected here
    // would be dead weight (and a leak surface).
    taskDefinition.addContainer('env-worker', {
      image: ecs.ContainerImage.fromEcrRepository(ecrHandle, 'latest'),
      portMappings: [{ containerPort: 8000, protocol: ecs.Protocol.TCP }],
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: 'env',
        logGroup,
      }),
    });

    this.alb = new elbv2.ApplicationLoadBalancer(this, 'EnvAlb', {
      vpc: props.vpc,
      internetFacing: false,
      securityGroup: props.albSg,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
    });

    const listener = this.alb.addListener('Http', {
      port: 80,
      protocol: elbv2.ApplicationProtocol.HTTP,
      open: false,
    });

    const service = new ecs.FargateService(this, 'EnvService', {
      cluster: props.cluster,
      taskDefinition,
      desiredCount: 2,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [props.envWorkerSg],
      assignPublicIp: false,
      enableExecuteCommand: true,
      // base=1 on FARGATE pins one always-on on-demand task as the floor;
      // weights 30/70 split everything above that floor 30% on-demand /
      // 70% spot. This is "70% spot above min" without exposing the whole
      // fleet to spot interruption storms.
      capacityProviderStrategies: [
        { capacityProvider: 'FARGATE', base: 1, weight: 30 },
        { capacityProvider: 'FARGATE_SPOT', weight: 70 },
      ],
    });

    // `targetType` is auto-inferred as IP for Fargate services — passing it
    // explicitly is rejected by the addTargets type. Don't add it back.
    const targetGroup = listener.addTargets('env', {
      port: 8000,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targets: [service],
      healthCheck: {
        path: '/health',
        healthyHttpCodes: '200',
        interval: Duration.seconds(30),
        timeout: Duration.seconds(5),
        healthyThresholdCount: 2,
        unhealthyThresholdCount: 3,
      },
    });

    const scaling = service.autoScaleTaskCount({
      minCapacity: 2,
      maxCapacity: 20,
    });

    scaling.scaleOnCpuUtilization('cpu', {
      targetUtilizationPercent: 60,
    });

    scaling.scaleOnRequestCount('rps', {
      requestsPerTarget: 50,
      targetGroup,
    });
  }
}
