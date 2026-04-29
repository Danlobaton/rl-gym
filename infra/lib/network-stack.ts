import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as logs from 'aws-cdk-lib/aws-logs';

export interface NetworkStackProps extends cdk.StackProps {}

export class NetworkStack extends cdk.Stack {
  public readonly vpc: ec2.Vpc;
  public readonly cluster: ecs.Cluster;
  public readonly coordinatorSg: ec2.SecurityGroup;
  public readonly envWorkerSg: ec2.SecurityGroup;
  public readonly albSg: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props?: NetworkStackProps) {
    super(scope, id, props);

    const flowLogGroup = new logs.LogGroup(this, 'VpcFlowLogs', {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // One NAT per AZ: a single NAT would create a cross-AZ data path and a
    // single point of failure for all egress-bearing tasks (coordinators).
    this.vpc = new ec2.Vpc(this, 'Vpc', {
      ipAddresses: ec2.IpAddresses.cidr('10.0.0.0/16'),
      maxAzs: 3,
      natGateways: 3,
      subnetConfiguration: [
        {
          name: 'public',
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: 'private',
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 22,
        },
      ],
      flowLogs: {
        cloudwatch: {
          destination: ec2.FlowLogDestination.toCloudWatchLogs(flowLogGroup),
          trafficType: ec2.FlowLogTrafficType.ALL,
        },
      },
    });
    this.vpc.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);

    this.cluster = new ecs.Cluster(this, 'Cluster', {
      vpc: this.vpc,
      containerInsightsV2: ecs.ContainerInsights.ENABLED,
      enableFargateCapacityProviders: true,
    });
    this.cluster.applyRemovalPolicy(cdk.RemovalPolicy.DESTROY);

    this.albSg = new ec2.SecurityGroup(this, 'AlbSg', {
      vpc: this.vpc,
      allowAllOutbound: false,
    });

    // No egress: env workers must not reach the internet or arbitrary AWS
    // endpoints. They talk to ECR/Logs/SSM via VPC endpoints, which the
    // endpoint SG gates separately on 443.
    this.envWorkerSg = new ec2.SecurityGroup(this, 'EnvWorkerSg', {
      vpc: this.vpc,
      allowAllOutbound: false,
    });

    this.coordinatorSg = new ec2.SecurityGroup(this, 'CoordinatorSg', {
      vpc: this.vpc,
      allowAllOutbound: true,
    });

    this.albSg.addIngressRule(
      this.coordinatorSg,
      ec2.Port.tcp(80),
      'coordinator → ALB',
    );
    this.albSg.addEgressRule(
      this.envWorkerSg,
      ec2.Port.tcp(8000),
      'ALB → env workers',
    );
    this.envWorkerSg.addIngressRule(
      this.albSg,
      ec2.Port.tcp(8000),
      'ALB → env workers',
    );

    const vpcEndpointSg = new ec2.SecurityGroup(this, 'VpcEndpointSg', {
      vpc: this.vpc,
      allowAllOutbound: false,
    });
    vpcEndpointSg.addIngressRule(
      ec2.Peer.ipv4(this.vpc.vpcCidrBlock),
      ec2.Port.tcp(443),
      'VPC → interface endpoints',
    );

    // Env workers have no internet egress, but they DO need to reach the VPC
    // interface endpoints (ECR API, ECR DKR, Logs, KMS, Secrets Manager, SQS)
    // on 443 — otherwise Fargate can't pull images or write task logs and
    // tasks never come up. Egress is scoped to the endpoint SG only.
    this.envWorkerSg.addEgressRule(
      vpcEndpointSg,
      ec2.Port.tcp(443),
      'env workers → VPC interface endpoints',
    );

    this.vpc.addGatewayEndpoint('S3Endpoint', {
      service: ec2.GatewayVpcEndpointAwsService.S3,
    });
    this.vpc.addGatewayEndpoint('DynamoDbEndpoint', {
      service: ec2.GatewayVpcEndpointAwsService.DYNAMODB,
    });

    const interfaceEndpoints: Array<[string, ec2.InterfaceVpcEndpointAwsService]> = [
      ['SqsEndpoint', ec2.InterfaceVpcEndpointAwsService.SQS],
      ['SecretsManagerEndpoint', ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER],
      ['EcrApiEndpoint', ec2.InterfaceVpcEndpointAwsService.ECR],
      ['EcrDkrEndpoint', ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER],
      ['CloudWatchLogsEndpoint', ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS],
      ['KmsEndpoint', ec2.InterfaceVpcEndpointAwsService.KMS],
    ];

    for (const [id, service] of interfaceEndpoints) {
      this.vpc.addInterfaceEndpoint(id, {
        service,
        subnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
        securityGroups: [vpcEndpointSg],
        privateDnsEnabled: true,
      });
    }
  }
}
