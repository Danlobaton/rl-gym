import * as cdk from 'aws-cdk-lib';
import { Duration } from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as appscaling from 'aws-cdk-lib/aws-applicationautoscaling';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';

export interface CoordinatorStackProps extends cdk.StackProps {
  vpc: ec2.IVpc;
  cluster: ecs.ICluster;
  coordinatorSg: ec2.ISecurityGroup;
  coordinatorEcrRepo: ecr.IRepository;
  tracesBucket: s3.IBucket;
  jobQueue: sqs.IQueue;
  runsTable: dynamodb.ITable;
  kmsKey: kms.IKey;
  anthropicSecret: secretsmanager.ISecret;
  albDnsName: string;
}

export class CoordinatorStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: CoordinatorStackProps) {
    super(scope, id, props);

    // Imported KMS handle for log-group encryption — same cross-stack-cycle
    // reasoning as the data resource handles below.
    const kmsHandleForLogs = kms.Key.fromKeyArn(this, 'KmsHandleForLogs', props.kmsKey.keyArn);
    const logGroup = new logs.LogGroup(this, 'CoordinatorLogs', {
      retention: logs.RetentionDays.THREE_MONTHS,
      encryptionKey: kmsHandleForLogs,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // Distinct execution role (image pull + log writes, runs before container
    // starts) and task role (what the application code assumes via the task
    // metadata endpoint). Splitting them is least-privilege 101: an
    // app-code RCE shouldn't be able to push images to ECR.
    const executionRole = new iam.Role(this, 'CoordinatorExecutionRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    });

    const taskRole = new iam.Role(this, 'CoordinatorTaskRole', {
      assumedBy: new iam.ServicePrincipal('ecs-tasks.amazonaws.com'),
    });

    const taskDefinition = new ecs.FargateTaskDefinition(this, 'CoordinatorTaskDef', {
      cpu: 512,
      memoryLimitMiB: 1024,
      executionRole,
      taskRole,
    });

    // All cross-stack data resources are imported as ARN-only handles. CDK
    // treats imported resources as external: grants only modify THIS stack's
    // role IAM policy, never the resource policies in DataStack. Without this,
    // every grant from CoordinatorStack would add a DataStack → CoordinatorStack
    // back-reference (role ARN in resource policies) → dep cycle.
    const ecrHandle = ecr.Repository.fromRepositoryAttributes(this, 'CoordinatorEcrHandle', {
      repositoryArn: props.coordinatorEcrRepo.repositoryArn,
      repositoryName: props.coordinatorEcrRepo.repositoryName,
    });
    const queueHandle = sqs.Queue.fromQueueArn(this, 'JobQueueHandle', props.jobQueue.queueArn);
    const bucketHandle = s3.Bucket.fromBucketArn(this, 'TracesBucketHandle', props.tracesBucket.bucketArn);
    const tableHandle = dynamodb.Table.fromTableArn(this, 'RunsTableHandle', props.runsTable.tableArn);
    const secretHandle = secretsmanager.Secret.fromSecretCompleteArn(this, 'AnthropicSecretHandle', props.anthropicSecret.secretArn);
    const keyHandle = kms.Key.fromKeyArn(this, 'KmsKeyHandle', props.kmsKey.keyArn);

    taskDefinition.addContainer('coordinator', {
      image: ecs.ContainerImage.fromEcrRepository(ecrHandle, 'latest'),
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: 'coordinator',
        logGroup,
      }),
      environment: {
        SQS_QUEUE_URL: props.jobQueue.queueUrl,
        S3_BUCKET: props.tracesBucket.bucketName,
        DYNAMODB_TABLE: props.runsTable.tableName,
        // ALB listener is HTTP/80 only; this is internal traffic over a
        // private VPC, so no TLS termination here.
        ENV_ALB_URL: `http://${props.albDnsName}`,
        AWS_REGION: this.region,
      },
      secrets: {
        // The imported secretHandle is critical: passing the live props.anthropicSecret
        // here would auto-grant the execution role on the SECRET'S policy in DataStack,
        // re-introducing the cycle.
        ANTHROPIC_API_KEY: ecs.Secret.fromSecretsManager(secretHandle),
      },
    });

    // Grants ride on the imported handles declared above the addContainer
    // call — see the comment there for why.
    queueHandle.grantConsumeMessages(taskRole);
    bucketHandle.grantPut(taskRole, 'agent=*/*');
    tableHandle.grant(taskRole, 'dynamodb:PutItem', 'dynamodb:UpdateItem');
    secretHandle.grantRead(taskRole);
    // Both Encrypt and Decrypt: SQS ChangeMessageVisibility and visibility
    // extensions during long episodes touch the CMK on both sides.
    keyHandle.grantEncryptDecrypt(taskRole);

    const service = new ecs.FargateService(this, 'CoordinatorService', {
      cluster: props.cluster,
      taskDefinition,
      desiredCount: 1,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [props.coordinatorSg],
      assignPublicIp: false,
      enableExecuteCommand: true,
      // Same shape as env service: floor of 1 on-demand, then 30/70
      // on-demand/spot above the floor.
      capacityProviderStrategies: [
        { capacityProvider: 'FARGATE', base: 1, weight: 30 },
        { capacityProvider: 'FARGATE_SPOT', weight: 70 },
      ],
    });

    const scaling = service.autoScaleTaskCount({
      minCapacity: 1,
      maxCapacity: 30,
    });

    // Step scaling on queue depth rather than target tracking: queue depth
    // is bursty (a job dump can spike from 0 to thousands in seconds), and
    // target tracking's PI controller smooths over exactly the spikes we want
    // to react to. Step intervals give predictable, bounded reactions per
    // alarm period.
    scaling.scaleOnMetric('QueueDepthScaling', {
      metric: props.jobQueue.metricApproximateNumberOfMessagesVisible({
        period: Duration.minutes(1),
        statistic: cloudwatch.Stats.MAXIMUM,
      }),
      adjustmentType: appscaling.AdjustmentType.CHANGE_IN_CAPACITY,
      scalingSteps: [
        { upper: 5, change: -1 },
        { lower: 20, change: +2 },
        { lower: 100, change: +5 },
      ],
      cooldown: Duration.seconds(60),
    });

    // Separate scale-in policy with a longer cooldown — we want to react
    // fast on spikes (60s) but drain conservatively (300s) to avoid
    // thrashing tasks mid-episode when the queue briefly empties.
    scaling.scaleOnMetric('QueueDepthScaleIn', {
      metric: props.jobQueue.metricApproximateNumberOfMessagesVisible({
        period: Duration.minutes(5),
        statistic: cloudwatch.Stats.MAXIMUM,
      }),
      adjustmentType: appscaling.AdjustmentType.CHANGE_IN_CAPACITY,
      scalingSteps: [
        { upper: 5, change: -1 },
        { upper: 1, change: -2 },
      ],
      cooldown: Duration.seconds(300),
    });
  }
}
