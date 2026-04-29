import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as sqs from 'aws-cdk-lib/aws-sqs';
import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as ecr from 'aws-cdk-lib/aws-ecr';
import * as glue from 'aws-cdk-lib/aws-glue';

export interface DataStackProps extends cdk.StackProps {}

export class DataStack extends cdk.Stack {
  public readonly kmsKey: kms.Key;
  public readonly tracesBucket: s3.Bucket;
  public readonly jobQueue: sqs.Queue;
  public readonly dlq: sqs.Queue;
  public readonly runsTable: dynamodb.Table;
  public readonly anthropicSecret: secretsmanager.Secret;
  public readonly envEcrRepo: ecr.Repository;
  public readonly coordinatorEcrRepo: ecr.Repository;

  constructor(scope: Construct, id: string, props?: DataStackProps) {
    super(scope, id, props);

    this.kmsKey = new kms.Key(this, 'DataKey', {
      enableKeyRotation: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      alias: 'alias/sregym-data',
    });

    // CloudWatch Logs in EnvStack and CoordinatorStack encrypt with this key.
    // Cross-stack KMS grants (via fromKeyArn handles) only mutate the role's
    // IAM policy — they CAN'T mutate the key policy here. So we must explicitly
    // authorize the Logs service principal in the key policy, otherwise log
    // writes fail at runtime even though IAM looks fine.
    this.kmsKey.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AllowCloudWatchLogsToUseKey',
      effect: iam.Effect.ALLOW,
      principals: [new iam.ServicePrincipal(`logs.${this.region}.amazonaws.com`)],
      actions: [
        'kms:Encrypt*',
        'kms:Decrypt*',
        'kms:ReEncrypt*',
        'kms:GenerateDataKey*',
        'kms:Describe*',
      ],
      resources: ['*'],
      conditions: {
        ArnLike: {
          // Scope to LogGroups in this account+region only.
          'kms:EncryptionContext:aws:logs:arn': `arn:aws:logs:${this.region}:${this.account}:log-group:*`,
        },
      },
    }));

    this.tracesBucket = new s3.Bucket(this, 'TracesBucket', {
      encryption: s3.BucketEncryption.KMS,
      encryptionKey: this.kmsKey,
      // S3 Bucket Keys collapse per-object KMS calls into a per-bucket data
      // key, slashing KMS request cost on high-volume trace writes.
      bucketKeyEnabled: true,
      versioned: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      enforceSSL: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          transitions: [
            {
              storageClass: s3.StorageClass.INTELLIGENT_TIERING,
              transitionAfter: cdk.Duration.days(30),
            },
            {
              storageClass: s3.StorageClass.DEEP_ARCHIVE,
              transitionAfter: cdk.Duration.days(365),
            },
          ],
        },
      ],
    });

    this.dlq = new sqs.Queue(this, 'JobDlq', {
      encryption: sqs.QueueEncryption.KMS,
      encryptionMasterKey: this.kmsKey,
      retentionPeriod: cdk.Duration.days(14),
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    this.jobQueue = new sqs.Queue(this, 'JobQueue', {
      encryption: sqs.QueueEncryption.KMS,
      encryptionMasterKey: this.kmsKey,
      visibilityTimeout: cdk.Duration.minutes(30),
      retentionPeriod: cdk.Duration.days(4),
      deadLetterQueue: {
        queue: this.dlq,
        maxReceiveCount: 3,
      },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    this.runsTable = new dynamodb.Table(this, 'RunsTable', {
      partitionKey: { name: 'run_id', type: dynamodb.AttributeType.STRING },
      billingMode: dynamodb.BillingMode.PAY_PER_REQUEST,
      encryption: dynamodb.TableEncryption.CUSTOMER_MANAGED,
      encryptionKey: this.kmsKey,
      pointInTimeRecoverySpecification: { pointInTimeRecoveryEnabled: true },
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    // Placeholder value: the secret must exist at deploy time so IAM grants
    // resolve, but the real key is rotated in by an operator post-deploy so
    // it never lives in source or CloudFormation parameters.
    this.anthropicSecret = new secretsmanager.Secret(this, 'AnthropicSecret', {
      secretName: 'sregym/anthropic-api-key',
      secretStringValue: cdk.SecretValue.unsafePlainText('REPLACE_ME'),
      encryptionKey: this.kmsKey,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    this.envEcrRepo = new ecr.Repository(this, 'EnvEcrRepo', {
      repositoryName: 'sregym/env',
      imageScanOnPush: true,
      encryption: ecr.RepositoryEncryption.KMS,
      encryptionKey: this.kmsKey,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    this.coordinatorEcrRepo = new ecr.Repository(this, 'CoordinatorEcrRepo', {
      repositoryName: 'sregym/coordinator',
      imageScanOnPush: true,
      encryption: ecr.RepositoryEncryption.KMS,
      encryptionKey: this.kmsKey,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
    });

    const glueDb = new glue.CfnDatabase(this, 'TracesDatabase', {
      catalogId: this.account,
      databaseInput: {
        name: 'sregym_traces',
      },
    });

    const tracesTable = new glue.CfnTable(this, 'TracesTable', {
      catalogId: this.account,
      databaseName: 'sregym_traces',
      tableInput: {
        name: 'traces',
        tableType: 'EXTERNAL_TABLE',
        // `agent` uses the `injected` projection type so Athena takes the
        // value straight from the WHERE clause — no S3 listing needed and
        // no pre-registered enum of agent names to maintain.
        parameters: {
          'projection.enabled': 'true',
          'projection.agent.type': 'injected',
          'projection.dt.type': 'date',
          'projection.dt.format': 'yyyy-MM-dd',
          'projection.dt.range': '2024-01-01,NOW',
          // \${agent} / \${dt} are Athena partition-projection placeholders,
          // not TS interpolation — escape the $ so the literal survives.
          'storage.location.template': `s3://${this.tracesBucket.bucketName}/agent=\${agent}/dt=\${dt}/`,
          classification: 'parquet',
          'parquet.compression': 'SNAPPY',
        },
        partitionKeys: [
          { name: 'agent', type: 'string' },
          { name: 'dt', type: 'string' },
        ],
        storageDescriptor: {
          location: `s3://${this.tracesBucket.bucketName}/`,
          inputFormat: 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
          outputFormat: 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
          serdeInfo: {
            serializationLibrary: 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe',
          },
          columns: [
            { name: 'run_id', type: 'string' },
            { name: 'seed', type: 'int' },
            { name: 'schema_version', type: 'string' },
            { name: 'run_started_at', type: 'double' },
            { name: 'agent_name', type: 'string' },
            { name: 'total_reward', type: 'double' },
            { name: 'reward_breakdown', type: 'map<string,double>' },
            { name: 'ground_truth', type: 'map<string,string>' },
            { name: 'task_meta', type: 'map<string,string>' },
            { name: 'agent_config', type: 'map<string,string>' },
            { name: 'diagnostics', type: 'map<string,string>' },
            { name: 'started_at', type: 'double' },
            { name: 'ended_at', type: 'double' },
            // steps stays JSON-encoded while the trace shape is in flux;
            // promoting to a struct/array column is a follow-up migration.
            { name: 'steps', type: 'string' },
          ],
        },
      },
    });
    tracesTable.addDependency(glueDb);
  }
}
