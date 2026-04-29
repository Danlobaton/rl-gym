#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { NetworkStack } from '../lib/network-stack';
import { DataStack } from '../lib/data-stack';
import { EnvStack } from '../lib/env-stack';
import { CoordinatorStack } from '../lib/coordinator-stack';

const app = new cdk.App();
const env = process.env.CDK_DEFAULT_ACCOUNT
  ? { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION || 'us-east-1' }
  : undefined;

const network = new NetworkStack(app, 'sregym-network', { env });

const data = new DataStack(app, 'sregym-data', { env });

const envStack = new EnvStack(app, 'sregym-env', {
  env,
  vpc: network.vpc,
  cluster: network.cluster,
  envWorkerSg: network.envWorkerSg,
  albSg: network.albSg,
  envEcrRepo: data.envEcrRepo,
  kmsKey: data.kmsKey,
});

new CoordinatorStack(app, 'sregym-coordinator', {
  env,
  vpc: network.vpc,
  cluster: network.cluster,
  coordinatorSg: network.coordinatorSg,
  coordinatorEcrRepo: data.coordinatorEcrRepo,
  tracesBucket: data.tracesBucket,
  jobQueue: data.jobQueue,
  runsTable: data.runsTable,
  kmsKey: data.kmsKey,
  anthropicSecret: data.anthropicSecret,
  albDnsName: envStack.alb.loadBalancerDnsName,
});

// Tag every resource with Project=sregym (cascades through the construct tree).
cdk.Tags.of(app).add('Project', 'sregym');
