#!/usr/bin/env node
import 'source-map-support/register';
import * as cdk from 'aws-cdk-lib';
import { TrademarkRiskStack } from '../lib/trademark-risk-stack';

const app = new cdk.App();

new TrademarkRiskStack(app, 'TrademarkRiskStack', {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION ?? 'us-east-1',
  },
  description: 'Single-box EC2 + ECR for the trademark-risk demo app.',
});
