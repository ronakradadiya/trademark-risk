# infra — AWS CDK deployment

Single-box EC2 (`t4g.small`, 2 vCPU ARM / 2 GB, ~$14/mo) + Elastic IP + ECR repo. HTTP-only on port 80 — add a domain + ACM cert later if you need TLS.

## One-time setup

```bash
cd infra
npm install
npm install -g aws-cdk           # if you don't already have `cdk` on PATH
aws configure                    # IAM user with AdminAccess is simplest for a demo
npx cdk bootstrap                # one-time per account+region
```

## Deploy (end-to-end, ~15–90 min depending on SQLite upload)

1. **Provision infra.** Creates the EC2 instance, EIP, and ECR repo.
   ```bash
   cd infra && npx cdk deploy
   ```
   Note the `PublicIp`, `EcrRepoUri`, and `InstanceId` outputs.

2. **Push app secrets** to SSM Parameter Store (free).
   ```bash
   aws ssm put-parameter --name /trademark-risk/OPENAI_API_KEY  --type SecureString --value "sk-..."
   aws ssm put-parameter --name /trademark-risk/SERPER_API_KEY  --type SecureString --value "..."
   ```

3. **Build + push the Docker image.** Run from repo root.
   ```bash
   ECR=<EcrRepoUri from step 1>
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin "$ECR"
   docker buildx build --platform linux/arm64 -t "$ECR:latest" --push .
   ```

4. **Upload the SQLite DB** to the instance (3.8 GB, slow step).
   ```bash
   INSTANCE=<InstanceId from step 1>
   aws ssm start-session --target "$INSTANCE"
   # inside the session:
   sudo mkdir -p /srv/trademark-risk/data && sudo chown ec2-user /srv/trademark-risk/data
   exit
   # from laptop, scp via SSM proxy:
   scp -o 'ProxyCommand=aws ssm start-session --target %h --document-name AWS-StartSSHSession --parameters portNumber=%p' \
     data/uspto.sqlite ec2-user@$INSTANCE:/srv/trademark-risk/data/uspto.sqlite
   ```

5. **Start the app.** SSM into the box and run:
   ```bash
   aws ssm start-session --target "$INSTANCE"
   # inside:
   ECR=<EcrRepoUri>
   docker pull "$ECR:latest"
   docker run -d --name tmrisk --restart unless-stopped \
     -p 80:3000 \
     -v /srv/trademark-risk/data:/data:ro \
     -e OPENAI_API_KEY="$(aws ssm get-parameter --name /trademark-risk/OPENAI_API_KEY --with-decryption --query Parameter.Value --output text)" \
     -e SERPER_API_KEY="$(aws ssm get-parameter --name /trademark-risk/SERPER_API_KEY --with-decryption --query Parameter.Value --output text)" \
     "$ECR:latest"
   ```

6. **Smoke test.**
   ```bash
   curl -sS -X POST "http://<PublicIp>/api/check" \
     -H 'content-type: application/json' \
     -d '{"brand_name":"NovaPay","applicant_name":"I420 LLC","class_code":36}' | jq .verdict
   ```

## Tear down

```bash
cd infra && npx cdk destroy
aws ssm delete-parameter --name /trademark-risk/OPENAI_API_KEY
aws ssm delete-parameter --name /trademark-risk/SERPER_API_KEY
```

The ECR repo is set to `emptyOnDelete: true` so the images go with it.
