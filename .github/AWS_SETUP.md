# AWS + GitHub Setup Guide

One-time setup to wire OIDC, ECR, SSM, and GitHub Secrets.

---

## 1. Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name nse-dashboard \
  --region us-east-1
# Note the repositoryUri — you'll need the registry prefix (account-id.dkr.ecr.us-east-1.amazonaws.com)
```

---

## 2. Add GitHub as OIDC Identity Provider in AWS

AWS Console → IAM → Identity providers → Add provider

| Field | Value |
|---|---|
| Provider type | OpenID Connect |
| Provider URL | `https://token.actions.githubusercontent.com` |
| Audience | `sts.amazonaws.com` |

Click **Get thumbprint**, then **Add provider**.

---

## 3. Create IAM Role for GitHub Actions

### Trust policy (scope to v2 branch only)

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<YOUR_ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
        },
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:jaideep2403/nse-trend-screener:ref:refs/heads/v2"
        }
      }
    }
  ]
}
```

### Attach these policies to the role

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRAuth",
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Sid": "ECRPush",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:CompleteLayerUpload",
        "ecr:InitiateLayerUpload",
        "ecr:PutImage",
        "ecr:UploadLayerPart",
        "ecr:BatchGetImage",
        "ecr:DescribeImages"
      ],
      "Resource": "arn:aws:ecr:us-east-1:<YOUR_ACCOUNT_ID>:repository/nse-dashboard"
    },
    {
      "Sid": "SSMDeploy",
      "Effect": "Allow",
      "Action": [
        "ssm:SendCommand",
        "ssm:GetCommandInvocation",
        "ssm:DescribeInstanceInformation"
      ],
      "Resource": [
        "arn:aws:ec2:us-east-1:<YOUR_ACCOUNT_ID>:instance/<YOUR_EC2_INSTANCE_ID>",
        "arn:aws:ssm:*::document/AWS-RunShellScript"
      ]
    }
  ]
}
```

Name the role `github-actions-nse-deploy` and note its ARN.

---

## 4. Give EC2 an Instance Profile for SSM + ECR pull

Create a new IAM Role for the EC2 instance (type: EC2):

Attach policies:
- `AmazonSSMManagedInstanceCore` (allows SSM agent to receive commands)
- ECR pull policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "ecr:GetAuthorizationToken",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer"
    ],
    "Resource": "*"
  }]
}
```

Attach this instance profile to your EC2 instance:
AWS Console → EC2 → Your instance → Actions → Security → Modify IAM role

---

## 5. Install SSM Agent on EC2 (Amazon Linux 2 / Ubuntu)

```bash
# Amazon Linux 2 (already installed, just enable):
sudo systemctl enable amazon-ssm-agent
sudo systemctl start amazon-ssm-agent

# Ubuntu:
sudo snap install amazon-ssm-agent --classic
sudo systemctl enable snap.amazon-ssm-agent.amazon-ssm-agent
sudo systemctl start snap.amazon-ssm-agent.amazon-ssm-agent
```

Verify: AWS Console → Systems Manager → Fleet Manager → your instance should appear.

---

## 6. Add GitHub Secrets

Go to: `https://github.com/jaideep2403/nse-trend-screener/settings/secrets/actions`

| Secret name | Value |
|---|---|
| `AWS_DEPLOY_ROLE_ARN` | `arn:aws:iam::<account-id>:role/github-actions-nse-deploy` |
| `EC2_INSTANCE_ID` | `i-xxxxxxxxxxxxxxxxx` (your EC2 instance ID) |

No SSH keys, no AWS access keys stored anywhere.

---

## 7. Configure GitHub Environment (Manual Approval Gate)

Go to: `https://github.com/jaideep2403/nse-trend-screener/settings/environments`

- Click **New environment** → name it `production`
- Enable **Required reviewers** → add yourself (`jaideep2403`)
- Optionally set **Wait timer** (e.g. 5 minutes) for a cooldown

After build passes, the deploy job will pause and email you for approval before touching EC2.

---

## 8. Secrets Rotation Calendar

| Credential | Action | Frequency |
|---|---|---|
| OIDC role trust | Review condition (branch scope) | Quarterly |
| EC2 instance profile | Rotate nothing — no keys, just IAM | — |
| GitHub Secrets audit | Settings → Secrets → review last-used dates | Quarterly |

OIDC credentials are **time-bound (15 min)** — nothing to rotate. The main hygiene task is ensuring the IAM role trust policy still has the correct branch scope.
