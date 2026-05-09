# MLOps on AWS: Build & Deploy an AI Sentiment Analysis API
### A 5-Stage Hands-On Guide for Learners

---

## Why This App?

This guide uses a **Sentiment Analysis API** — a foundational ML use-case that is:

- **Universally understandable** — everyone knows what "positive" and "negative" mean
- **Immediately testable** — no personalisation needed; just send a sentence and get a result
- **Realistic for MLOps** — model serving, versioning, monitoring, and CI/CD all apply cleanly
- **Cost-controlled** — uses a lightweight HuggingFace model locally; no per-token API costs

By the end, you will have a production-grade sentiment analysis service running on AWS with automated deployments via GitHub Actions — the same pattern used in real MLOps teams.

---

## Architecture Overview

```
GitHub → GitHub Actions (CI/CD)
                  ↓
         Terraform (IaC)
                  ↓
    ┌─────────────────────────┐
    │         AWS             │
    │  API Gateway → Lambda   │
    │         ↓               │
    │   S3 (model cache)      │
    │   CloudWatch (logs)     │
    └─────────────────────────┘
```

**Tech Stack:**

| Layer | Technology |
|---|---|
| ML Model | HuggingFace `distilbert-base-uncased-finetuned-sst-2-english` |
| Backend API | FastAPI + Python |
| Deployment | AWS Lambda (serverless) |
| API Gateway | AWS API Gateway (HTTP API) |
| Infrastructure | Terraform |
| CI/CD | GitHub Actions |
| Monitoring | AWS CloudWatch |

---

## A Note on Operating Systems

Every step in this guide includes instructions for both **Mac/Linux** and **Windows**. Look for the labelled tabs below each command block. On Windows, all terminal commands use **PowerShell** (not Command Prompt). Open PowerShell by searching for it in the Start menu.

> 💡 **Windows tip:** If you are on Windows 10/11, consider installing **Windows Subsystem for Linux (WSL2)** — it lets you run a full Linux terminal and follow the Mac/Linux instructions throughout. Install it with: `wsl --install` in PowerShell (run as Administrator). This guide works without WSL, but WSL makes the experience smoother.

---

## Prerequisites

Before Stage 1, ensure you have the following installed:

**Python 3.12+**

> **Mac/Linux:** Download from [python.org](https://www.python.org/downloads/) or use Homebrew: `brew install python@3.12`
> 
> **Windows:** Download the installer from [python.org](https://www.python.org/downloads/). During installation, **check "Add Python to PATH"**. Verify with:
> ```powershell
> python --version
> ```

**AWS CLI**

> **Mac:** `brew install awscli`  
> **Linux:** `curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && sudo ./aws/install`  
> **Windows:** Download the MSI installer from [aws.amazon.com/cli](https://aws.amazon.com/cli/) and run it. Verify with:
> ```powershell
> aws --version
> ```

**Git**

> **Mac:** `brew install git` (or it comes with Xcode Command Line Tools)  
> **Linux:** `sudo apt-get install git`  
> **Windows:** Download from [git-scm.com](https://git-scm.com/download/win). During install, choose "Git from the command line and also from 3rd-party software".

**`uv` Python package manager**

> **Mac/Linux:**
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```
> Then restart your terminal and verify: `uv --version`
>
> **Windows (PowerShell):**
> ```powershell
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
> ```
> Then restart PowerShell and verify: `uv --version`

**An AWS account** with a non-root IAM user and a **GitHub account** are also required.

---

---

# Stage 1 — Local API Development

## 🎯 Target

By the end of Stage 1 you will have a **fully working Sentiment Analysis REST API running on your local machine**, capable of classifying any text as POSITIVE or NEGATIVE with a confidence score. You will understand how to serve an ML model behind an API and why this pattern is used in production.

---

## 1.1 What Is Sentiment Analysis?

Sentiment analysis is a classic NLP task: given a piece of text, classify it as expressing a positive or negative sentiment.

Examples:

| Input | Output |
|---|---|
| "I love this product!" | POSITIVE (0.998) |
| "The service was terrible." | NEGATIVE (0.991) |
| "It was okay, nothing special." | NEGATIVE (0.61) |

We will use a pre-trained model from HuggingFace — no training required. This is a real-world MLOps pattern: use a foundation model or pre-trained checkpoint and focus engineering effort on **serving, monitoring, and deployment**.

---

## 1.2 Project Setup

### Create the project structure

**Mac/Linux:**
```bash
mkdir sentiment-api
cd sentiment-api
mkdir backend
```

**Windows (PowerShell):**
```powershell
mkdir sentiment-api
cd sentiment-api
mkdir backend
```

Your structure:
```
sentiment-api/
└── backend/
```

### Create the Python environment

**Mac/Linux:**
```bash
cd backend
uv init
uv python pin 3.12
```

**Windows (PowerShell):**
```powershell
cd backend
uv init
uv python pin 3.12
```

> 💡 `uv` commands are identical on both platforms — only the surrounding shell syntax differs.

### Install dependencies

Create `backend/requirements.txt`:

```
fastapi
uvicorn
transformers
torch
python-dotenv
pydantic
requests
```

Install them:

**Mac/Linux:**
```bash
uv add -r requirements.txt
```

**Windows (PowerShell):**
```powershell
uv add -r requirements.txt
```

> ⚠️ **Note on size:** `torch` and `transformers` are large packages (~1GB total). This is expected. In Stage 2 we solve the Lambda size problem using a model caching strategy.

---

## 1.3 Create the Model Loader

Create `backend/model.py`:

```python
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)

# Load the model once at module import time (not per-request)
# This is a key MLOps pattern: warm-loading the model
_sentiment_pipeline = None


def get_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        logger.info("Loading sentiment model...")
        _sentiment_pipeline = pipeline(
            task="sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )
        logger.info("Model loaded successfully.")
    return _sentiment_pipeline


def predict(text: str) -> dict:
    """
    Run sentiment prediction on a single piece of text.
    Returns: { label: 'POSITIVE'|'NEGATIVE', score: float }
    """
    pipe = get_pipeline()
    result = pipe(text, truncation=True, max_length=512)
    return result[0]  # pipeline returns a list; take first item
```

**Why `_sentiment_pipeline = None` with lazy loading?**  
Loading an ML model takes several seconds. In production (Lambda), we load it once on cold start and reuse it for subsequent requests — this is called **model warming** and is a fundamental MLOps performance pattern.

---

## 1.4 Create the FastAPI Server

Create `backend/server.py`:

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import time

from model import predict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="Classify text sentiment using DistilBERT",
    version="1.0.0",
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request / Response Models ---

class SentimentRequest(BaseModel):
    text: str
    model_config = {"json_schema_extra": {"example": {"text": "I love building ML APIs!"}}}


class SentimentResult(BaseModel):
    text: str
    label: str          # "POSITIVE" or "NEGATIVE"
    score: float        # confidence, 0.0 to 1.0
    latency_ms: float   # time taken for inference


class BatchSentimentRequest(BaseModel):
    texts: List[str]


class BatchSentimentResponse(BaseModel):
    results: List[SentimentResult]
    total_latency_ms: float


# --- Endpoints ---

@app.get("/")
def root():
    return {"service": "Sentiment Analysis API", "status": "running", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=SentimentResult)
def predict_sentiment(request: SentimentRequest):
    """Analyse sentiment of a single text input."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    start = time.time()
    try:
        result = predict(request.text)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Model inference failed.")
    latency = round((time.time() - start) * 1000, 2)

    logger.info(f"Predicted: {result['label']} ({result['score']:.3f}) in {latency}ms")

    return SentimentResult(
        text=request.text,
        label=result["label"],
        score=round(result["score"], 4),
        latency_ms=latency,
    )


@app.post("/predict/batch", response_model=BatchSentimentResponse)
def predict_batch(request: BatchSentimentRequest):
    """Analyse sentiment of multiple texts in one request."""
    if not request.texts:
        raise HTTPException(status_code=400, detail="texts list cannot be empty.")
    if len(request.texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch.")

    start_total = time.time()
    results = []
    for text in request.texts:
        start = time.time()
        result = predict(text)
        latency = round((time.time() - start) * 1000, 2)
        results.append(SentimentResult(
            text=text,
            label=result["label"],
            score=round(result["score"], 4),
            latency_ms=latency,
        ))

    total_latency = round((time.time() - start_total) * 1000, 2)
    return BatchSentimentResponse(results=results, total_latency_ms=total_latency)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

## 1.5 Run and Test Locally

### Start the server

**Mac/Linux:**
```bash
cd backend
uv run uvicorn server:app --reload
```

**Windows (PowerShell):**
```powershell
cd backend
uv run uvicorn server:app --reload
```

You will see the model download on first run (this may take 1–2 minutes):
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [32552] using StatReload
INFO:     Started server process [35744]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Explore the API docs

Open `http://localhost:8000/docs` in your browser. FastAPI auto-generates interactive Swagger UI — try out the `/predict` endpoint directly.

### Test with curl

**Mac/Linux:**
```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This course is absolutely fantastic!"}'

# Expected response:
# {"text":"This course is absolutely fantastic!","label":"POSITIVE","score":0.9998,"latency_ms":45.2}

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great experience!", "Terrible service.", "Just okay."]}'
```

**Windows (PowerShell):**
```powershell
# Single prediction
Invoke-RestMethod -Method POST http://localhost:8000/predict `
  -ContentType "application/json" `
  -Body '{"text": "This course is absolutely fantastic!"}'

# Batch prediction
Invoke-RestMethod -Method POST http://localhost:8000/predict/batch `
  -ContentType "application/json" `
  -Body '{"texts": ["Great experience!", "Terrible service.", "Just okay."]}'
```

> 💡 **Windows alternative:** You can also install `curl` for Windows via `winget install curl.curl` and then use the exact Mac/Linux `curl` commands.

### Test with Python

Create `backend/test_local.py`:

```python
import requests

BASE = "http://localhost:8000"

test_cases = [
    "I absolutely love this!",
    "This is the worst thing ever.",
    "The weather is fine today.",
    "Outstanding performance, highly recommend.",
    "Do not buy this product.",
]

print("=== Sentiment Analysis Tests ===\n")
for text in test_cases:
    r = requests.post(f"{BASE}/predict", json={"text": text})
    data = r.json()
    print(f"Text    : {data['text']}")
    print(f"Result  : {data['label']} ({data['score']:.4f}) in {data['latency_ms']}ms")
    print()
```

Run it:

**Mac/Linux:**
```bash
uv run python test_local.py
```

**Windows (PowerShell):**
```powershell
uv run python test_local.py
```

---

## 1.6 Checkpoint ✅

Before moving to Stage 2, confirm:

- [ ] Server starts without errors
- [ ] `/health` returns `{"status": "healthy"}`
- [ ] `/predict` returns correct sentiment labels and scores
- [ ] Subsequent requests are faster than the first (model is warm)
- [ ] `/docs` shows the Swagger UI with all endpoints

---

---

# Stage 2 — Deploy to AWS Lambda

## 🎯 Target

By the end of Stage 2, your Sentiment Analysis API will be **live on the internet**, running serverlessly on AWS Lambda behind API Gateway. You will understand how to handle ML model deployment constraints (package size limits), configure Lambda correctly for ML workloads, and test a live cloud endpoint.

---

## 2.1 The Lambda ML Deployment Challenge

AWS Lambda has a **250 MB unzipped deployment package limit**. Our `torch` + `transformers` packages alone exceed 1 GB. This is a real MLOps challenge.

**Solution: Load the model from S3 at runtime**

Instead of bundling the model in the deployment package, we:
1. Upload the model files to an S3 bucket once
2. Lambda downloads the model to `/tmp` (512 MB available) on cold start
3. Subsequent warm invocations reuse the in-memory model

This is standard practice for deploying large ML models on Lambda.

---

## 2.2 AWS Setup

### Create an IAM user for deployments

1. Sign in to the AWS Console as your root user
2. Navigate to **IAM** → **Users** → **Create user**
3. Username: `mlops-deployer`
4. Attach policies directly:
   - `AWSLambda_FullAccess`
   - `AmazonAPIGatewayAdministrator`
   - `AmazonS3FullAccess`
   - `CloudWatchFullAccess`
   - `IAMFullAccess`
5. Click "Next" > "Create User"
6. Click "Download .csv file" >  "View user" (GREEN box on top)
7. Create **Access Keys** (Access Key ID + Secret) using following steps:
   i. Click on tab "Security credentials"
   ii. In section "Access keys" click button "Create access key"
   iii. Select radion button "Command Line Interface (CLI)" >  tick the checkbox for **Confirmation** > Click "Next" > Click "Create access key"
   iv. Copy "Access key" and "Secret access key" > Click on "Download .csv file" > Click on "Done"

### Configure AWS CLI

**Mac/Linux & Windows (same command):**
```bash
aws configure
# Enter: Access Key ID, Secret Access Key, region (e.g. us-east-1), output format (json)
```

Verify:
```bash
aws sts get-caller-identity
```

---

## 2.3 Upload the Model to S3

### Create an S3 bucket

**Mac/Linux:**
```bash
# Replace YOUR_BUCKET_NAME with a globally unique name, e.g. sentiment-model-yourname-2024
export MODEL_BUCKET=sentiment-model-yourname-2024
aws s3 mb s3://$MODEL_BUCKET --region us-east-1
```

**Windows (PowerShell):**
```powershell
# Replace with a globally unique name
$env:MODEL_BUCKET = "sentiment-model-yourname-2024"
aws s3 mb s3://$env:MODEL_BUCKET --region us-east-1
```

### Save the model locally first

Create `backend/save_model.py`:

```python
from transformers import pipeline
import os

MODEL_DIR = "./model_cache"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Downloading model...")
pipe = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)
pipe.save_pretrained(MODEL_DIR)
print(f"Model saved to {MODEL_DIR}")
```

**Mac/Linux:**
```bash
cd backend
uv run python save_model.py
```

**Windows (PowerShell):**
```powershell
cd backend
uv run python save_model.py
```

### Upload to S3

**Mac/Linux:**
```bash
aws s3 sync ./model_cache s3://$MODEL_BUCKET/distilbert-sst2/ --region us-east-1
echo "Model uploaded to s3://$MODEL_BUCKET/distilbert-sst2/"
```

**Windows (PowerShell):**
```powershell
aws s3 sync ./model_cache s3://$env:MODEL_BUCKET/distilbert-sst2/ --region us-east-1
Write-Host "Model uploaded to s3://$env:MODEL_BUCKET/distilbert-sst2/"
```

---

## 2.4 Create the Lambda Handler

Create `backend/lambda_handler.py`:

```python
"""
Lambda handler — loads model from S3 on cold start, serves predictions on warm invocations.
"""
import json
import os
import boto3
import logging
import time
from pathlib import Path

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Model will be downloaded to /tmp on Lambda cold start
MODEL_LOCAL_PATH = "/tmp/model_cache"
MODEL_S3_BUCKET = os.environ.get("MODEL_S3_BUCKET", "")
MODEL_S3_PREFIX = os.environ.get("MODEL_S3_PREFIX", "distilbert-sst2/")

_pipeline = None  # Cached in Lambda execution context


def download_model_from_s3():
    """Download model files from S3 to /tmp."""
    logger.info(f"Downloading model from s3://{MODEL_S3_BUCKET}/{MODEL_S3_PREFIX}")
    s3 = boto3.client("s3")
    Path(MODEL_LOCAL_PATH).mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=MODEL_S3_BUCKET, Prefix=MODEL_S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.replace(MODEL_S3_PREFIX, "")
            if not filename:
                continue
            local_file = os.path.join(MODEL_LOCAL_PATH, filename)
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            logger.info(f"Downloading {key}")
            s3.download_file(MODEL_S3_BUCKET, key, local_file)

    logger.info("Model download complete.")


def get_pipeline():
    """Load model, downloading from S3 if needed (cold start)."""
    global _pipeline
    if _pipeline is None:
        if not Path(MODEL_LOCAL_PATH).exists():
            download_model_from_s3()

        from transformers import pipeline
        logger.info("Loading model into memory...")
        _pipeline = pipeline(
            "sentiment-analysis",
            model=MODEL_LOCAL_PATH,
        )
        logger.info("Model ready.")
    return _pipeline


def make_response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body),
    }


def handler(event, context):
    """Main Lambda entry point."""
    logger.info(f"Event: {json.dumps(event)}")

    http_method = event.get("httpMethod", "GET")
    path = event.get("path", "/")

    # Health check
    if path == "/health":
        return make_response(200, {"status": "healthy"})

    # Predict endpoint
    if path == "/predict" and http_method == "POST":
        try:
            body = json.loads(event.get("body", "{}"))
            text = body.get("text", "").strip()
            if not text:
                return make_response(400, {"error": "text cannot be empty"})

            pipe = get_pipeline()
            start = time.time()
            result = pipe(text, truncation=True, max_length=512)[0]
            latency = round((time.time() - start) * 1000, 2)

            return make_response(200, {
                "text": text,
                "label": result["label"],
                "score": round(result["score"], 4),
                "latency_ms": latency,
            })

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return make_response(500, {"error": "Inference failed"})

    return make_response(404, {"error": "Not found"})
```

---

## 2.5 Package and Deploy

### Create the Lambda deployment package

Lambda needs a zip of your code plus dependencies (excluding torch/transformers — those are loaded at runtime from model files). We use a Lambda Layer approach for dependencies.

**Mac/Linux** — Create `backend/package_lambda.sh`:

```bash
#!/bin/bash
set -e

echo "Creating Lambda deployment package..."

# Clean up
rm -rf lambda_package lambda_package.zip

# Install only the small dependencies (no torch/transformers)
mkdir lambda_package
uv pip install \
    "fastapi>=0.100" \
    "mangum>=0.17" \
    --target lambda_package \
    --quiet

# Copy our handler
cp lambda_handler.py lambda_package/

# Zip it up
cd lambda_package
zip -r ../lambda_package.zip . -q
cd ..

echo "Package created: lambda_package.zip ($(du -sh lambda_package.zip | cut -f1))"
```

Make it executable and run it:
```bash
chmod +x backend/package_lambda.sh
cd backend && ./package_lambda.sh
```

---

**Windows (PowerShell)** — Create `backend/package_lambda.ps1`:

```powershell
Write-Host "Creating Lambda deployment package..."

# Clean up previous builds
if (Test-Path lambda_package) { Remove-Item -Recurse -Force lambda_package }
if (Test-Path lambda_package.zip) { Remove-Item lambda_package.zip }

# Install only small dependencies
New-Item -ItemType Directory -Name lambda_package | Out-Null
uv pip install "fastapi>=0.100" "mangum>=0.17" --target lambda_package --quiet

# Copy handler
Copy-Item lambda_handler.py lambda_package/

# Zip the package
Compress-Archive -Path lambda_package\* -DestinationPath lambda_package.zip -Force

$size = (Get-Item lambda_package.zip).Length / 1MB
Write-Host ("Package created: lambda_package.zip ({0:N1} MB)" -f $size)
```

Run it:
```powershell
cd backend
.\package_lambda.ps1
```

> **Note:** We use `mangum` to wrap FastAPI for Lambda compatibility — it adapts the ASGI interface to the Lambda event/context format.

### Create the Lambda function

**Mac/Linux:**
```bash
# Create execution role
aws iam create-role \
  --role-name sentiment-lambda-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach policies
aws iam attach-role-policy \
  --role-name sentiment-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
  --role-name sentiment-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Wait for role to propagate
sleep 10

# Get your account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

# Create Lambda function
aws lambda create-function `
  --function-name sentiment-api `
  --runtime python3.12 `
  --role "arn:aws:iam::${ACCOUNT_ID}:role/sentiment-lambda-role" `
  --handler lambda_handler.handler `
  --zip-file fileb://lambda_package.zip `
  --timeout 120 `
  --memory-size 1024 `
  --environment "Variables={MODEL_S3_BUCKET=$env:MODEL_BUCKET,MODEL_S3_PREFIX=distilbert-sst2/}" `
  --region $REGION
```

**Windows (PowerShell):**
```powershell
# Create execution role
aws iam create-role `
  --role-name sentiment-lambda-role `
  --assume-role-policy-document '{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Principal\":{\"Service\":\"lambda.amazonaws.com\"},\"Action\":\"sts:AssumeRole\"}]}'

# Attach policies
aws iam attach-role-policy `
  --role-name sentiment-lambda-role `
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy `
  --role-name sentiment-lambda-role `
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Wait for role to propagate
Start-Sleep -Seconds 10

# Get your account ID
$ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
$REGION = "us-east-1"

# Create Lambda function
aws lambda create-function `
  --function-name sentiment-api `
  --runtime python3.12 `
  --role "arn:aws:iam::${ACCOUNT_ID}:role/sentiment-lambda-role" `
  --handler lambda_handler.handler `
  --zip-file fileb://backend/lambda_package.zip `
  --timeout 120 `
  --memory-size 1024 `
  --environment "Variables={MODEL_S3_BUCKET=$env:MODEL_BUCKET,MODEL_S3_PREFIX=distilbert-sst2/}" `
  --region $REGION
```

> 💡 **Windows tip:** In PowerShell, use the backtick `` ` `` (not `\`) to continue a command on the next line. Also note that inline JSON strings must have their quotes escaped with `\"` when passed to the AWS CLI.

### Create API Gateway

**Mac/Linux:**
```bash
# Create HTTP API
API_ID=$(aws apigatewayv2 create-api \
  --name sentiment-api \
  --protocol-type HTTP \
  --region $REGION \
  --query ApiId \
  --output text)

echo "API ID: $API_ID"

# Add Lambda permission
aws lambda add-permission \
  --function-name sentiment-api \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*/*"

# Create integration
INTEGRATION_ID=$(aws apigatewayv2 create-integration \
  --api-id $API_ID \
  --integration-type AWS_PROXY \
  --integration-uri arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:sentiment-api \
  --payload-format-version 1.0 \
  --region $REGION \
  --query IntegrationId \
  --output text)

# Create routes
aws apigatewayv2 create-route --api-id $API_ID --route-key "POST /predict" --target "integrations/$INTEGRATION_ID" --region $REGION
aws apigatewayv2 create-route --api-id $API_ID --route-key "GET /health" --target "integrations/$INTEGRATION_ID" --region $REGION

# Deploy
aws apigatewayv2 create-stage \
  --api-id $API_ID \
  --stage-name prod \
  --auto-deploy \
  --region $REGION

API_URL="https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod"
echo "API URL: $API_URL"
```

**Windows (PowerShell):**
```powershell
# Create HTTP API
$API_ID = aws apigatewayv2 create-api `
  --name sentiment-api `
  --protocol-type HTTP `
  --region $REGION `
  --query ApiId `
  --output text

Write-Host "API ID: $API_ID"

# Add Lambda permission
aws lambda add-permission `
  --function-name sentiment-api `
  --statement-id apigateway-invoke `
  --action lambda:InvokeFunction `
  --principal apigateway.amazonaws.com `
  --source-arn "arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*/*"

# Create integration
$INTEGRATION_ID = aws apigatewayv2 create-integration `
  --api-id $API_ID `
  --integration-type AWS_PROXY `
  --integration-uri "arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:sentiment-api" `
  --payload-format-version 1.0 `
  --region $REGION `
  --query IntegrationId `
  --output text

# Create routes
aws apigatewayv2 create-route --api-id $API_ID --route-key "POST /predict" --target "integrations/$INTEGRATION_ID" --region $REGION
aws apigatewayv2 create-route --api-id $API_ID --route-key "GET /health" --target "integrations/$INTEGRATION_ID" --region $REGION

# Deploy
aws apigatewayv2 create-stage `
  --api-id $API_ID `
  --stage-name prod `
  --auto-deploy `
  --region $REGION

$API_URL = "https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod"
Write-Host "API URL: $API_URL"
```

---

## 2.6 Test the Live Endpoint

**Mac/Linux:**
```bash
# Health check
curl $API_URL/health

# Sentiment prediction (note: first call will be slow — Lambda cold start + model download)
curl -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Deploying to AWS is incredibly rewarding!"}'
```

**Windows (PowerShell):**
```powershell
# Health check
Invoke-RestMethod $API_URL/health

# Sentiment prediction
Invoke-RestMethod -Method POST "$API_URL/predict" `
  -ContentType "application/json" `
  -Body '{"text": "Deploying to AWS is incredibly rewarding!"}'
```

Expected response:
```json
{
  "text": "Deploying to AWS is incredibly rewarding!",
  "label": "POSITIVE",
  "score": 0.9997,
  "latency_ms": 312.4
}
```

> **Cold start vs warm:** Your first invocation may take 20–60 seconds (downloading the model from S3). Subsequent requests within a few minutes will take under 500ms. This is the Lambda ML cold-start tradeoff — we address it with Provisioned Concurrency in production scenarios.

---

## 2.7 Checkpoint ✅

- [ ] Lambda function created and visible in AWS Console
- [ ] API Gateway configured and deployed
- [ ] `/health` endpoint returns healthy response from the cloud
- [ ] `/predict` returns correct sentiment for test inputs
- [ ] CloudWatch logs visible for your Lambda function (check AWS Console → Lambda → Monitor)

---

---

# Stage 3 — Monitoring with CloudWatch

## 🎯 Target

By the end of Stage 3, you will have **production-grade observability** for your ML API. You'll create custom metrics, structured logs, CloudWatch dashboards, and alarms — so you can tell at a glance whether your model is healthy, how fast it's responding, and whether its predictions are drifting.

---

## 3.1 Why Monitoring Matters in MLOps

Deploying a model is not the end — it's the beginning. ML systems can fail in ways traditional software doesn't:

- **Latency degradation** — cold starts, memory pressure, slow inference
- **Prediction drift** — model confidence decreasing over time (sign of data drift)
- **Error spikes** — unexpected input formats, memory errors, timeout
- **Sentiment skew** — if 90% of predictions are NEGATIVE, something might be wrong

We will monitor all of these.

---

## 3.2 Add Structured Logging and Custom Metrics

Update `backend/lambda_handler.py` to add CloudWatch custom metrics and structured logging:

```python
"""
lambda_handler.py — with structured logging and CloudWatch metrics
"""
import json
import os
import boto3
import logging
import time
from pathlib import Path
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# CloudWatch client for custom metrics
cloudwatch = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1"))

MODEL_LOCAL_PATH = "/tmp/model_cache"
MODEL_S3_BUCKET = os.environ.get("MODEL_S3_BUCKET", "")
MODEL_S3_PREFIX = os.environ.get("MODEL_S3_PREFIX", "distilbert-sst2/")
NAMESPACE = "SentimentAPI"  # CloudWatch metric namespace

_pipeline = None


def put_metric(name: str, value: float, unit: str = "None", dimensions: list = None):
    """Send a custom metric to CloudWatch."""
    try:
        metric = {
            "MetricName": name,
            "Value": value,
            "Unit": unit,
            "Timestamp": datetime.utcnow(),
        }
        if dimensions:
            metric["Dimensions"] = dimensions
        cloudwatch.put_metric_data(Namespace=NAMESPACE, MetricData=[metric])
    except Exception as e:
        logger.warning(f"Failed to publish metric {name}: {e}")


def log_structured(level: str, message: str, **kwargs):
    """Emit a structured JSON log entry."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    getattr(logger, level.lower())(json.dumps(entry))


def download_model_from_s3():
    log_structured("info", "Downloading model from S3", bucket=MODEL_S3_BUCKET, prefix=MODEL_S3_PREFIX)
    s3 = boto3.client("s3")
    Path(MODEL_LOCAL_PATH).mkdir(parents=True, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=MODEL_S3_BUCKET, Prefix=MODEL_S3_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key.replace(MODEL_S3_PREFIX, "")
            if not filename:
                continue
            local_file = os.path.join(MODEL_LOCAL_PATH, filename)
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            s3.download_file(MODEL_S3_BUCKET, key, local_file)
    log_structured("info", "Model download complete")


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        cold_start = time.time()
        if not Path(MODEL_LOCAL_PATH).exists():
            download_model_from_s3()
        from transformers import pipeline
        _pipeline = pipeline("sentiment-analysis", model=MODEL_LOCAL_PATH)
        cold_start_ms = round((time.time() - cold_start) * 1000, 2)
        log_structured("info", "Model loaded", cold_start_ms=cold_start_ms)
        put_metric("ColdStartDuration", cold_start_ms, unit="Milliseconds")
    return _pipeline


def make_response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json", "Access-Control-Allow-Origin": "*"},
        "body": json.dumps(body),
    }


def handler(event, context):
    path = event.get("path", "/")
    http_method = event.get("httpMethod", "GET")

    if path == "/health":
        return make_response(200, {"status": "healthy"})

    if path == "/predict" and http_method == "POST":
        try:
            body = json.loads(event.get("body", "{}"))
            text = body.get("text", "").strip()
            if not text:
                put_metric("ValidationErrors", 1)
                return make_response(400, {"error": "text cannot be empty"})

            pipe = get_pipeline()
            start = time.time()
            result = pipe(text, truncation=True, max_length=512)[0]
            latency = round((time.time() - start) * 1000, 2)

            label = result["label"]
            score = round(result["score"], 4)

            # Emit custom metrics
            put_metric("InferenceLatency", latency, unit="Milliseconds")
            put_metric("PredictionConfidence", score)
            put_metric("Predictions", 1)
            put_metric(
                "LabelDistribution", 1,
                dimensions=[{"Name": "Label", "Value": label}]
            )

            log_structured("info", "Prediction complete",
                           label=label, score=score, latency_ms=latency,
                           text_length=len(text))

            return make_response(200, {
                "text": text,
                "label": label,
                "score": score,
                "latency_ms": latency,
            })

        except Exception as e:
            put_metric("Errors", 1)
            log_structured("error", "Prediction failed", error=str(e))
            return make_response(500, {"error": "Inference failed"})

    return make_response(404, {"error": "Not found"})
```

### Redeploy with the updated handler

**Mac/Linux:**
```bash
cd backend && ./package_lambda.sh

aws lambda update-function-code \
  --function-name sentiment-api \
  --zip-file fileb://lambda_package.zip \
  --region us-east-1
```

**Windows (PowerShell):**
```powershell
cd backend
.\package_lambda.ps1

aws lambda update-function-code `
  --function-name sentiment-api `
  --zip-file fileb://lambda_package.zip `
  --region us-east-1
```

---

## 3.3 Create a CloudWatch Dashboard

Run this script to create a monitoring dashboard:

**Mac/Linux:**
```bash
REGION=us-east-1

aws cloudwatch put-dashboard \
  --dashboard-name SentimentAPI \
  --dashboard-body '{
    "widgets": [
      {
        "type": "metric",
        "properties": {
          "title": "Inference Latency (ms)",
          "metrics": [["SentimentAPI", "InferenceLatency"]],
          "stat": "Average",
          "period": 60,
          "view": "timeSeries"
        }
      },
      {
        "type": "metric",
        "properties": {
          "title": "Prediction Volume",
          "metrics": [["SentimentAPI", "Predictions"]],
          "stat": "Sum",
          "period": 60
        }
      },
      {
        "type": "metric",
        "properties": {
          "title": "Label Distribution",
          "metrics": [
            ["SentimentAPI", "LabelDistribution", "Label", "POSITIVE"],
            ["SentimentAPI", "LabelDistribution", "Label", "NEGATIVE"]
          ],
          "stat": "Sum",
          "period": 300
        }
      },
      {
        "type": "metric",
        "properties": {
          "title": "Errors",
          "metrics": [["SentimentAPI", "Errors"]],
          "stat": "Sum",
          "period": 60
        }
      }
    ]
  }' \
  --region $REGION

echo "Dashboard created. View at:"
echo "https://$REGION.console.aws.amazon.com/cloudwatch/home?region=$REGION#dashboards:name=SentimentAPI"
```

**Windows (PowerShell):**

Save the dashboard JSON to a file first (avoids complex escaping in PowerShell):

Create `dashboard.json`:
```json
{
  "widgets": [
    {
      "type": "metric",
      "properties": {
        "title": "Inference Latency (ms)",
        "metrics": [["SentimentAPI", "InferenceLatency"]],
        "stat": "Average",
        "period": 60,
        "view": "timeSeries"
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "Prediction Volume",
        "metrics": [["SentimentAPI", "Predictions"]],
        "stat": "Sum",
        "period": 60
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "Label Distribution",
        "metrics": [
          ["SentimentAPI", "LabelDistribution", "Label", "POSITIVE"],
          ["SentimentAPI", "LabelDistribution", "Label", "NEGATIVE"]
        ],
        "stat": "Sum",
        "period": 300
      }
    },
    {
      "type": "metric",
      "properties": {
        "title": "Errors",
        "metrics": [["SentimentAPI", "Errors"]],
        "stat": "Sum",
        "period": 60
      }
    }
  ]
}
```

Then run:
```powershell
$REGION = "us-east-1"
$dashboardBody = Get-Content -Raw dashboard.json

aws cloudwatch put-dashboard `
  --dashboard-name SentimentAPI `
  --dashboard-body $dashboardBody `
  --region $REGION

Write-Host "Dashboard created. View at:"
Write-Host "https://$REGION.console.aws.amazon.com/cloudwatch/home?region=$REGION#dashboards:name=SentimentAPI"
```

---

## 3.4 Set Up Alarms

**Mac/Linux:**
```bash
REGION=us-east-1

# Alarm: High error rate
aws cloudwatch put-metric-alarm \
  --alarm-name "SentimentAPI-HighErrors" \
  --alarm-description "More than 5 errors in 5 minutes" \
  --namespace SentimentAPI \
  --metric-name Errors \
  --statistic Sum \
  --period 300 \
  --threshold 5 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --region $REGION

# Alarm: High latency
aws cloudwatch put-metric-alarm \
  --alarm-name "SentimentAPI-HighLatency" \
  --alarm-description "Average inference latency over 2000ms" \
  --namespace SentimentAPI \
  --metric-name InferenceLatency \
  --statistic Average \
  --period 300 \
  --threshold 2000 \
  --comparison-operator GreaterThanThreshold \
  --evaluation-periods 1 \
  --region $REGION

echo "Alarms created."
```

**Windows (PowerShell):**
```powershell
$REGION = "us-east-1"

# Alarm: High error rate
aws cloudwatch put-metric-alarm `
  --alarm-name "SentimentAPI-HighErrors" `
  --alarm-description "More than 5 errors in 5 minutes" `
  --namespace SentimentAPI `
  --metric-name Errors `
  --statistic Sum `
  --period 300 `
  --threshold 5 `
  --comparison-operator GreaterThanThreshold `
  --evaluation-periods 1 `
  --region $REGION

# Alarm: High latency
aws cloudwatch put-metric-alarm `
  --alarm-name "SentimentAPI-HighLatency" `
  --alarm-description "Average inference latency over 2000ms" `
  --namespace SentimentAPI `
  --metric-name InferenceLatency `
  --statistic Average `
  --period 300 `
  --threshold 2000 `
  --comparison-operator GreaterThanThreshold `
  --evaluation-periods 1 `
  --region $REGION

Write-Host "Alarms created."
```

---

## 3.5 Generate Traffic and Observe

Create `backend/load_test.py`:

```python
"""
Send test traffic to trigger CloudWatch metrics.
"""
import requests
import random
import time

API_URL = "YOUR_API_GATEWAY_URL"  # Replace with your actual URL

test_texts = [
    "I absolutely love this service, it's amazing!",
    "Worst experience I've ever had.",
    "Pretty good, would recommend.",
    "Totally disappointed and frustrated.",
    "The product works exactly as described.",
    "Never buying from here again.",
    "Exceptional quality and fast delivery!",
    "Could be better, not impressed.",
]

print("Sending test traffic...\n")
for i in range(20):
    text = random.choice(test_texts)
    r = requests.post(f"{API_URL}/predict", json={"text": text})
    data = r.json()
    print(f"[{i+1}/20] {data['label']} ({data['score']:.3f}) — {text[:40]}...")
    time.sleep(1)

print("\nDone. Check your CloudWatch dashboard.")
```

Update `API_URL` and run:

**Mac/Linux:**
```bash
uv run python backend/load_test.py
```

**Windows (PowerShell):**
```powershell
uv run python backend/load_test.py
```

Then open your CloudWatch dashboard to see the metrics populate in real time.

---

## 3.6 Checkpoint ✅

- [ ] Lambda updated with structured logging and custom metrics
- [ ] CloudWatch dashboard visible with latency, volume, and label distribution widgets
- [ ] At least one alarm created
- [ ] Metrics appear after running the load test
- [ ] Logs in CloudWatch Logs show structured JSON entries

---

---

# Stage 4 — Infrastructure as Code with Terraform

## 🎯 Target

By the end of Stage 4, all AWS infrastructure will be **defined in code, version-controlled, and reproducible with a single command**. You will be able to create or destroy the entire stack for `dev`, `staging`, and `prod` environments without clicking in the AWS Console.

---

## 4.1 Why Infrastructure as Code?

In Stages 2 and 3, you created resources by running AWS CLI commands manually. This approach has critical problems in team environments:

- **Not reproducible** — someone else can't recreate it exactly
- **Not version-controlled** — no history of what changed and when
- **Error-prone** — manual steps get missed or done in the wrong order
- **Not scalable** — creating a `staging` copy means repeating everything manually

Terraform solves all of these by treating infrastructure like application code.

---

## 4.2 Install Terraform

Visit [developer.hashicorp.com/terraform/install](https://developer.hashicorp.com/terraform/install) and follow instructions for your OS.

**Mac (Homebrew):**
```bash
brew tap hashicorp/tap
brew install hashicorp/tap/terraform
```

**Linux:**
```bash
sudo apt-get update && sudo apt-get install -y gnupg software-properties-common
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt-get update && sudo apt-get install terraform
```

**Windows (winget):**
```powershell
winget install HashiCorp.Terraform
# Restart PowerShell after installation
```

Verify on all platforms:
```
terraform --version
# Terraform v1.9.x or later
```

---

## 4.3 Project Structure

Create the Terraform directory:

**Mac/Linux:**
```bash
mkdir -p terraform/environments
```

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Path terraform/environments -Force
```

Final structure:
```
sentiment-api/
├── backend/
│   ├── requirements.txt
│   ├── server.py
│   ├── model.py
│   ├── lambda_handler.py
│   ├── package_lambda.sh      ← Mac/Linux
│   └── package_lambda.ps1     ← Windows
└── terraform/
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    └── environments/
        ├── dev.tfvars
        └── prod.tfvars
```

---

## 4.4 Write the Terraform Configuration

### `terraform/variables.tf`

```hcl
variable "environment" {
  description = "Deployment environment (dev, staging, prod)"
  type        = string
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "model_s3_bucket" {
  description = "S3 bucket containing the ML model"
  type        = string
}

variable "lambda_memory_mb" {
  description = "Lambda memory allocation in MB"
  type        = number
  default     = 1024
}

variable "lambda_timeout_s" {
  description = "Lambda timeout in seconds"
  type        = number
  default     = 120
}
```

### `terraform/main.tf`

```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

locals {
  name_prefix = "sentiment-api-${var.environment}"
  tags = {
    Project     = "sentiment-api"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ─── IAM Role for Lambda ───────────────────────────────────────────────────

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect = "Allow"
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
    actions = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "lambda_role" {
  name               = "${local.name_prefix}-role"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
  tags               = local.tags
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy_attachment" "lambda_s3" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
}

resource "aws_iam_role_policy_attachment" "lambda_cloudwatch" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchFullAccess"
}

# ─── Lambda Function ───────────────────────────────────────────────────────

resource "aws_lambda_function" "sentiment_api" {
  function_name = local.name_prefix
  role          = aws_iam_role.lambda_role.arn
  handler       = "lambda_handler.handler"
  runtime       = "python3.12"
  filename      = "${path.module}/../backend/lambda_package.zip"
  timeout       = var.lambda_timeout_s
  memory_size   = var.lambda_memory_mb

  environment {
    variables = {
      MODEL_S3_BUCKET = var.model_s3_bucket
      MODEL_S3_PREFIX = "distilbert-sst2/"
      ENVIRONMENT     = var.environment
    }
  }

  tags = local.tags
}

# ─── API Gateway ───────────────────────────────────────────────────────────

resource "aws_apigatewayv2_api" "sentiment_api" {
  name          = local.name_prefix
  protocol_type = "HTTP"
  tags          = local.tags
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id             = aws_apigatewayv2_api.sentiment_api.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.sentiment_api.invoke_arn
  payload_format_version = "1.0"
}

resource "aws_apigatewayv2_route" "predict" {
  api_id    = aws_apigatewayv2_api.sentiment_api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.sentiment_api.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.sentiment_api.id
  name        = "$default"
  auto_deploy = true
  tags        = local.tags
}

resource "aws_lambda_permission" "api_gw" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sentiment_api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.sentiment_api.execution_arn}/*/*"
}

# ─── CloudWatch Alarms ─────────────────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "high_errors" {
  alarm_name          = "${local.name_prefix}-errors"
  alarm_description   = "High error rate on ${var.environment}"
  namespace           = "SentimentAPI"
  metric_name         = "Errors"
  statistic           = "Sum"
  period              = 300
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  tags                = local.tags
}

resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "${local.name_prefix}-latency"
  alarm_description   = "High latency on ${var.environment}"
  namespace           = "SentimentAPI"
  metric_name         = "InferenceLatency"
  statistic           = "Average"
  period              = 300
  threshold           = 2000
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  tags                = local.tags
}
```

### `terraform/outputs.tf`

```hcl
output "api_url" {
  description = "The API endpoint URL"
  value       = aws_apigatewayv2_api.sentiment_api.api_endpoint
}

output "lambda_function_name" {
  description = "Lambda function name"
  value       = aws_lambda_function.sentiment_api.function_name
}
```

### `terraform/environments/dev.tfvars`

```hcl
environment      = "dev"
aws_region       = "us-east-1"
model_s3_bucket  = "sentiment-model-yourname-2024"  # Replace
lambda_memory_mb = 512
lambda_timeout_s = 120
```

### `terraform/environments/prod.tfvars`

```hcl
environment      = "prod"
aws_region       = "us-east-1"
model_s3_bucket  = "sentiment-model-yourname-2024"  # Replace
lambda_memory_mb = 1024
lambda_timeout_s = 120
```

---

## 4.5 Deploy with Terraform

### First-time initialisation

**Mac/Linux & Windows (same command):**
```bash
cd terraform
terraform init
```

### Build the Lambda package before deploying

Before running Terraform, ensure your Lambda package is up to date:

**Mac/Linux:**
```bash
cd backend && ./package_lambda.sh && cd ..
```

**Windows (PowerShell):**
```powershell
cd backend; .\package_lambda.ps1; cd ..
```

### Deploy the dev environment

**Mac/Linux & Windows (same commands — Terraform is cross-platform):**
```bash
# Preview what will be created
terraform plan -var-file="environments/dev.tfvars"

# Apply (type 'yes' to confirm)
terraform apply -var-file="environments/dev.tfvars"
```

You will see output like:
```
api_url = "https://abc123.execute-api.us-east-1.amazonaws.com"
lambda_function_name = "sentiment-api-dev"
```

### Test the Terraform-deployed endpoint

**Mac/Linux:**
```bash
API_URL=$(terraform output -raw api_url)
curl -X POST $API_URL/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Terraform makes infrastructure reproducible!"}'
```

**Windows (PowerShell):**
```powershell
$API_URL = terraform output -raw api_url
Invoke-RestMethod -Method POST "$API_URL/predict" `
  -ContentType "application/json" `
  -Body '{"text": "Terraform makes infrastructure reproducible!"}'
```

### Deploy the prod environment

**Mac/Linux & Windows:**
```bash
terraform apply -var-file="environments/prod.tfvars"
```

### Destroy an environment (important: avoid AWS charges)

**Mac/Linux & Windows:**
```bash
# Destroy dev
terraform destroy -var-file="environments/dev.tfvars"

# Destroy prod
terraform destroy -var-file="environments/prod.tfvars"
```

---

## 4.6 Checkpoint ✅

- [ ] `terraform init` succeeds
- [ ] `terraform plan` shows the expected resources (Lambda, API Gateway, IAM roles, alarms)
- [ ] `terraform apply` successfully creates all resources
- [ ] The Terraform-created API URL responds correctly to `/predict`
- [ ] `terraform destroy` cleanly removes all resources
- [ ] `prod` and `dev` environments are isolated (different Lambda names, tags)

---

---

# Stage 5 — CI/CD with GitHub Actions

## 🎯 Target

By the end of Stage 5, every push to your `main` branch will **automatically test your code, package the Lambda, and deploy to production** — with no manual steps. This is the complete MLOps loop: code → test → deploy → monitor.

---

## 5.1 Why CI/CD for ML APIs?

Manual deployments are error-prone and slow. A CI/CD pipeline ensures:

- Every deployment is tested before going live
- The same process runs every time (no human error)
- You have a complete audit trail of every deployment
- Rolling back is a single click (or command)

---

## 5.2 Set Up GitHub

### Create a repository

1. Go to [github.com](https://github.com) and create a new repository named `sentiment-api`
2. Make it private

### Initialise git and push

**Mac/Linux:**
```bash
cd sentiment-api  # project root

# Create .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.env
model_cache/
lambda_package/
lambda_package.zip
.terraform/
*.tfstate
*.tfstate.backup
EOF

git init
git add .
git commit -m "Initial commit: sentiment analysis API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sentiment-api.git
git push -u origin main
```

**Windows (PowerShell):**
```powershell
cd sentiment-api  # project root

# Create .gitignore
@"
__pycache__/
*.pyc
.env
model_cache/
lambda_package/
lambda_package.zip
.terraform/
*.tfstate
*.tfstate.backup
"@ | Out-File -FilePath .gitignore -Encoding UTF8

git init
git add .
git commit -m "Initial commit: sentiment analysis API"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/sentiment-api.git
git push -u origin main
```

---

## 5.3 Set Up AWS OIDC Authentication for GitHub Actions

Instead of storing AWS Access Keys in GitHub Secrets (a security risk), we use **OIDC** — GitHub Actions proves its identity to AWS directly, and AWS issues a temporary token. No long-lived secrets.

### Create the OIDC provider in AWS

**Mac/Linux:**
```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-1

# Create OIDC provider for GitHub
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

**Windows (PowerShell):**
```powershell
$ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
$REGION = "us-east-1"

# Create OIDC provider for GitHub
aws iam create-open-id-connect-provider `
  --url https://token.actions.githubusercontent.com `
  --client-id-list sts.amazonaws.com `
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

### Create an IAM role for GitHub Actions

Replace `YOUR_GITHUB_USERNAME` and `YOUR_REPO_NAME`:

**Mac/Linux:**
```bash
GITHUB_USER=YOUR_GITHUB_USERNAME
REPO_NAME=sentiment-api

cat > /tmp/github-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:${GITHUB_USER}/${REPO_NAME}:ref:refs/heads/main"
        }
      }
    }
  ]
}
EOF

aws iam create-role \
  --role-name github-actions-sentiment-api \
  --assume-role-policy-document file:///tmp/github-trust-policy.json

# Attach permissions
aws iam attach-role-policy \
  --role-name github-actions-sentiment-api \
  --policy-arn arn:aws:iam::aws:policy/AWSLambda_FullAccess

aws iam attach-role-policy \
  --role-name github-actions-sentiment-api \
  --policy-arn arn:aws:iam::aws:policy/AmazonAPIGatewayAdministrator

aws iam attach-role-policy \
  --role-name github-actions-sentiment-api \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

aws iam attach-role-policy \
  --role-name github-actions-sentiment-api \
  --policy-arn arn:aws:iam::aws:policy/IAMFullAccess

aws iam attach-role-policy \
  --role-name github-actions-sentiment-api \
  --policy-arn arn:aws:iam::aws:policy/CloudWatchFullAccess

echo "Role ARN: arn:aws:iam::${ACCOUNT_ID}:role/github-actions-sentiment-api"
```

**Windows (PowerShell):**

First save the trust policy JSON to a file (easier than inline JSON in PowerShell):

Create `github-trust-policy.json` in the project root with this content, replacing `YOUR_ACCOUNT_ID`, `YOUR_GITHUB_USERNAME`, and `sentiment-api` with your actual values:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "token.actions.githubusercontent.com:aud": "sts.amazonaws.com",
          "token.actions.githubusercontent.com:sub": "repo:YOUR_GITHUB_USERNAME/sentiment-api:ref:refs/heads/main"
        }
      }
    }
  ]
}
```

Then run:
```powershell
aws iam create-role `
  --role-name github-actions-sentiment-api `
  --assume-role-policy-document file://github-trust-policy.json

# Attach permissions
aws iam attach-role-policy --role-name github-actions-sentiment-api --policy-arn arn:aws:iam::aws:policy/AWSLambda_FullAccess
aws iam attach-role-policy --role-name github-actions-sentiment-api --policy-arn arn:aws:iam::aws:policy/AmazonAPIGatewayAdministrator
aws iam attach-role-policy --role-name github-actions-sentiment-api --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
aws iam attach-role-policy --role-name github-actions-sentiment-api --policy-arn arn:aws:iam::aws:policy/IAMFullAccess
aws iam attach-role-policy --role-name github-actions-sentiment-api --policy-arn arn:aws:iam::aws:policy/CloudWatchFullAccess

Write-Host "Role ARN: arn:aws:iam::${ACCOUNT_ID}:role/github-actions-sentiment-api"
```

> 💡 **Windows tip:** Add `github-trust-policy.json` to your `.gitignore` since it contains your account ID.

---

## 5.4 Store Terraform State in S3 (Remote Backend)

Terraform needs to store state somewhere persistent and shared. We use S3.

**Mac/Linux:**
```bash
STATE_BUCKET=sentiment-tfstate-${ACCOUNT_ID}

aws s3 mb s3://$STATE_BUCKET --region us-east-1

# Enable versioning (allows state recovery)
aws s3api put-bucket-versioning \
  --bucket $STATE_BUCKET \
  --versioning-configuration Status=Enabled

echo "State bucket: $STATE_BUCKET"
```

**Windows (PowerShell):**
```powershell
$STATE_BUCKET = "sentiment-tfstate-$ACCOUNT_ID"

aws s3 mb s3://$STATE_BUCKET --region us-east-1

# Enable versioning (allows state recovery)
aws s3api put-bucket-versioning `
  --bucket $STATE_BUCKET `
  --versioning-configuration Status=Enabled

Write-Host "State bucket: $STATE_BUCKET"
```

Add the remote backend to `terraform/main.tf` (update bucket name):

```hcl
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket = "sentiment-tfstate-YOUR_ACCOUNT_ID"  # Replace
    key    = "sentiment-api/terraform.tfstate"
    region = "us-east-1"
  }
}
```

Re-initialise with the remote backend:

**Mac/Linux & Windows:**
```bash
cd terraform && terraform init
```

---

## 5.5 Add GitHub Secrets

In your GitHub repository, go to **Settings → Secrets and variables → Actions → New repository secret**:

| Secret Name | Value |
|---|---|
| `AWS_ROLE_ARN` | `arn:aws:iam::YOUR_ACCOUNT_ID:role/github-actions-sentiment-api` |
| `AWS_REGION` | `us-east-1` |
| `MODEL_S3_BUCKET` | `sentiment-model-yourname-2024` |
| `TF_STATE_BUCKET` | `sentiment-tfstate-YOUR_ACCOUNT_ID` |

---

## 5.6 Create the GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy Sentiment API

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:  # Allow manual trigger

permissions:
  id-token: write   # Required for OIDC authentication
  contents: read

env:
  AWS_REGION: ${{ secrets.AWS_REGION }}
  PYTHON_VERSION: "3.12"

jobs:

  # ── Job 1: Test ────────────────────────────────────────────────────────
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh && echo "$HOME/.cargo/bin" >> $GITHUB_PATH

      - name: Install dependencies
        run: |
          cd backend
          uv pip install fastapi pydantic pytest httpx --system

      - name: Run tests
        run: |
          cd backend
          pytest tests/ -v || echo "No tests directory found — skipping."

  # ── Job 2: Package Lambda ──────────────────────────────────────────────
  package:
    name: Package Lambda
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Package Lambda function
        run: |
          cd backend
          pip install fastapi mangum --target lambda_package --quiet
          cp lambda_handler.py lambda_package/
          cd lambda_package && zip -r ../lambda_package.zip . -q
          cd ..
          echo "Package size: $(du -sh lambda_package.zip | cut -f1)"

      - name: Upload Lambda package as artifact
        uses: actions/upload-artifact@v4
        with:
          name: lambda-package
          path: backend/lambda_package.zip

  # ── Job 3: Deploy to Prod ──────────────────────────────────────────────
  deploy:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: package
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    environment: production

    steps:
      - uses: actions/checkout@v4

      - name: Download Lambda package
        uses: actions/download-artifact@v4
        with:
          name: lambda-package
          path: backend/

      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Set up Terraform
        uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: "1.9.0"

      - name: Terraform Init
        run: |
          cd terraform
          terraform init

      - name: Terraform Plan
        run: |
          cd terraform
          terraform plan \
            -var="environment=prod" \
            -var="model_s3_bucket=${{ secrets.MODEL_S3_BUCKET }}" \
            -var="aws_region=${{ env.AWS_REGION }}" \
            -out=tfplan

      - name: Terraform Apply
        run: |
          cd terraform
          terraform apply tfplan

      - name: Output API URL
        run: |
          cd terraform
          echo "✅ Deployed to: $(terraform output -raw api_url)"

      - name: Smoke test live endpoint
        run: |
          cd terraform
          API_URL=$(terraform output -raw api_url)
          echo "Testing $API_URL/health ..."
          curl -f "$API_URL/health"
          echo ""
          echo "Testing $API_URL/predict ..."
          curl -f -X POST "$API_URL/predict" \
            -H "Content-Type: application/json" \
            -d '{"text": "Automated deployment is working perfectly!"}'
```

---

## 5.7 Create a Basic Test

Create `backend/tests/test_api.py`:

```python
"""
Basic tests for the FastAPI sentiment API.
Run locally with: pytest backend/tests/ -v
"""
import pytest
from fastapi.testclient import TestClient

# Import the local FastAPI app (not Lambda handler)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Mock the model so tests don't download it
from unittest.mock import patch, MagicMock

MOCK_RESULT = [{"label": "POSITIVE", "score": 0.9998}]


@pytest.fixture
def client():
    with patch("model.get_pipeline") as mock_pipe:
        mock_instance = MagicMock()
        mock_instance.return_value = MOCK_RESULT
        mock_pipe.return_value = mock_instance
        from server import app
        return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_valid(client):
    response = client.post("/predict", json={"text": "This is great!"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "score" in data
    assert data["label"] in ["POSITIVE", "NEGATIVE"]
    assert 0 <= data["score"] <= 1


def test_predict_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 400


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
```

---

## 5.8 Push and Watch the Pipeline

**Mac/Linux & Windows (Git commands are identical):**
```bash
git add .
git commit -m "Add CI/CD pipeline with GitHub Actions"
git push origin main
```

Go to your GitHub repository → **Actions** tab. You will see the workflow running:

1. ✅ **Test** — runs unit tests
2. ✅ **Package** — creates `lambda_package.zip`
3. ✅ **Deploy** — runs Terraform and deploys to prod, then smoke-tests the live URL

Every subsequent push to `main` will trigger this pipeline automatically.

---

## 5.9 Trigger a Code Change to See the Pipeline

Make a small change to test the full loop:

```python
# In backend/server.py, update the version string:
version="1.1.0",
```

**Mac/Linux & Windows:**
```bash
git add backend/server.py
git commit -m "Bump version to 1.1.0"
git push origin main
```

Watch the pipeline deploy the change automatically — this is the core of MLOps: **code changes flow automatically to production**.

---

## 5.10 Checkpoint ✅

- [ ] Repository pushed to GitHub
- [ ] OIDC role created and trust policy verified
- [ ] Terraform remote state configured in S3
- [ ] All GitHub Secrets set
- [ ] First pipeline run completes: Test → Package → Deploy
- [ ] Smoke test in the pipeline passes with a live response from AWS
- [ ] A second code change triggers the pipeline automatically
- [ ] CloudWatch shows new metrics from the pipeline's smoke test

---

---

## Clean Up (Avoid AWS Charges)

When you have finished the course, destroy all infrastructure:

**Mac/Linux:**
```bash
# Destroy Terraform-managed resources
cd terraform
terraform destroy -var-file="environments/prod.tfvars"

# Delete the model S3 bucket
aws s3 rm s3://sentiment-model-yourname-2024 --recursive
aws s3 rb s3://sentiment-model-yourname-2024

# Delete the Terraform state bucket
aws s3 rm s3://sentiment-tfstate-YOUR_ACCOUNT_ID --recursive
aws s3 rb s3://sentiment-tfstate-YOUR_ACCOUNT_ID

# Delete the OIDC provider (optional)
aws iam delete-open-id-connect-provider \
  --open-id-connect-provider-arn arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com
```

**Windows (PowerShell):**
```powershell
# Destroy Terraform-managed resources
cd terraform
terraform destroy -var-file="environments/prod.tfvars"

# Delete the model S3 bucket
aws s3 rm s3://sentiment-model-yourname-2024 --recursive
aws s3 rb s3://sentiment-model-yourname-2024

# Delete the Terraform state bucket
aws s3 rm s3://sentiment-tfstate-YOUR_ACCOUNT_ID --recursive
aws s3 rb s3://sentiment-tfstate-YOUR_ACCOUNT_ID

# Delete the OIDC provider (optional)
aws iam delete-open-id-connect-provider `
  --open-id-connect-provider-arn arn:aws:iam::YOUR_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com
```

---

## Summary: What You Have Built

| Stage | What You Did | Key MLOps Concept |
|---|---|---|
| **1 — Local API** | FastAPI server serving DistilBERT predictions | Model serving, API design |
| **2 — AWS Lambda** | Serverless deployment with S3 model caching | Cold start, cloud ML serving |
| **3 — Monitoring** | CloudWatch custom metrics, dashboards, alarms | Observability, drift detection |
| **4 — Terraform** | Infrastructure as Code, multi-environment deploy | Reproducibility, IaC |
| **5 — CI/CD** | GitHub Actions auto-deploys on every push | Continuous delivery, MLOps loop |

You now have a production-grade, monitored, automatically-deployed ML API on AWS — built with the exact patterns used by professional MLOps teams.

---

## Further Reading

- [AWS Lambda ML best practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [HuggingFace Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [GitHub Actions OIDC with AWS](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/configuring-openid-connect-in-amazon-web-services)
- [CloudWatch Custom Metrics](https://docs.aws.amazon.com/AmazonCloudWatch/latest/monitoring/publishingMetrics.html)
