# Deploy a Python Web App to AWS Lambda — Step-by-Step Guide

---

## What You'll Build

A **Smart Bookmark Manager** — a lightweight web app where you can:

- Save URLs with a title and tags
- List and search your saved bookmarks
- Delete bookmarks
- Check a live health status endpoint

Everything runs in a single Python (FastAPI) container. The frontend is a single HTML file served by FastAPI itself — no Node.js, no build step, no external auth service, no paid AI API.

**Skills you'll practise:** Docker multi-stage builds, AWS account setup, IAM users & permissions, Amazon ECR, AWS Lambda container images, Lambda Function URLs, CloudWatch monitoring, cost control.

---

## Prerequisites

| Tool | Check |
|------|-------|
| Python 3.11+ | `python --version` |
| Docker Desktop | `docker --version` |
| AWS CLI v2 | `aws --version` |
| A text editor | VS Code recommended |

---

## Part 1: Set Up Your AWS Account

### Step 1: Create an AWS Account

1. Visit [aws.amazon.com](https://aws.amazon.com) and click **Create an AWS Account**.
2. Enter your email address and choose a password.
3. Select **Personal** account type — do **not** select the "free tier only" sandbox; you want full access (no subscription cost, just enter payment details).
4. Enter payment information (required by AWS; expect $0 for this course).
5. Verify your phone number via SMS.
6. Choose **Basic Support – Free**.

You now have an AWS **root account**. Treat it like your building's master key — powerful and dangerous. Use it only for one-time setup.

### Step 2: Secure the Root Account with MFA

1. Sign in, click your account name (top right) → **Security credentials**.
2. Under **Multi-factor authentication (MFA)**, click **Assign MFA device**.
3. Name: `root-mfa` → select **Authenticator app**.
4. Scan the QR code with Google Authenticator or Authy.
5. Enter two consecutive codes → **Add MFA**.

### Step 3: Set Up Budget Alerts — Do This Before Anything Else

AWS charges for what you provision. Lambda's free tier is generous, but it pays to have guardrails.

1. In the AWS Console, search **Billing** → **Billing and Cost Management**.
2. Click **Budgets** in the left menu → **Create budget**.
3. Choose **Use a template (simplified)** → **Monthly cost budget**.
4. Create three budgets — repeat the process for each:

| Budget Name | Amount |
|-------------|--------|
| `early-warning` | $1 |
| `caution-budget` | $5 |
| `stop-budget` | $10 |

Use your email address for all three. If you hit $10, stop and review what's running.

### Step 4: Create an IAM User for Daily Work

Never use your root account for day-to-day work.

1. Search **IAM** → **Users** → **Create user**.
2. Username: `devuser`
3. Check **Provide user access to the AWS Management Console**.
4. Select **I want to create an IAM user**.
5. Choose **Custom password** and set a strong password.
6. Uncheck **Users must create a new password at next sign-in**.
7. Click **Next**.

### Step 5: Create a Permission Group and Attach It

1. On the permissions page, click **Add user to group** → **Create group**.
2. Group name: `AppDeployAccess`
3. Search for and check these policies:
   - `AWSLambda_FullAccess` — deploy and manage Lambda functions
   - `AmazonEC2ContainerRegistryFullAccess` — push Docker images to ECR
   - `CloudWatchLogsFullAccess` — view application logs
   - `IAMUserChangePassword` — manage own password
   - `IAMFullAccess` — needed so Lambda can create its own execution role
4. Click **Create user group**.
5. Select the `AppDeployAccess` group → **Next** → **Create user**.
6. **Important:** Click **Download .csv file** and save it somewhere safe.

### Step 6: Sign In as the IAM User

1. Sign out of the root account.
2. Open the sign-in URL from the CSV (format: `https://<account-id>.signin.aws.amazon.com/console`).
3. Sign in with username `devuser` and the password you set.

**Checkpoint:** You should see `devuser @ <Account-ID>` in the top right corner.

---

## Part 2: Build the Application

### Project Structure

Create a folder called `bookmark-app` and set it up like this:

```
bookmark-app/
├── main.py           # FastAPI app (API + serves frontend)
├── requirements.txt  # Python dependencies
├── static/
│   └── index.html    # Single-page frontend
├── Dockerfile
└── .dockerignore
```

### Step 1: Write the Backend (`main.py`)

```python
import os
import uuid
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Bookmark Manager")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory store (good enough for a learning project) ---
bookmarks: dict[str, dict] = {}


class Bookmark(BaseModel):
    url: str
    title: str
    tags: Optional[str] = ""


@app.get("/health")
def health():
    return {"status": "healthy", "bookmark_count": len(bookmarks)}


@app.get("/api/bookmarks")
def list_bookmarks(q: Optional[str] = None):
    """Return all bookmarks, or filter by search query."""
    items = list(bookmarks.values())
    if q:
        q_lower = q.lower()
        items = [
            b for b in items
            if q_lower in b["title"].lower()
            or q_lower in b["url"].lower()
            or q_lower in b.get("tags", "").lower()
        ]
    return items


@app.post("/api/bookmarks", status_code=201)
def add_bookmark(bookmark: Bookmark):
    """Add a new bookmark."""
    bm_id = str(uuid.uuid4())[:8]
    bookmarks[bm_id] = {
        "id": bm_id,
        "url": bookmark.url,
        "title": bookmark.title,
        "tags": bookmark.tags,
    }
    return bookmarks[bm_id]


@app.delete("/api/bookmarks/{bm_id}")
def delete_bookmark(bm_id: str):
    """Delete a bookmark by ID."""
    if bm_id not in bookmarks:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    del bookmarks[bm_id]
    return {"deleted": bm_id}


# --- Serve the frontend (must be LAST) ---
static_dir = Path("static")
if static_dir.exists():
    @app.get("/")
    def serve_root():
        return FileResponse(static_dir / "index.html")

    app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

### Step 2: Write the Frontend (`static/index.html`)

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Bookmark Manager</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; background: #f5f7fa; }
    h1 { color: #2d3748; }
    .card { background: white; border-radius: 10px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
    input, button { padding: 8px 14px; border-radius: 6px; border: 1px solid #cbd5e0; font-size: 14px; }
    input { width: 100%; box-sizing: border-box; margin-bottom: 8px; }
    button { background: #4f6ef7; color: white; border: none; cursor: pointer; }
    button:hover { background: #3b5bdb; }
    .del-btn { background: #e53e3e; padding: 4px 10px; font-size: 12px; }
    .del-btn:hover { background: #c53030; }
    .bm { display: flex; justify-content: space-between; align-items: center; padding: 10px 0; border-bottom: 1px solid #eee; }
    .bm:last-child { border-bottom: none; }
    .bm-info a { font-weight: 600; color: #4f6ef7; text-decoration: none; }
    .bm-info a:hover { text-decoration: underline; }
    .tags { font-size: 12px; color: #718096; margin-top: 2px; }
    #status { font-size: 13px; color: #48bb78; margin-bottom: 8px; }
  </style>
</head>
<body>
  <h1>📑 Bookmark Manager</h1>
  <div id="status">Loading health status...</div>

  <div class="card">
    <h3>Add Bookmark</h3>
    <input id="url"   placeholder="URL (https://...)" />
    <input id="title" placeholder="Title" />
    <input id="tags"  placeholder="Tags (comma separated, optional)" />
    <button onclick="addBookmark()">Save</button>
  </div>

  <div class="card">
    <h3>My Bookmarks</h3>
    <input id="search" placeholder="Search..." oninput="loadBookmarks()" />
    <div id="list">Loading...</div>
  </div>

  <script>
    const API = '';   // same origin

    async function loadHealth() {
      const r = await fetch(`${API}/health`);
      const d = await r.json();
      document.getElementById('status').textContent =
        `✅ Service healthy · ${d.bookmark_count} bookmark(s) saved`;
    }

    async function loadBookmarks() {
      const q = document.getElementById('search').value;
      const r = await fetch(`${API}/api/bookmarks?q=${encodeURIComponent(q)}`);
      const items = await r.json();
      const list = document.getElementById('list');
      if (!items.length) { list.innerHTML = '<p style="color:#718096">No bookmarks yet.</p>'; return; }
      list.innerHTML = items.map(b => `
        <div class="bm">
          <div class="bm-info">
            <a href="${b.url}" target="_blank">${b.title}</a>
            ${b.tags ? `<div class="tags">🏷 ${b.tags}</div>` : ''}
          </div>
          <button class="del-btn" onclick="deleteBookmark('${b.id}')">Delete</button>
        </div>
      `).join('');
    }

    async function addBookmark() {
      const url   = document.getElementById('url').value.trim();
      const title = document.getElementById('title').value.trim();
      const tags  = document.getElementById('tags').value.trim();
      if (!url || !title) { alert('URL and Title are required'); return; }
      await fetch(`${API}/api/bookmarks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, title, tags })
      });
      ['url','title','tags'].forEach(id => document.getElementById(id).value = '');
      loadBookmarks();
      loadHealth();
    }

    async function deleteBookmark(id) {
      await fetch(`${API}/api/bookmarks/${id}`, { method: 'DELETE' });
      loadBookmarks();
      loadHealth();
    }

    loadHealth();
    loadBookmarks();
  </script>
</body>
</html>
```

### Step 3: Write `requirements.txt`

```
fastapi
uvicorn[standard]
pydantic
```

---

## Part 3: Create the Dockerfile

### Step 1: Write the Dockerfile

Create `Dockerfile` in the project root:

```dockerfile
# ─── Single-stage build (no Node.js needed — frontend is plain HTML) ───────
FROM python:3.12-slim

WORKDIR /app

# ── Lambda Web Adapter ──────────────────────────────────────────────────────
# Drops a binary into /opt/extensions. It's inert when you run the container
# locally with `docker run`; it only activates when Lambda invokes the image.
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.0.0 \
     /lambda-adapter /opt/extensions/lambda-adapter

# Tell the adapter which port FastAPI listens on
ENV PORT=8000
# Enable streaming responses (needed for any streaming endpoint)
ENV AWS_LWA_INVOKE_MODE=response_stream
# ── end of Lambda Web Adapter section ──────────────────────────────────────

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and frontend
COPY main.py .
COPY static/ static/

# Health check (used locally; Lambda does not invoke this)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 2: Write `.dockerignore`

```
__pycache__
*.pyc
*.pyo
.env
.git
.gitignore
README.md
.DS_Store
*.log
```

---

## Part 4: Install Docker Desktop

1. Visit [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop).
2. Download the version for your OS (Apple Silicon or Intel for Mac; Windows 10/11 for Windows).
3. Run the installer. On Windows, accept the WSL2 prompts.
4. Start Docker Desktop (whale icon in menu bar / system tray).

**Verify Docker works:**

```bash
docker --version         # e.g. Docker version 26.x.x
docker run hello-world   # should print "Hello from Docker!"
```

**Checkpoint:** Docker Desktop is running and `hello-world` succeeded.

---

## Part 5: Build and Test Locally

### Step 1: Build the Image

Run this from inside the `bookmark-app/` folder:

```bash
docker build -t bookmark-app .
```

This takes 1–2 minutes the first time (downloading the Python base image).

> **Apple Silicon (M1/M2/M3/M4) note:** For local testing you can build natively. Before pushing to ECR you will rebuild with `--platform linux/amd64` — Lambda runs AMD64 by default.

### Step 2: Run the Container Locally

```bash
docker run -p 8000:8000 bookmark-app
```

### Step 3: Test the App

1. Open your browser at [http://localhost:8000](http://localhost:8000) — you should see the Bookmark Manager UI.
2. Open [http://localhost:8000/health](http://localhost:8000/health) — you should see `{"status":"healthy","bookmark_count":0}`.
3. Open [http://localhost:8000/docs](http://localhost:8000/docs) — FastAPI's auto-generated API explorer.
4. Add a couple of bookmarks via the UI. Search for one. Delete one.

**To stop:** Press `Ctrl+C` in the terminal.

**Checkpoint:** The app works fully in a local Docker container.

---

## Part 6: Configure the AWS CLI

### Step 1: Create CLI Access Keys

1. In the AWS Console, go to **IAM** → **Users** → click `devuser`.
2. Click **Security credentials** tab → **Create access key**.
3. Select **Command Line Interface (CLI)** → check the confirmation → **Next**.
4. Description: `cli-access` → **Create access key**.
5. **Critical:** Copy or download both:
   - Access key ID (e.g. `AKIAIOSFODNN7EXAMPLE`)
   - Secret access key

### Step 2: Install the AWS CLI

- **Mac:** `brew install awscli` or download from [aws.amazon.com/cli](https://aws.amazon.com/cli/)
- **Windows:** Download the MSI installer from [aws.amazon.com/cli](https://aws.amazon.com/cli/)

**Verify:**
```bash
aws --version   # e.g. aws-cli/2.x.x
```

### Step 3: Configure the CLI

```bash
aws configure
```

Enter when prompted:

| Prompt | Value |
|--------|-------|
| AWS Access Key ID | (paste your key) |
| AWS Secret Access Key | (paste your secret) |
| Default region name | Choose the closest: `us-east-1` (US East), `us-west-2` (US West), `eu-west-1` (Europe), `ap-southeast-1` (Asia) |
| Default output format | `json` |

> **Remember your region.** Your ECR repository, Lambda function, and AWS CLI must all use the same region throughout this guide.

**Verify:**
```bash
aws sts get-caller-identity
```

You should see your account ID, user ID, and ARN — confirming the CLI is authenticated.

---

## Part 7: Push the Image to Amazon ECR

**ECR (Elastic Container Registry)** is AWS's private Docker image registry — think of it as GitHub for your Docker images.

### Step 1: Create an ECR Repository

1. In the AWS Console, search **ECR** and click **Get started** (or **Create repository**).
2. Confirm the region (top right) matches your CLI region.
3. Settings:
   - Visibility: **Private**
   - Repository name: `bookmark-app`
   - Leave all other settings as default.
4. Click **Create repository**.

### Step 2: Set Environment Variables

Before running the push commands, set two variables so the commands below work without editing:

**Mac/Linux:**
```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=$(aws configure get region)
echo "Account: $AWS_ACCOUNT_ID  Region: $AWS_REGION"
```

**Windows PowerShell:**
```powershell
$env:AWS_ACCOUNT_ID = (aws sts get-caller-identity --query Account --output text)
$env:AWS_REGION     = (aws configure get region)
Write-Host "Account: $env:AWS_ACCOUNT_ID  Region: $env:AWS_REGION"
```

### Step 3: Authenticate Docker to ECR

**Mac/Linux:**
```bash
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

**Windows PowerShell:**
```powershell
aws ecr get-login-password --region $env:AWS_REGION `
  | docker login --username AWS --password-stdin `
    "$env:AWS_ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com"
```

You should see `Login Succeeded`.

### Step 4: Build for Linux/AMD64, Tag, and Push

> **Apple Silicon users:** The `--platform linux/amd64` flag is **critical**. Without it, Lambda will fail with an "exec format error" because Lambda runs AMD64 by default.

**Mac/Linux:**
```bash
# 1. Build for AMD64
docker build --platform linux/amd64 -t bookmark-app .

# 2. Tag for ECR
docker tag bookmark-app:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/bookmark-app:latest

# 3. Push to ECR
docker push \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/bookmark-app:latest
```

**Windows PowerShell:**
```powershell
# 1. Build for AMD64
docker build --platform linux/amd64 -t bookmark-app .

# 2. Tag for ECR
docker tag bookmark-app:latest `
  "$env:AWS_ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/bookmark-app:latest"

# 3. Push to ECR
docker push `
  "$env:AWS_ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/bookmark-app:latest"
```

The push takes 2–5 minutes depending on your internet speed.

**Checkpoint:** In the ECR console, click your `bookmark-app` repository — you should see an image tagged `latest`.

---

## Part 8: Deploy to AWS Lambda

AWS Lambda runs your container on demand. You pay only when it receives a request. With the free tier (1 million requests/month, 400,000 GB-seconds/month), expect **$0/month** for course usage.

### Step 1: Open Lambda

1. In the AWS Console, search **Lambda**.
2. Confirm the region (top right) matches the region you used for ECR.
3. Click **Create function**.

### Step 2: Configure the Function

1. Select **Container image** (not "Author from scratch").
2. **Function name:** `bookmark-app`
3. **Container image URI:** click **Browse images** → select repository `bookmark-app` → tag `latest` → **Select image**.
4. Click **Create function**.

Lambda takes 30–60 seconds to provision. Wait until the function page fully loads.

### Step 3: Tune Memory and Timeout

1. On the function page, click the **Configuration** tab.
2. Click **General configuration** (left sidebar) → **Edit**.
3. Set:
   - **Memory:** `512 MB` (plenty for FastAPI + static files)
   - **Timeout:** `30 sec` (no streaming AI calls here, so 30 s is sufficient)
4. Click **Save**.

### Step 4: Cap Concurrency (Cost Protection)

By default, Lambda can spin up thousands of containers under load. For a learning project, cap it at 5.

1. Still in the **Configuration** tab, click **Concurrency and recursion detection** (left sidebar).
2. On the **Concurrency** card, click **Edit**.
3. Select **Reserve concurrency** and enter `5`.
4. Click **Save**.

> **Do not touch Provisioned Concurrency.** That's a separate, paid feature that pre-warms containers — it costs money even when idle. We only want the free **Reserved** concurrency ceiling.

### Step 5: Create the Function URL

A Function URL gives your Lambda a public HTTPS endpoint — no API Gateway needed, no extra cost.

1. Still in the **Configuration** tab, click **Function URL** (left sidebar) → **Create function URL**.
2. **Auth type:** `NONE` — our app doesn't need AWS IAM auth; it's a learning project.
3. Expand **Additional settings**:
   - **CORS:** leave unchecked (FastAPI already handles CORS via its middleware).
   - **Invoke mode:** **`RESPONSE_STREAM`** — important if you ever add streaming endpoints; harmless to set now.
4. Click **Save**.

You'll now see a **Function URL** on the function overview page:

```
https://<random-id>.lambda-url.<region>.on.aws/
```

This is your public URL — share it, bookmark it, or open it in your browser.

### Step 6: Test the Live Deployment

1. Click the **Function URL** link.
2. **First request will take 10–30 seconds** — this is a "cold start" while Lambda boots the container. All subsequent requests within ~15 minutes will be fast (under a second).
3. Add some bookmarks. Search. Delete. Everything should work exactly as it did locally.
4. Visit `<your-function-url>/health` — confirm the health endpoint responds.
5. Visit `<your-function-url>/docs` — confirm the FastAPI docs page loads.

**Checkpoint:** Your app is live on the public internet, served by AWS Lambda, with HTTPS and zero monthly cost.

---

## Part 9: Monitor with CloudWatch

### View Logs

1. Open your Lambda function in the AWS Console.
2. Click the **Monitor** tab → **View CloudWatch logs**.
3. Click the most recent **log stream** to see uvicorn startup logs and per-request access logs.

### View Metrics

The **Monitor** tab shows charts for:

- **Invocations** — total number of requests received
- **Duration** — how long each request took (in milliseconds)
- **Error count** — number of 4xx/5xx responses
- **Throttles** — requests rejected due to concurrency cap (should be zero)

---

## Part 10: Update Your Application

When you make code changes:

**Mac/Linux:**
```bash
# 1. Rebuild and push to ECR
docker build --platform linux/amd64 -t bookmark-app .
docker tag bookmark-app:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/bookmark-app:latest
docker push \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/bookmark-app:latest

# 2. Tell Lambda to use the new image (one-liner)
aws lambda update-function-code \
  --function-name bookmark-app \
  --image-uri $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/bookmark-app:latest \
  --region $AWS_REGION
```

**Windows PowerShell:**
```powershell
docker build --platform linux/amd64 -t bookmark-app .
docker tag bookmark-app:latest `
  "$env:AWS_ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/bookmark-app:latest"
docker push `
  "$env:AWS_ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/bookmark-app:latest"

aws lambda update-function-code `
  --function-name bookmark-app `
  --image-uri "$env:AWS_ACCOUNT_ID.dkr.ecr.$env:AWS_REGION.amazonaws.com/bookmark-app:latest" `
  --region $env:AWS_REGION
```

Lambda redeploys in 10–30 seconds. The next request will run the new image (and may cold-start once).

Alternatively via the Console: Lambda → function → **Image** tab → **Deploy new image** → browse to `latest` → **Save**.

---

## Cost Reference

| Service | Expected Cost |
|---------|---------------|
| Lambda compute | **$0/month** — free tier: 1M requests + 400K GB-seconds/month, forever |
| Lambda Function URL | **$0** — no extra charge beyond Lambda's own |
| ECR storage | ~$0.05–$0.10/month for one ~250 MB image |
| CloudWatch Logs | **$0** — first 5 GB ingested/month is free |
| **Total** | **~$0.10/month or less** |

### Emergency Cost Control

If you hit a budget alert:

1. Lambda → your function → **Actions** → **Throttle** — sets reserved concurrency to 0, stopping all invocations immediately.
2. Check CloudWatch Logs for unexpected traffic patterns.
3. Clean up old ECR image versions: ECR → repository → select old images → **Delete**.

---

## Troubleshooting

### Docker Issues

**"Cannot connect to the Docker daemon"**
Docker Desktop is not running. Look for the whale icon in your menu bar (Mac) or system tray (Windows) and start it.

**"exec format error" in CloudWatch Logs**
You forgot `--platform linux/amd64` when building. Rebuild with that flag and push again.

**Windows: "image manifest media type not supported"**
Go to Docker Desktop → Settings → General → uncheck **Use containerd for pulling and storing images**. Rebuild.

### AWS Issues

**"Unauthorized" when pushing to ECR**
Re-run the `aws ecr get-login-password | docker login ...` command. The temporary token expires after 12 hours.

**"Access Denied" in Lambda Console**
Your IAM user is missing a policy. Sign in as root → IAM → Users → `devuser` → **Add permissions** → attach the relevant policy.

### Application Issues

**Lambda returns 502 / "Internal Server Error"**
Open CloudWatch Logs and find the Python traceback. The most common cause is a missing environment variable or an import error.

**Page loads but API calls fail**
Open the browser developer console (F12) → Network tab. Look for 4xx errors. Check that the URL in the fetch calls matches your actual endpoint path (`/api/bookmarks`).

**App works locally but not on Lambda**
Confirm the Dockerfile copies both `main.py` and the `static/` folder. Run `docker run -p 8000:8000 bookmark-app` locally with the same image you pushed to ECR.

---

## What You've Learned

By completing this guide you have:

- Created a production AWS account with MFA and cost guardrails
- Set up least-privilege IAM users and groups
- Built a real FastAPI web app with a working frontend
- Containerized the app with Docker (multi-stage awareness, Lambda Web Adapter)
- Pushed images to Amazon ECR
- Deployed a container to AWS Lambda with a public HTTPS Function URL
- Monitored logs and metrics in CloudWatch
- Learned the update cycle: build → push → `update-function-code`

---

## Architecture Overview

```
Browser
  │
  │ HTTPS  (Lambda Function URL — free, no API Gateway)
  ▼
AWS Lambda
  └─ Docker container
       ├─ Lambda Web Adapter  (translates Lambda invocations → HTTP)
       ├─ uvicorn             (Python ASGI server)
       └─ FastAPI app
            ├─ GET  /              → serves static/index.html
            ├─ GET  /health        → JSON health check
            ├─ GET  /api/bookmarks → list / search bookmarks
            ├─ POST /api/bookmarks → add bookmark
            └─ DELETE /api/bookmarks/{id} → remove bookmark
```

---

## Next Steps

Once you're comfortable with this pattern, here are natural progressions:

**Persistence:** Replace the in-memory dict with Amazon DynamoDB so bookmarks survive container restarts.

**Custom domain:** Put a CloudFront distribution in front of the Function URL and attach your own domain (free TLS via ACM).

**Automatic deployments:** Add a GitHub Actions workflow that runs `docker build → push → update-function-code` on every push to `main`.

**Secrets management:** Move sensitive values out of plain environment variables into AWS Secrets Manager or Parameter Store.

**Monitoring:** Add CloudWatch Alarms for error rate or duration spikes — get an email when something breaks.

---

## Resources

- [AWS Lambda Container Image Support](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [AWS Lambda Web Adapter (GitHub)](https://github.com/awslabs/aws-lambda-web-adapter)
- [Lambda Function URLs documentation](https://docs.aws.amazon.com/lambda/latest/dg/urls-configuration.html)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Amazon ECR documentation](https://docs.aws.amazon.com/ecr/)
- [AWS Free Tier](https://aws.amazon.com/free/)
