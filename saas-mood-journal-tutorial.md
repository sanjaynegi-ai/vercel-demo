# Building a SaaS AI Mood Journal — Full Tutorial

> **Why this app?**  
> A **Mood Journal** powered by AI is the perfect teaching app. Everyone understands what it does instantly. It needs a real-time streaming AI response (same tech as before), a reason to sign in (your journal is private), and a reason to pay (unlimited entries). It maps 1:1 to every lesson in the original curriculum — without the awkwardness of a "business idea generator" that no one would actually pay for.

---

## What You'll Build Across 3 Parts

| Part | What You Build | Key Concepts |
|---|---|---|
| **STAGE 1** | Full-stack app: Next.js + FastAPI + streaming AI | SSE, Vercel deployment |
| **STAGE 2** | Authentication with Clerk | JWT, protected routes |
| **STAGE 3** | Subscription paywall with Clerk Billing | Pricing table, plan protection |

The app lets users type how they're feeling → AI streams back an empathetic reflection and a suggested action → stored behind a login → gated behind a monthly subscription.

---

# STAGE 1: Build the Full-Stack Mood Journal

## What You'll Build

An AI-powered Mood Journal that:
- Has a modern React frontend built with Next.js (Pages Router)
- Uses TypeScript for type safety
- Connects to a FastAPI Python backend
- Streams AI responses in real-time using Server-Sent Events (SSE)
- Renders beautifully formatted Markdown responses
- Deploys to production on Vercel

## Prerequisites

- Node.js 18+ installed
- Python 3.9+ installed
- Vercel CLI installed (`npm i -g vercel`)
- An OpenAI API key

---

## Step 1: Create Your Next.js Project

### Create the Project

Open your terminal and run:

```bash
npx create-next-app mood-journal --ts --eslint --tailwind --no-src-dir --no-app
```

This creates a Next.js project with:
- **Pages Router** — stable, battle-tested routing
- **TypeScript** — type safety
- **ESLint** — code quality checks
- **Tailwind CSS** — utility-first styling

### Open Your Project

```bash
cd mood-journal
```

Open the folder in your editor (e.g. Cursor: File → Open Folder → select `mood-journal`).

### Project Structure

```
mood-journal/
├── pages/
│   ├── _app.tsx        # App wrapper
│   ├── _document.tsx   # HTML structure
│   ├── index.tsx       # Homepage → "/"
│   └── api/            # Delete this folder
├── styles/
│   └── globals.css     # Global styles + Tailwind
├── public/
├── package.json
└── tsconfig.json
```

### Clean Up

Delete the `pages/api` folder — we're using a Python FastAPI backend instead of Next.js API routes.

---

## Step 2: Set Up the Python Backend

### Create the API Folder

In your project root, create a new folder called `api`.

### Create `requirements.txt`

In the project root, create `requirements.txt`:

```
fastapi
uvicorn
openai
```

### Create `api/index.py`

```python
import os
from fastapi import FastAPI  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from openai import OpenAI  # type: ignore

app = FastAPI()

@app.get("/api")
def reflect(mood: str = "I feel okay"):
    client = OpenAI()
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a compassionate journaling assistant. "
                "When a user shares how they feel, you respond with: "
                "1. A warm, empathetic reflection (2-3 sentences). "
                "2. A practical suggested action they can take right now. "
                "Format your response with Markdown headings and bullet points."
            )
        },
        {
            "role": "user",
            "content": f"Today I feel: {mood}"
        }
    ]
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        stream=True
    )

    def event_stream():
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                lines = text.split("\n")
                for line in lines[:-1]:
                    yield f"data: {line}\n\n"
                    yield "data:  \n"
                yield f"data: {lines[-1]}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

**What's happening here:**
- The `/api` endpoint accepts a `mood` query parameter
- It sends the mood to OpenAI with a system prompt that shapes a compassionate response
- It streams the response back to the browser using Server-Sent Events (SSE)

---

## Step 3: Install Frontend Dependencies

```bash
npm install react-markdown remark-gfm remark-breaks
```

---

## Step 4: Create the App Pages

### Update `pages/_app.tsx`

```typescript
import type { AppProps } from 'next/app';
import '../styles/globals.css';

export default function MyApp({ Component, pageProps }: AppProps) {
  return <Component {...pageProps} />;
}
```

### Update `pages/_document.tsx`

```typescript
import { Html, Head, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <title>AI Mood Journal</title>
        <meta name="description" content="Your private AI-powered mood journal" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
```

### Replace `pages/index.tsx`

```typescript
"use client"

import { useEffect, useState, FormEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';

export default function Home() {
    const [mood, setMood] = useState<string>('');
    const [reflection, setReflection] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const handleSubmit = (e: FormEvent) => {
        e.preventDefault();
        if (!mood.trim()) return;

        setReflection('');
        setIsLoading(true);

        const encoded = encodeURIComponent(mood);
        const evt = new EventSource(`/api?mood=${encoded}`);
        let buffer = '';

        evt.onmessage = (e) => {
            buffer += e.data;
            setReflection(buffer);
        };
        evt.onerror = () => {
            evt.close();
            setIsLoading(false);
        };
    };

    return (
        <main className="min-h-screen bg-gradient-to-br from-violet-50 to-purple-100 dark:from-gray-900 dark:to-gray-800">
            <div className="container mx-auto px-4 py-12 max-w-2xl">

                {/* Header */}
                <header className="text-center mb-10">
                    <h1 className="text-5xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent mb-3">
                        Mood Journal
                    </h1>
                    <p className="text-gray-500 dark:text-gray-400 text-lg">
                        Share how you feel. Get an honest, caring reflection.
                    </p>
                </header>

                {/* Input Form */}
                <form onSubmit={handleSubmit} className="mb-8">
                    <textarea
                        value={mood}
                        onChange={(e) => setMood(e.target.value)}
                        placeholder="How are you feeling today? Write freely..."
                        rows={4}
                        className="w-full p-4 rounded-xl border border-purple-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-400 resize-none text-base"
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !mood.trim()}
                        className="mt-3 w-full bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 disabled:opacity-50 text-white font-semibold py-3 px-6 rounded-xl transition-all"
                    >
                        {isLoading ? 'Reflecting...' : 'Get Reflection →'}
                    </button>
                </form>

                {/* AI Reflection Card */}
                {(reflection || isLoading) && (
                    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
                        {isLoading && !reflection ? (
                            <div className="animate-pulse text-gray-400 text-center py-6">
                                Thinking about what you shared...
                            </div>
                        ) : (
                            <div className="markdown-content text-gray-700 dark:text-gray-300">
                                <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
                                    {reflection}
                                </ReactMarkdown>
                            </div>
                        )}
                    </div>
                )}

            </div>
        </main>
    );
}
```

---

## Step 5: Fix Markdown Styles

Tailwind resets all default HTML styles, which breaks Markdown rendering. Add this to the **bottom** of `styles/globals.css` (do not replace existing content):

```css
@layer base {
  .markdown-content h1 { font-size: 2em; font-weight: bold; margin: 0.67em 0; }
  .markdown-content h2 { font-size: 1.5em; font-weight: bold; margin: 0.83em 0; }
  .markdown-content h3 { font-size: 1.17em; font-weight: bold; margin: 1em 0; }
  .markdown-content p  { margin: 1em 0; }
  .markdown-content ul { list-style-type: disc; padding-left: 2em; margin: 1em 0; }
  .markdown-content ol { list-style-type: decimal; padding-left: 2em; margin: 1em 0; }
  .markdown-content li { margin: 0.25em 0; }
  .markdown-content strong { font-weight: bold; }
  .markdown-content em { font-style: italic; }
  .markdown-content hr { border: 0; border-top: 1px solid #e5e7eb; margin: 2em 0; }
}
```

---

## Step 6: Link Project to Vercel

```bash
vercel link
```

Follow the prompts:
- Set up and link? → **Yes**
- Which scope? → Your personal account
- Link to existing project? → **No**
- Project name? → `mood-journal`
- Directory? → Press Enter (current directory)

---

## Step 7: Add Your OpenAI Key to Vercel

```bash
vercel env add OPENAI_API_KEY
```

Paste your key, select **Preview** and **Production** environments (not Development). Mark as sensitive when prompted.

---

## Step 8: Deploy

```bash
vercel --prod
```

Visit the production URL. Type how you're feeling and click **Get Reflection**. You'll see the AI response stream in real time!

---

## What You've Learned

- Full-stack architecture with Next.js (frontend) + FastAPI (backend)
- Server-Sent Events (SSE) for real-time streaming responses
- Passing user input from the frontend to a Python API
- Rendering Markdown with `react-markdown`
- Deploying a full-stack app to Vercel

## Troubleshooting

**Streaming not working locally** — SSE can behave oddly on `localhost`. Deploy to Vercel to test.  
**API not responding** — Check your OpenAI API key and account credits.  
**Markdown looks unstyled** — Make sure you appended the CSS and didn't overwrite `globals.css`.  
**Module not found** — Run `npm install` again.

---
---

# STAGE 2: Adding User Authentication with Clerk

## Why Authentication?

Your journal entries are personal. Without authentication, anyone could read anyone else's reflections. Adding auth means every user gets their own private journal, and your backend knows exactly who is making each request.

## What You'll Add

- Sign-in with Google, GitHub, or Email
- A landing page for signed-out users
- A protected `/journal` route for signed-in users
- Secure JWT tokens passed from the browser to your Python backend
- Backend verification of every API request

## Prerequisites

- Completed STAGE 1 (working Mood Journal deployed to Vercel)

---

## Step 1: Create a Clerk Account

1. Go to [clerk.com](https://clerk.com) and click **Sign Up**
2. After signing in, click **Create Application**
3. Set:
   - **Application name:** mood-journal
   - **Sign-in options:** Email, Google, GitHub
4. Click **Create Application**

You'll land on the Clerk dashboard showing your API keys. Keep this tab open.

---

## Step 2: Install Clerk Dependencies

```bash
npm install @clerk/nextjs@6.39.0
npm install @microsoft/fetch-event-source
```

> **Why `fetch-event-source`?**  
> The native browser `EventSource` API cannot send custom headers. To attach our JWT auth token to the streaming request, we need `@microsoft/fetch-event-source`, which supports headers.

---

## Step 3: Set Up Environment Variables

Create `.env.local` in your project root:

```bash
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=your_publishable_key_here
CLERK_SECRET_KEY=your_secret_key_here
CLERK_JWKS_URL=your_jwks_url_here
```

**Where to find each value in the Clerk Dashboard:**
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` and `CLERK_SECRET_KEY` → **Configure → API Keys**
- `CLERK_JWKS_URL` → Also on the **API Keys** page, scroll down to "Advanced"

**Add `.env.local` to `.gitignore`:**

Open `.gitignore` and add:
```
.env.local
```

> **What is a JWKS URL?**  
> When a user signs in, Clerk issues a JWT (JSON Web Token) — a cryptographically signed token proving the user's identity. Your Python backend uses the JWKS (JSON Web Key Set) URL to fetch Clerk's public keys and independently verify that any incoming JWT is genuine, without contacting Clerk for every request.

---

## Step 4: Wrap Your App with ClerkProvider

Update `pages/_app.tsx`:

```typescript
import { ClerkProvider } from '@clerk/nextjs';
import type { AppProps } from 'next/app';
import '../styles/globals.css';

export default function MyApp({ Component, pageProps }: AppProps) {
  return (
    <ClerkProvider {...pageProps}>
      <Component {...pageProps} />
    </ClerkProvider>
  );
}
```

---

## Step 5: Create the Protected Journal Page

Create `pages/journal.tsx` — this is where authenticated users write their mood entries:

```typescript
"use client"

import { useState, FormEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { useAuth, UserButton } from '@clerk/nextjs';
import { fetchEventSource } from '@microsoft/fetch-event-source';

export default function Journal() {
    const { getToken } = useAuth();
    const [mood, setMood] = useState<string>('');
    const [reflection, setReflection] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (!mood.trim()) return;

        setReflection('');
        setIsLoading(true);

        const jwt = await getToken();
        if (!jwt) {
            setReflection('You must be signed in to use your journal.');
            setIsLoading(false);
            return;
        }

        let buffer = '';
        const encoded = encodeURIComponent(mood);

        await fetchEventSource(`/api?mood=${encoded}`, {
            headers: { Authorization: `Bearer ${jwt}` },
            onmessage(ev) {
                buffer += ev.data;
                setReflection(buffer);
            },
            onerror(err) {
                console.error('Stream error:', err);
                setIsLoading(false);
            },
            onclose() {
                setIsLoading(false);
            }
        });
    };

    return (
        <main className="min-h-screen bg-gradient-to-br from-violet-50 to-purple-100 dark:from-gray-900 dark:to-gray-800">

            {/* User Menu */}
            <div className="absolute top-4 right-4">
                <UserButton showName={true} />
            </div>

            <div className="container mx-auto px-4 py-12 max-w-2xl">

                <header className="text-center mb-10">
                    <h1 className="text-5xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent mb-3">
                        My Mood Journal
                    </h1>
                    <p className="text-gray-500 dark:text-gray-400 text-lg">
                        Your private space to reflect.
                    </p>
                </header>

                <form onSubmit={handleSubmit} className="mb-8">
                    <textarea
                        value={mood}
                        onChange={(e) => setMood(e.target.value)}
                        placeholder="How are you feeling today? Write freely..."
                        rows={4}
                        className="w-full p-4 rounded-xl border border-purple-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-400 resize-none text-base"
                    />
                    <button
                        type="submit"
                        disabled={isLoading || !mood.trim()}
                        className="mt-3 w-full bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 disabled:opacity-50 text-white font-semibold py-3 px-6 rounded-xl transition-all"
                    >
                        {isLoading ? 'Reflecting...' : 'Get Reflection →'}
                    </button>
                </form>

                {(reflection || isLoading) && (
                    <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
                        {isLoading && !reflection ? (
                            <div className="animate-pulse text-gray-400 text-center py-6">
                                Thinking about what you shared...
                            </div>
                        ) : (
                            <div className="markdown-content text-gray-700 dark:text-gray-300">
                                <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
                                    {reflection}
                                </ReactMarkdown>
                            </div>
                        )}
                    </div>
                )}

            </div>
        </main>
    );
}
```

---

## Step 6: Update the Landing Page

Replace `pages/index.tsx` with a landing page for signed-out users:

```typescript
"use client"

import Link from 'next/link';
import { SignInButton, SignedIn, SignedOut, UserButton } from '@clerk/nextjs';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-violet-50 to-purple-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-12 max-w-4xl">

        {/* Navigation */}
        <nav className="flex justify-between items-center mb-16">
          <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-200">
            🌿 Mood Journal
          </h1>
          <div>
            <SignedOut>
              <SignInButton mode="modal">
                <button className="bg-violet-600 hover:bg-violet-700 text-white font-medium py-2 px-6 rounded-lg transition-colors">
                  Sign In
                </button>
              </SignInButton>
            </SignedOut>
            <SignedIn>
              <div className="flex items-center gap-4">
                <Link
                  href="/journal"
                  className="bg-violet-600 hover:bg-violet-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
                >
                  Open Journal
                </Link>
                <UserButton showName={true} />
              </div>
            </SignedIn>
          </div>
        </nav>

        {/* Hero */}
        <div className="text-center py-20">
          <h2 className="text-6xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent mb-6">
            Your Daily
            <br />
            AI Mood Journal
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-400 mb-12 max-w-xl mx-auto">
            Write how you feel. Receive a compassionate AI reflection and a practical step forward — privately, every day.
          </p>

          <SignedOut>
            <SignInButton mode="modal">
              <button className="bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white font-bold py-4 px-10 rounded-xl text-lg transition-all transform hover:scale-105 shadow-lg">
                Start Journaling Free
              </button>
            </SignInButton>
          </SignedOut>
          <SignedIn>
            <Link href="/journal">
              <button className="bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white font-bold py-4 px-10 rounded-xl text-lg transition-all transform hover:scale-105 shadow-lg">
                Open My Journal →
              </button>
            </Link>
          </SignedIn>
        </div>

      </div>
    </main>
  );
}
```

---

## Step 7: Update the Python Backend to Verify Auth

### Update `requirements.txt`

```
fastapi
uvicorn
openai
fastapi-clerk-auth
```

### Replace `api/index.py`

```python
import os
from fastapi import FastAPI, Depends  # type: ignore
from fastapi.responses import StreamingResponse  # type: ignore
from fastapi_clerk_auth import ClerkConfig, ClerkHTTPBearer, HTTPAuthorizationCredentials  # type: ignore
from openai import OpenAI  # type: ignore

app = FastAPI()

clerk_config = ClerkConfig(jwks_url=os.getenv("CLERK_JWKS_URL"))
clerk_guard = ClerkHTTPBearer(clerk_config)

@app.get("/api")
def reflect(
    mood: str = "I feel okay",
    creds: HTTPAuthorizationCredentials = Depends(clerk_guard)
):
    user_id = creds.decoded["sub"]  # The authenticated user's ID
    # user_id can be used to: log entries, apply limits, personalise responses

    client = OpenAI()
    prompt = [
        {
            "role": "system",
            "content": (
                "You are a compassionate journaling assistant. "
                "When a user shares how they feel, respond with: "
                "1. A warm empathetic reflection (2-3 sentences). "
                "2. A practical suggested action they can take right now. "
                "Format with Markdown headings and bullet points."
            )
        },
        {
            "role": "user",
            "content": f"Today I feel: {mood}"
        }
    ]
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        stream=True
    )

    def event_stream():
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                lines = text.split("\n")
                for line in lines[:-1]:
                    yield f"data: {line}\n\n"
                    yield "data:  \n"
                yield f"data: {lines[-1]}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

---

## Step 8: Add Clerk Keys to Vercel

```bash
vercel env add NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
```
Paste your publishable key → select **all environments**.

```bash
vercel env add CLERK_SECRET_KEY
```
Paste your secret key → select **Preview** and **Production** only.

```bash
vercel env add CLERK_JWKS_URL
```
Paste your JWKS URL → select **Preview** and **Production** only.

---

## Step 9: Deploy

```bash
vercel --prod
```

Test the flow:
1. Visit your production URL — you see the landing page
2. Click **Start Journaling Free** → Clerk sign-in modal appears
3. Sign in with Google or GitHub
4. You're redirected to the landing page — click **Open My Journal**
5. Write how you feel and submit
6. The AI reflection streams in real time ✅

> **Note on JWT timeout:** If you see a `403` error after ~60 seconds, it means the JWT expired mid-stream. This is a known issue with long responses. The fix: request a fresh token before each stream call, or use shorter AI responses during testing.

---

## How the Security Works

```
Browser                     Next.js (Vercel)          Python (Vercel)
  │                               │                         │
  │── User signs in ──────────────▶ Clerk issues JWT        │
  │                               │                         │
  │── POST /api?mood=... ─────────────────────────────────▶ │
  │   (Authorization: Bearer JWT) │                         │
  │                               │                    Verifies JWT
  │                               │                 using JWKS public keys
  │                               │                         │
  │◀── Streaming AI response ──────────────────────────────-│
```

No session cookies. No server-side storage. Each request is stateless and cryptographically verified.

---

## Troubleshooting

**"Unauthorized" errors** — Check all 3 environment variables are added to Vercel. Re-deploy after adding them.  
**Sign-in modal not appearing** — Confirm `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` starts with `pk_`.  
**`ClerkProvider` error** — Make sure `_app.tsx` wraps with `<ClerkProvider>`.  
**API not authenticating** — Verify `CLERK_JWKS_URL` is correct and `fastapi-clerk-auth` is in `requirements.txt`.

---
---

# STAGE 3: Adding a Subscription Paywall with Clerk Billing

## Why a Paywall?

Authentication proves *who* the user is. Billing proves they're a *paying* customer. Together, they form the core of any SaaS. We'll gate the journal behind a monthly subscription — free users see a pricing page, paid users get full access.

## What You'll Add

- A subscription plan in Clerk Billing
- A `<Protect>` wrapper that checks subscription status before showing the journal
- A built-in `<PricingTable>` component shown to non-subscribers
- Subscription management via the user menu

## Prerequisites

- Completed Day 3 Part 1 (authentication working and deployed)

---

## Step 1: Enable Clerk Billing

1. Go to [Clerk Dashboard](https://dashboard.clerk.com)
2. Select your **mood-journal** application
3. Click **Configure** (top nav)
4. Click **Billing → Subscription Plans** (left sidebar)
5. Click **Get Started** / **Enable Billing**

---

## Step 2: Create Your Subscription Plan

1. Click **Create Plan**
2. Fill in:
   - **Name:** Premium Journal
   - **Key:** `premium_journal` ← copy this exactly, you'll use it in code
   - **Price:** $5.00/month (or whatever you like)
   - **Description:** Unlimited private AI reflections
3. Optional: Toggle on **Annual billing** and set a discounted annual price
4. Click **Save**

---

## Step 3: Update the Journal Page with Subscription Protection

Replace `pages/journal.tsx` entirely:

```typescript
"use client"

import { useState, FormEvent } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { useAuth, UserButton, Protect, PricingTable } from '@clerk/nextjs';
import { fetchEventSource } from '@microsoft/fetch-event-source';

// The actual journal — only rendered for subscribers
function JournalApp() {
    const { getToken } = useAuth();
    const [mood, setMood] = useState<string>('');
    const [reflection, setReflection] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(false);

    const handleSubmit = async (e: FormEvent) => {
        e.preventDefault();
        if (!mood.trim()) return;

        setReflection('');
        setIsLoading(true);

        const jwt = await getToken();
        if (!jwt) {
            setReflection('Authentication required.');
            setIsLoading(false);
            return;
        }

        let buffer = '';
        const encoded = encodeURIComponent(mood);

        await fetchEventSource(`/api?mood=${encoded}`, {
            headers: { Authorization: `Bearer ${jwt}` },
            onmessage(ev) {
                buffer += ev.data;
                setReflection(buffer);
            },
            onerror(err) {
                console.error('Stream error:', err);
                setIsLoading(false);
            },
            onclose() {
                setIsLoading(false);
            }
        });
    };

    return (
        <div className="container mx-auto px-4 py-12 max-w-2xl">
            <header className="text-center mb-10">
                <h1 className="text-5xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent mb-3">
                    My Mood Journal
                </h1>
                <p className="text-gray-500 dark:text-gray-400 text-lg">
                    Your private space to reflect.
                </p>
            </header>

            <form onSubmit={handleSubmit} className="mb-8">
                <textarea
                    value={mood}
                    onChange={(e) => setMood(e.target.value)}
                    placeholder="How are you feeling today? Write freely..."
                    rows={4}
                    className="w-full p-4 rounded-xl border border-purple-200 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100 shadow-sm focus:outline-none focus:ring-2 focus:ring-violet-400 resize-none text-base"
                />
                <button
                    type="submit"
                    disabled={isLoading || !mood.trim()}
                    className="mt-3 w-full bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 disabled:opacity-50 text-white font-semibold py-3 px-6 rounded-xl transition-all"
                >
                    {isLoading ? 'Reflecting...' : 'Get Reflection →'}
                </button>
            </form>

            {(reflection || isLoading) && (
                <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
                    {isLoading && !reflection ? (
                        <div className="animate-pulse text-gray-400 text-center py-6">
                            Thinking about what you shared...
                        </div>
                    ) : (
                        <div className="markdown-content text-gray-700 dark:text-gray-300">
                            <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>
                                {reflection}
                            </ReactMarkdown>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

// The paywall wrapper — shown to authenticated but non-subscribed users
function PaywallPage() {
    return (
        <div className="container mx-auto px-4 py-12 max-w-3xl">
            <header className="text-center mb-12">
                <h1 className="text-5xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent mb-4">
                    Unlock Your Journal
                </h1>
                <p className="text-gray-500 dark:text-gray-400 text-lg">
                    Subscribe to get unlimited private AI reflections
                </p>
            </header>
            <PricingTable />
        </div>
    );
}

// The page — Protect checks subscription, shows one of the above
export default function Journal() {
    return (
        <main className="min-h-screen bg-gradient-to-br from-violet-50 to-purple-100 dark:from-gray-900 dark:to-gray-800">

            {/* User Menu */}
            <div className="absolute top-4 right-4">
                <UserButton showName={true} />
            </div>

            <Protect
                plan="premium_journal"
                fallback={<PaywallPage />}
            >
                <JournalApp />
            </Protect>

        </main>
    );
}
```

**Key concept — `<Protect>`:**  
`<Protect plan="premium_journal">` checks whether the signed-in user has an active subscription to the `premium_journal` plan. If they do, it renders `<JournalApp />`. If they don't, it renders the `fallback` — the `<PaywallPage />` with Clerk's built-in `<PricingTable />`.

---

## Step 4: Update the Landing Page for Subscribers

Update `pages/index.tsx` to show a different CTA for subscribers vs free users:

```typescript
"use client"

import Link from 'next/link';
import { SignInButton, SignedIn, SignedOut, UserButton } from '@clerk/nextjs';

export default function Home() {
  return (
    <main className="min-h-screen bg-gradient-to-br from-violet-50 to-purple-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-12 max-w-4xl">

        <nav className="flex justify-between items-center mb-16">
          <h1 className="text-2xl font-bold text-gray-800 dark:text-gray-200">
            🌿 Mood Journal
          </h1>
          <div>
            <SignedOut>
              <SignInButton mode="modal">
                <button className="bg-violet-600 hover:bg-violet-700 text-white font-medium py-2 px-6 rounded-lg transition-colors">
                  Sign In
                </button>
              </SignInButton>
            </SignedOut>
            <SignedIn>
              <div className="flex items-center gap-4">
                <Link href="/journal" className="bg-violet-600 hover:bg-violet-700 text-white font-medium py-2 px-6 rounded-lg transition-colors">
                  Open Journal
                </Link>
                <UserButton showName={true} />
              </div>
            </SignedIn>
          </div>
        </nav>

        <div className="text-center py-20">
          <h2 className="text-6xl font-bold bg-gradient-to-r from-violet-600 to-purple-600 bg-clip-text text-transparent mb-6">
            Your Daily
            <br />
            AI Mood Journal
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-400 mb-8 max-w-xl mx-auto">
            Write how you feel. Get a compassionate AI reflection and a practical step forward — privately, every day.
          </p>

          {/* Pricing preview for signed-out visitors */}
          <SignedOut>
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-lg rounded-2xl p-8 max-w-sm mx-auto mb-10 shadow-lg">
              <h3 className="text-2xl font-bold mb-2 text-gray-800 dark:text-gray-100">Premium Journal</h3>
              <p className="text-4xl font-bold text-violet-600 mb-4">$5<span className="text-lg text-gray-500">/month</span></p>
              <ul className="text-left text-gray-600 dark:text-gray-400 space-y-2 mb-6">
                <li>✓ Unlimited daily reflections</li>
                <li>✓ Private AI-powered insights</li>
                <li>✓ Cancel anytime</li>
              </ul>
            </div>
            <SignInButton mode="modal">
              <button className="bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white font-bold py-4 px-10 rounded-xl text-lg transition-all transform hover:scale-105 shadow-lg">
                Start Journaling Free
              </button>
            </SignInButton>
          </SignedOut>

          <SignedIn>
            <Link href="/journal">
              <button className="bg-gradient-to-r from-violet-600 to-purple-600 hover:from-violet-700 hover:to-purple-700 text-white font-bold py-4 px-10 rounded-xl text-lg transition-all transform hover:scale-105 shadow-lg">
                Open My Journal →
              </button>
            </Link>
          </SignedIn>
        </div>

      </div>
    </main>
  );
}
```

---

## Step 5: (Optional) Configure a Payment Provider

By default, Clerk uses its own built-in payment gateway — no setup needed for testing.

To connect Stripe for real payments:
1. Clerk Dashboard → **Configure** → **Billing** → **Settings**
2. Select **Stripe** and follow the setup wizard

For this tutorial, the built-in gateway is fine.

---

## Step 6: Deploy

```bash
vercel --prod
```

---

## Step 7: Test the Full SaaS Flow

1. Visit your production URL
2. Click **Start Journaling Free** → sign in
3. Click **Open My Journal** → you see the **Pricing Table** (no subscription yet)
4. Click **Subscribe** → complete the test payment
5. After subscribing, you're taken to the journal ✅
6. Click the user avatar (top right) → **Manage account** → **Subscriptions** to manage billing

---

## How It All Fits Together

```
User visits /journal
       │
       ▼
 Signed in? ──No──▶ Redirected to landing page (sign in)
       │
      Yes
       │
       ▼
 Has premium_journal plan? ──No──▶ PricingTable shown
       │
      Yes
       │
       ▼
 JournalApp rendered ──▶ Types mood ──▶ JWT sent to Python API
                                              │
                                        Verifies JWT
                                              │
                                        Streams AI reflection
```

---

## Troubleshooting

**Always seeing the pricing table after subscribing** — Sign out and back in to refresh the session token. Also verify the plan key is exactly `premium_journal`.  
**"Plan not found" error** — Check Billing is enabled in the Clerk Dashboard and the plan key matches your code.  
**PricingTable not rendering** — Make sure `@clerk/nextjs` is up to date and Billing is enabled for your app.

---

## Congratulations! 🎉

You've shipped a complete, production SaaS with:

| Feature | How |
|---|---|
| ✅ Full-stack app | Next.js + FastAPI |
| ✅ Real-time streaming | Server-Sent Events |
| ✅ User authentication | Clerk |
| ✅ JWT-secured API | `fastapi-clerk-auth` |
| ✅ Subscription paywall | Clerk Billing + `<Protect>` |
| ✅ Subscription management | Clerk `<UserButton>` |
| ✅ Production deployment | Vercel |

### What to Build Next

- Save journal entries to a database (Supabase or PlanetScale)
- Add a history page showing past entries per user
- Send a weekly summary email (Resend + cron job)
- Add multiple subscription tiers (Basic: 5 entries/month, Premium: unlimited)
- Add mood tracking charts over time
