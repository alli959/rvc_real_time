# Background Generation + Real-Time Progress

## Problem

Audio processing (voice swap, vocal split, TTS, voice convert) currently runs synchronously through the request chain: Frontend → Cloudflare → Laravel → Voice Engine → response. Cloudflare's free-tier 100-second timeout kills long-running requests (multi-voice songs can take 30-120s). Additionally, the frontend shows fake progress percentages that don't reflect actual processing steps.

## Solution

Convert audio processing to an async job-based system with real-time progress polling. Users can optionally run jobs in the background and continue browsing, receiving a notification when complete.

## Architecture

```
┌─────────┐     ┌───────────┐     ┌──────────────┐     ┌────────────┐
│ Frontend │────▶│ Laravel   │────▶│ Voice Engine │────▶│ MinIO (S3) │
│ (Next.js)│◀───│ (PHP API) │◀────│ (FastAPI)    │     │ (local)    │
└─────────┘     └───────────┘     └──────────────┘     └────────────┘
     │ polls          │                    │
     │ GET /jobs/{id} │  progress webhook  │
     └────────────────┘◀───────────────────┘
```

### Request Flow

1. **Submit**: Frontend POST `/api/audio/process` — always async. Laravel returns job UUID immediately (202 Accepted).
2. **Dispatch**: Laravel creates `JobQueue` record (status=`queued`), fires async HTTP POST to voice engine with `callback_url` and `job_uuid`.
3. **Process**: Voice engine receives job, processes in a background thread.
4. **Progress**: Voice engine POST `{callback_url}/internal/jobs/{uuid}/progress` back to Laravel at each step (updates `progress`, `progress_message`).
5. **Complete**: Voice engine uploads output WAV to MinIO via S3 API at `user-generations/{user_id}/{job_uuid}.wav`, then POST `{callback_url}/internal/jobs/{uuid}/complete` with the S3 key.
6. **Poll**: Frontend polls `GET /api/jobs/{uuid}` every 2-3 seconds, updates UI with real progress.
7. **Retrieve**: Frontend gets signed download URL from completed job response, plays/downloads.

### Foreground vs Background Mode

- **Foreground** (default): User stays on page and watches real-time progress from polling. If Cloudflare cuts the initial submission request, the job still runs server-side and appears in their jobs list.
- **Background**: User clicks "Run in Background" button. Job submits, toast notification appears, user navigates freely. Floating progress widget shows active jobs from any page.

Both modes use the same backend (always async). The difference is purely frontend UX — foreground keeps the page in a "processing" state; background dismisses it immediately.

## Components

### 1. Laravel Backend Changes

#### Modified Endpoint: Async Process

Modify `AudioProcessingController::process()` to **always** be async:
- Create `JobQueue` record with status `queued`
- Fire-and-forget HTTP POST to voice engine with `callback_url` (Laravel's internal URL) and `job_uuid`
- Return `202 Accepted` with `{ job_id: uuid, status: "queued" }` immediately
- The voice engine processes in a background thread and reports back via webhooks

The old synchronous path is removed. All processing is async.

#### New Endpoint: Progress Webhook (internal)

`POST /internal/jobs/{uuid}/progress`

Protected by shared secret header: voice engine sends `X-Internal-Token: {VOICE_ENGINE_INTERNAL_TOKEN}` (configured in both `.env` files). Laravel middleware rejects requests without valid token.

```php
// Request body
{
  "progress": 45,          // 0-100
  "message": "Converting voice 1 (Lexi)...",
  "step": "convert_voice", // machine-readable step ID
  "step_number": 3,        // current step
  "total_steps": 6         // total steps
}
```

Updates `JobQueue` record: `progress`, `progress_message`, `step_number`, `total_steps`.

#### New Endpoint: Completion Webhook (internal)

`POST /internal/jobs/{uuid}/complete`

Same `X-Internal-Token` auth as progress webhook.

```php
// Request body (success)
{
  "status": "completed",
  "output_path": "user-generations/1/abc123.wav",  // S3/MinIO key
  "sample_rate": 44100,
  "duration": 245.3
}

// Request body (failure)
{
  "status": "failed",
  "error": "Audio buffer is not finite everywhere"
}
```

Updates `JobQueue`: status, output_path, completed_at, error_message. Duration and sample_rate stored in the existing `parameters` JSON field.

#### Modified: GET /api/jobs/{uuid}

Already exists. Returns job with progress info. `output_url` is a **signed temporary URL** computed at read time (not stored in DB), generated from `output_path` via `Storage::disk('s3')->temporaryUrl()`:

```json
{
  "id": "abc-123",
  "status": "processing",
  "progress": 45,
  "progress_message": "Converting voice 1 (Lexi)...",
  "step_number": 3,
  "total_steps": 6,
  "output_url": null,
  "created_at": "2026-05-03T23:00:00Z",
  "saved": false
}
```

When completed:
```json
{
  "id": "abc-123",
  "status": "completed",
  "progress": 100,
  "progress_message": "Complete",
  "output_url": "http://localhost:9000/morphvox/user-generations/1/abc-123.wav?X-Amz-Expires=3600&...",
  "duration": 245.3,
  "sample_rate": 44100,
  "saved": false,
  "created_at": "2026-05-03T23:00:00Z",
  "completed_at": "2026-05-03T23:01:30Z"
}
```

#### New Endpoint: Save/Unsave Job Output

`POST /api/jobs/{uuid}/save` — sets `saved=true`, prevents auto-deletion  
`POST /api/jobs/{uuid}/unsave` — sets `saved=false`, subject to 24h cleanup

#### New Endpoint: Cancel Job

`POST /api/jobs/{uuid}/cancel`

1. Sets job status to `cancelled` in DB
2. Voice engine polls status before each pipeline step (see status endpoint below)

#### New Internal Endpoint: Job Status Check

`GET /internal/jobs/{uuid}/status`

Same `X-Internal-Token` auth. Returns:
```json
{ "status": "processing" }  // or "cancelled"
```

Voice engine polls this before each major pipeline step. If `cancelled`, it stops immediately, cleans temp files, and exits without sending a completion webhook.

#### Scheduled Command: Cleanup Old Generations

`php artisan jobs:cleanup` (runs hourly via cron/scheduler)
- Delete files for jobs where `saved=false` AND `completed_at < now() - 24 hours`
- Update job status to indicate file removed (keep record for history)

### 2. Voice Engine Changes

#### Async Processing Endpoint

Modify `/audio/process` to accept `callback_url`, `job_uuid`, and `user_id` parameters:

- If `callback_url` present: process in background thread, return `{ status: "accepted", job_uuid }` immediately (202)
- If no `callback_url`: behave synchronously as before (for manual testing/debugging only — Laravel always sends callback_url in production)

#### Progress Callback System

At each pipeline step, POST progress to Laravel:

```python
def report_progress(callback_url, job_uuid, progress, message, step, step_number, total_steps):
    requests.post(f"{callback_url}/internal/jobs/{job_uuid}/progress", json={
        "progress": progress,
        "message": message,
        "step": step,
        "step_number": step_number,
        "total_steps": total_steps,
    }, timeout=5)
```

Progress steps for Voice Swap (multi-voice, 2 voices):

| Step | Progress | Message |
|------|----------|---------|
| 1 | 5% | Downloading audio... |
| 2 | 15% | Separating main vocal (HP5)... |
| 3 | 30% | Converting voice 1 ({model_name})... |
| 4 | 45% | Extracting backing vocal... |
| 5 | 60% | Converting voice 2 ({model_name})... |
| 6 | 75% | Generating clean instrumental (HP3)... |
| 7 | 90% | Mixing final output... |
| 8 | 100% | Complete |

For Vocal Split:

| Step | Progress | Message |
|------|----------|---------|
| 1 | 20% | Separating vocals from instrumental... |
| 2 | 90% | Finalizing... |
| 3 | 100% | Complete |

For Voice Convert:

| Step | Progress | Message |
|------|----------|---------|
| 1 | 20% | Loading model... |
| 2 | 50% | Converting voice... |
| 3 | 90% | Finalizing... |
| 4 | 100% | Complete |

For TTS:

| Step | Progress | Message |
|------|----------|---------|
| 1 | 20% | Generating speech... |
| 2 | 60% | Applying voice conversion... |
| 3 | 90% | Finalizing... |
| 4 | 100% | Complete |

#### Output File Saving

On completion, voice engine uploads output to MinIO via `boto3` S3 client:
- Bucket: `morphvox`
- Key: `user-generations/{user_id}/{job_uuid}.wav`
- Uses same AWS credentials configured in voice engine environment
- Reports completion to Laravel with the S3 key as `output_path`

#### Cancellation

Voice engine calls `GET {callback_url}/internal/jobs/{uuid}/status` before each major pipeline step (separation, conversion, mixing). If response returns `status: "cancelled"`, processing stops immediately, temp files are cleaned, and no completion webhook is sent.

### 3. Frontend Changes

#### Job Submission (Modified)

All processing pages (Song Remix, Voice Convert, TTS, Vocal Splitter) change from:
```
await audioProcessingApi.process(payload) → show result
```
To:
```
const { job_id } = await audioProcessingApi.submitJob(payload)
→ start polling(job_id)
→ show progress from polling
→ on complete: show result
```

#### New: AudioJobProvider (React Context)

Global context (in layout) that manages all active jobs:

```typescript
interface ActiveJob {
  id: string;
  type: 'swap' | 'split' | 'convert' | 'tts';
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  progressMessage: string;
  stepNumber: number;
  totalSteps: number;
  outputUrl: string | null;
  modelName: string;
  saved: boolean;
  createdAt: string;
  completedAt: string | null;
}

interface AudioJobContextValue {
  activeJobs: ActiveJob[];
  submitJob: (payload: AudioProcessRequest) => Promise<string>; // returns job_id
  cancelJob: (jobId: string) => Promise<void>;
  saveJob: (jobId: string) => Promise<void>;
  unsaveJob: (jobId: string) => Promise<void>;
  dismissJob: (jobId: string) => void; // remove from active list
}
```

- On mount: fetches recent jobs (last 24h) from `GET /api/jobs?status=processing,completed`
- Polls active (non-complete) jobs every 2-3 seconds
- Stops polling for each job when status reaches terminal state

#### New: FloatingJobsWidget (Component)

Rendered in the dashboard layout, always visible:

- **Collapsed state**: Small badge in bottom-right corner showing count of active jobs (e.g., "2 ⚡")
- **Expanded state** (on click): Shows list of jobs with:
  - Progress bar + current step message
  - Model name + type icon
  - Cancel button (for in-progress)
  - Play / Download / Save buttons (for completed)
  - "Save" toggle to prevent auto-deletion
  - Dismiss button to remove from list
- Auto-expands when a new job completes (with subtle animation)
- Plays a subtle notification sound on completion (optional, can be disabled)

#### Modified: Processing Pages

Each page (Song Remix, Voice Convert, etc.) gets:
- New "Run in Background" button alongside the main action button
- When "Run in Background" is clicked:
  - Submit job via `submitJob()` from context
  - Show "Job submitted! Track progress in the corner" toast
  - Reset the page form state (ready for next job)
- When main button is clicked (foreground):
  - Submit job the same way but stay on page showing real progress
  - Progress bar uses actual polling data instead of fake percentages
  - On complete: show result inline (same as today)

### 4. Storage

Output files are stored in MinIO (S3-compatible, running locally on the Vast.ai instance):

```
Bucket: morphvox
Key: user-generations/{user_id}/{job_uuid}.wav
```

- Voice engine uploads via `boto3` S3 client (same credentials in voice engine `.env`)
- Laravel serves via `Storage::disk('s3')->temporaryUrl()` (signed URLs, 1-hour expiry)
- No direct nginx exposure of user files — all access through signed URLs

### 5. Security

- **Webhook auth**: Voice engine sends `X-Internal-Token` header with a shared secret (set in both Laravel `.env` and voice engine `.env`). Laravel middleware validates token on all `/internal/*` routes.
- **File access**: Users never access MinIO directly. Laravel generates signed temporary URLs (1-hour expiry) at read time.
- **Job access**: Users can only poll/cancel their own jobs (existing `user_id` check).
- **Admin**: Can see all jobs and generate download URLs for any user's files via admin panel.

### 6. Database Changes

Add to `jobs_queue` table:
```sql
ALTER TABLE jobs_queue ADD COLUMN step_number INT DEFAULT 0;
ALTER TABLE jobs_queue ADD COLUMN total_steps INT DEFAULT 1;
ALTER TABLE jobs_queue ADD COLUMN saved BOOLEAN DEFAULT FALSE;
```

Notes:
- `output_path` column already exists — used to store the S3 key
- `output_url` is NOT stored — computed as signed URL at read time
- `sample_rate` and `duration` stored in existing `parameters` JSON column
- `progress` and `progress_message` columns already exist
- `updated_at` (Eloquent timestamp) is auto-updated on every progress webhook — used by stale job reaper

### 7. Stale Job Reaper

A scheduled command (`php artisan jobs:reap-stale`, runs every 5 minutes) marks jobs as `failed` if:
- Status is `processing` AND `updated_at < now() - 10 minutes` (no progress update in 10 min)
- Status is `queued` AND `created_at < now() - 5 minutes` (never started)

This handles voice engine crashes or network failures where no completion webhook fires.

### 8. Usage Tracking

The completion webhook handler records a `UsageEvent` (same as the current sync controller does on success). This ensures usage tracking isn't lost in the async flow.

### 8. Admin Visibility

The existing admin Jobs page already displays all jobs. No additional admin work is needed for MVP — admins can already view status, and the signed URL generation gives them download access to any output.

## Success Criteria

1. User submits a voice swap for a 5-minute song → gets immediate confirmation
2. User navigates to another page → sees progress in floating widget
3. Processing completes → user gets notification, can play/download
4. Cloudflare timeout no longer affects generation success
5. Progress messages accurately reflect current pipeline step
6. Files auto-delete after 24h unless saved
7. All generation types (swap, split, convert, TTS) support this flow

## Out of Scope (for now)

- Queue system (only one job at a time per user — concurrent jobs can be added later)
- Shared/public gallery of generations
- WebSocket/SSE for sub-second updates (polling at 2-3s is fine)
- Audio streaming (progressive download while still generating)
