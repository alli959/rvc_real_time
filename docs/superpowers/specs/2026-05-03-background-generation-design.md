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
3. **Process**: Voice engine receives job, submits to ThreadPoolExecutor, returns 202 immediately.
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

**Route key change**: Override `JobQueue::getRouteKeyName()` to return `'uuid'`. All job routes use UUID in URLs (not integer ID). Existing routes (`{job}`) will resolve via UUID after this change.

#### Modified Endpoint: Async Process

Modify `AudioProcessingController::process()` to **always** be async:
- Create `JobQueue` record with status `queued` (skip `pending` — job is immediately dispatched)
- POST to voice engine with `callback_url` and `job_uuid` (see §callback_url below)
- **callback_url**: Send full webhook URLs in the dispatch payload to avoid path construction issues. Laravel sends:
  ```json
  {
    "progress_url": "http://localhost/api/internal/jobs/{uuid}/progress",
    "complete_url": "http://localhost/api/internal/jobs/{uuid}/complete",
    "status_url": "http://localhost/api/internal/jobs/{uuid}/status"
  }
  ```
  Voice engine uses these URLs directly (no path construction). Value of base URL comes from `APP_URL` env var in Laravel.
- **Check response**: if voice engine returns non-2xx (down, 500, timeout), immediately mark job as `failed` with error "Voice engine unavailable" and return 503 to frontend
- If voice engine returns 202, return `202 Accepted` with `{ job_id: uuid, status: "queued" }` to frontend
- Use a short HTTP timeout (5s) for the dispatch call — we're only waiting for the 202 acknowledgment, not processing

The old synchronous path is removed. All processing is async.

#### Concurrency Guard

Wrapped in `DB::transaction()`. Before creating a new job, check if user already has an active job (status in `queued`, `processing`):
- Use `SELECT ... FOR UPDATE` to lock the user's active jobs row, preventing race conditions from double-clicks or frontend retries
- If yes: return `429 Too Many Requests` with `{ error: "You have an active job in progress", active_job_id: uuid }`
- Frontend shows this to user and offers to cancel the existing job or wait
- Admin users bypass this limit
- The entire check + job creation happens atomically within the transaction

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
  "total_steps": 8         // total steps (varies by job type)
}
```

Updates `JobQueue` record: `progress`, `progress_message`, `step_number`, `total_steps`.

**Ordering guard**: Only update if incoming `step_number >= current step_number` on the record. Prevents stale/reordered updates from overwriting newer state.

**Terminal state guard**: If job is in a terminal state (`completed`, `failed`, `cancelled`), return 200 OK but do not update. If UUID doesn't exist, return 404. Only accept progress updates when status is `queued` or `processing`.

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

**Idempotency**: If job is already `completed` or `failed`, ignore duplicate webhook calls (return 200 OK but don't update). This handles network retries safely.

**Cancel race**: If a completion webhook arrives for a `cancelled` job (voice engine finished before checking cancellation status), accept the completion — override status to `completed`. The user gets their result; cancellation was "too late" and that's fine. Better to give the user a completed file than to discard work already done.

#### Modified: GET /api/jobs/{uuid}

Already exists. Returns job with progress info. `output_url` is a **Laravel proxy endpoint** (`/api/jobs/{uuid}/stream`) — only populated when job is completed and file exists:

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
  "output_url": "/api/jobs/abc-123/stream",
  "duration": 245.3,
  "sample_rate": 44100,
  "saved": false,
  "created_at": "2026-05-03T23:00:00Z",
  "completed_at": "2026-05-03T23:01:30Z"
}
```

Note: `output_url` is a **Laravel proxy endpoint** (not a direct MinIO presigned URL). MinIO runs on localhost:9000 which is inaccessible to remote users. Laravel streams the file through `GET /api/jobs/{uuid}/stream` which internally fetches from MinIO and pipes to the response. This avoids exposing MinIO to the internet while working through Cloudflare.

#### New Endpoint: Stream Job Output

`GET /api/jobs/{uuid}/stream`

Auth: Job owner or admin. Returns the audio file streamed from MinIO.

- Uses `Storage::disk('s3')->readStream($job->output_path)` piped to `response()->stream()`
- Response headers:
  - `Content-Type: audio/wav`
  - `Content-Length: {file_size_bytes}`
  - `Content-Disposition: inline; filename="{job_uuid}.wav"` (or `attachment` if `?download=1` query param)
  - `Accept-Ranges: bytes` (enables browser seeking in audio player)
- Supports HTTP Range requests for partial content (206) — required for `<audio>` element seeking
- Memory-safe: streams in chunks (8KB), never loads full file into PHP memory
- Returns 404 if `output_path` is null or file doesn't exist in MinIO
- Returns 403 if user doesn't own the job (and isn't admin)

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
- Delete files for jobs where `status='failed'` AND `output_path IS NOT NULL` AND `updated_at < now() - 24 hours` (handles orphaned files from failed webhook delivery)
- Update job status to indicate file removed (keep record for history)

### 2. Voice Engine Changes

#### Execution Model

Processing runs in a **`concurrent.futures.ThreadPoolExecutor`** with `max_workers=1`. This is acceptable because:
- ML inference is GPU-bound (releases GIL during CUDA operations)
- UVR5 separation and RVC inference both use PyTorch which releases GIL for GPU ops
- FastAPI's async event loop remains responsive for health checks and new request handling
- Single worker ensures one GPU job at a time (matches concurrency guard in Laravel)

The main event loop stays free to serve health checks, status endpoints, and accept the next job dispatch.

**Busy rejection**: If the executor already has a job running (or queued internally), reject new submissions with `503 Service Unavailable` + `{ "error": "GPU busy", "retry_after": 30 }`. Laravel receives this and marks the job `failed` with error "Voice engine busy, please try again". This prevents the internal executor queue from growing and avoids stale reaper false-positives on legitimately queued jobs from other users.

Implementation: Track with a simple `AtomicInteger` or `threading.Lock` counter. Before submitting to executor, check count. This gives immediate feedback to Laravel.

#### Async Processing Endpoint

Modify `/audio/process` to accept `callback_url`, `job_uuid`, and `user_id` parameters:

- If `callback_url` present: submit to ThreadPoolExecutor, return `{ status: "accepted", job_uuid }` immediately (202)
- If no `callback_url`: behave synchronously as before (for manual testing/debugging only — Laravel always sends callback_url in production)

#### Progress Callback System

At each pipeline step, POST progress to Laravel using the `progress_url` received in the dispatch payload. Failures are non-fatal (logged but don't stop processing):

```python
def report_progress(progress_url, progress, message, step, step_number, total_steps):
    try:
        requests.post(progress_url, json={
            "progress": progress,
            "message": message,
            "step": step,
            "step_number": step_number,
            "total_steps": total_steps,
        }, headers={"X-Internal-Token": INTERNAL_TOKEN}, timeout=5)
    except Exception:
        logger.warning(f"Failed to report progress to {progress_url}, continuing")
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

#### Heartbeat for Long-Running Steps

For steps that may exceed 5 minutes (e.g., voice conversion on 10+ minute audio), the processing thread sends a periodic heartbeat every 3 minutes. This re-sends the current progress (same `step_number`) to reset `updated_at` in Laravel and prevent the stale reaper from false-killing the job.

```python
import threading

class HeartbeatTimer:
    def __init__(self, interval, callback):
        self._timer = None
        self.interval = interval
        self.callback = callback
    
    def start(self):
        self._timer = threading.Timer(self.interval, self._run)
        self._timer.daemon = True
        self._timer.start()
    
    def _run(self):
        self.callback()
        self.start()  # reschedule
    
    def stop(self):
        if self._timer:
            self._timer.cancel()
```

Usage: Start heartbeat before each long step, stop it after. The callback re-calls `report_progress()` with the same values.

#### Output File Saving

On completion, voice engine uploads output to MinIO via `boto3` S3 client (new dependency — add to `requirements.txt`):
- Bucket: `morphvox`
- Key: `user-generations/{user_id}/{job_uuid}.wav`
- Uses same AWS credentials configured in voice engine environment
- **Retry**: Up to 3 attempts with 2s backoff on upload failure
- **If upload still fails**: report `failed` status to Laravel with error "Failed to save output file"
- Reports completion to Laravel with the S3 key as `output_path`

#### Completion Webhook Delivery

Voice engine POSTs completion to Laravel with retry:
- Up to 3 attempts with exponential backoff (2s, 4s, 8s)
- If all retries fail: log error. File exists in MinIO but job stays `processing` — stale reaper will mark it failed after 15 min. This is acceptable for MVP; the file can be recovered manually by admin if needed.

#### Cancellation

Voice engine calls `GET {status_url}` (received in dispatch payload) before each major pipeline step (separation, conversion, mixing). If response returns `status: "cancelled"`, processing stops immediately and temp files are cleaned.

**If status check fails** (network error): assume NOT cancelled and continue processing. Better to complete a cancelled job than abort a running one due to transient network issue.

**Cancellation latency**: Cancellation takes effect at the next step boundary. A single step (e.g., voice conversion) can take 30-60s. The UI should document this: "Cancellation may take up to 60 seconds to take effect."

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
  type: 'audio_swap' | 'audio_split' | 'audio_convert' | 'tts';  // matches backend JobQueue::TYPE_* constants
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

- On mount: fetches active jobs (`status=queued,processing`) + last 5 completed jobs
- **Polling strategy**: 
  - First poll at 1s after submission (catches fast TTS/convert jobs)
  - Then every 3s while `queued` or `processing`
  - Stops polling when status reaches terminal state
- Stops polling for each job when status reaches terminal state

#### New: FloatingJobsWidget (Component)

Rendered in the dashboard layout, always visible:

- **Collapsed state**: Small badge in bottom-right corner showing count of non-dismissed jobs (active + recently completed, e.g., "1 ⚡" for an in-progress job, "1 ✓" for a just-completed one)
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
- **Quick-job optimization (TTS, Voice Convert)**: First poll fires at 1s. If job completes on first poll, user experience is nearly identical to the old synchronous flow — no jarring "submitted!" → "complete!" flicker.
- If `429 Too Many Requests` returned: show inline error "You already have a generation in progress. Wait for it to finish or cancel it." with link to the active job in the floating widget.

### 4. Storage

Output files are stored in MinIO (S3-compatible, running locally on the Vast.ai instance):

```
Bucket: morphvox  (set via AWS_BUCKET env var; code default is 'voiceforge' but production .env overrides to 'morphvox')
Key: user-generations/{user_id}/{job_uuid}.wav
```

- Voice engine uploads via `boto3` S3 client (same credentials in voice engine `.env`)
- Laravel proxies downloads via `GET /api/jobs/{uuid}/stream` — streams directly from MinIO through PHP (avoids exposing MinIO to the internet)
- For large files (>100MB), consider adding `Content-Length` and `Accept-Ranges` headers to support resumable downloads
- No direct nginx/internet exposure of MinIO — all access through Laravel proxy

### 5. Security

- **Webhook auth**: Voice engine sends `X-Internal-Token` header with a shared secret (≥32 random bytes, set in both Laravel `.env` and voice engine `.env`). Laravel middleware validates token on all `/internal/*` routes.
- **Internal route isolation**: **Add** an nginx rule to deny external access to `/internal/*` routes. Only requests from `127.0.0.1` (voice engine on same host) are allowed. Example: `location /api/internal/ { allow 127.0.0.1; deny all; }`. The `X-Internal-Token` header provides a second layer of defense.
- **XSS prevention**: `progress_message` from webhooks is stored as plain text. Frontend renders it with React's default escaping (no `dangerouslySetInnerHTML`).
- **File access**: Users never access MinIO directly. Laravel proxies file downloads through `GET /api/jobs/{uuid}/stream`.
- **Job access**: Users can only poll/cancel their own jobs (existing `user_id` check).
- **Admin**: Can see all jobs and download any user's files via admin panel.

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
- Status is `processing` AND `updated_at < now() - 15 minutes` (no progress update in 15 min)
- Status is `queued` AND `created_at < now() - 5 minutes` (never started)

This handles voice engine crashes or network failures where no completion webhook fires.

The 15-minute threshold is safe because the voice engine sends progress updates at each step boundary (typically every 10-60s for a step). For very long audio files where a single step might approach 10+ minutes, the voice engine sends a periodic heartbeat (progress update with same step_number) every 5 minutes within long-running steps.

### 8. Usage Tracking

The completion webhook handler records a `UsageEvent` (same as the current sync controller does on success). This ensures usage tracking isn't lost in the async flow.

### 9. Admin Visibility

The existing admin Jobs page already displays all jobs. No additional admin work is needed for MVP — admins can already view status, and the signed URL generation gives them download access to any output.

### 10. Database Indexes

Add composite index for the stale-reaper query:
```sql
CREATE INDEX idx_jobs_status_updated ON jobs_queue (status, updated_at);
```

Note: `(user_id, status)` index already exists in the original migration.

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
