# Background Generation + Real-Time Progress Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert synchronous audio processing to async job-based system with real-time progress polling and background generation support.

**Architecture:** Laravel dispatches jobs to voice engine via HTTP, voice engine reports progress/completion via internal webhooks, frontend polls for updates. Files stored in MinIO, served via Laravel proxy.

**Tech Stack:** Laravel 11 (PHP), FastAPI (Python), Next.js 14 (React), MinIO (S3), MariaDB, supervisord

---

## File Structure

### Laravel Backend (apps/api/)
| File | Responsibility |
|------|---------------|
| `database/migrations/2026_05_04_000001_add_async_job_columns.php` | Add step_number, total_steps, saved columns + index |
| `app/Models/JobQueue.php` | Add getRouteKeyName, new fillables, scopes |
| `app/Http/Controllers/Api/AudioProcessingController.php` | Rewrite to async dispatch |
| `app/Http/Controllers/Api/JobController.php` | Add stream, save/unsave, modify cancel + show |
| `app/Http/Controllers/Internal/JobWebhookController.php` | Progress + completion webhooks |
| `routes/api.php` | New internal routes, stream route |
| `app/Console/Commands/ReapStaleJobs.php` | Stale job reaper (15min threshold) |
| `app/Console/Commands/CleanupJobFiles.php` | 24h file cleanup |
| `app/Console/Kernel.php` | Schedule commands |

### Voice Engine (services/voice-engine/)
| File | Responsibility |
|------|---------------|
| `app/job_runner.py` | ThreadPoolExecutor wrapper, busy tracking, heartbeat |
| `app/s3_client.py` | boto3 MinIO upload with retry |
| `app/webhook_client.py` | Progress/completion reporting with retry |
| `app/http_api.py` | Modify /audio/process for async dispatch |
| `requirements.txt` | Add boto3 |

### Frontend (apps/web/src/)
| File | Responsibility |
|------|---------------|
| `contexts/audio-job-context.tsx` | AudioJobProvider — polling, state management |
| `components/floating-jobs-widget.tsx` | Floating progress widget |
| `lib/api.ts` | Add job submission/polling/stream API functions |
| `app/dashboard/song-remix/page.tsx` | Integrate async submission + progress |
| `components/providers.tsx` | Wrap with AudioJobProvider |

### Infrastructure
| File | Responsibility |
|------|---------------|
| `infra/nginx/internal-block.conf` | Block /api/internal/* from external |

---

## Chunk 1: Laravel Backend — Database, Model & Internal Webhooks

### Task 1: Database Migration

**Files:**
- Create: `apps/api/database/migrations/2026_05_04_000001_add_async_job_columns.php`

- [ ] **Step 1: Create migration**

```php
<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('jobs_queue', function (Blueprint $table) {
            $table->unsignedInteger('step_number')->default(0)->after('progress_message');
            $table->unsignedInteger('total_steps')->default(1)->after('step_number');
            $table->boolean('saved')->default(false)->after('worker_id');
            $table->index(['status', 'updated_at'], 'idx_jobs_status_updated');
        });
    }

    public function down(): void
    {
        Schema::table('jobs_queue', function (Blueprint $table) {
            $table->dropIndex('idx_jobs_status_updated');
            $table->dropColumn(['step_number', 'total_steps', 'saved']);
        });
    }
};
```

- [ ] **Step 2: Run migration**

Run: `cd apps/api && php artisan migrate`
Expected: Migration runs successfully, adds 3 columns and 1 index.

- [ ] **Step 3: Commit**

```bash
git add apps/api/database/migrations/2026_05_04_000001_add_async_job_columns.php
git commit -m "feat: add async job columns (step_number, total_steps, saved)"
```

### Task 2: JobQueue Model Updates

**Files:**
- Modify: `apps/api/app/Models/JobQueue.php`

- [ ] **Step 1: Add getRouteKeyName override and new fillables**

Add to `JobQueue` model:
```php
// After existing $casts array:
public function getRouteKeyName(): string
{
    return 'uuid';
}
```

Add to `$fillable` array: `'step_number'`, `'total_steps'`, `'saved'`

Add to `$casts` array: `'saved' => 'boolean'`

Add new scopes:
```php
public function scopeActive($query)
{
    return $query->whereIn('status', [self::STATUS_QUEUED, self::STATUS_PROCESSING]);
}

public function scopeStale($query, int $processingMinutes = 15, int $queuedMinutes = 5)
{
    return $query->where(function ($q) use ($processingMinutes, $queuedMinutes) {
        $q->where(function ($sub) use ($processingMinutes) {
            $sub->where('status', self::STATUS_PROCESSING)
                ->where('updated_at', '<', now()->subMinutes($processingMinutes));
        })->orWhere(function ($sub) use ($queuedMinutes) {
            $sub->where('status', self::STATUS_QUEUED)
                ->where('created_at', '<', now()->subMinutes($queuedMinutes));
        });
    });
}
```

Add helper:
```php
public function isTerminal(): bool
{
    return in_array($this->status, [
        self::STATUS_COMPLETED,
        self::STATUS_FAILED,
        self::STATUS_CANCELLED,
    ]);
}
```

- [ ] **Step 2: Verify model loads without errors**

Run: `cd apps/api && php artisan tinker --execute="new \App\Models\JobQueue();"` 
Expected: No errors

- [ ] **Step 3: Commit**

```bash
git add apps/api/app/Models/JobQueue.php
git commit -m "feat: add UUID route key, async scopes, and new columns to JobQueue"
```

### Task 3: Internal Webhook Controller

**Files:**
- Create: `apps/api/app/Http/Controllers/Internal/JobWebhookController.php`

- [ ] **Step 1: Create the controller**

```php
<?php

namespace App\Http\Controllers\Internal;

use App\Http\Controllers\Controller;
use App\Models\JobQueue;
use App\Models\UsageEvent;
use Illuminate\Http\Request;

class JobWebhookController extends Controller
{
    /**
     * Receive progress update from voice engine.
     * POST /api/internal/jobs/{job}/progress
     */
    public function progress(Request $request, JobQueue $job)
    {
        // Terminal state guard
        if ($job->isTerminal()) {
            return response()->json(['status' => 'ignored', 'reason' => 'terminal_state']);
        }

        $validated = $request->validate([
            'progress' => 'required|integer|min:0|max:100',
            'message' => 'required|string|max:255',
            'step' => 'required|string|max:50',
            'step_number' => 'required|integer|min:0',
            'total_steps' => 'required|integer|min:1',
        ]);

        // Ordering guard: only update if step_number >= current
        if ($validated['step_number'] < $job->step_number) {
            return response()->json(['status' => 'ignored', 'reason' => 'stale_update']);
        }

        $job->update([
            'status' => JobQueue::STATUS_PROCESSING,
            'progress' => $validated['progress'],
            'progress_message' => $validated['message'],
            'step_number' => $validated['step_number'],
            'total_steps' => $validated['total_steps'],
        ]);

        return response()->json(['status' => 'ok']);
    }

    /**
     * Receive completion notification from voice engine.
     * POST /api/internal/jobs/{job}/complete
     */
    public function complete(Request $request, JobQueue $job)
    {
        // Idempotency: ignore if already completed or failed
        if (in_array($job->status, [JobQueue::STATUS_COMPLETED, JobQueue::STATUS_FAILED])) {
            return response()->json(['status' => 'ignored', 'reason' => 'already_terminal']);
        }

        $validated = $request->validate([
            'status' => 'required|in:completed,failed',
            'output_path' => 'nullable|string|max:500',
            'output_paths' => 'nullable|array',
            'output_paths.*' => 'string|max:500',
            'sample_rate' => 'nullable|integer',
            'duration' => 'nullable|numeric',
            'error' => 'nullable|string|max:1000',
        ]);

        if ($validated['status'] === 'completed') {
            // Cancel race: accept completion even if cancelled
            $params = $job->parameters ?? [];
            if ($validated['sample_rate'] ?? null) {
                $params['sample_rate'] = $validated['sample_rate'];
            }
            if ($validated['duration'] ?? null) {
                $params['duration'] = $validated['duration'];
            }
            if ($validated['output_paths'] ?? null) {
                $params['output_paths'] = $validated['output_paths'];
            }

            $job->update([
                'status' => JobQueue::STATUS_COMPLETED,
                'output_path' => $validated['output_path'],
                'parameters' => $params,
                'progress' => 100,
                'progress_message' => 'Complete',
                'completed_at' => now(),
            ]);

            // Record usage event
            UsageEvent::create([
                'user_id' => $job->user_id,
                'event_type' => 'audio_processing',
                'metadata' => [
                    'job_uuid' => $job->uuid,
                    'type' => $job->type,
                    'duration' => $validated['duration'] ?? null,
                ],
            ]);
        } else {
            $job->update([
                'status' => JobQueue::STATUS_FAILED,
                'error_message' => $validated['error'] ?? 'Unknown error',
                'completed_at' => now(),
            ]);
        }

        return response()->json(['status' => 'ok']);
    }

    /**
     * Return job status for cancellation polling.
     * GET /api/internal/jobs/{job}/status
     */
    public function status(JobQueue $job)
    {
        return response()->json([
            'status' => $job->status,
        ]);
    }
}
```

- [ ] **Step 2: Register routes**

Add to `routes/api.php` inside the existing `Route::prefix('internal')->middleware('internal.token')` group:

```php
// Job webhooks (voice engine → Laravel)
Route::post('/jobs/{job}/progress', [\App\Http\Controllers\Internal\JobWebhookController::class, 'progress']);
Route::post('/jobs/{job}/complete', [\App\Http\Controllers\Internal\JobWebhookController::class, 'complete']);
Route::get('/jobs/{job}/status', [\App\Http\Controllers\Internal\JobWebhookController::class, 'status']);
```

- [ ] **Step 3: Test routes resolve**

Run: `cd apps/api && php artisan route:list --path=internal`
Expected: Shows the 3 new internal routes with correct controller methods.

- [ ] **Step 4: Commit**

```bash
git add apps/api/app/Http/Controllers/Internal/JobWebhookController.php apps/api/routes/api.php
git commit -m "feat: add internal webhook endpoints for job progress/completion"
```

### Task 4: Async Dispatch in AudioProcessingController

**Files:**
- Modify: `apps/api/app/Http/Controllers/Api/AudioProcessingController.php`

- [ ] **Step 1: Rewrite the process() method for async dispatch**

Replace the synchronous `Http::timeout(300)->post(...)` block (around lines 171-260) with:

```php
// Concurrency guard (wrapped in transaction)
$activeJob = null;
$job = \DB::transaction(function () use ($user, $validated, $voiceModel, $voiceCount, $mode, $sampleRate, $voiceConfigs, &$activeJob) {
    $activeJob = JobQueue::forUser($user->id)
        ->active()
        ->lockForUpdate()
        ->first();

    if ($activeJob && !$user->is_admin) {
        return null; // Will return 429 below
    }

    // Map mode to job type
    $typeMap = [
        'swap' => JobQueue::TYPE_AUDIO_SWAP,
        'split' => JobQueue::TYPE_AUDIO_SPLIT,
        'convert' => JobQueue::TYPE_AUDIO_CONVERT,
    ];

    return JobQueue::create([
        'user_id' => $user->id,
        'voice_model_id' => $voiceModel?->id,
        'type' => $typeMap[$mode] ?? JobQueue::TYPE_AUDIO_SWAP,
        'status' => JobQueue::STATUS_QUEUED,
        'parameters' => $validated,
    ]);
});

if (!$job) {
    return response()->json([
        'error' => 'You have an active job in progress',
        'active_job_id' => $activeJob->uuid,
    ], 429);
}

// Build webhook URLs
$baseUrl = config('app.url');
$webhookUrls = [
    'progress_url' => "{$baseUrl}/api/internal/jobs/{$job->uuid}/progress",
    'complete_url' => "{$baseUrl}/api/internal/jobs/{$job->uuid}/complete",
    'status_url' => "{$baseUrl}/api/internal/jobs/{$job->uuid}/status",
];

// Build voice engine payload
$payload = [
    'job_uuid' => $job->uuid,
    'user_id' => $user->id,
    ...$webhookUrls,
    // ... existing payload fields (audio, mode, model_path, etc.)
];

// Dispatch to voice engine (5s timeout — only waiting for 202 ack)
$voiceEngineUrl = config('services.voice_engine.url', 'http://localhost:8001');
try {
    $response = Http::timeout(5)->post("{$voiceEngineUrl}/audio/process", $payload);
} catch (\Exception $e) {
    $job->update([
        'status' => JobQueue::STATUS_FAILED,
        'error_message' => 'Voice engine unavailable',
        'completed_at' => now(),
    ]);
    return response()->json([
        'error' => 'Voice engine unavailable',
        'job_id' => $job->uuid,
    ], 503);
}

if (!$response->successful()) {
    $detail = $response->json('error') ?? $response->json('detail') ?? 'Unknown error';
    $job->update([
        'status' => JobQueue::STATUS_FAILED,
        'error_message' => is_string($detail) ? $detail : json_encode($detail),
        'completed_at' => now(),
    ]);
    return response()->json([
        'error' => 'Voice engine rejected job',
        'message' => $detail,
        'job_id' => $job->uuid,
    ], $response->status());
}

return response()->json([
    'job_id' => $job->uuid,
    'status' => 'queued',
], 202);
```

- [ ] **Step 2: Verify controller syntax**

Run: `cd apps/api && php -l app/Http/Controllers/Api/AudioProcessingController.php`
Expected: No parse errors

- [ ] **Step 3: Commit**

```bash
git add apps/api/app/Http/Controllers/Api/AudioProcessingController.php
git commit -m "feat: rewrite AudioProcessingController to async dispatch"
```

### Task 5: Stream Endpoint & Job Controller Updates

**Files:**
- Modify: `apps/api/app/Http/Controllers/Api/JobController.php`
- Modify: `apps/api/routes/api.php`

- [ ] **Step 1: Add stream method to JobController**

```php
/**
 * Stream job output file from MinIO.
 * GET /api/jobs/{job}/stream
 */
public function stream(Request $request, JobQueue $job)
{
    if ($job->user_id !== $request->user()->id && !$request->user()->is_admin) {
        abort(403, 'Access denied');
    }

    if (!$job->isCompleted()) {
        abort(422, 'Job is not completed');
    }

    // Determine which file to stream
    $track = $request->query('track');
    $path = $job->output_path;

    if ($track && $job->type === JobQueue::TYPE_AUDIO_SPLIT) {
        $outputPaths = $job->parameters['output_paths'] ?? [];
        $pathMap = [];
        foreach ($outputPaths as $p) {
            if (str_contains($p, '_vocals.')) $pathMap['vocals'] = $p;
            if (str_contains($p, '_instrumental.')) $pathMap['instrumental'] = $p;
        }
        $path = $pathMap[$track] ?? $path;
    }

    if (!$path || !Storage::disk('s3')->exists($path)) {
        abort(404, 'Output file not found');
    }

    $size = Storage::disk('s3')->size($path);
    $download = $request->query('download') === '1';
    $filename = basename($path);
    $disposition = $download ? "attachment; filename=\"{$filename}\"" : "inline; filename=\"{$filename}\"";

    $headers = [
        'Content-Type' => 'audio/wav',
        'Content-Length' => $size,
        'Content-Disposition' => $disposition,
        'Accept-Ranges' => 'bytes',
    ];

    // Handle Range requests
    $range = $request->header('Range');
    if ($range && preg_match('/bytes=(\d+)-(\d*)/', $range, $matches)) {
        $start = (int) $matches[1];
        $end = $matches[2] !== '' ? (int) $matches[2] : $size - 1;
        $length = $end - $start + 1;

        return response()->stream(function () use ($path, $start, $length) {
            $stream = Storage::disk('s3')->readStream($path);
            fseek($stream, $start);
            $remaining = $length;
            while ($remaining > 0 && !feof($stream)) {
                $chunk = fread($stream, min(8192, $remaining));
                echo $chunk;
                $remaining -= strlen($chunk);
                flush();
            }
            fclose($stream);
        }, 206, array_merge($headers, [
            'Content-Length' => $length,
            'Content-Range' => "bytes {$start}-{$end}/{$size}",
        ]));
    }

    return response()->stream(function () use ($path) {
        $stream = Storage::disk('s3')->readStream($path);
        while (!feof($stream)) {
            echo fread($stream, 8192);
            flush();
        }
        fclose($stream);
    }, 200, $headers);
}
```

- [ ] **Step 2: Update cancel() to support in-progress jobs**

The cancel method already exists at line 136, but needs to handle the async case (cancelling an in-progress job that the voice engine is currently processing):

```php
/**
 * Cancel a job. For queued jobs, marks immediately as cancelled.
 * For in-progress jobs, marks cancelled — voice engine checks status and stops.
 * Note: cancellation of in-progress jobs may take up to 60 seconds.
 */
public function cancel(Request $request, JobQueue $job)
{
    if ($job->user_id !== $request->user()->id) {
        abort(403, 'Access denied');
    }

    if (in_array($job->status, [JobQueue::STATUS_COMPLETED, JobQueue::STATUS_FAILED, JobQueue::STATUS_CANCELLED])) {
        abort(422, 'Cannot cancel a finished job');
    }

    $job->update([
        'status' => JobQueue::STATUS_CANCELLED,
        'completed_at' => now(),
    ]);

    return response()->json([
        'status' => 'cancelled',
        'message' => $job->status === JobQueue::STATUS_PROCESSING
            ? 'Cancellation requested. May take up to 60 seconds to stop.'
            : 'Job cancelled',
    ]);
}
```

- [ ] **Step 3: Add save/unsave methods**

```php
/**
 * Save job output (prevents auto-deletion).
 */
public function save(Request $request, JobQueue $job)
{
    if ($job->user_id !== $request->user()->id) {
        abort(403, 'Access denied');
    }
    $job->update(['saved' => true]);
    return response()->json(['status' => 'saved']);
}

/**
 * Unsave job output (subject to 24h cleanup).
 */
public function unsave(Request $request, JobQueue $job)
{
    if ($job->user_id !== $request->user()->id) {
        abort(403, 'Access denied');
    }
    $job->update(['saved' => false]);
    return response()->json(['status' => 'unsaved']);
}
```

- [ ] **Step 3: Update show() to include new fields and output_url**

Modify the `show()` response to include:
```php
$data = $job->toArray();
$data['output_url'] = null;
$data['output_urls'] = null;

if ($job->isCompleted() && $job->output_path) {
    $data['output_url'] = "/api/jobs/{$job->uuid}/stream";
    
    if ($job->type === JobQueue::TYPE_AUDIO_SPLIT && !empty($job->parameters['output_paths'])) {
        $data['output_urls'] = [
            'vocals' => "/api/jobs/{$job->uuid}/stream?track=vocals",
            'instrumental' => "/api/jobs/{$job->uuid}/stream?track=instrumental",
        ];
    }
}

return response()->json($data);
```

- [ ] **Step 4: Register new routes**

Add to `routes/api.php` inside the authenticated jobs prefix group:
```php
Route::get('/{job}/stream', [JobController::class, 'stream']);
Route::post('/{job}/save', [JobController::class, 'save']);
Route::post('/{job}/unsave', [JobController::class, 'unsave']);
```

- [ ] **Step 5: Verify routes**

Run: `cd apps/api && php artisan route:list --path=jobs`
Expected: Shows stream, save, unsave routes.

- [ ] **Step 6: Commit**

```bash
git add apps/api/app/Http/Controllers/Api/JobController.php apps/api/routes/api.php
git commit -m "feat: add stream, save/unsave endpoints for async jobs"
```

### Task 6: Scheduled Commands (Stale Reaper + Cleanup)

**Files:**
- Create: `apps/api/app/Console/Commands/ReapStaleJobs.php`
- Create: `apps/api/app/Console/Commands/CleanupJobFiles.php`
- Modify: `apps/api/app/Console/Kernel.php` (or `routes/console.php`)

- [ ] **Step 1: Create ReapStaleJobs command**

```php
<?php

namespace App\Console\Commands;

use App\Models\JobQueue;
use Illuminate\Console\Command;

class ReapStaleJobs extends Command
{
    protected $signature = 'jobs:reap-stale';
    protected $description = 'Mark stale jobs as failed (no progress in 15min or never started in 5min)';

    public function handle(): int
    {
        $count = JobQueue::stale()->update([
            'status' => JobQueue::STATUS_FAILED,
            'error_message' => 'Job timed out (no progress received)',
            'completed_at' => now(),
        ]);

        if ($count > 0) {
            $this->info("Marked {$count} stale jobs as failed.");
        }

        return 0;
    }
}
```

- [ ] **Step 2: Create CleanupJobFiles command**

```php
<?php

namespace App\Console\Commands;

use App\Models\JobQueue;
use Illuminate\Console\Command;
use Illuminate\Support\Facades\Storage;

class CleanupJobFiles extends Command
{
    protected $signature = 'jobs:cleanup';
    protected $description = 'Delete output files for unsaved jobs older than 24 hours';

    public function handle(): int
    {
        $disk = Storage::disk('s3');
        $count = 0;

        // Completed unsaved jobs older than 24h
        $jobs = JobQueue::where('saved', false)
            ->where('status', JobQueue::STATUS_COMPLETED)
            ->where('completed_at', '<', now()->subHours(24))
            ->whereNotNull('output_path')
            ->get();

        // Failed jobs with orphaned files older than 24h
        $failedJobs = JobQueue::where('status', JobQueue::STATUS_FAILED)
            ->where('updated_at', '<', now()->subHours(24))
            ->whereNotNull('output_path')
            ->get();

        $allJobs = $jobs->merge($failedJobs);

        foreach ($allJobs as $job) {
            // Delete primary output
            if ($job->output_path && $disk->exists($job->output_path)) {
                $disk->delete($job->output_path);
                $count++;
            }

            // Delete multi-output files
            $outputPaths = $job->parameters['output_paths'] ?? [];
            foreach ($outputPaths as $path) {
                if ($disk->exists($path)) {
                    $disk->delete($path);
                    $count++;
                }
            }

            $job->update(['output_path' => null, 'parameters' => array_diff_key($job->parameters ?? [], ['output_paths' => ''])]);
        }

        if ($count > 0) {
            $this->info("Deleted {$count} output files from {$allJobs->count()} jobs.");
        }

        return 0;
    }
}
```

- [ ] **Step 3: Register in scheduler**

Add to `routes/console.php` (or `app/Console/Kernel.php` schedule method):
```php
use Illuminate\Support\Facades\Schedule;

Schedule::command('jobs:reap-stale')->everyFiveMinutes();
Schedule::command('jobs:cleanup')->hourly();
```

- [ ] **Step 4: Test commands exist**

Run: `cd apps/api && php artisan jobs:reap-stale && php artisan jobs:cleanup`
Expected: Both run without errors (0 jobs affected on fresh DB).

- [ ] **Step 5: Commit**

```bash
git add apps/api/app/Console/Commands/ apps/api/routes/console.php
git commit -m "feat: add stale job reaper and file cleanup scheduled commands"
```

---

## Chunk 2: Voice Engine — Async Processing, S3 Upload & Webhooks

### Task 7: Add boto3 Dependency

**Files:**
- Modify: `services/voice-engine/requirements.txt`

- [ ] **Step 1: Add boto3 to requirements**

Add line: `boto3>=1.34.0`

- [ ] **Step 2: Install**

Run: `cd services/voice-engine && pip install boto3`
Expected: Successfully installed.

- [ ] **Step 3: Commit**

```bash
git add services/voice-engine/requirements.txt
git commit -m "feat: add boto3 dependency for S3/MinIO uploads"
```

### Task 8: Webhook Client Module

**Files:**
- Create: `services/voice-engine/app/webhook_client.py`

- [ ] **Step 1: Create webhook_client.py**

```python
"""Client for reporting progress and completion back to Laravel."""

import logging
import time
import requests

logger = logging.getLogger(__name__)

INTERNAL_TOKEN = None  # Set at startup from env


def init(token: str):
    """Initialize the webhook client with the internal token."""
    global INTERNAL_TOKEN
    INTERNAL_TOKEN = token


def _headers():
    return {"X-Internal-Token": INTERNAL_TOKEN, "Content-Type": "application/json"}


def report_progress(progress_url: str, progress: int, message: str, step: str, step_number: int, total_steps: int):
    """Report progress to Laravel. Non-fatal on failure."""
    try:
        requests.post(progress_url, json={
            "progress": progress,
            "message": message,
            "step": step,
            "step_number": step_number,
            "total_steps": total_steps,
        }, headers=_headers(), timeout=5)
    except Exception as e:
        logger.warning(f"Failed to report progress: {e}")


def report_completion(complete_url: str, output_path: str, sample_rate: int = None,
                      duration: float = None, output_paths: list = None):
    """Report successful completion to Laravel with retry."""
    payload = {"status": "completed", "output_path": output_path}
    if sample_rate:
        payload["sample_rate"] = sample_rate
    if duration:
        payload["duration"] = duration
    if output_paths:
        payload["output_paths"] = output_paths

    _post_with_retry(complete_url, payload, retries=3)


def report_failure(complete_url: str, error: str):
    """Report failure to Laravel with retry."""
    _post_with_retry(complete_url, {"status": "failed", "error": error}, retries=3)


def check_cancelled(status_url: str) -> bool:
    """Check if job has been cancelled. Returns False on network error (safe default)."""
    try:
        resp = requests.get(status_url, headers=_headers(), timeout=5)
        if resp.ok:
            return resp.json().get("status") == "cancelled"
    except Exception as e:
        logger.warning(f"Failed to check cancellation status: {e}")
    return False


def _post_with_retry(url: str, payload: dict, retries: int = 3):
    """POST with exponential backoff retry."""
    for attempt in range(retries):
        try:
            resp = requests.post(url, json=payload, headers=_headers(), timeout=10)
            if resp.ok:
                return
            logger.warning(f"Webhook POST to {url} returned {resp.status_code}, attempt {attempt + 1}/{retries}")
        except Exception as e:
            logger.warning(f"Webhook POST to {url} failed: {e}, attempt {attempt + 1}/{retries}")

        if attempt < retries - 1:
            time.sleep(2 ** (attempt + 1))  # 2s, 4s

    logger.error(f"All {retries} webhook retries failed for {url}")
```

- [ ] **Step 2: Commit**

```bash
git add services/voice-engine/app/webhook_client.py
git commit -m "feat: add webhook client for progress/completion reporting"
```

### Task 9: S3 Upload Client

**Files:**
- Create: `services/voice-engine/app/s3_client.py`

- [ ] **Step 1: Create s3_client.py**

```python
"""S3/MinIO client for uploading job output files."""

import logging
import os
import time
import boto3
from botocore.config import Config

logger = logging.getLogger(__name__)

_client = None
_bucket = None


def init():
    """Initialize S3 client from environment variables."""
    global _client, _bucket
    _client = boto3.client(
        's3',
        endpoint_url=os.environ.get('AWS_ENDPOINT', 'http://localhost:9000'),
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
        config=Config(signature_version='s3v4'),
    )
    _bucket = os.environ.get('AWS_BUCKET', 'morphvox')
    logger.info(f"S3 client initialized: endpoint={os.environ.get('AWS_ENDPOINT')}, bucket={_bucket}")


def upload_file(local_path: str, s3_key: str, retries: int = 3) -> bool:
    """Upload a local file to S3/MinIO with retry. Returns True on success."""
    for attempt in range(retries):
        try:
            _client.upload_file(local_path, _bucket, s3_key)
            logger.info(f"Uploaded {local_path} → s3://{_bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.warning(f"S3 upload failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    
    logger.error(f"All {retries} upload attempts failed for {s3_key}")
    return False
```

- [ ] **Step 2: Commit**

```bash
git add services/voice-engine/app/s3_client.py
git commit -m "feat: add S3/MinIO upload client with retry"
```

### Task 10: Job Runner (ThreadPoolExecutor Wrapper)

**Files:**
- Create: `services/voice-engine/app/job_runner.py`

- [ ] **Step 1: Create job_runner.py**

```python
"""Async job runner using ThreadPoolExecutor with busy tracking and heartbeat."""

import logging
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class HeartbeatTimer:
    """Periodic timer that re-sends progress to prevent stale reaping."""

    def __init__(self, interval: float, callback):
        self._timer = None
        self.interval = interval
        self.callback = callback
        self._running = False

    def start(self):
        self._running = True
        self._schedule()

    def _schedule(self):
        if self._running:
            self._timer = threading.Timer(self.interval, self._run)
            self._timer.daemon = True
            self._timer.start()

    def _run(self):
        if self._running:
            try:
                self.callback()
            except Exception as e:
                logger.warning(f"Heartbeat callback failed: {e}")
            self._schedule()

    def stop(self):
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None


class JobRunner:
    """Single-worker executor with busy tracking."""

    def __init__(self):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._busy = False
        self._lock = threading.Lock()

    @property
    def is_busy(self) -> bool:
        return self._busy

    def submit(self, fn, *args, **kwargs) -> bool:
        """Submit a job. Returns False if busy (caller should return 503)."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True

        def wrapper():
            try:
                fn(*args, **kwargs)
            except Exception as e:
                logger.exception(f"Job failed with unhandled exception: {e}")
            finally:
                with self._lock:
                    self._busy = False

        self._executor.submit(wrapper)
        return True


# Global singleton
runner = JobRunner()
```

- [ ] **Step 2: Commit**

```bash
git add services/voice-engine/app/job_runner.py
git commit -m "feat: add ThreadPoolExecutor job runner with busy tracking"
```

### Task 11: Modify /audio/process for Async Dispatch

**Files:**
- Modify: `services/voice-engine/app/http_api.py`

- [ ] **Step 1: Add imports and initialization at top of http_api.py**

Near existing imports, add:
```python
from app.webhook_client import report_progress, report_completion, report_failure, check_cancelled
from app.webhook_client import init as init_webhook
from app.s3_client import init as init_s3, upload_file
from app.job_runner import runner, HeartbeatTimer
```

In the startup/init section:
```python
init_webhook(os.environ.get('INTERNAL_TOKEN', ''))
init_s3()
```

- [ ] **Step 2: Modify the /audio/process endpoint**

At the start of the endpoint handler, check for async params:
```python
progress_url = request_data.get('progress_url')
complete_url = request_data.get('complete_url')
status_url = request_data.get('status_url')
job_uuid = request_data.get('job_uuid')
user_id = request_data.get('user_id')

if progress_url:
    # Async mode: submit to executor
    if runner.is_busy:
        return JSONResponse(
            status_code=503,
            content={"error": "GPU busy", "retry_after": 30}
        )
    
    # Submit background processing
    accepted = runner.submit(
        _process_job_async,
        request_data, progress_url, complete_url, status_url, job_uuid, user_id
    )
    if not accepted:
        return JSONResponse(status_code=503, content={"error": "GPU busy"})
    
    return JSONResponse(status_code=202, content={"status": "accepted", "job_uuid": job_uuid})

# Otherwise: synchronous mode (legacy/testing)
# ... existing code continues unchanged ...
```

- [ ] **Step 3: Create _process_job_async function**

This wraps the existing processing pipeline with progress reporting, cancellation checks, S3 upload, and completion webhook. The function must extract the same parameters the sync `process_audio()` handler uses (starting at line 2358 of http_api.py) and call the same core functions:

**Key existing functions to call (all in http_api.py):**
- `separate_vocals(audio, sample_rate, model_name, agg)` — UVR5 separation (HP5 for main, HP3 for all)
- `convert_vocal(vocals, voice_config, voice_num)` — RVC voice conversion
- `ensure_length(audio, target_length)` — pad/trim audio to match original duration
- `check_training_active()` — verify GPU not in use by training

**Parameters to extract from request_data:**
- `mode` — "swap", "split", or "convert"
- `audio` — base64-encoded audio data
- `voice_configs` — list of {model_path, pitch, ...} for each voice
- `pitch_shift_all` — global pitch shift value
- `f0_up_key` — per-voice pitch (deprecated, use pitch_shift_all)
- `youtube_url` — optional, download instead of using audio field
- `separation_model` — HP5_only_main_vocal or HP3_all_vocals

**Total steps by mode:**
- `split`: 2 steps (decode + separate)
- `convert`: 2 steps (decode + convert)
- `swap` single voice: 4 steps (decode → separate → convert → mix)
- `swap` multi-voice (N voices): 2 + N + 2 steps (decode → N×separate+convert → final separate → mix)

```python
def _process_job_async(request_data: dict, progress_url: str, complete_url: str,
                       status_url: str, job_uuid: str, user_id: int):
    """Background job processing with progress reporting."""
    try:
        import librosa
        import tempfile
        import soundfile as sf
        
        # Extract params from request_data (mirrors process_audio at line 2377)
        mode = request_data.get('mode', 'swap')
        audio_data = request_data.get('audio', '')
        voice_configs = request_data.get('voice_configs', [])
        pitch_shift_all = request_data.get('pitch_shift_all', 0)
        youtube_url = request_data.get('youtube_url')
        separation_model = request_data.get('separation_model', 'HP5_only_main_vocal')
        
        voice_count = len(voice_configs)
        
        # Calculate total_steps based on mode
        if mode == 'split':
            total_steps = 2
        elif mode == 'convert':
            total_steps = 2
        elif mode == 'swap' and voice_count <= 1:
            total_steps = 4  # decode, separate, convert, mix
        else:
            total_steps = 2 + voice_count + 2  # decode, N*(sep+convert), final_sep, mix
        
        step = 0
        
        # --- STEP: Decode/download audio ---
        step += 1
        report_progress(progress_url, 5, "Loading audio...", "decode", step, total_steps)
        if check_cancelled(status_url):
            return
        
        if youtube_url:
            # Download from YouTube (uses yt-dlp subprocess)
            import subprocess
            tmp_path = f"/tmp/{job_uuid}_download.wav"
            result = subprocess.run(
                ['yt-dlp', '-x', '--audio-format', 'wav', '-o', tmp_path, youtube_url],
                capture_output=True, timeout=120
            )
            if result.returncode != 0:
                report_failure(complete_url, f"YouTube download failed: {result.stderr[:200]}")
                return
            audio, sr = librosa.load(tmp_path, sr=None, mono=False)
        else:
            # Decode base64 audio (same logic as sync path lines 2394-2450)
            if ',' in audio_data and audio_data.startswith('data:'):
                audio_data = audio_data.split(',')[1]
            audio_bytes = base64.b64decode(audio_data)
            audio_buffer = io.BytesIO(audio_bytes)
            audio, sr = sf.read(audio_buffer, dtype='float32')
        
        # --- MODE: split ---
        if mode == 'split':
            step += 1
            report_progress(progress_url, 50, "Separating vocals...", "separate", step, total_steps)
            if check_cancelled(status_url):
                return
            
            vocals, instrumental = separate_vocals(audio, sr, separation_model, agg=10)
            
            # Upload both files
            vocals_key = f"user-generations/{user_id}/{job_uuid}_vocals.wav"
            inst_key = f"user-generations/{user_id}/{job_uuid}_instrumental.wav"
            
            # Write to temp files for upload
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, vocals, sr)
                upload_file(f.name, vocals_key)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, instrumental, sr)
                upload_file(f.name, inst_key)
            
            report_completion(complete_url, vocals_key, 
                            output_paths=[vocals_key, inst_key],
                            sample_rate=sr, duration=len(vocals)/sr)
            return
        
        # --- MODE: convert ---
        if mode == 'convert':
            step += 1
            report_progress(progress_url, 50, "Converting voice...", "convert", step, total_steps)
            if check_cancelled(status_url):
                return
            
            converted = convert_vocal(audio, voice_configs[0], 1)
            
            output_key = f"user-generations/{user_id}/{job_uuid}.wav"
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                sf.write(f.name, converted, sr)
                if not upload_file(f.name, output_key):
                    report_failure(complete_url, "Failed to save output file")
                    return
            
            report_completion(complete_url, output_key, sample_rate=sr, duration=len(converted)/sr)
            return
        
        # --- MODE: swap (single or multi-voice) ---
        if voice_count <= 1:
            # Single voice swap: separate → convert → mix
            step += 1
            report_progress(progress_url, 25, "Separating vocals...", "separate", step, total_steps)
            if check_cancelled(status_url):
                return
            vocals, instrumental = separate_vocals(audio, sr, separation_model, agg=10)
            
            step += 1
            report_progress(progress_url, 50, "Converting voice...", "convert", step, total_steps)
            if check_cancelled(status_url):
                return
            converted = convert_vocal(vocals, voice_configs[0], 1)
            
            step += 1
            report_progress(progress_url, 80, "Mixing audio...", "mix", step, total_steps)
            # Mix converted vocals with instrumental
            target_length = len(instrumental)
            converted = ensure_length(converted, target_length)
            output = instrumental + converted
            
        else:
            # Multi-voice Voice Layers pipeline (mirrors lines 2857-2990 of http_api.py)
            # Uses HP5 iteratively for each voice, HP3 for clean instrumental
            target_length = len(audio)
            audio_for_hp3 = audio.copy()
            converted_vocals_list = []
            
            # Extract and convert main voice (HP5)
            step += 1
            report_progress(progress_url, int(15), "Extracting main voice...", "separate_v1", step, total_steps)
            if check_cancelled(status_url):
                return
            vocals_1, inst_minus_1 = separate_vocals(audio, sr, "HP5_only_main_vocal", agg=10)
            vocals_1 = ensure_length(vocals_1, target_length)
            converted_1 = convert_vocal(vocals_1, voice_configs[0], 1)
            converted_vocals_list.append(converted_1)
            del vocals_1
            
            # Iteratively extract backing voices
            current_input = inst_minus_1
            for voice_idx in range(1, voice_count):
                step += 1
                pct = int(15 + (65 * voice_idx / voice_count))
                report_progress(progress_url, pct, f"Extracting voice {voice_idx+1}...", f"separate_v{voice_idx+1}", step, total_steps)
                if check_cancelled(status_url):
                    return
                
                vocals_n, inst_minus_n = separate_vocals(current_input, sr, "HP5_only_main_vocal", agg=10)
                vocals_n = ensure_length(vocals_n, target_length)
                converted_n = convert_vocal(vocals_n, voice_configs[voice_idx], voice_idx + 1)
                converted_vocals_list.append(converted_n)
                del vocals_n
                current_input = inst_minus_n
            
            # Clean instrumental with HP3 (remove ALL voices)
            step += 1
            report_progress(progress_url, 80, "Creating clean instrumental...", "final_separate", step, total_steps)
            if check_cancelled(status_url):
                return
            _, instrumental_clean = separate_vocals(audio_for_hp3, sr, "HP3_all_vocals", agg=10)
            instrumental_clean = ensure_length(instrumental_clean, target_length)
            
            # Mix all converted vocals with clean instrumental
            step += 1
            report_progress(progress_url, 90, "Mixing all voices...", "mix", step, total_steps)
            output = instrumental_clean
            for cv in converted_vocals_list:
                cv = ensure_length(cv, target_length)
                output = output + cv
        
        # --- Upload final output ---
        output_key = f"user-generations/{user_id}/{job_uuid}.wav"
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, output, sr)
            if not upload_file(f.name, output_key):
                report_failure(complete_url, "Failed to save output file")
                return
        
        duration = len(output) / sr
        report_completion(complete_url, output_key, sample_rate=sr, duration=duration)
        
    except Exception as e:
        logger.exception(f"Job {job_uuid} failed: {e}")
        try:
            report_failure(complete_url, str(e)[:500])
        except:
            pass
```

**Integration notes for the implementer:**
- This function runs in the ThreadPoolExecutor from `job_runner.py` (not async)
- The `separate_vocals()`, `convert_vocal()`, and `ensure_length()` functions already exist in `http_api.py` — import or reference them directly
- The audio mixing (`output = instrumental + converted`) is numpy addition; ensure arrays are the same length first with `ensure_length()`
- Memory management: call `gc.collect()` and `torch.cuda.empty_cache()` between heavy steps (existing `clear_memory()` pattern at line 2876)
- For YouTube downloads, the existing sync path uses `yt-dlp` — replicate that logic

- [ ] **Step 4: Verify voice engine starts without import errors**

Run: `cd services/voice-engine && python -c "from app.webhook_client import report_progress; from app.s3_client import init; from app.job_runner import runner; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add services/voice-engine/app/http_api.py
git commit -m "feat: add async dispatch to /audio/process endpoint"
```

---

## Chunk 3: Frontend — AudioJobProvider, FloatingWidget & Page Integration

### Task 12: API Functions for Jobs

**Files:**
- Modify: `apps/web/src/lib/api.ts`

- [ ] **Step 1: Add job API functions**

Add near the bottom of `api.ts`:

```typescript
// =============================================================================
// Job Queue API (Background Generation)
// =============================================================================

export interface JobResponse {
  id: number;
  uuid: string;
  type: 'audio_swap' | 'audio_split' | 'audio_convert' | 'tts';
  status: 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: number;
  progress_message: string | null;
  step_number: number;
  total_steps: number;
  output_url: string | null;
  output_urls: Record<string, string> | null;
  saved: boolean;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
  voice_model?: { id: number; uuid: string; name: string; slug: string } | null;
}

export const jobsApi = {
  list: async (params?: { status?: string }): Promise<JobResponse[]> => {
    const queryStr = params?.status ? `?status=${params.status}` : '';
    const response = await api.get(`/jobs${queryStr}`);
    return response.data.data || response.data;
  },

  get: async (uuid: string): Promise<JobResponse> => {
    const response = await api.get(`/jobs/${uuid}`);
    return response.data;
  },

  cancel: async (uuid: string): Promise<void> => {
    await api.post(`/jobs/${uuid}/cancel`);
  },

  save: async (uuid: string): Promise<void> => {
    await api.post(`/jobs/${uuid}/save`);
  },

  unsave: async (uuid: string): Promise<void> => {
    await api.post(`/jobs/${uuid}/unsave`);
  },

  getStreamUrl: (uuid: string, track?: string): string => {
    const base = `${API_URL}/jobs/${uuid}/stream`;
    return track ? `${base}?track=${track}` : base;
  },
};
```

- [ ] **Step 2: Commit**

```bash
git add apps/web/src/lib/api.ts
git commit -m "feat: add job queue API functions"
```

### Task 13: AudioJobProvider Context

**Files:**
- Create: `apps/web/src/contexts/audio-job-context.tsx`

- [ ] **Step 1: Create the context provider**

```typescript
'use client';

import { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import { jobsApi, JobResponse } from '@/lib/api';

export interface ActiveJob {
  id: string;
  type: JobResponse['type'];
  status: JobResponse['status'];
  progress: number;
  progressMessage: string;
  stepNumber: number;
  totalSteps: number;
  outputUrl: string | null;
  outputUrls: Record<string, string> | null;
  modelName: string;
  saved: boolean;
  createdAt: string;
  completedAt: string | null;
  errorMessage: string | null;
}

interface AudioJobContextValue {
  activeJobs: ActiveJob[];
  submitJob: (submitFn: () => Promise<{ job_id: string }>) => Promise<string>;
  cancelJob: (jobId: string) => Promise<void>;
  saveJob: (jobId: string) => Promise<void>;
  unsaveJob: (jobId: string) => Promise<void>;
  dismissJob: (jobId: string) => void;
}

const AudioJobContext = createContext<AudioJobContextValue | null>(null);

export function useAudioJobs() {
  const ctx = useContext(AudioJobContext);
  if (!ctx) throw new Error('useAudioJobs must be used within AudioJobProvider');
  return ctx;
}

function mapJob(job: JobResponse): ActiveJob {
  return {
    id: job.uuid,
    type: job.type,
    status: job.status,
    progress: job.progress,
    progressMessage: job.progress_message || '',
    stepNumber: job.step_number,
    totalSteps: job.total_steps,
    outputUrl: job.output_url,
    outputUrls: job.output_urls,
    modelName: job.voice_model?.name || 'Unknown',
    saved: job.saved,
    createdAt: job.created_at,
    completedAt: job.completed_at,
    errorMessage: job.error_message,
  };
}

export function AudioJobProvider({ children }: { children: React.ReactNode }) {
  const [jobs, setJobs] = useState<ActiveJob[]>([]);
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());
  const pollRef = useRef<NodeJS.Timeout | null>(null);
  const isFirstPoll = useRef<Record<string, boolean>>({});

  // Fetch initial jobs on mount
  useEffect(() => {
    const fetchInitial = async () => {
      try {
        const active = await jobsApi.list({ status: 'processing' });
        const queued = await jobsApi.list({ status: 'queued' });
        const all = [...active, ...queued];
        setJobs(all.map(mapJob));
      } catch (e) {
        console.error('Failed to fetch initial jobs:', e);
      }
    };
    fetchInitial();
  }, []);

  // Poll active jobs
  useEffect(() => {
    const activeJobIds = jobs.filter(j => j.status === 'queued' || j.status === 'processing').map(j => j.id);
    
    if (activeJobIds.length === 0) {
      if (pollRef.current) clearInterval(pollRef.current);
      return;
    }

    const poll = async () => {
      for (const id of activeJobIds) {
        try {
          const updated = await jobsApi.get(id);
          setJobs(prev => prev.map(j => j.id === id ? mapJob(updated) : j));
        } catch (e) {
          console.error(`Failed to poll job ${id}:`, e);
        }
      }
    };

    // First poll at 1s for quick jobs
    const hasNewJobs = activeJobIds.some(id => isFirstPoll.current[id]);
    if (hasNewJobs) {
      const timeout = setTimeout(() => {
        poll();
        activeJobIds.forEach(id => { isFirstPoll.current[id] = false; });
      }, 1000);
      return () => clearTimeout(timeout);
    }

    pollRef.current = setInterval(poll, 3000);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [jobs]);

  const submitJob = useCallback(async (submitFn: () => Promise<{ job_id: string }>) => {
    const { job_id } = await submitFn();
    isFirstPoll.current[job_id] = true;
    
    // Add placeholder job
    setJobs(prev => [...prev, {
      id: job_id,
      type: 'audio_swap',
      status: 'queued',
      progress: 0,
      progressMessage: 'Queued...',
      stepNumber: 0,
      totalSteps: 1,
      outputUrl: null,
      outputUrls: null,
      modelName: '',
      saved: false,
      createdAt: new Date().toISOString(),
      completedAt: null,
      errorMessage: null,
    }]);

    return job_id;
  }, []);

  const cancelJob = useCallback(async (jobId: string) => {
    const response = await jobsApi.cancel(jobId);
    setJobs(prev => prev.map(j => j.id === jobId ? { ...j, status: 'cancelled' as const } : j));
    // If the job was in-progress, show latency warning
    if (response.message?.includes('up to 60 seconds')) {
      toast.info('Cancellation requested — may take up to 60 seconds to fully stop.');
    }
  }, []);

  const saveJob = useCallback(async (jobId: string) => {
    await jobsApi.save(jobId);
    setJobs(prev => prev.map(j => j.id === jobId ? { ...j, saved: true } : j));
  }, []);

  const unsaveJob = useCallback(async (jobId: string) => {
    await jobsApi.unsave(jobId);
    setJobs(prev => prev.map(j => j.id === jobId ? { ...j, saved: false } : j));
  }, []);

  const dismissJob = useCallback((jobId: string) => {
    setDismissed(prev => new Set([...prev, jobId]));
    setJobs(prev => prev.filter(j => j.id !== jobId));
  }, []);

  const visibleJobs = jobs.filter(j => !dismissed.has(j.id));

  return (
    <AudioJobContext.Provider value={{ activeJobs: visibleJobs, submitJob, cancelJob, saveJob, unsaveJob, dismissJob }}>
      {children}
    </AudioJobContext.Provider>
  );
}
```

- [ ] **Step 2: Wrap providers with AudioJobProvider**

In `apps/web/src/components/providers.tsx`, add:
```typescript
import { AudioJobProvider } from '@/contexts/audio-job-context';
```

Wrap inside the existing providers (inside `AuthProvider`):
```tsx
<AuthProvider>
  <AudioJobProvider>
    {children}
  </AudioJobProvider>
</AuthProvider>
```

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd apps/web && npx tsc --noEmit`
Expected: No type errors.

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/contexts/audio-job-context.tsx apps/web/src/components/providers.tsx
git commit -m "feat: add AudioJobProvider context with polling"
```

### Task 14: FloatingJobsWidget Component

**Files:**
- Create: `apps/web/src/components/floating-jobs-widget.tsx`

- [ ] **Step 1: Create the widget**

```typescript
'use client';

import { useState } from 'react';
import { useAudioJobs, ActiveJob } from '@/contexts/audio-job-context';
import { jobsApi } from '@/lib/api';
import {
  X, Loader2, CheckCircle2, AlertCircle, Download, Play, Pause,
  Trash2, Bookmark, BookmarkCheck, XCircle,
} from 'lucide-react';

function JobTypeIcon({ type }: { type: ActiveJob['type'] }) {
  const labels: Record<string, string> = {
    audio_swap: '🎤',
    audio_split: '✂️',
    audio_convert: '🔄',
    tts: '💬',
  };
  return <span className="text-sm">{labels[type] || '🎵'}</span>;
}

function JobCard({ job }: { job: ActiveJob }) {
  const { cancelJob, saveJob, unsaveJob, dismissJob } = useAudioJobs();
  const [audioPlaying, setAudioPlaying] = useState(false);

  const isActive = job.status === 'queued' || job.status === 'processing';
  const isComplete = job.status === 'completed';
  const isFailed = job.status === 'failed';

  return (
    <div className="p-3 bg-gray-800 rounded-lg border border-gray-700 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <JobTypeIcon type={job.type} />
          <span className="text-sm font-medium text-gray-200 truncate max-w-[120px]">
            {job.modelName}
          </span>
        </div>
        <button onClick={() => dismissJob(job.id)} className="text-gray-500 hover:text-gray-300">
          <X size={14} />
        </button>
      </div>

      {isActive && (
        <>
          <div className="w-full bg-gray-700 rounded-full h-2">
            <div
              className="bg-purple-500 h-2 rounded-full transition-all duration-500"
              style={{ width: `${job.progress}%` }}
            />
          </div>
          <div className="flex items-center justify-between">
            <span className="text-xs text-gray-400">{job.progressMessage}</span>
            <button onClick={() => cancelJob(job.id)} className="text-xs text-red-400 hover:text-red-300">
              Cancel
            </button>
          </div>
        </>
      )}

      {isComplete && job.outputUrl && (
        <div className="flex items-center gap-2">
          <a
            href={`${job.outputUrl}${job.outputUrl.includes('?') ? '&' : '?'}download=1`}
            className="flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300"
          >
            <Download size={12} /> Download
          </a>
          <button
            onClick={() => job.saved ? unsaveJob(job.id) : saveJob(job.id)}
            className={`flex items-center gap-1 text-xs ${job.saved ? 'text-yellow-400' : 'text-gray-400 hover:text-yellow-400'}`}
          >
            {job.saved ? <BookmarkCheck size={12} /> : <Bookmark size={12} />}
            {job.saved ? 'Saved' : 'Save'}
          </button>
        </div>
      )}

      {isFailed && (
        <span className="text-xs text-red-400">{job.errorMessage || 'Processing failed'}</span>
      )}
    </div>
  );
}

export function FloatingJobsWidget() {
  const { activeJobs } = useAudioJobs();
  const [expanded, setExpanded] = useState(false);

  if (activeJobs.length === 0) return null;

  const inProgress = activeJobs.filter(j => j.status === 'queued' || j.status === 'processing');
  const completed = activeJobs.filter(j => j.status === 'completed');

  const badgeText = inProgress.length > 0
    ? `${inProgress.length} ⚡`
    : `${completed.length} ✓`;

  if (!expanded) {
    return (
      <button
        onClick={() => setExpanded(true)}
        className="fixed bottom-6 right-6 z-50 bg-purple-600 hover:bg-purple-500 text-white px-3 py-2 rounded-full shadow-lg flex items-center gap-2 transition-all"
      >
        {inProgress.length > 0 && <Loader2 size={14} className="animate-spin" />}
        <span className="text-sm font-medium">{badgeText}</span>
      </button>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 w-80 max-h-96 bg-gray-900 border border-gray-700 rounded-xl shadow-2xl flex flex-col">
      <div className="flex items-center justify-between p-3 border-b border-gray-700">
        <span className="text-sm font-semibold text-gray-200">Generations</span>
        <button onClick={() => setExpanded(false)} className="text-gray-400 hover:text-gray-200">
          <X size={16} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {activeJobs.map(job => (
          <JobCard key={job.id} job={job} />
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Add FloatingJobsWidget to dashboard layout**

In `apps/web/src/components/dashboard-layout.tsx`, import and render the widget at the end of the component (after main content, inside the outermost wrapper):

```typescript
import { FloatingJobsWidget } from './floating-jobs-widget';
```

Add before closing tag: `<FloatingJobsWidget />`

- [ ] **Step 3: Verify TypeScript compiles**

Run: `cd apps/web && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add apps/web/src/components/floating-jobs-widget.tsx apps/web/src/components/dashboard-layout.tsx
git commit -m "feat: add FloatingJobsWidget for background job progress"
```

### Task 15: Integrate Song Remix Page with Async Submission

**Files:**
- Modify: `apps/web/src/app/dashboard/song-remix/page.tsx`

- [ ] **Step 1: Import useAudioJobs hook**

```typescript
import { useAudioJobs } from '@/contexts/audio-job-context';
```

- [ ] **Step 2: Replace synchronous processing with async submission**

In the `handleProcessing` function, replace the current `audioProcessingApi.process()` call and fake progress with:

```typescript
const { submitJob } = useAudioJobs();

// In handleProcessing:
try {
  setIsProcessing(true);
  setProcessingProgress(0);
  setProcessingStatus('Submitting...');

  const jobId = await submitJob(async () => {
    const response = await api.post('/audio/process', payload);
    return { job_id: response.data.job_id };
  });

  // If foreground mode: poll and show progress inline
  setProcessingStatus('Processing...');
  const pollInterval = setInterval(async () => {
    try {
      const job = await jobsApi.get(jobId);
      setProcessingProgress(job.progress);
      setProcessingStatus(job.progress_message || 'Processing...');
      
      if (job.status === 'completed') {
        clearInterval(pollInterval);
        setIsProcessing(false);
        // Set result URL for audio player
        setOutputUrl(job.output_url);
      } else if (job.status === 'failed') {
        clearInterval(pollInterval);
        setIsProcessing(false);
        setError(job.error_message || 'Processing failed');
      }
    } catch (e) { /* continue polling */ }
  }, 3000);

  // First poll at 1s
  setTimeout(async () => { /* same poll logic */ }, 1000);

} catch (err: any) {
  if (err.response?.status === 429) {
    setError('You already have a generation in progress. Wait for it to finish or cancel it.');
  } else {
    setError(err.response?.data?.error || 'Failed to submit job');
  }
  setIsProcessing(false);
}
```

- [ ] **Step 3: Add "Run in Background" button**

Next to the existing submit button, add:
```tsx
<button
  onClick={handleBackgroundSubmit}
  className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm"
  disabled={isProcessing}
>
  Run in Background
</button>
```

The `handleBackgroundSubmit` function submits the job but doesn't poll inline — instead shows a toast and resets the form:

```tsx
const handleBackgroundSubmit = async () => {
  try {
    setIsProcessing(true);
    const formData = buildFormData(); // same payload as handleSubmit
    const response = await submitAudioJob(formData);
    
    // Don't poll here — the AudioJobProvider's global polling handles it
    // Just show a toast notification and reset
    toast.success('Job submitted! Check the floating widget for progress.');
    setIsProcessing(false);
    // Form stays as-is so user can tweak and submit again
  } catch (err: any) {
    if (err.response?.status === 429) {
      setError('You already have a generation in progress. Wait for it to finish or cancel it.');
    } else {
      setError(err.response?.data?.error || 'Failed to submit job');
    }
    setIsProcessing(false);
  }
};
```

- [ ] **Step 4: Verify TypeScript compiles**

Run: `cd apps/web && npx tsc --noEmit`
Expected: No errors.

- [ ] **Step 5: Commit**

```bash
git add apps/web/src/app/dashboard/song-remix/page.tsx
git commit -m "feat: integrate Song Remix page with async job submission"
```

---

## Chunk 4: Infrastructure & Deployment

### Task 16: Nginx Internal Route Blocking

**Files:**
- Create: `infra/nginx/internal-block.conf` (snippet to include)

- [ ] **Step 1: Create nginx config snippet**

```nginx
# Block external access to internal webhook routes
# Only allow from localhost (voice engine on same host)
location /api/internal/ {
    allow 127.0.0.1;
    deny all;
    
    # If allowed, proxy to Laravel
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

- [ ] **Step 2: Document where to include it**

The production nginx config on the Vast.ai server needs this location block added before the general `/api` location block (order matters in nginx — more specific paths first).

- [ ] **Step 3: Commit**

```bash
git add infra/nginx/internal-block.conf
git commit -m "feat: add nginx config to block external /api/internal access"
```

### Task 17: Environment Variables

**Files:**
- Modify: `apps/api/.env.example`
- Modify: `services/voice-engine/.env.example`
- Verify: `apps/api/config/services.php` (ensure `internal_token` key reads from env)

- [ ] **Step 1: Add to Laravel .env**

```env
INTERNAL_TOKEN=<generate-32-random-bytes-hex>
```

Also ensure `services.internal_token` config reads from this (check `config/services.php`).

- [ ] **Step 2: Add to voice engine .env**

```env
INTERNAL_TOKEN=<same-value-as-laravel>
AWS_ENDPOINT=http://localhost:9000
AWS_ACCESS_KEY_ID=<minio-key>
AWS_SECRET_ACCESS_KEY=<minio-secret>
AWS_BUCKET=morphvox
```

- [ ] **Step 3: Verify config reads the env var**

Run: `cd apps/api && php artisan tinker --execute="echo config('services.internal_token');"`
Expected: Outputs the token value (or add `'internal_token' => env('INTERNAL_TOKEN')` to `config/services.php` if missing).

- [ ] **Step 4: Commit .env.example updates**

```bash
git add apps/api/.env.example services/voice-engine/.env.example
git commit -m "docs: add INTERNAL_TOKEN and S3 env vars to examples"
```

### Task 18: End-to-End Integration Test

- [ ] **Step 1: Run full pipeline test**

On the Vast.ai server:
1. Build frontend: `cd apps/web && npm run build && cp -r .next/static .next/standalone/.next/static && cp -r public .next/standalone/public`
2. Run migration: `cd apps/api && php artisan migrate`
3. Restart services: `supervisorctl restart voice-engine nextjs laravel-worker`
4. Test via curl:

```bash
# Submit a job
curl -X POST https://morphvox.net/api/audio/process \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"mode":"convert","audio":"<base64>","model_id":1}' \
  -w "\n%{http_code}"
# Expected: 202 with {"job_id":"...", "status":"queued"}

# Poll job status
curl https://morphvox.net/api/jobs/$JOB_UUID \
  -H "Authorization: Bearer $TOKEN"
# Expected: Shows progress updates, eventually status=completed

# Stream output
curl https://morphvox.net/api/jobs/$JOB_UUID/stream \
  -H "Authorization: Bearer $TOKEN" -o output.wav
# Expected: Downloads WAV file
```

- [ ] **Step 2: Test cancellation**

```bash
# Submit a long job (voice swap with YouTube URL)
# While processing, cancel:
curl -X POST https://morphvox.net/api/jobs/$JOB_UUID/cancel \
  -H "Authorization: Bearer $TOKEN"
# Expected: Job eventually shows cancelled status
```

- [ ] **Step 3: Test concurrency guard**

```bash
# Submit a job, then immediately submit another
# Expected: Second returns 429
```

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete background generation integration"
```

---

## Execution Notes

- **Deploy order matters**: Database migration must run before code that uses new columns
- **Next.js standalone**: After every build, copy `.next/static` → `.next/standalone/.next/static` and `public` → `.next/standalone/public`
- **Supervisor restart**: After voice engine changes, restart via `supervisorctl restart voice-engine`
- **The voice engine `_process_job_async` function** (Task 11 Step 3) needs detailed implementation that integrates with the existing ~200-line pipeline. The skeleton is provided; the implementer should:
  1. Extract the existing sync processing into a helper function
  2. Insert `report_progress()` calls at each step boundary
  3. Insert `check_cancelled()` calls before each major step
  4. Add S3 upload at the end
  5. Wrap in try/except with `report_failure()` on error
