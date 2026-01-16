# Implementation Plan: UI + Backend Job/Logging System

## Overview

This document outlines the implementation plan for the comprehensive job/logging system improvements including accurate progress reporting, admin logs page, system metrics, assets management, and checkpoint system.

---

## 1. Progress Reporting System

### 1.1 Data Contracts

#### Job Progress Schema (Backend → UI)

```typescript
interface JobProgress {
  job_id: string;
  job_type: JobType;
  status: JobStatus;
  progress_percent: number; // 0-100
  
  current_step: {
    key: string;           // stable identifier e.g., "preprocessing"
    label: string;         // user-friendly e.g., "Preprocessing Audio"
    step_index: number;    // 0-based
    step_count: number;    // total steps
    step_progress_percent: number; // 0-100 within this step
  };
  
  details: JobDetails;     // type-specific details
  log_stream?: string;     // optional log stream identifier
  
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error?: string;
}

type JobStatus = 'queued' | 'starting' | 'running' | 'finalizing' | 
                 'succeeded' | 'failed' | 'canceled' | 'paused';

type JobType = 'training_rvc' | 'training_fine_tune' | 'generate_song' | 
               'tts_generate' | 'voice_convert' | 'vocal_split' | 
               'preprocess_audio' | 'download_model';
```

#### Training Job Details

```typescript
interface TrainingJobDetails {
  exp_name: string;
  phase: 'dataset_prep' | 'feature_extraction' | 'training' | 'evaluation' | 'packaging';
  current_epoch: number;
  total_epochs: number;
  current_step: number;      // within epoch
  total_steps: number;       // within epoch
  gpu_utilization?: number;  // 0-100
  vram_usage_mb?: number;
  checkpoint_status: 'idle' | 'requested' | 'saving' | 'saved' | 'failed';
  last_checkpoint_at?: string;
  last_checkpoint_path?: string;
}
```

#### Generation Job Details

```typescript
interface GenerationJobDetails {
  phase: 'input_prep' | 'model_load' | 'inference' | 'postprocess' | 'export';
  input_file?: string;
  output_artifacts: Array<{
    type: string;
    path: string;
    url?: string;
  }>;
}
```

### 1.2 Progress Computation

Training step weights:
- Preprocessing: 10%
- F0 Extraction: 15%  
- Feature Extraction: 15%
- Training: 50%
- Index Building: 5%
- Packaging: 5%

Formula: `progress = Σ(step_weight × step_completion)`

---

## 2. Admin Logs Page

### 2.1 Service Discovery (from docker-compose.prod.yml)

Services detected:
1. **nginx** - Reverse proxy
2. **api** - Laravel API backend
3. **api-worker** - Queue worker
4. **web** - Next.js frontend
5. **voice-engine** - Python ML service
6. **db** - MariaDB database
7. **redis** - Cache/queue
8. **minio** - Object storage

### 2.2 Log Source Discovery Rules

For each container, check these directories in order:
1. `/app/storage/logs`
2. `/app/logs`
3. `/var/log`
4. `/logs`
5. `/data/logs`
6. Compose-defined mount points

File patterns to match:
- `*.log`
- `*error*`
- `*debug*`
- `*access*`
- `*warning*`

Priority ranking:
1. Error logs (highest)
2. Application logs
3. Access logs
4. Debug logs (lowest)

### 2.3 Service-Specific Log Sources

| Service | Log Sources |
|---------|-------------|
| nginx | `/var/log/nginx/access.log`, `/var/log/nginx/error.log` |
| api | `/app/storage/logs/laravel.log`, php-fpm logs |
| api-worker | `/app/storage/logs/laravel.log`, worker stdout |
| web | Container stdout |
| voice-engine | `/app/logs/*.log`, Container stdout |
| db | Container stdout, `/var/log/mysql/*.log` |
| redis | Container stdout |
| minio | Container stdout |

### 2.4 API Endpoints

```
GET /admin/logs/services          # List available services
GET /admin/logs/sources/:service  # Get log sources for service
WS  /admin/logs/stream/:service/:source  # Stream log tail
GET /admin/logs/download/:service/:source?lines=N
```

---

## 3. System Metrics

### 3.1 Metrics to Collect

```typescript
interface SystemMetrics {
  cpu: {
    usage_percent: number;
    load_avg: [number, number, number];
    cores: Array<{ core: number; usage_percent: number }>;
  };
  memory: {
    used_bytes: number;
    total_bytes: number;
    swap_used_bytes: number;
    swap_total_bytes: number;
  };
  disk: Array<{
    mount: string;
    used_bytes: number;
    total_bytes: number;
  }>;
  network: {
    rx_bytes_per_sec: number;
    tx_bytes_per_sec: number;
  };
  gpu?: {
    name: string;
    utilization_percent: number;
    vram_used_mb: number;
    vram_total_mb: number;
    temperature_c: number;
  };
  timestamp: string;
}
```

### 3.2 Collection Method

- Host metrics via `/proc` filesystem or `psutil`
- GPU metrics via `nvidia-smi` 
- Update frequency: 1 second
- WebSocket streaming to UI

---

## 4. Admin Assets Page

### 4.1 Asset Registry Schema

```typescript
interface Asset {
  id: string;
  name: string;
  type: 'model' | 'service' | 'component';
  resource_type: 'cpu' | 'gpu' | 'ram';
  status: 'running' | 'stopped' | 'loading' | 'error';
  container?: string;      // Docker container name
  process_name?: string;   // Process identifier
  dependencies: string[];  // Other asset IDs
  
  metrics: {
    ram_usage_mb?: number;
    vram_usage_mb?: number;
    cpu_percent?: number;
  };
}
```

### 4.2 Known Heavy Assets

1. **Bark TTS** - GPU heavy (~4GB VRAM)
2. **RVC Models** - GPU heavy (~2GB VRAM each)
3. **HuBERT** - GPU heavy (~1GB VRAM)
4. **RMVPE** - GPU heavy (~500MB VRAM)
5. **UVR5** - GPU heavy (~2GB VRAM)

### 4.3 API Endpoints

```
GET  /admin/assets           # List all assets
POST /admin/assets/:id/start
POST /admin/assets/:id/stop
GET  /admin/assets/:id/status
```

---

## 5. Checkpoint System

### 5.1 Checkpoint Naming Convention

Format: `{model_name}-v{version}-e{epoch}-s{step}-{date}.pth`

Examples:
- `anton-v0.5-e100-s4300-20240115.pth`
- `bjarni-v1.0-e200-s8600-20240116.pth`

### 5.2 Checkpoint Request Flow

```
Admin clicks "Save checkpoint & cancel"
    ↓
Backend sets checkpoint_status = 'requested'
    ↓
Training loop detects request, saves checkpoint
    ↓
Backend sets checkpoint_status = 'saving'
    ↓
Checkpoint write completes atomically
    ↓
Backend sets checkpoint_status = 'saved'
    ↓
Job status → 'canceled'
```

### 5.3 Atomic Checkpoint Save

1. Write to `.tmp` file
2. Verify file integrity
3. Rename to final path (atomic on POSIX)
4. Update metadata

---

## 6. File Structure

### Backend (voice-engine)

```
services/voice-engine/app/
├── admin/
│   ├── __init__.py
│   ├── logs_api.py          # Log streaming endpoints
│   ├── metrics_api.py       # System metrics
│   ├── assets_api.py        # Asset management
│   └── log_discovery.py     # Log source discovery
├── jobs/
│   ├── __init__.py
│   ├── progress.py          # Progress contract
│   ├── checkpoint.py        # Checkpoint management
│   └── types.py             # Job type definitions
└── trainer/
    └── pipeline.py          # Updated with checkpoint support
```

### Frontend (web)

```
apps/web/src/app/
├── admin/
│   ├── layout.tsx
│   ├── page.tsx             # Admin dashboard
│   ├── logs/
│   │   └── page.tsx         # Logs page with tabs
│   ├── assets/
│   │   └── page.tsx         # Assets management
│   └── jobs/
│       └── page.tsx         # Admin jobs view
└── dashboard/
    └── jobs/
        └── page.tsx         # Updated user jobs view
```

---

## 7. Implementation Order

1. **Phase 1: Backend Infrastructure**
   - Job progress contract types
   - Checkpoint system in trainer
   - Log discovery module
   - System metrics collector

2. **Phase 2: API Layer**
   - Admin API endpoints (Laravel)
   - Voice engine admin routes
   - WebSocket log streaming

3. **Phase 3: Frontend - Admin**
   - Admin layout/routing
   - Logs page with service tabs
   - System metrics dashboard
   - Assets management page

4. **Phase 4: Frontend - User**
   - Enhanced job progress UI
   - Background job notifications
   - Job detail pages

5. **Phase 5: Integration & Testing**
   - End-to-end testing
   - Progress accuracy verification
   - Checkpoint save/resume testing
