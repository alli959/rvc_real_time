<?php

/**
 * Unified Storage Paths Configuration
 * 
 * This file provides centralized path configuration for the Laravel API.
 * All paths are derived from STORAGE_ROOT environment variable with sensible defaults.
 * 
 * See docs/STORAGE_LAYOUT.md for full documentation.
 */

return [
    /*
    |--------------------------------------------------------------------------
    | Storage Root
    |--------------------------------------------------------------------------
    |
    | The root directory for all unified storage. Inside containers, this is
    | typically /storage. For local development, it defaults to the storage
    | directory relative to the repo root.
    |
    */
    'root' => env('STORAGE_ROOT', '/storage'),

    /*
    |--------------------------------------------------------------------------
    | Logs Directory
    |--------------------------------------------------------------------------
    |
    | Service-specific log files are written here.
    |
    */
    'logs' => env('STORAGE_ROOT', '/storage') . '/logs/api',

    /*
    |--------------------------------------------------------------------------
    | Data Directories
    |--------------------------------------------------------------------------
    |
    | Runtime data directories for uploads, preprocessing, training, and outputs.
    |
    */
    'data' => [
        'root' => env('STORAGE_ROOT', '/storage') . '/data',
        'uploads' => env('STORAGE_ROOT', '/storage') . '/data/uploads',
        'preprocess' => env('STORAGE_ROOT', '/storage') . '/data/preprocess',
        'training' => env('STORAGE_ROOT', '/storage') . '/data/training',
        'outputs' => env('STORAGE_ROOT', '/storage') . '/data/outputs',
    ],

    /*
    |--------------------------------------------------------------------------
    | Assets Directory
    |--------------------------------------------------------------------------
    |
    | Shared non-user assets like HuBERT, RMVPE, pretrained models.
    |
    */
    'assets' => [
        'root' => env('STORAGE_ROOT', '/storage') . '/assets',
        'hubert' => env('STORAGE_ROOT', '/storage') . '/assets/hubert',
        'rmvpe' => env('STORAGE_ROOT', '/storage') . '/assets/rmvpe',
        'pretrained_v2' => env('STORAGE_ROOT', '/storage') . '/assets/pretrained_v2',
        'uvr5_weights' => env('STORAGE_ROOT', '/storage') . '/assets/uvr5_weights',
        'bark' => env('STORAGE_ROOT', '/storage') . '/assets/bark',
        'whisper' => env('STORAGE_ROOT', '/storage') . '/assets/whisper',
        'index' => env('STORAGE_ROOT', '/storage') . '/assets/index',
    ],

    /*
    |--------------------------------------------------------------------------
    | Models Directory
    |--------------------------------------------------------------------------
    |
    | User voice models directory. Each model has:
    |   - <model_name>.pth (weights)
    |   - <model_name>.index (FAISS index, optional)
    |   - <model_name>/ (directory for metadata, images)
    |
    */
    'models' => env('VOICE_MODELS_PATH', env('STORAGE_ROOT', '/storage') . '/models'),

    /*
    |--------------------------------------------------------------------------
    | Path Translation
    |--------------------------------------------------------------------------
    |
    | Map paths between different container contexts.
    | API container path -> Voice engine container path
    |
    */
    'translation' => [
        // API sees /storage/models, voice-engine also sees /storage/models
        // No translation needed with unified storage
        'api_to_voice_engine' => [
            '/storage/' => '/storage/',
        ],
        // Legacy path translations (for backward compatibility)
        'legacy' => [
            '/var/www/html/storage/models/' => '/storage/models/',
            '/app/assets/models/' => '/storage/models/',
        ],
    ],
];
