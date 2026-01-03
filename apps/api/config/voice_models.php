<?php

return [
    /*
    |--------------------------------------------------------------------------
    | Voice Models Storage Configuration
    |--------------------------------------------------------------------------
    |
    | Configure where voice models are stored. Supported: "local", "s3"
    |
    */

    'storage' => env('VOICE_MODELS_STORAGE', 'local'),

    /*
    |--------------------------------------------------------------------------
    | Local Storage Configuration
    |--------------------------------------------------------------------------
    |
    | Path to the local directory containing voice models.
    | Can be absolute or relative to the Laravel app root.
    |
    */

    'local' => [
        'path' => env('VOICE_MODELS_LOCAL_PATH', '../../services/voice-engine/assets/models'),
    ],

    /*
    |--------------------------------------------------------------------------
    | S3/Cloud Storage Configuration
    |--------------------------------------------------------------------------
    |
    | Configuration for S3-compatible storage (AWS S3, MinIO, etc.)
    | Uses the default S3 filesystem disk settings unless overridden.
    |
    */

    's3' => [
        'disk' => env('VOICE_MODELS_S3_DISK', 's3'),
        'prefix' => env('VOICE_MODELS_S3_PREFIX', 'models'),
        'url_expiration' => env('VOICE_MODELS_S3_URL_EXPIRATION', 60), // minutes
    ],

    /*
    |--------------------------------------------------------------------------
    | Supported Model File Extensions
    |--------------------------------------------------------------------------
    */

    'model_extensions' => ['pth', 'onnx'],
    'index_extensions' => ['index'],

    /*
    |--------------------------------------------------------------------------
    | Default Engine
    |--------------------------------------------------------------------------
    */

    'default_engine' => env('VOICE_MODELS_DEFAULT_ENGINE', 'rvc'),
];
