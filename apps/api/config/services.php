<?php

return [

    /*
    |--------------------------------------------------------------------------
    | Third Party Services
    |--------------------------------------------------------------------------
    |
    | This file is for storing the credentials for third party services such
    | as Mailgun, Postmark, AWS and more. This file provides the de facto
    | location for this type of information, allowing packages to have
    | a conventional file to locate the various service credentials.
    |
    */

    'postmark' => [
        'token' => env('POSTMARK_TOKEN'),
    ],

    'ses' => [
        'key' => env('AWS_ACCESS_KEY_ID'),
        'secret' => env('AWS_SECRET_ACCESS_KEY'),
        'region' => env('AWS_DEFAULT_REGION', 'us-east-1'),
    ],

    'resend' => [
        'key' => env('RESEND_KEY'),
    ],

    'slack' => [
        'notifications' => [
            'bot_user_oauth_token' => env('SLACK_BOT_USER_OAUTH_TOKEN'),
            'channel' => env('SLACK_BOT_USER_DEFAULT_CHANNEL'),
        ],
    ],

    /*
    |--------------------------------------------------------------------------
    | OAuth Providers
    |--------------------------------------------------------------------------
    |
    | Configuration for OAuth authentication providers (Google, GitHub).
    |
    */

    'google' => [
        'client_id' => env('GOOGLE_CLIENT_ID'),
        'client_secret' => env('GOOGLE_CLIENT_SECRET'),
        'redirect' => env('GOOGLE_REDIRECT_URI'),
    ],

    'github' => [
        'client_id' => env('GITHUB_CLIENT_ID'),
        'client_secret' => env('GITHUB_CLIENT_SECRET'),
        'redirect' => env('GITHUB_REDIRECT_URI'),
    ],

    /*
    |--------------------------------------------------------------------------
    | Voice Engine Service
    |--------------------------------------------------------------------------
    |
    | Configuration for the Python voice engine service that handles
    | RVC inference and real-time voice conversion.
    |
    */

    'voice_engine' => [
        'url' => env('VOICE_ENGINE_URL', 'http://voice-engine:8001'),
        'base_url' => env('VOICE_ENGINE_URL', 'http://voice-engine:8001'),
        'ws_url' => env('VOICE_ENGINE_WS_URL', 'ws://voice-engine:8765'),
        'timeout' => env('VOICE_ENGINE_TIMEOUT', 300),
        'storage_endpoint' => env('VOICE_ENGINE_STORAGE_ENDPOINT', 'http://minio:9000'),
    ],

    /*
    |--------------------------------------------------------------------------
    | Trainer Service
    |--------------------------------------------------------------------------
    |
    | Configuration for the RVC training service that handles
    | model training jobs.
    |
    */

    'trainer' => [
        'url' => env('TRAINER_URL', 'http://trainer:8002'),
        'timeout' => env('TRAINER_TIMEOUT', 600),
    ],

    /*
    |--------------------------------------------------------------------------
    | Preprocessor Service
    |--------------------------------------------------------------------------
    |
    | Configuration for the audio preprocessing service that handles
    | slicing, resampling, and feature extraction.
    |
    */

    'preprocessor' => [
        'url' => env('PREPROCESSOR_URL', 'http://preprocess:8003'),
        'timeout' => env('PREPROCESSOR_TIMEOUT', 300),
    ],

];
