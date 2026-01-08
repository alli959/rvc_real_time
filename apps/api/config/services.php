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
    | Voice Engine Service
    |--------------------------------------------------------------------------
    |
    | Configuration for the Python voice engine service that handles
    | RVC inference and real-time voice conversion.
    |
    */

    'voice_engine' => [
        'base_url' => env('VOICE_ENGINE_URL', 'http://voice-engine:8001'),
        'ws_url' => env('VOICE_ENGINE_WS_URL', 'ws://voice-engine:8765'),
        'timeout' => env('VOICE_ENGINE_TIMEOUT', 300),
        'storage_endpoint' => env('VOICE_ENGINE_STORAGE_ENDPOINT', 'http://minio:9000'),
    ],

];
