<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Api\AuthController;
use App\Http\Controllers\Api\VoiceModelController;
use App\Http\Controllers\Api\JobController;

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
|
| Here is where you can register API routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "api" middleware group. Make something great!
|
*/

// ==========================================================================
// Public Routes (No Authentication Required)
// ==========================================================================

Route::prefix('auth')->group(function () {
    Route::post('/register', [AuthController::class, 'register']);
    Route::post('/login', [AuthController::class, 'login']);
    Route::get('/invitation/{token}', [AuthController::class, 'checkInvitation']);
    Route::post('/register-with-invite', [AuthController::class, 'registerWithInvitation']);
});

// Voice models - unified endpoint for all models (system + user-uploaded)
// Supports filtering by type=system or type=user
// These routes support optional authentication - user is resolved from bearer token if provided
Route::prefix('voice-models')->group(function () {
    Route::get('/', [VoiceModelController::class, 'index']);
    Route::get('/stats', [VoiceModelController::class, 'stats']);
    Route::get('/{slug}', [VoiceModelController::class, 'show']);
});

// Alias: /models routes point to the same controller
Route::get('/models', [VoiceModelController::class, 'index']);
Route::get('/models/{voiceModel}', [VoiceModelController::class, 'show']);

// ==========================================================================
// Protected Routes (Authentication Required)
// ==========================================================================

Route::middleware('auth:sanctum')->group(function () {
    // ------------------------------------------------------------------
    // Auth Management
    // ------------------------------------------------------------------
    Route::prefix('auth')->group(function () {
        Route::post('/logout', [AuthController::class, 'logout']);
        Route::get('/me', [AuthController::class, 'me']);
        Route::post('/refresh', [AuthController::class, 'refresh']);
    });

    // ------------------------------------------------------------------
    // Voice Models Management
    // ------------------------------------------------------------------
    Route::prefix('voice-models')->group(function () {
        // My models (user-uploaded)
        Route::get('/my', [VoiceModelController::class, 'myModels']);
        
        // Create new model
        Route::post('/', [VoiceModelController::class, 'store']);
        
        // Model-specific actions
        Route::put('/{voiceModel}', [VoiceModelController::class, 'update']);
        Route::patch('/{voiceModel}', [VoiceModelController::class, 'update']);
        Route::delete('/{voiceModel}', [VoiceModelController::class, 'destroy']);
        
        // Pre-signed URLs for direct S3 uploads/downloads
        Route::post('/{voiceModel}/upload-urls', [VoiceModelController::class, 'getUploadUrls']);
        Route::post('/{voiceModel}/confirm-upload', [VoiceModelController::class, 'confirmUpload']);
        Route::get('/{voiceModel}/download-urls', [VoiceModelController::class, 'getDownloadUrls']);
    });

    // Alias: /models routes
    Route::prefix('models')->group(function () {
        Route::get('/my', [VoiceModelController::class, 'myModels']);
        Route::post('/', [VoiceModelController::class, 'store']);
        Route::put('/{voiceModel}', [VoiceModelController::class, 'update']);
        Route::delete('/{voiceModel}', [VoiceModelController::class, 'destroy']);
        Route::post('/{voiceModel}/upload-urls', [VoiceModelController::class, 'getUploadUrls']);
        Route::post('/{voiceModel}/confirm-upload', [VoiceModelController::class, 'confirmUpload']);
        Route::get('/{voiceModel}/download-urls', [VoiceModelController::class, 'getDownloadUrls']);
    });

    // ------------------------------------------------------------------
    // Job Queue Management
    // ------------------------------------------------------------------
    Route::prefix('jobs')->group(function () {
        Route::get('/', [JobController::class, 'index']);
        Route::get('/{job}', [JobController::class, 'show']);
        
        // Create inference job
        Route::post('/inference', [JobController::class, 'createInference']);
        
        // Get upload URL for input audio
        Route::post('/{job}/upload-url', [JobController::class, 'getUploadUrl']);
        
        // Start processing (after upload complete)
        Route::post('/{job}/start', [JobController::class, 'start']);
        
        // Cancel a job
        Route::post('/{job}/cancel', [JobController::class, 'cancel']);
        
        // Get output download URL
        Route::get('/{job}/output', [JobController::class, 'getOutput']);
    });

    // ------------------------------------------------------------------
    // Admin Routes
    // ------------------------------------------------------------------
    Route::middleware('role:admin')->prefix('admin')->group(function () {
        // User management
        Route::get('/users', [AuthController::class, 'listUsers']);
        Route::put('/users/{user}', [AuthController::class, 'updateUser']);
        Route::delete('/users/{user}', [AuthController::class, 'deleteUser']);
        
        // Voice model admin actions
        Route::post('/voice-models/sync', [VoiceModelController::class, 'sync']);
        Route::get('/voice-models/config', [VoiceModelController::class, 'config']);
        
        // All jobs
        Route::get('/jobs', [JobController::class, 'adminIndex']);
        
        // System stats
        Route::get('/stats', [JobController::class, 'systemStats']);
    });
});

// ==========================================================================
// Health Check
// ==========================================================================

Route::get('/health', function () {
    return response()->json([
        'status' => 'healthy',
        'timestamp' => now()->toIso8601String(),
        'version' => config('app.version', '1.0.0'),
    ]);
});
