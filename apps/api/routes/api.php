<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Api\AuthController;
use App\Http\Controllers\Api\VoiceModelController;
use App\Http\Controllers\Api\SystemVoiceModelController;
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
});

// Public model browsing (user-uploaded public models)
Route::get('/models', [VoiceModelController::class, 'index']);
Route::get('/models/{model}', [VoiceModelController::class, 'show']);

// System voice models (server-side models from local dir or S3)
Route::prefix('voice-models')->group(function () {
    Route::get('/', [SystemVoiceModelController::class, 'index']);
    Route::get('/stats', [SystemVoiceModelController::class, 'stats']);
    Route::get('/config', [SystemVoiceModelController::class, 'config']);
    Route::get('/{slug}', [SystemVoiceModelController::class, 'show']);
});

// ==========================================================================
// Protected Routes (Authentication Required)
// ==========================================================================

Route::middleware('auth:sanctum')->group(function () {
    // ------------------------------------------------------------------
    // System Voice Models Management (Admin)
    // ------------------------------------------------------------------
    Route::prefix('voice-models')->group(function () {
        Route::post('/sync', [SystemVoiceModelController::class, 'sync']);
        Route::patch('/{slug}', [SystemVoiceModelController::class, 'update']);
    });

    // ------------------------------------------------------------------
    // Auth Management
    // ------------------------------------------------------------------
    Route::prefix('auth')->group(function () {
        Route::post('/logout', [AuthController::class, 'logout']);
        Route::get('/me', [AuthController::class, 'me']);
        Route::post('/refresh', [AuthController::class, 'refresh']);
    });

    // ------------------------------------------------------------------
    // Voice Models Management (User-uploaded)
    // ------------------------------------------------------------------
    Route::prefix('models')->group(function () {
        // My models
        Route::get('/my', [VoiceModelController::class, 'myModels']);
        
        // Create new model
        Route::post('/', [VoiceModelController::class, 'store']);
        
        // Model-specific actions
        Route::put('/{model}', [VoiceModelController::class, 'update']);
        Route::delete('/{model}', [VoiceModelController::class, 'destroy']);
        
        // Pre-signed URLs for direct S3 uploads/downloads
        Route::post('/{model}/upload-urls', [VoiceModelController::class, 'getUploadUrls']);
        Route::post('/{model}/confirm-upload', [VoiceModelController::class, 'confirmUpload']);
        Route::get('/{model}/download-urls', [VoiceModelController::class, 'getDownloadUrls']);
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
        
        // All models (including private)
        Route::get('/models', [VoiceModelController::class, 'adminIndex']);
        
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
