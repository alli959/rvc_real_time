<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Api\AuthController;
use App\Http\Controllers\Api\OAuthController;
use App\Http\Controllers\Api\VoiceModelController;
use App\Http\Controllers\Api\ModelUploadController;
use App\Http\Controllers\Api\TTSController;
use App\Http\Controllers\Api\JobController;
use App\Http\Controllers\Api\RoleRequestController;
use App\Http\Controllers\Api\YouTubeController;
use App\Http\Controllers\Api\AudioProcessingController;
use App\Http\Controllers\Api\TrainerController;

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

// OAuth routes - separate from auth prefix to match OAuth provider callback URLs
Route::prefix('oauth')->group(function () {
    Route::get('/{provider}/redirect', [OAuthController::class, 'redirect'])
        ->where('provider', 'google|github');
    Route::get('/{provider}/callback', [OAuthController::class, 'callback'])
        ->where('provider', 'google|github');
    Route::post('/{provider}/callback', [OAuthController::class, 'handleCode'])
        ->where('provider', 'google|github');
});

// Voice models - unified endpoint for all models (system + user-uploaded)
// Supports filtering by type=system or type=user
// These routes support optional authentication - user is resolved from bearer token if provided
Route::prefix('voice-models')->group(function () {
    Route::get('/', [VoiceModelController::class, 'index']);
    Route::get('/stats', [VoiceModelController::class, 'stats']);
    // Use a regex constraint to prevent matching reserved words like "my"
    // Allow dots in slugs (e.g., "anton-0.5-model")
    Route::get('/{slug}', [VoiceModelController::class, 'show'])->where('slug', '^(?!my$)[a-zA-Z0-9_.-]+$');
});

// Alias: /models routes point to the same controller
Route::get('/models', [VoiceModelController::class, 'index']);
Route::get('/models/{voiceModel}', [VoiceModelController::class, 'show'])->where('voiceModel', '^(?!my$)[a-zA-Z0-9_.-]+$');

// TTS voices list (public - no auth required)
Route::get('/tts/voices', [TTSController::class, 'getVoices']);

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
        
        // Create new model (metadata only, for presigned upload flow)
        Route::post('/', [VoiceModelController::class, 'store']);
        
        // Direct file upload (multipart form)
        Route::post('/upload', [ModelUploadController::class, 'upload']);
        
        // Model-specific actions
        Route::put('/{voiceModel}', [VoiceModelController::class, 'update']);
        Route::patch('/{voiceModel}', [VoiceModelController::class, 'update']);
        Route::delete('/{voiceModel}', [VoiceModelController::class, 'destroy']);
        
        // Image upload
        Route::post('/{voiceModel}/image', [VoiceModelController::class, 'uploadImage']);
        
        // File management for existing models
        Route::post('/{voiceModel}/files', [ModelUploadController::class, 'uploadFiles']);
        Route::post('/{voiceModel}/replace', [ModelUploadController::class, 'replaceModel']);
        
        // Pre-signed URLs for direct S3 uploads/downloads
        Route::post('/{voiceModel}/upload-urls', [VoiceModelController::class, 'getUploadUrls']);
        Route::post('/{voiceModel}/confirm-upload', [VoiceModelController::class, 'confirmUpload']);
        Route::get('/{voiceModel}/download-urls', [VoiceModelController::class, 'getDownloadUrls']);
    });

    // Alias: /models routes
    Route::prefix('models')->group(function () {
        Route::get('/my', [VoiceModelController::class, 'myModels']);
        Route::post('/', [VoiceModelController::class, 'store']);
        Route::post('/upload', [ModelUploadController::class, 'upload']);
        Route::put('/{voiceModel}', [VoiceModelController::class, 'update']);
        Route::delete('/{voiceModel}', [VoiceModelController::class, 'destroy']);
        Route::post('/{voiceModel}/image', [VoiceModelController::class, 'uploadImage']);
        Route::post('/{voiceModel}/files', [ModelUploadController::class, 'uploadFiles']);
        Route::post('/{voiceModel}/replace', [ModelUploadController::class, 'replaceModel']);
        Route::post('/{voiceModel}/upload-urls', [VoiceModelController::class, 'getUploadUrls']);
        Route::post('/{voiceModel}/confirm-upload', [VoiceModelController::class, 'confirmUpload']);
        Route::get('/{voiceModel}/download-urls', [VoiceModelController::class, 'getDownloadUrls']);
    });

    // ------------------------------------------------------------------
    // Text-to-Speech (authenticated endpoints)
    // ------------------------------------------------------------------
    Route::prefix('tts')->group(function () {
        // /tts/voices is public - see routes above middleware group
        Route::post('/generate', [TTSController::class, 'generate']);
        Route::post('/stream', [TTSController::class, 'stream']);
    });

    // ------------------------------------------------------------------
    // Audio Processing
    // ------------------------------------------------------------------
    Route::prefix('audio')->group(function () {
        Route::post('/process', [AudioProcessingController::class, 'process']);
    });

    // ------------------------------------------------------------------
    // YouTube Song Search & Download
    // ------------------------------------------------------------------
    Route::prefix('youtube')->group(function () {
        Route::post('/search', [YouTubeController::class, 'search']);
        Route::post('/download', [YouTubeController::class, 'download']);
        Route::get('/info/{videoId}', [YouTubeController::class, 'info']);
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
    // Role Requests
    // ------------------------------------------------------------------
    Route::prefix('role-requests')->group(function () {
        Route::get('/available-roles', [RoleRequestController::class, 'getAvailableRoles']);
        Route::get('/my', [RoleRequestController::class, 'myRequests']);
        Route::post('/', [RoleRequestController::class, 'store']);
        Route::delete('/{roleRequest}', [RoleRequestController::class, 'cancel']);
    });

    // ------------------------------------------------------------------
    // Trainer / Model Scanning
    // ------------------------------------------------------------------
    Route::prefix('trainer')->group(function () {
        // Health & info
        Route::get('/health', [TrainerController::class, 'health']);
        Route::get('/languages', [TrainerController::class, 'languages']);
        
        // Phonemes & prompts
        Route::get('/phonemes/{language}', [TrainerController::class, 'phonemes']);
        Route::get('/prompts/{language}', [TrainerController::class, 'prompts']);
        Route::post('/prompts/{language}/for-phonemes', [TrainerController::class, 'promptsForPhonemes']);
        
        // Model scanning
        Route::post('/scan/{voiceModel}', [TrainerController::class, 'scanModel']);
        Route::get('/readiness/{voiceModel}', [TrainerController::class, 'getModelReadiness']);
        Route::post('/gaps/{voiceModel}', [TrainerController::class, 'analyzeGaps']);
        
        // Inference testing
        Route::post('/test/{voiceModel}', [TrainerController::class, 'testModelInference']);
        
        // Training jobs
        Route::post('/upload', [TrainerController::class, 'uploadAudio']);
        Route::post('/start', [TrainerController::class, 'startTraining']);
        Route::get('/jobs', [TrainerController::class, 'listTrainingJobs']);
        Route::get('/jobs/{jobId}', [TrainerController::class, 'trainingStatus']);
        Route::post('/jobs/{jobId}/cancel', [TrainerController::class, 'cancelTraining']);
        
        // Recording wizard
        Route::post('/wizard/sessions', [TrainerController::class, 'createWizardSession']);
        Route::get('/wizard/sessions/{sessionId}', [TrainerController::class, 'getWizardSession']);
        Route::post('/wizard/sessions/{sessionId}/start', [TrainerController::class, 'startWizardSession']);
        Route::get('/wizard/sessions/{sessionId}/prompt', [TrainerController::class, 'getWizardPrompt']);
        Route::post('/wizard/sessions/{sessionId}/submit', [TrainerController::class, 'submitWizardRecording']);
        Route::post('/wizard/sessions/{sessionId}/next', [TrainerController::class, 'wizardNext']);
        Route::post('/wizard/sessions/{sessionId}/previous', [TrainerController::class, 'wizardPrevious']);
        Route::post('/wizard/sessions/{sessionId}/skip', [TrainerController::class, 'wizardSkip']);
        Route::post('/wizard/sessions/{sessionId}/pause', [TrainerController::class, 'pauseWizardSession']);
        Route::post('/wizard/sessions/{sessionId}/complete', [TrainerController::class, 'completeWizardSession']);
        Route::delete('/wizard/sessions/{sessionId}', [TrainerController::class, 'cancelWizardSession']);

        // Model training data (cumulative recordings from all sessions)
        Route::get('/model/{modelSlug}/recordings', [TrainerController::class, 'getModelRecordings']);
        Route::get('/model/{modelSlug}/category-status', [TrainerController::class, 'getCategoryStatus']);
        Route::post('/model/{modelSlug}/train', [TrainerController::class, 'trainModel']);
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
        Route::post('/voice-models/scan-all', [TrainerController::class, 'scanAllModels']);
        
        // Role requests management
        Route::get('/role-requests', [RoleRequestController::class, 'adminIndex']);
        Route::post('/role-requests/{roleRequest}/approve', [RoleRequestController::class, 'approve']);
        Route::post('/role-requests/{roleRequest}/reject', [RoleRequestController::class, 'reject']);
        
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
