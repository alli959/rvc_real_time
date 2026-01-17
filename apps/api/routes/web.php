<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Admin\AdminAuthController;
use App\Http\Controllers\Admin\DashboardController;
use App\Http\Controllers\Admin\UserController;
use App\Http\Controllers\Admin\VoiceModelAdminController;
use App\Http\Controllers\Admin\JobsAdminController;
use App\Http\Controllers\Admin\LogsAdminController;
use App\Http\Controllers\Admin\MetricsAdminController;
use App\Http\Controllers\Admin\AssetsAdminController;
use App\Http\Controllers\Admin\VoiceEngineProxyController;

/**
 * Define admin routes once so we can mount them on:
 * - /admin/* in local
 * - admin.morphvox.net/* in production
 */
$adminRoutes = function () {

    // Auth (no middleware)
    Route::get('/login', [AdminAuthController::class, 'showLogin'])->name('login');
    Route::post('/login', [AdminAuthController::class, 'login'])->name('login.post');
    Route::post('/logout', [AdminAuthController::class, 'logout'])->name('logout');

    // Protected admin area
    Route::middleware(['auth', 'role:admin'])->group(function () {
        // Default admin site (dashboard)
        Route::get('/', [DashboardController::class, 'index'])->name('dashboard');

        // User invitations (must be before resource route)
        Route::get('/users/invite', [UserController::class, 'showInvite'])->name('users.invite');
        Route::post('/users/invite', [UserController::class, 'sendInvite'])->name('users.invite.send');

        // Users CRUD
        Route::resource('/users', UserController::class)->names('users');

        // Voice models + resync
        Route::get('/voice-models', [VoiceModelAdminController::class, 'index'])->name('models.index');
        Route::post('/voice-models/sync', [VoiceModelAdminController::class, 'sync'])->name('models.sync');
        Route::post('/voice-models/scan-languages', [VoiceModelAdminController::class, 'scanAllLanguages'])->name('models.scan-languages');
        Route::get('/voice-models/{voiceModel}/edit', [VoiceModelAdminController::class, 'edit'])->name('models.edit');
        Route::put('/voice-models/{voiceModel}', [VoiceModelAdminController::class, 'update'])->name('models.update');
        Route::post('/voice-models/{voiceModel}/scan-languages', [VoiceModelAdminController::class, 'scanModelLanguages'])->name('models.scan-model-languages');
        Route::post('/voice-models/{voiceModel}/transfer-ownership', [VoiceModelAdminController::class, 'transferOwnership'])->name('models.transfer-ownership');
        Route::post('/voice-models/{voiceModel}/test-inference', [VoiceModelAdminController::class, 'testModelInference'])->name('models.test-inference');
        Route::post('/voice-models/{voiceModel}/extract-model', [VoiceModelAdminController::class, 'extractModel'])->name('models.extract-model');
        Route::post('/voice-models/{voiceModel}/checkpoint', [VoiceModelAdminController::class, 'requestCheckpoint'])->name('models.checkpoint');

        // Per-user access control for models
        Route::get('/voice-models/{voiceModel}/access', [VoiceModelAdminController::class, 'editAccess'])->name('models.access.edit');
        Route::put('/voice-models/{voiceModel}/access', [VoiceModelAdminController::class, 'updateAccess'])->name('models.access.update');

        // Jobs queue
        Route::get('/jobs', [JobsAdminController::class, 'index'])->name('jobs.index');
        
        // System Logs
        Route::get('/logs', [LogsAdminController::class, 'index'])->name('logs.index');
        Route::get('/logs/laravel', [LogsAdminController::class, 'laravelLogs'])->name('logs.laravel');
        Route::get('/logs/worker', [LogsAdminController::class, 'workerLogs'])->name('logs.worker');
        Route::get('/logs/services', [LogsAdminController::class, 'services'])->name('logs.services');
        Route::get('/logs/service/{service}', [LogsAdminController::class, 'serviceLogs'])->name('logs.service');
        
        // System Metrics
        Route::get('/metrics', [MetricsAdminController::class, 'index'])->name('metrics.index');
        
        // System Assets
        Route::get('/assets', [AssetsAdminController::class, 'index'])->name('assets.index');
        
        // Voice Engine Proxy Routes (browser can't reach internal Docker hostnames)
        Route::prefix('proxy')->name('proxy.')->group(function () {
            Route::get('/metrics', [VoiceEngineProxyController::class, 'metrics'])->name('metrics');
            Route::get('/assets', [VoiceEngineProxyController::class, 'assets'])->name('assets');
            Route::get('/assets/by-category', [VoiceEngineProxyController::class, 'assetsByCategory'])->name('assets.byCategory');
            Route::post('/assets/{assetId}/{action}', [VoiceEngineProxyController::class, 'assetAction'])->name('assets.action');
            Route::get('/logs/services', [VoiceEngineProxyController::class, 'logsServices'])->name('logs.services');
            Route::get('/logs/{service}', [VoiceEngineProxyController::class, 'logsByService'])->name('logs.service');
            Route::get('/trainer/logs', [VoiceEngineProxyController::class, 'trainerLogs'])->name('trainer.logs');
            Route::get('/trainer/logs/{jobId}', [VoiceEngineProxyController::class, 'trainerJobLogs'])->name('trainer.job.logs');
        });
    });
};

// Local: use /admin prefix so artisan serve works with localhost
if (app()->environment('local')) {
    Route::prefix('admin')->name('admin.')->group($adminRoutes);

    // Optional root redirect for local dev:
    Route::get('/', fn() => redirect('/admin'))->name('root');
    
    // Global login redirect (only needed in local since admin uses prefix)
    Route::get('/login', fn() => redirect()->route('admin.login'))->name('login');
} else {
    // Production: bind to admin subdomain (no prefix)
    // IMPORTANT: Register admin domain routes FIRST
    Route::domain(config('admin.domain', 'admin.morphvox.net'))
        ->name('admin.')
        ->group($adminRoutes);
    
    // Global login redirect for main domain only (to avoid redirect loop on admin subdomain)
    Route::domain(config('admin.main_domain', 'morphvox.net'))
        ->group(function () {
            Route::get('/login', fn() => redirect()->route('admin.login'))->name('login');
        });
}
