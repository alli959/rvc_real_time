<?php

use Illuminate\Support\Facades\Route;
use App\Http\Controllers\Admin\AdminAuthController;
use App\Http\Controllers\Admin\DashboardController;
use App\Http\Controllers\Admin\UserController;
use App\Http\Controllers\Admin\VoiceModelAdminController;

/**
 * IMPORTANT:
 * Laravel's default auth middleware redirects guests to route('login').
 * So we MUST provide a route named "login".
 *
 * We'll make it redirect to the admin login route.
 */
Route::get('/login', fn() => redirect()->route('admin.login'))->name('login');

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
        Route::get('/voice-models/{voiceModel}/edit', [VoiceModelAdminController::class, 'edit'])->name('models.edit');
        Route::put('/voice-models/{voiceModel}', [VoiceModelAdminController::class, 'update'])->name('models.update');

        // Per-user access control for models
        Route::get('/voice-models/{voiceModel}/access', [VoiceModelAdminController::class, 'editAccess'])->name('models.access.edit');
        Route::put('/voice-models/{voiceModel}/access', [VoiceModelAdminController::class, 'updateAccess'])->name('models.access.update');
    });
};

// Local: use /admin prefix so artisan serve works with localhost
if (app()->environment('local')) {
    Route::prefix('admin')->name('admin.')->group($adminRoutes);

    // Optional root redirect for local dev:
    Route::get('/', fn() => redirect('/admin'))->name('root');
} else {
    // Production: bind to admin subdomain (no prefix)
    Route::domain(config('admin.domain', 'admin.morphvox.net'))
        ->name('admin.')
        ->group($adminRoutes);
}
