<?php

namespace App\Providers;

use Illuminate\Support\ServiceProvider;
use App\Services\VoiceEngineService;
use App\Services\StorageService;

class AppServiceProvider extends ServiceProvider
{
    /**
     * Register any application services.
     */
    public function register(): void
    {
        $this->app->singleton(VoiceEngineService::class, function ($app) {
            return new VoiceEngineService();
        });

        $this->app->singleton(StorageService::class, function ($app) {
            return new StorageService();
        });
    }

    /**
     * Bootstrap any application services.
     */
    public function boot(): void
    {
        // Trust proxy headers when behind reverse proxy (nginx)
        if (config('app.env') === 'production' || env('FORCE_HTTPS')) {
            \Illuminate\Support\Facades\URL::forceScheme('https');
            
            // Trust all proxies for now (nginx container)
            request()->server->set('HTTPS', 'on');
        }
    }
}
