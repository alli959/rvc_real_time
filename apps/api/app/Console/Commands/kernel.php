<?php

namespace App\Console\Commands;

use Illuminate\Console\Scheduling\Schedule;
use Illuminate\Foundation\Console\Kernel as ConsoleKernel;
use App\Console\Commands\SyncVoiceModels;
class Kernel extends ConsoleKernel
{
    protected $commands = [
        SyncVoiceModels::class,
    ];

    protected function schedule(Schedule $schedule): void
        {
            // Sync models every hour
            $schedule->command('voice-models:sync')->hourly();
            
            // Or daily with pruning
            $schedule->command('voice-models:sync --prune')->dailyAt('03:00');
        }
    }