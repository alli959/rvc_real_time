<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Adds training run and dataset version tracking columns to voice_models table.
     */
    public function up(): void
    {
        Schema::table('voice_models', function (Blueprint $table) {
            // Current training state
            $table->foreignId('current_run_id')
                  ->nullable()
                  ->after('id')
                  ->constrained('training_runs')
                  ->onDelete('set null');
            
            $table->foreignId('current_dataset_version_id')
                  ->nullable()
                  ->after('current_run_id')
                  ->constrained('dataset_versions')
                  ->onDelete('set null');
            
            // Aggregate training stats
            $table->unsignedInteger('total_training_epochs')
                  ->default(0)
                  ->after('current_dataset_version_id')
                  ->comment('Sum of epochs across all completed runs');
            
            $table->unsignedInteger('total_training_runs')
                  ->default(0)
                  ->after('total_training_epochs')
                  ->comment('Count of completed training runs');
            
            $table->unsignedBigInteger('total_training_seconds')
                  ->default(0)
                  ->after('total_training_runs')
                  ->comment('Total training time in seconds');
            
            // Best model tracking
            $table->foreignId('best_checkpoint_id')
                  ->nullable()
                  ->after('total_training_seconds')
                  ->constrained('training_checkpoints')
                  ->onDelete('set null');
            
            // Last training info
            $table->timestamp('last_trained_at')
                  ->nullable()
                  ->after('best_checkpoint_id');
            
            // Index for quick lookups
            $table->index('current_run_id');
            $table->index('current_dataset_version_id');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('voice_models', function (Blueprint $table) {
            $table->dropForeign(['current_run_id']);
            $table->dropForeign(['current_dataset_version_id']);
            $table->dropForeign(['best_checkpoint_id']);
            
            $table->dropColumn([
                'current_run_id',
                'current_dataset_version_id',
                'total_training_epochs',
                'total_training_runs',
                'total_training_seconds',
                'best_checkpoint_id',
                'last_trained_at',
            ]);
        });
    }
};
