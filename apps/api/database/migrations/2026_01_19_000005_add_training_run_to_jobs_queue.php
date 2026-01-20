<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Adds training run reference to jobs_queue for better job tracking.
     */
    public function up(): void
    {
        Schema::table('jobs_queue', function (Blueprint $table) {
            $table->foreignId('training_run_id')
                  ->nullable()
                  ->after('type')
                  ->constrained('training_runs')
                  ->onDelete('set null');
            
            // Resume-related fields
            $table->boolean('is_resume')->default(false)->after('training_run_id');
            $table->foreignId('resume_from_checkpoint_id')
                  ->nullable()
                  ->after('is_resume')
                  ->constrained('training_checkpoints')
                  ->onDelete('set null');
            
            // Index for efficient lookups
            $table->index('training_run_id');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('jobs_queue', function (Blueprint $table) {
            $table->dropForeign(['training_run_id']);
            $table->dropForeign(['resume_from_checkpoint_id']);
            
            $table->dropColumn([
                'training_run_id',
                'is_resume',
                'resume_from_checkpoint_id',
            ]);
        });
    }
};
