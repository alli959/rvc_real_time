<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Training runs track individual training sessions with full lineage support.
     * Supports branching (creating new runs from existing checkpoints) and resuming.
     */
    public function up(): void
    {
        Schema::create('training_runs', function (Blueprint $table) {
            $table->id();
            $table->uuid('uuid')->unique();
            $table->foreignId('voice_model_id')->constrained('voice_models')->onDelete('cascade');
            $table->foreignId('job_queue_id')->nullable()->constrained('jobs_queue')->onDelete('set null');
            
            // Lineage - for branching support
            $table->foreignId('parent_run_id')->nullable()->constrained('training_runs')->onDelete('set null');
            $table->unsignedBigInteger('parent_checkpoint_id')->nullable()->comment('Checkpoint branched from');
            $table->unsignedInteger('run_number')->comment('Sequential run number per model');
            $table->string('branch_name', 100)->default('main');
            
            // Mode and status
            $table->enum('mode', ['new', 'resume', 'continue', 'branch'])->default('new');
            $table->enum('status', ['pending', 'running', 'paused', 'completed', 'failed', 'cancelled'])->default('pending');
            
            // Configuration snapshot (immutable after start)
            $table->char('config_hash', 16)->comment('Short hash of config for naming');
            $table->json('config_snapshot')->comment('Full training config frozen at start');
            $table->unsignedInteger('sample_rate');
            $table->unsignedSmallInteger('batch_size');
            $table->string('f0_method', 20)->default('rmvpe');
            $table->enum('version', ['v1', 'v2'])->default('v2');
            $table->boolean('use_pitch_guidance')->default(true);
            
            // Epoch tracking
            $table->unsignedInteger('starting_epoch')->default(0)->comment('Epoch number to start from');
            $table->unsignedInteger('current_epoch')->default(0);
            $table->unsignedInteger('target_epochs');
            $table->unsignedInteger('save_every_epoch')->default(10);
            
            // Dataset reference
            $table->foreignId('dataset_version_id')->constrained('dataset_versions');
            $table->char('dataset_hash', 64)->comment('Dataset manifest hash at training start');
            
            // Progress tracking
            $table->decimal('best_loss', 10, 6)->nullable();
            $table->unsignedBigInteger('best_checkpoint_id')->nullable();
            $table->unsignedInteger('training_time_seconds')->default(0);
            $table->unsignedInteger('checkpoint_count')->default(0);
            
            // Voice engine job tracking
            $table->string('voice_engine_job_id', 50)->nullable()->comment('Job ID from voice-engine');
            
            // Directory paths (relative to model directory)
            $table->string('run_directory', 500)->nullable()->comment('runs/run_001_main');
            
            // Timestamps
            $table->timestamp('started_at')->nullable();
            $table->timestamp('last_activity_at')->nullable();
            $table->timestamp('completed_at')->nullable();
            $table->timestamps();
            
            // Error info
            $table->text('error_message')->nullable();
            $table->json('error_details')->nullable();
            
            // Indexes
            $table->index(['voice_model_id', 'status'], 'idx_model_status');
            $table->index(['voice_model_id', 'run_number'], 'idx_model_run');
            $table->index('voice_engine_job_id', 'idx_ve_job');
        });
        
        // Add foreign key for best_checkpoint_id after training_checkpoints is created
        // This will be done in a separate migration
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('training_runs');
    }
};
