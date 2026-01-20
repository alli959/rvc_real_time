<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Training checkpoints track all saved model states during training.
     * Supports milestone checkpoints, best checkpoints, and archiving.
     */
    public function up(): void
    {
        Schema::create('training_checkpoints', function (Blueprint $table) {
            $table->id();
            $table->uuid('uuid')->unique();
            $table->foreignId('training_run_id')->constrained('training_runs')->onDelete('cascade');
            
            // Position in training
            $table->unsignedInteger('epoch');
            $table->unsignedBigInteger('global_step');
            
            // Naming
            $table->string('checkpoint_name', 200)->comment('Human-readable name with all metadata');
            $table->string('short_name', 50)->comment('Short reference name like ep0020');
            
            // File paths (relative to run directory)
            $table->string('generator_path', 500);
            $table->string('discriminator_path', 500);
            $table->string('checkpoint_directory', 500)->comment('Directory containing G/D files');
            $table->unsignedBigInteger('file_size_bytes')->default(0);
            
            // Training metrics at this checkpoint
            $table->decimal('loss_g', 10, 6)->nullable()->comment('Generator loss');
            $table->decimal('loss_d', 10, 6)->nullable()->comment('Discriminator loss');
            $table->decimal('loss_mel', 10, 6)->nullable()->comment('Mel spectrogram loss');
            $table->decimal('loss_kl', 10, 6)->nullable()->comment('KL divergence loss');
            $table->decimal('loss_fm', 10, 6)->nullable()->comment('Feature matching loss');
            
            // Flags
            $table->boolean('is_milestone')->default(false)->comment('Keep during cleanup (e.g., every 50 epochs)');
            $table->boolean('is_best')->default(false)->comment('Best loss checkpoint');
            $table->boolean('is_final')->default(false)->comment('Final checkpoint of completed training');
            $table->boolean('is_archived')->default(false)->comment('Soft-deleted, files may be removed');
            $table->boolean('is_exported')->default(false)->comment('Has been exported to final model');
            
            // Metadata
            $table->json('metadata')->nullable()->comment('Additional checkpoint info');
            $table->string('notes', 500)->nullable()->comment('User notes');
            
            // Timestamps
            $table->timestamp('created_at')->useCurrent();
            $table->timestamp('archived_at')->nullable();
            $table->timestamp('exported_at')->nullable();
            
            // Indexes
            $table->index(['training_run_id', 'epoch'], 'idx_run_epoch');
            $table->index(['training_run_id', 'is_best'], 'idx_run_best');
            $table->index(['training_run_id', 'is_archived'], 'idx_run_archived');
            $table->unique(['training_run_id', 'epoch', 'global_step'], 'uk_run_epoch_step');
        });
        
        // Add foreign key for best_checkpoint_id in training_runs
        Schema::table('training_runs', function (Blueprint $table) {
            $table->foreign('best_checkpoint_id')
                  ->references('id')
                  ->on('training_checkpoints')
                  ->onDelete('set null');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('training_runs', function (Blueprint $table) {
            $table->dropForeign(['best_checkpoint_id']);
        });
        
        Schema::dropIfExists('training_checkpoints');
    }
};
