<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Dataset versions track snapshots of training audio for reproducibility.
     * Each version has a manifest hash that uniquely identifies the audio content.
     */
    public function up(): void
    {
        Schema::create('dataset_versions', function (Blueprint $table) {
            $table->id();
            $table->uuid('uuid')->unique();
            $table->foreignId('voice_model_id')->constrained('voice_models')->onDelete('cascade');
            
            // Version tracking
            $table->unsignedInteger('version_number');
            $table->char('manifest_hash', 64)->comment('SHA256 hash of audio file list + sizes');
            
            // Audio statistics
            $table->unsignedInteger('audio_count')->default(0);
            $table->decimal('total_duration_seconds', 10, 2)->default(0);
            $table->unsignedInteger('segment_count')->default(0)->comment('Preprocessed segment count');
            
            // Preprocessing config
            $table->unsignedInteger('sample_rate')->default(48000);
            $table->json('preprocessing_config')->nullable();
            
            // Paths (relative to model directory)
            $table->string('manifest_path', 500)->nullable();
            $table->string('directory_path', 500)->nullable()->comment('datasets/v001_<hash>');
            
            // Status
            $table->enum('status', ['pending', 'preprocessing', 'ready', 'failed'])->default('pending');
            $table->text('error_message')->nullable();
            $table->timestamp('preprocessed_at')->nullable();
            
            $table->timestamps();
            
            // Unique version per model
            $table->unique(['voice_model_id', 'version_number'], 'uk_model_version');
            $table->index(['voice_model_id', 'status'], 'idx_model_status');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('dataset_versions');
    }
};
