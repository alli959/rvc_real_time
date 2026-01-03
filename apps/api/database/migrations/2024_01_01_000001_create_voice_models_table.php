<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('voice_models', function (Blueprint $table) {
            $table->id();
            $table->uuid('uuid')->unique();
            $table->foreignId('user_id')->nullable()->constrained()->nullOnDelete();
            
            // Model info
            $table->string('name');
            $table->string('slug')->unique();
            $table->text('description')->nullable();
            $table->string('avatar')->nullable();
            
            // Engine type (rvc, tts, etc.)
            $table->string('engine')->default('rvc');
            
            // Visibility: public, private, unlisted
            $table->string('visibility')->default('private');
            
            // Storage paths (relative to bucket)
            $table->string('model_path')->nullable(); // e.g. users/{userId}/models/{modelId}/model.pth
            $table->string('index_path')->nullable(); // e.g. users/{userId}/models/{modelId}/index.faiss
            $table->string('config_path')->nullable();
            
            // Model metadata
            $table->json('metadata')->nullable(); // f0_method, sample_rate, version, etc.
            $table->json('tags')->nullable();
            
            // Status: pending, ready, failed, training
            $table->string('status')->default('pending');
            
            // Stats
            $table->unsignedBigInteger('usage_count')->default(0);
            $table->unsignedBigInteger('download_count')->default(0);
            
            // Consent/legal
            $table->boolean('has_consent')->default(false);
            $table->text('consent_notes')->nullable();
            
            $table->timestamps();
            $table->softDeletes();
            
            // Indexes
            $table->index(['user_id', 'visibility']);
            $table->index(['visibility', 'status']);
            $table->index('engine');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('voice_models');
    }
};
