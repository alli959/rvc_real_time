<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('jobs_queue', function (Blueprint $table) {
            $table->id();
            $table->uuid('uuid')->unique();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();
            $table->foreignId('voice_model_id')->nullable()->constrained()->nullOnDelete();
            
            // Job type: inference, training, preprocessing
            $table->string('type');
            
            // Status: pending, queued, processing, completed, failed, cancelled
            $table->string('status')->default('pending');
            
            // Input/output storage paths
            $table->string('input_path')->nullable();
            $table->string('output_path')->nullable();
            
            // Job parameters (pitch, f0_method, etc.)
            $table->json('parameters')->nullable();
            
            // Progress tracking
            $table->unsignedTinyInteger('progress')->default(0);
            $table->text('progress_message')->nullable();
            
            // Timing
            $table->timestamp('started_at')->nullable();
            $table->timestamp('completed_at')->nullable();
            
            // Error handling
            $table->text('error_message')->nullable();
            $table->json('error_details')->nullable();
            
            // Worker info
            $table->string('worker_id')->nullable();
            
            $table->timestamps();
            
            // Indexes
            $table->index(['user_id', 'status']);
            $table->index(['type', 'status']);
            $table->index('status');
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('jobs_queue');
    }
};
