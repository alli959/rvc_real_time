<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('usage_events', function (Blueprint $table) {
            $table->id();
            $table->foreignId('user_id')->constrained()->cascadeOnDelete();
            $table->foreignId('voice_model_id')->nullable()->constrained()->nullOnDelete();
            $table->foreignId('job_id')->nullable()->references('id')->on('jobs_queue')->nullOnDelete();
            
            // Event type: inference, training, download, api_call
            $table->string('event_type');
            
            // Usage metrics
            $table->unsignedInteger('audio_seconds')->default(0);
            $table->unsignedInteger('tokens_used')->default(0);
            
            // Billing info (for future)
            $table->decimal('cost', 10, 6)->default(0);
            $table->string('billing_period')->nullable();
            
            $table->json('metadata')->nullable();
            
            $table->timestamp('created_at');
            
            // Indexes for billing queries
            $table->index(['user_id', 'created_at']);
            $table->index(['user_id', 'event_type', 'created_at']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('usage_events');
    }
};
