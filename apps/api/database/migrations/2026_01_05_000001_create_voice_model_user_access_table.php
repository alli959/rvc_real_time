<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::create('voice_model_user_access', function (Blueprint $table) {
            $table->id();
            $table->foreignId('voice_model_id')->constrained('voice_models')->cascadeOnDelete();
            $table->foreignId('user_id')->constrained('users')->cascadeOnDelete();
            $table->boolean('can_view')->default(true);
            $table->boolean('can_use')->default(true);
            $table->timestamps();

            $table->unique(['voice_model_id', 'user_id']);
        });
    }

    public function down(): void
    {
        Schema::dropIfExists('voice_model_user_access');
    }
};
