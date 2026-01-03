<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        Schema::create('local_voice_models', function (Blueprint $table) {
            $table->id();
            $table->string('slug')->unique(); // folder name, used as identifier
            $table->string('name'); // display name
            $table->text('description')->nullable();
            $table->string('model_file'); // .pth filename
            $table->string('model_path'); // full path to .pth
            $table->string('index_file')->nullable(); // .index filename
            $table->string('index_path')->nullable(); // full path to .index
            $table->boolean('has_index')->default(false);
            $table->bigInteger('size_bytes')->default(0);
            $table->string('type')->default('local'); // local, symlink, remote
            $table->string('engine')->default('rvc'); // rvc, so-vits-svc, etc
            $table->json('metadata')->nullable(); // extra info (training epochs, sample rate, etc)
            $table->boolean('is_active')->default(true); // can disable without deleting
            $table->boolean('is_featured')->default(false);
            $table->unsignedInteger('usage_count')->default(0);
            $table->timestamp('last_synced_at')->nullable();
            $table->timestamps();
            
            $table->index('is_active');
            $table->index('engine');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::dropIfExists('local_voice_models');
    }
};
