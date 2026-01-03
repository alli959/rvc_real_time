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
        // Rename table to reflect it's for system/server models (not user-uploaded)
        Schema::rename('local_voice_models', 'system_voice_models');
        
        Schema::table('system_voice_models', function (Blueprint $table) {
            // Change 'type' to 'storage_type' for clarity (local, s3)
            $table->renameColumn('type', 'storage_type');
            
            // Add S3-specific fields
            $table->string('storage_path')->nullable()->after('engine'); // relative path in storage
            $table->string('index_storage_path')->nullable()->after('storage_path');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('system_voice_models', function (Blueprint $table) {
            $table->dropColumn(['storage_path', 'index_storage_path']);
            $table->renameColumn('storage_type', 'type');
        });
        
        Schema::rename('system_voice_models', 'local_voice_models');
    }
};
