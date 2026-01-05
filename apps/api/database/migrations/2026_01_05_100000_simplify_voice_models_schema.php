<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;

/**
 * Simplify voice_models table:
 * - Drop redundant columns added during "merge" migration
 * - Keep it simple: user models vs system models are distinguished by user_id (NULL = system)
 * - Remove 'source' column (redundant with user_id check)
 * - Remove duplicate file columns (model_file/index_file redundant with model_path/index_path)
 */
return new class extends Migration
{
    public function up(): void
    {
        // Drop unused local_voice_models table if it exists
        Schema::dropIfExists('local_voice_models');
        
        // Drop unused system_voice_models table if it exists
        Schema::dropIfExists('system_voice_models');

        // Simplify voice_models table
        // First, drop any indexes that reference columns we want to remove
        // Use raw SQL since Laravel 11 removed Doctrine
        try {
            DB::statement('ALTER TABLE voice_models DROP INDEX voice_models_source_slug_unique');
        } catch (\Exception $e) {
            // Index might not exist, that's fine
        }

        // Now check and drop columns
        $columnsToDrop = [];
        
        if (Schema::hasColumn('voice_models', 'model_file')) {
            $columnsToDrop[] = 'model_file';
        }
        if (Schema::hasColumn('voice_models', 'index_file')) {
            $columnsToDrop[] = 'index_file';
        }
        if (Schema::hasColumn('voice_models', 'source')) {
            $columnsToDrop[] = 'source';
        }
        if (Schema::hasColumn('voice_models', 'storage_path')) {
            $columnsToDrop[] = 'storage_path';
        }
        if (Schema::hasColumn('voice_models', 'index_storage_path')) {
            $columnsToDrop[] = 'index_storage_path';
        }

        if (!empty($columnsToDrop)) {
            Schema::table('voice_models', function (Blueprint $table) use ($columnsToDrop) {
                $table->dropColumn($columnsToDrop);
            });
        }

        // Add a unique index on just slug if it doesn't exist
        if (!Schema::hasIndex('voice_models', 'voice_models_slug_unique')) {
            Schema::table('voice_models', function (Blueprint $table) {
                $table->unique('slug');
            });
        }
    }

    public function down(): void
    {
        Schema::table('voice_models', function (Blueprint $table) {
            $table->string('source')->default('user')->after('user_id');
            $table->string('model_file')->nullable()->after('avatar');
            $table->string('index_file')->nullable()->after('model_file');
            $table->string('storage_path')->nullable()->after('storage_type');
            $table->string('index_storage_path')->nullable()->after('storage_path');
        });
    }
};
