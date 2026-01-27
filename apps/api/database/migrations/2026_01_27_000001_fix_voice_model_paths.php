<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Support\Facades\DB;

return new class extends Migration
{
    /**
     * Run the migrations.
     * 
     * Fix voice model paths that are missing the /var/www/html/storage/ prefix.
     * This migration corrects paths saved by the training service before the fix.
     */
    public function up(): void
    {
        $storageBasePath = config('voice_models.local.path', '/var/www/html/storage/models');
        
        // Fix model_path for models with relative paths
        DB::table('voice_models')
            ->where('storage_type', 'local')
            ->where('model_path', 'NOT LIKE', '/var/www/html/storage/%')
            ->whereNotNull('model_path')
            ->update([
                'model_path' => DB::raw("CONCAT('{$storageBasePath}/', model_path)"),
                'updated_at' => now(),
            ]);
        
        // Fix index_path for models with relative paths  
        DB::table('voice_models')
            ->where('storage_type', 'local')
            ->where('index_path', 'NOT LIKE', '/var/www/html/storage/%')
            ->whereNotNull('index_path')
            ->update([
                'index_path' => DB::raw("CONCAT('{$storageBasePath}/', index_path)"),
                'updated_at' => now(),
            ]);
            
        // Mark has_index=1 for models with non-null index paths
        DB::table('voice_models')
            ->whereNotNull('index_path')
            ->where('has_index', 0)
            ->update([
                'has_index' => 1,
                'updated_at' => now(),
            ]);
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        // Cannot reverse - we'd lose information about which paths were originally relative
        // This is a data correction migration that shouldn't be rolled back
    }
};
