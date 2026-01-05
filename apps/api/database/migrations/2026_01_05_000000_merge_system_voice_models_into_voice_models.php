<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\DB;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Str;

return new class extends Migration
{
    public function up(): void
    {
        // 1) Extend voice_models schema so it can also store system models.
        Schema::table('voice_models', function (Blueprint $table) {
            if (!Schema::hasColumn('voice_models', 'source')) {
                $table->string('source')->default('user')->after('user_id'); // user | system
            }

            // Fields used by the storage scanner (system models)
            if (!Schema::hasColumn('voice_models', 'model_file')) {
                $table->string('model_file')->nullable()->after('avatar');
            }
            if (!Schema::hasColumn('voice_models', 'index_file')) {
                $table->string('index_file')->nullable()->after('model_file');
            }
            if (!Schema::hasColumn('voice_models', 'has_index')) {
                $table->boolean('has_index')->default(false)->after('index_path');
            }
            if (!Schema::hasColumn('voice_models', 'size_bytes')) {
                $table->unsignedBigInteger('size_bytes')->default(0)->after('has_index');
            }
            if (!Schema::hasColumn('voice_models', 'storage_type')) {
                $table->string('storage_type')->nullable()->after('size_bytes'); // local | s3
            }
            if (!Schema::hasColumn('voice_models', 'storage_path')) {
                $table->string('storage_path')->nullable()->after('storage_type');
            }
            if (!Schema::hasColumn('voice_models', 'index_storage_path')) {
                $table->string('index_storage_path')->nullable()->after('storage_path');
            }
            if (!Schema::hasColumn('voice_models', 'is_active')) {
                $table->boolean('is_active')->default(true)->after('index_storage_path');
            }
            if (!Schema::hasColumn('voice_models', 'is_featured')) {
                $table->boolean('is_featured')->default(false)->after('is_active');
            }
            if (!Schema::hasColumn('voice_models', 'last_synced_at')) {
                $table->timestamp('last_synced_at')->nullable()->after('is_featured');
            }
        });

        // Replace unique(slug) with unique(source, slug)
        // (voice_models.slug was originally unique for user models; system models also have slugs)
        Schema::table('voice_models', function (Blueprint $table) {
            // Drop the old unique index if it exists.
            try {
                $table->dropUnique(['slug']);
            } catch (\Throwable $e) {
                // ignore - index might already be dropped in some environments
            }

            // Create the new composite unique.
            try {
                $table->unique(['source', 'slug']);
            } catch (\Throwable $e) {
                // ignore
            }
        });

        // 2) If the legacy system table exists, migrate its rows into voice_models.
        if (Schema::hasTable('system_voice_models')) {
            $rows = DB::table('system_voice_models')->get();

            foreach ($rows as $row) {
                $exists = DB::table('voice_models')
                    ->where('source', 'system')
                    ->where('slug', $row->slug)
                    ->exists();

                if ($exists) {
                    continue;
                }

                DB::table('voice_models')->insert([
                    'uuid' => (string) Str::uuid(),
                    'user_id' => null,
                    'source' => 'system',

                    'name' => $row->name,
                    'slug' => $row->slug,
                    'description' => $row->description,
                    'avatar' => null,

                    'model_file' => $row->model_file,
                    'model_path' => $row->model_path,
                    'index_file' => $row->index_file,
                    'index_path' => $row->index_path,
                    'has_index' => (bool) $row->has_index,
                    'size_bytes' => (int) $row->size_bytes,
                    'storage_type' => $row->storage_type,
                    'storage_path' => $row->storage_path,
                    'index_storage_path' => $row->index_storage_path,

                    'engine' => $row->engine ?? 'rvc',
                    'visibility' => 'public',
                    'status' => 'ready',
                    'metadata' => $row->metadata,
                    'tags' => null,

                    'is_active' => (bool) $row->is_active,
                    'is_featured' => (bool) $row->is_featured,
                    'usage_count' => (int) $row->usage_count,
                    'download_count' => 0,
                    'last_synced_at' => $row->last_synced_at,

                    'has_consent' => false,
                    'consent_notes' => null,

                    'created_at' => $row->created_at ?? now(),
                    'updated_at' => $row->updated_at ?? now(),
                    'deleted_at' => null,
                ]);
            }

            Schema::dropIfExists('system_voice_models');
        }
    }

    public function down(): void
    {
        // This migration is a one-way consolidation.
        // We do not attempt to re-create the legacy system table on rollback.
    }
};
