<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('jobs_queue', function (Blueprint $table) {
            $table->unsignedInteger('step_number')->default(0)->after('progress_message');
            $table->unsignedInteger('total_steps')->default(1)->after('step_number');
            $table->boolean('saved')->default(false)->after('worker_id');
            $table->index(['status', 'updated_at'], 'idx_jobs_status_updated');
        });
    }

    public function down(): void
    {
        Schema::table('jobs_queue', function (Blueprint $table) {
            $table->dropIndex('idx_jobs_status_updated');
            $table->dropColumn(['step_number', 'total_steps', 'saved']);
        });
    }
};
