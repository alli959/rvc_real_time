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
        Schema::table('training_runs', function (Blueprint $table) {
            $table->json('metadata')->nullable()->after('error_details')->comment('Additional metadata like voice_engine_job_id');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('training_runs', function (Blueprint $table) {
            $table->dropColumn('metadata');
        });
    }
};
