<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
use Illuminate\Support\Facades\DB;

return new class extends Migration
{
    /**
     * Run the migrations.
     */
    public function up(): void
    {
        // MySQL requires dropping and recreating ENUM columns to change values
        // First convert to varchar, then back to enum with new values
        DB::statement("ALTER TABLE training_runs MODIFY COLUMN status VARCHAR(20) NOT NULL DEFAULT 'pending'");
        
        // Update any 'running' values to 'training'
        DB::statement("UPDATE training_runs SET status = 'training' WHERE status = 'running'");
        
        // Now create the new enum
        DB::statement("ALTER TABLE training_runs MODIFY COLUMN status ENUM('pending', 'preparing', 'training', 'paused', 'completed', 'failed', 'cancelled') NOT NULL DEFAULT 'pending'");
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        DB::statement("ALTER TABLE training_runs MODIFY COLUMN status VARCHAR(20) NOT NULL DEFAULT 'pending'");
        DB::statement("UPDATE training_runs SET status = 'running' WHERE status = 'training'");
        DB::statement("UPDATE training_runs SET status = 'pending' WHERE status = 'preparing'");
        DB::statement("ALTER TABLE training_runs MODIFY COLUMN status ENUM('pending', 'running', 'paused', 'completed', 'failed', 'cancelled') NOT NULL DEFAULT 'pending'");
    }
};
