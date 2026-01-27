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
        Schema::table('voice_models', function (Blueprint $table) {
            // Inference test results
            $table->float('inference_test_score')->nullable()->after('metadata');
            $table->json('inference_test_results')->nullable()->after('inference_test_score');
            $table->timestamp('inference_tested_at')->nullable()->after('inference_test_results');
            
            // Per-language inference scores
            $table->float('en_inference_score')->nullable()->after('inference_tested_at');
            $table->float('is_inference_score')->nullable()->after('en_inference_score');
        });
    }

    /**
     * Reverse the migrations.
     */
    public function down(): void
    {
        Schema::table('voice_models', function (Blueprint $table) {
            $table->dropColumn([
                'inference_test_score',
                'inference_test_results',
                'inference_tested_at',
                'en_inference_score',
                'is_inference_score',
            ]);
        });
    }
};
