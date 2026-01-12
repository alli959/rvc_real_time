<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

/**
 * Add language readiness scores to voice_models table.
 * 
 * Stores phoneme coverage analysis results for EN and IS languages:
 * - Overall readiness score (0-100)
 * - Phoneme coverage percentage
 * - Missing phonemes list
 * - Last scan timestamp
 */
return new class extends Migration
{
    public function up(): void
    {
        Schema::table('voice_models', function (Blueprint $table) {
            // English language readiness
            $table->decimal('en_readiness_score', 5, 2)->nullable()->after('metadata')
                ->comment('English language readiness score 0-100');
            $table->decimal('en_phoneme_coverage', 5, 2)->nullable()
                ->comment('English phoneme coverage percentage');
            $table->json('en_missing_phonemes')->nullable()
                ->comment('List of missing English phonemes');
            
            // Icelandic language readiness
            $table->decimal('is_readiness_score', 5, 2)->nullable()
                ->comment('Icelandic language readiness score 0-100');
            $table->decimal('is_phoneme_coverage', 5, 2)->nullable()
                ->comment('Icelandic phoneme coverage percentage');
            $table->json('is_missing_phonemes')->nullable()
                ->comment('List of missing Icelandic phonemes');
            
            // Full scan results (detailed JSON)
            $table->json('language_scan_results')->nullable()
                ->comment('Full language scan results JSON');
            $table->timestamp('language_scanned_at')->nullable()
                ->comment('When the model was last scanned for language readiness');
        });
    }

    public function down(): void
    {
        Schema::table('voice_models', function (Blueprint $table) {
            $table->dropColumn([
                'en_readiness_score',
                'en_phoneme_coverage',
                'en_missing_phonemes',
                'is_readiness_score',
                'is_phoneme_coverage',
                'is_missing_phonemes',
                'language_scan_results',
                'language_scanned_at',
            ]);
        });
    }
};
