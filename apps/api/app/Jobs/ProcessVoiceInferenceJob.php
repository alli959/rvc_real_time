<?php

namespace App\Jobs;

use App\Models\JobQueue;
use App\Models\UsageEvent;
use App\Services\VoiceEngineService;
use App\Services\StorageService;
use Illuminate\Bus\Queueable;
use Illuminate\Contracts\Queue\ShouldQueue;
use Illuminate\Foundation\Bus\Dispatchable;
use Illuminate\Queue\InteractsWithQueue;
use Illuminate\Queue\SerializesModels;
use Illuminate\Support\Facades\Log;

class ProcessVoiceInferenceJob implements ShouldQueue
{
    use Dispatchable, InteractsWithQueue, Queueable, SerializesModels;

    public int $tries = 3;
    public int $timeout = 600; // 10 minutes
    public int $backoff = 30;

    protected JobQueue $job;

    /**
     * Create a new job instance.
     */
    public function __construct(JobQueue $job)
    {
        $this->job = $job;
    }

    /**
     * Execute the job.
     */
    public function handle(VoiceEngineService $voiceEngine, StorageService $storage): void
    {
        Log::info('Starting voice inference job', ['job_id' => $this->job->id]);
        
        try {
            // Mark as processing
            $this->job->markProcessing();
            
            // Get the model
            $model = $this->job->voiceModel;
            if (!$model) {
                throw new \Exception('Voice model not found');
            }
            
            // Prepare paths
            $inputPath = $storage->getInternalPath($this->job->input_path);
            $outputPath = $storage->getInternalPath($this->job->output_path);
            $modelPath = $storage->getInternalPath($model->model_path);
            $indexPath = $model->index_path ? $storage->getInternalPath($model->index_path) : null;
            
            // Get parameters
            $params = $this->job->parameters ?? [];
            
            // Call voice engine
            $result = $voiceEngine->processAudio([
                'input_path' => $inputPath,
                'output_path' => $outputPath,
                'model_path' => $modelPath,
                'index_path' => $indexPath,
                'pitch' => $params['pitch'] ?? 0,
                'index_rate' => $params['index_rate'] ?? 0.75,
                'filter_radius' => $params['filter_radius'] ?? 3,
                'resample_sr' => $params['resample_sr'] ?? 0,
                'rms_mix_rate' => $params['rms_mix_rate'] ?? 0.25,
                'protect' => $params['protect'] ?? 0.33,
                'f0_method' => $params['f0_method'] ?? 'rmvpe',
            ]);
            
            if (!$result['success']) {
                throw new \Exception($result['error'] ?? 'Voice engine processing failed');
            }
            
            // Get output file info for usage tracking
            $outputMeta = $storage->getMetadata($this->job->output_path);
            $audioSeconds = $result['data']['duration'] ?? 0;
            
            // Record usage
            UsageEvent::recordInference(
                $this->job->user_id,
                $this->job->voice_model_id,
                $audioSeconds,
                $this->job->id
            );
            
            // Increment model usage count
            $model->increment('usage_count');
            
            // Mark completed
            $this->job->markCompleted();
            
            Log::info('Voice inference job completed', [
                'job_id' => $this->job->id,
                'duration' => $audioSeconds,
            ]);
            
        } catch (\Exception $e) {
            Log::error('Voice inference job failed', [
                'job_id' => $this->job->id,
                'error' => $e->getMessage(),
            ]);
            
            $this->job->markFailed($e->getMessage());
            
            throw $e;
        }
    }

    /**
     * Handle a job failure.
     */
    public function failed(\Throwable $exception): void
    {
        Log::error('Voice inference job permanently failed', [
            'job_id' => $this->job->id,
            'error' => $exception->getMessage(),
        ]);
        
        $this->job->markFailed($exception->getMessage());
    }
}
