<?php

namespace App\Http\Controllers\Api;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Cache;

/**
 * Controller for YouTube Song Search functionality.
 * 
 * Proxies requests to the voice engine's YouTube endpoints
 * for searching and downloading songs for vocal processing.
 */
class YouTubeController extends Controller
{
    /**
     * Search YouTube for songs
     * 
     * @param Request $request
     * @return \Illuminate\Http\JsonResponse
     */
    public function search(Request $request)
    {
        $validated = $request->validate([
            'query' => 'required|string|max:200',
            'max_results' => 'nullable|integer|min:1|max:25',
        ]);

        $query = $validated['query'];
        $maxResults = $validated['max_results'] ?? 10;

        // Cache search results for 1 hour
        $cacheKey = 'youtube_search_' . md5($query . $maxResults);
        
        if (Cache::has($cacheKey)) {
            return response()->json(Cache::get($cacheKey));
        }

        try {
            $voiceEngineUrl = config('services.voice_engine.base_url', 'http://localhost:8001');
            
            $response = Http::timeout(30)->post("{$voiceEngineUrl}/youtube/search", [
                'query' => $query,
                'max_results' => $maxResults,
            ]);

            if (!$response->successful()) {
                return response()->json([
                    'error' => 'YouTube search failed',
                    'message' => $response->json('detail') ?? 'Unknown error',
                ], $response->status());
            }

            $result = $response->json();
            
            // Cache successful results
            Cache::put($cacheKey, $result, now()->addHour());

            return response()->json($result);

        } catch (\Exception $e) {
            return response()->json([
                'error' => 'YouTube search failed',
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Download audio from YouTube video
     * 
     * Returns base64 encoded audio for use in audio processing
     * 
     * @param Request $request
     * @return \Illuminate\Http\JsonResponse
     */
    public function download(Request $request)
    {
        $validated = $request->validate([
            'video_id' => 'required|string|max:20',
            'use_cache' => 'nullable|boolean',
        ]);

        $videoId = $validated['video_id'];
        $useCache = $validated['use_cache'] ?? true;

        try {
            $voiceEngineUrl = config('services.voice_engine.base_url', 'http://localhost:8001');
            
            // YouTube downloads can take a while
            $response = Http::timeout(120)->post("{$voiceEngineUrl}/youtube/download", [
                'video_id' => $videoId,
                'use_cache' => $useCache,
            ]);

            if (!$response->successful()) {
                return response()->json([
                    'error' => 'YouTube download failed',
                    'message' => $response->json('detail') ?? 'Unknown error',
                ], $response->status());
            }

            return response()->json($response->json());

        } catch (\Exception $e) {
            return response()->json([
                'error' => 'YouTube download failed',
                'message' => $e->getMessage(),
            ], 500);
        }
    }

    /**
     * Get information about a YouTube video
     * 
     * @param string $videoId
     * @return \Illuminate\Http\JsonResponse
     */
    public function info(string $videoId)
    {
        try {
            $voiceEngineUrl = config('services.voice_engine.base_url', 'http://localhost:8001');
            
            $response = Http::timeout(30)->get("{$voiceEngineUrl}/youtube/info/{$videoId}");

            if (!$response->successful()) {
                return response()->json([
                    'error' => 'Failed to get video info',
                    'message' => $response->json('detail') ?? 'Unknown error',
                ], $response->status());
            }

            return response()->json($response->json());

        } catch (\Exception $e) {
            return response()->json([
                'error' => 'Failed to get video info',
                'message' => $e->getMessage(),
            ], 500);
        }
    }
}
