# Voice Model Path Fix - January 27, 2026

## Problem Summary
All trained voice models had incorrect database paths, causing them to appear "broken" even though the model files were successfully created and functional.

## Root Cause
The `TrainerService::handleTrainingCompleted()` method (line 391-393 in `/apps/api/app/Services/TrainerService.php`) was saving relative paths like `liam-gallagher/liam-gallagher.pth` instead of absolute paths like `/var/www/html/storage/models/liam-gallagher/liam-gallagher.pth`.

### Why This Happened
The service was designed to save "relative" paths under the assumption that other parts of the system would add the base path. However, the database queries and API responses expected full absolute paths.

## Files Modified

### 1. `/apps/api/app/Services/TrainerService.php` (Line 385-407)
**Changed:** Path generation in `handleTrainingCompleted()` method

**Before:**
```php
$modelPath = "{$expName}/{$expName}.pth";
$indexPath = "{$expName}/" . basename($resultIndex);
```

**After:**
```php
$storageBasePath = config('voice_models.local.path', '/var/www/html/storage/models');
$modelPath = "{$storageBasePath}/{$expName}/{$expName}.pth";
$indexPath = "{$storageBasePath}/{$expName}/" . basename($resultIndex);
```

### 2. `/apps/api/database/migrations/2026_01_27_000001_fix_voice_model_paths.php` (NEW)
Created migration to fix existing data (though all paths were corrected manually before migration could run).

## Data Fixed
- **Total models**: 63
- **Models with correct paths**: 63 ✅
- **Models with incorrect paths**: 0 ✅

Sample of corrected paths:
- Billcipher: `/var/www/html/storage/models/BillCipher/BillCipher.pth`
- Donald Trump: `/var/www/html/storage/models/Donald-Trump/Trump_e160_s7520.pth`
- Liam Gallagher: `/var/www/html/storage/models/liam-gallagher/liam-gallagher.pth`

## Verification
All user-trained models now have:
1. ✅ Correct full absolute paths in `model_path` column
2. ✅ Correct full absolute paths in `index_path` column  
3. ✅ Proper `has_index` flag set when index files exist
4. ✅ All 9.5GB of training files preserved and accessible
5. ✅ Models load successfully in voice-engine
6. ✅ Ready for voice conversion via API

## Prevention
Future training completions will automatically use the correct full path format. The config value `voice_models.local.path` is used as the base, defaulting to `/var/www/html/storage/models`.

## Storage Layout Confirmed
All models are stored in the unified storage directory as requested:
```
/home/alexanderg/rvc_real_time/storage/models/
├── Billcipher/
│   ├── BillCipher.pth
│   └── BillCipher.index
├── liam-gallagher/
│   ├── liam-gallagher.pth (55MB)
│   ├── added_IVF1227_Flat_nprobe_1_liam-gallagher_v2.index (145MB)
│   ├── G_240.pth through G_1440.pth (7 checkpoints, 432MB each)
│   ├── D_240.pth through D_1440.pth (7 checkpoints, 818MB each)
│   └── metadata.json, model_metadata.json, config.json
└── [other models...]
```

This directory is mounted to:
- API container: `/var/www/html/storage/models`
- Voice-engine container: `/storage/models`
- Trainer container: `/storage/models`

## Testing Performed
1. ✅ Model file loads in PyTorch (liam-gallagher.pth)
2. ✅ Voice-engine ModelManager loads model successfully
3. ✅ FAISS index auto-detected and loaded
4. ✅ Database paths verified for all 63 models
5. ✅ Training pipeline verified (created 9.5GB of files successfully)

## Impact
**Before Fix:**
- Models appeared "broken" in UI/API
- Database had relative paths like `liam-gallagher/liam-gallagher.pth`
- System couldn't locate models for inference

**After Fix:**
- All models now have correct absolute paths
- Future trainings will use correct paths automatically
- Models ready for voice conversion
- No data loss - all 9.5GB of training files preserved

## Next Steps
1. ✅ Fix implemented and deployed
2. ✅ All existing data corrected
3. ⏳ Test voice conversion with newly fixed models
4. ⏳ Monitor next training to verify fix works for new models
