# RVC Voice Model Trainer - Technical Design Document

## Overview

The RVC Voice Model Trainer is a comprehensive feature set that enables users to create, analyze, and improve voice models with a focus on **English (en)** and **Icelandic (is)** language readiness. This document outlines the architecture, data flows, and implementation details.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RVC Trainer System                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐ │
│  │  Upload Flow    │  │  Recording      │  │  Model Scanner              │ │
│  │                 │  │  Wizard         │  │                             │ │
│  │  • ZIP upload   │  │                 │  │  • Analyze existing models  │ │
│  │  • Audio files  │  │  • EN prompts   │  │  • Language readiness score │ │
│  │  • Validation   │  │  • IS prompts   │  │  • Gap analysis             │ │
│  └────────┬────────┘  │  • Auto-slice   │  └──────────────┬──────────────┘ │
│           │           └────────┬────────┘                 │                │
│           │                    │                          │                │
│           ▼                    ▼                          ▼                │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                      Training Pipeline                                 ││
│  │                                                                        ││
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────┐ ││
│  │   │ Preprocess   │→ │ F0 Extract   │→ │ Feature      │→ │ Train     │ ││
│  │   │ (Slicer)     │  │ (RMVPE)      │  │ (HuBERT)     │  │ (RVC)     │ ││
│  │   └──────────────┘  └──────────────┘  └──────────────┘  └───────────┘ ││
│  │                                                                        ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                       │                                     │
│                                       ▼                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                      Versioning & Metadata                             ││
│  │                                                                        ││
│  │   • model_metadata.json (version, language scores, prompts used)       ││
│  │   • Incremental updates (v1.0, v1.1, v2.0...)                         ││
│  │   • Training history & provenance                                      ││
│  │                                                                        ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1. Language Readiness Scoring Rubric

### Scoring Components (Total: 100%)

| Component | Weight | Description | Measurement |
|-----------|--------|-------------|-------------|
| Phoneme Coverage | 35% | Coverage of target language phonemes | % of phonemes present |
| Vowel Variation | 20% | Range of vowel sounds | Unique vowels / target |
| Pitch Variation | 15% | F0 range diversity | Octave range coverage |
| Speaking Rate | 10% | Varied speech tempo | Rate variance |
| Prosody Quality | 10% | Natural intonation patterns | Contour analysis |
| Audio Quality | 10% | SNR, clarity, consistency | dB metrics |

### Language-Specific Phoneme Sets

#### English (en) - 44 phonemes
```
Consonants (24):
  p, b, t, d, k, g,           # Plosives
  tʃ, dʒ,                      # Affricates
  f, v, θ, ð, s, z, ʃ, ʒ, h,  # Fricatives
  m, n, ŋ,                     # Nasals
  l, r,                        # Liquids
  w, j                         # Glides

Vowels (20):
  iː, ɪ, e, æ, ɑː, ɒ, ɔː, ʊ, uː, ʌ,  # Monophthongs
  ə, ɜː,                              # Mid vowels
  eɪ, aɪ, ɔɪ, aʊ, əʊ, ɪə, eə, ʊə     # Diphthongs
```

#### Icelandic (is) - 33 phonemes
```
Consonants (22):
  p, pʰ, t, tʰ, c, cʰ, k, kʰ,  # Plosives (unaspirated/aspirated)
  f, v, θ, ð, s, ç, x, h,       # Fricatives
  m, n, ɲ, ŋ,                   # Nasals
  l, r                          # Liquids

Vowels (11):
  iː, ɪ, eː, ɛ, aː, a, ɔː, ɔ, uː, ʏ, œ  # Basic vowels
  
Diphthongs:
  ei, au, ou, ai                         # Common diphthongs
```

## 2. Recording Wizard Prompts

### English Prompt Set (200+ prompts organized by phoneme groups)

#### Balanced Coverage Prompts
```json
{
  "en_basic": [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "The five boxing wizards jump quickly."
  ],
  "en_plosives": [
    "Peter Piper picked a peck of pickled peppers.",
    "Big bad bears bounce beautiful balls.",
    "Tiny Tim took two teddy bears to town.",
    "Danny decided to dig ditches down deep."
  ],
  "en_fricatives": [
    "She sells seashells by the seashore.",
    "The thistle sifter sifted thistles through.",
    "Fresh French fried fish.",
    "Vincent vowed vengeance very viciously."
  ],
  "en_vowels": [
    "Each eagle easily eats eager eels.",
    "I scream, you scream, we all scream for ice cream.",
    "The rain in Spain stays mainly in the plain."
  ],
  "en_diphthongs": [
    "My bike tire is quite nice and white.",
    "How now brown cow?",
    "Roy enjoys his toys with joy."
  ]
}
```

### Icelandic Prompt Set (150+ prompts)

```json
{
  "is_basic": [
    "Kæri vinur, komdu sæll og blessaður.",
    "Það er gott veður í dag.",
    "Ísland er land íss og elds.",
    "Hundurinn hlaupir um garðinn."
  ],
  "is_aspirated": [
    "Þú þarft þetta þrjátíu þúsund.",
    "Kettlingurinn klifrar á köttum.",
    "Pottur og panna í púkanum."
  ],
  "is_fricatives": [
    "Sjórinn syngur sína sögu.",
    "Hvað heitir þessi hestur?",
    "Fljótt og vel."
  ],
  "is_special_vowels": [
    "Göngum saman yfir fjöllin.",
    "Sýndu mér yndislega hluti.",
    "Fjörður er fullt af fiskum."
  ]
}
```

## 3. Model Metadata Schema

```json
{
  "$schema": "https://morphvox.ai/schemas/model-metadata-v1.json",
  "model_id": "uuid-v4",
  "name": "Custom Voice Model",
  "version": "1.0.0",
  "created_at": "2024-01-15T12:00:00Z",
  "updated_at": "2024-01-15T12:00:00Z",
  
  "training_config": {
    "sample_rate": 40000,
    "f0_method": "rmvpe",
    "epochs": 200,
    "batch_size": 8,
    "save_frequency": 50,
    "pretrained_base": "assets/pretrained_v2/f0G40k.pth"
  },
  
  "audio_source": {
    "type": "upload|recording|youtube",
    "total_duration_seconds": 600,
    "num_clips": 45,
    "avg_clip_duration": 13.3
  },
  
  "language_readiness": {
    "en": {
      "overall_score": 87.5,
      "phoneme_coverage": 92.0,
      "vowel_variation": 85.0,
      "pitch_variation": 88.0,
      "speaking_rate": 80.0,
      "prosody_quality": 85.0,
      "audio_quality": 90.0,
      "missing_phonemes": ["ʒ", "ð"],
      "weak_phonemes": ["θ", "ŋ"]
    },
    "is": {
      "overall_score": 45.0,
      "phoneme_coverage": 35.0,
      "vowel_variation": 50.0,
      "pitch_variation": 88.0,
      "missing_phonemes": ["þ", "ð", "æ", "ö"],
      "weak_phonemes": ["p_h", "t_h", "k_h"]
    }
  },
  
  "version_history": [
    {
      "version": "1.0.0",
      "date": "2024-01-15T12:00:00Z",
      "changes": "Initial training with uploaded audio",
      "audio_added_seconds": 600,
      "prompts_recorded": []
    }
  ],
  
  "consent": {
    "voice_owner_consent": true,
    "consent_date": "2024-01-15T12:00:00Z",
    "consent_type": "self|recorded_consent|written_consent",
    "consent_file": null
  },
  
  "files": {
    "model_pth": "model_v1.0.0.pth",
    "model_index": "model_v1.0.0.index",
    "training_log": "training_log.json"
  }
}
```

## 4. Training Pipeline Integration

### Pipeline Steps

```python
# Step 1: Preprocess
def preprocess_dataset(
    exp_dir: str,
    trainset_dir: str,
    sr: int = 40000,
    n_threads: int = 4
) -> dict:
    """
    Slice audio into chunks, resample to target SR.
    Creates: logs/{exp_dir}/0_gt_wavs/, 1_16k_wavs/
    """

# Step 2: Extract F0
def extract_f0_features(
    exp_dir: str,
    f0_method: str = "rmvpe",
    device: str = "cuda:0"
) -> dict:
    """
    Extract pitch contours using RMVPE.
    Creates: logs/{exp_dir}/2a_f0/, 2b_f0nsf/
    """

# Step 3: Extract HuBERT Features
def extract_hubert_features(
    exp_dir: str,
    device: str = "cuda:0"
) -> dict:
    """
    Extract speaker embeddings.
    Creates: logs/{exp_dir}/3_feature256/
    """

# Step 4: Train Model
def train_model(
    exp_dir: str,
    sr: int = 40000,
    epochs: int = 200,
    batch_size: int = 8,
    save_every_epoch: int = 50,
    pretrain_G: str = None,
    pretrain_D: str = None,
    gpus: str = "0",
    version: str = "v2"
) -> dict:
    """
    Train RVC model with pretrained weights.
    Creates: logs/{exp_dir}/G_*.pth, D_*.pth
    """

# Step 5: Train Index
def train_index(exp_dir: str) -> str:
    """
    Build FAISS index for voice matching.
    Creates: logs/{exp_dir}/added_*.index
    """
```

## 5. API Endpoints

### Training Endpoints

```
POST /api/v1/train/upload
  - Upload audio files or ZIP
  - Returns: job_id, upload validation

POST /api/v1/train/start
  - Start training pipeline
  - Body: {exp_name, config, audio_paths}
  - Returns: job_id, status

GET /api/v1/train/status/{job_id}
  - Get training progress
  - Returns: step, progress%, logs

POST /api/v1/train/cancel/{job_id}
  - Cancel training job

GET /api/v1/train/prompts/{language}
  - Get recording prompts for language
  - Returns: prompt sets by category

POST /api/v1/train/analyze-recording
  - Analyze recorded audio for phoneme coverage
  - Returns: coverage report, suggestions
```

### Scanner Endpoints

```
POST /api/v1/scanner/analyze
  - Analyze model for language readiness
  - Body: {model_path, languages: ["en", "is"]}
  - Returns: readiness scores, gaps

POST /api/v1/scanner/suggest-prompts
  - Get prompts to fill coverage gaps
  - Body: {model_metadata, target_language}
  - Returns: prioritized prompts

GET /api/v1/scanner/compare/{model_id_a}/{model_id_b}
  - Compare two model versions
  - Returns: diff report
```

## 6. Directory Structure

```
services/voice-engine/
├── app/
│   ├── trainer/
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Training orchestration
│   │   ├── preprocess.py        # Audio preprocessing
│   │   ├── f0_extractor.py      # Pitch extraction
│   │   ├── feature_extractor.py # HuBERT features
│   │   ├── model_trainer.py     # RVC training
│   │   └── index_builder.py     # FAISS index
│   │
│   ├── analyzer/
│   │   ├── __init__.py
│   │   ├── phoneme_analyzer.py  # Phoneme coverage
│   │   ├── language_scorer.py   # Language readiness
│   │   ├── audio_quality.py     # SNR, clarity
│   │   └── prompt_suggester.py  # Gap filling
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── en_prompts.json      # English prompts
│   │   ├── is_prompts.json      # Icelandic prompts
│   │   └── prompt_loader.py     # Prompt utilities
│   │
│   └── trainer_api.py           # HTTP endpoints
│
├── assets/
│   ├── pretrained_v2/           # Base models
│   ├── hubert/                  # HuBERT model
│   ├── rmvpe/                   # RMVPE model
│   └── phoneme_data/            # IPA mappings
│
└── logs/
    └── {exp_name}/              # Training outputs
        ├── 0_gt_wavs/
        ├── 1_16k_wavs/
        ├── 2a_f0/
        ├── 2b_f0nsf/
        ├── 3_feature256/
        ├── model_metadata.json
        └── *.pth, *.index
```

## 7. Versioning System

### Version Format: `MAJOR.MINOR.PATCH`

- **MAJOR**: Significant changes to voice character
- **MINOR**: Added language coverage, improved quality
- **PATCH**: Bug fixes, metadata updates

### Incremental Update Flow

```
1. User requests improvement for language X
2. System analyzes current model → identifies gaps
3. Wizard presents targeted prompts
4. User records additional audio
5. System combines: existing_features + new_features
6. Trains incremental update (faster than full retrain)
7. Creates new version: v1.0.0 → v1.1.0
8. Updates model_metadata.json
```

## 8. Implementation Priority

### Phase 1: Core Pipeline (Week 1)
- [ ] Training pipeline module
- [ ] Preprocess, F0, HuBERT extraction
- [ ] Basic model training
- [ ] Index building

### Phase 2: Analysis (Week 2)
- [ ] Phoneme analyzer with phonemizer
- [ ] Language readiness scorer
- [ ] Audio quality analyzer
- [ ] Model scanner API

### Phase 3: Wizard (Week 3)
- [ ] Prompt loader and manager
- [ ] Recording wizard API
- [ ] Coverage tracker
- [ ] Gap-filling suggestions

### Phase 4: Polish (Week 4)
- [ ] Versioning system
- [ ] Metadata management
- [ ] UI integration
- [ ] Documentation

## 9. Dependencies

### Required Python Packages
```
phonemizer>=3.2.1      # Phoneme extraction (espeak backend)
epitran>=1.24          # IPA transcription
librosa>=0.10.0        # Audio analysis
praat-parselmouth>=0.4 # F0/prosody analysis
faiss-cpu>=1.7.4       # Index building
torch>=2.0.0           # Training
numpy>=1.24.0
scipy>=1.11.0
soundfile>=0.12.0
```

### System Dependencies
```bash
# For phonemizer
apt-get install espeak-ng

# For Icelandic (optional, improves accuracy)
apt-get install festival
```

## 10. Ethics & Consent

### Guardrails

1. **Voice Consent**: Required acknowledgment before training
2. **Usage Limits**: Rate limiting on training jobs
3. **Watermarking**: Optional audio watermark for generated voices
4. **Audit Trail**: Log all training with timestamps

### Consent Verification
```python
class ConsentVerification:
    TYPES = ["self_voice", "recorded_consent", "written_consent"]
    
    def verify(self, consent_type: str, evidence: Optional[bytes]) -> bool:
        # Self-voice: User attests it's their own voice
        # Recorded: Audio file of consent
        # Written: Document upload
        pass
```
