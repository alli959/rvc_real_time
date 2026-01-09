# Input Audio Files

Place your input audio files here for local mode processing.

## Supported Formats

- WAV (.wav) - Recommended
- FLAC (.flac)
- MP3 (.mp3)
- OGG (.ogg)
- Any format supported by soundfile/librosa

## Example

```bash
# Copy your audio file here
cp /path/to/your/audio.wav ./input/

# Then run conversion
python main.py --mode local \
  --model ./assets/models/BillCipher/BillCipher.pth \
  --index ./assets/models/BillCipher/BillCipher.index \
  --input ./input/audio.wav \
  --output ./outputs/converted.wav
```

## Tips

- Use mono or stereo audio at any sample rate (will be automatically resampled)
- Shorter files process faster for testing
- Clean audio without background noise produces better results
- Recommended: 16kHz or 44.1kHz sample rate input

## Sample Input

For testing, you can use any audio file with speech. The model works best with:
- Clear vocal recordings
- Minimal background noise
- Single speaker audio
- For best results, use clean vocal audio with minimal background noise
