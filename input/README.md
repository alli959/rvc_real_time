# Input Audio Files

Place your input audio files here for local mode processing.

## Supported Formats

- WAV (.wav)
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
- For best results, use clean vocal audio with minimal background noise
