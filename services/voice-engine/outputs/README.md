# Output Audio Files

Converted audio files will be saved here when using local mode.

## Output Format

- Default format: WAV (PCM)
- Sample rate: Determined by model (typically 40kHz or 48kHz)
- Channels: Mono

## Example Output

After running:
```bash
python main.py --mode local \
  --input ./input/audio.wav \
  --output ./outputs/converted.wav
```

Your converted file will be: `./outputs/converted.wav`
