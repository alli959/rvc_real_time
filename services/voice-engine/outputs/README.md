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
  --model ./assets/models/BillCipher/BillCipher.pth \
  --index ./assets/models/BillCipher/BillCipher.index \
  --input ./input/audio.wav \
  --output ./outputs/converted.wav
```

Your converted file will be: `./outputs/converted.wav`

## File Naming

When processing multiple files, you can use patterns:
```bash
# Single file
--output ./outputs/converted.wav

# Timestamped output
--output ./outputs/output_$(date +%Y%m%d_%H%M%S).wav
```

## Notes

- Output files are overwritten if they already exist
- Make sure you have write permissions to this directory
- Large files may take longer to process
