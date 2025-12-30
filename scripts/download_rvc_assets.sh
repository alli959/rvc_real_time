#!/usr/bin/env bash
set -euo pipefail

# Downloads the two auxiliary checkpoints that RVC inference requires:
#  - HuBERT (content encoder)
#  - RMVPE (pitch extractor)

mkdir -p assets/hubert assets/rmvpe

echo "Downloading hubert_base.pt -> assets/hubert/hubert_base.pt"
curl -L -o assets/hubert/hubert_base.pt https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt

echo "Downloading rmvpe.pt -> assets/rmvpe/rmvpe.pt"
curl -L -o assets/rmvpe/rmvpe.pt https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt

echo "Done."
