#!/bin/bash

# =============================================================================
# Upload Voice Model to S3/MinIO
# =============================================================================
# Usage: ./upload_model_to_s3.sh <model_folder_path> [model_name]
#
# Example:
#   ./upload_model_to_s3.sh ./BillCipher
#   ./upload_model_to_s3.sh ./my_model CustomName
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_DIR="$(dirname "$0")/../infra/compose"
BUCKET="${AWS_BUCKET:-morphvox}"
PREFIX="models"

# Check arguments
if [ -z "$1" ]; then
    echo -e "${RED}Error: Model folder path required${NC}"
    echo "Usage: $0 <model_folder_path> [model_name]"
    exit 1
fi

MODEL_PATH="$1"
MODEL_NAME="${2:-$(basename "$MODEL_PATH")}"

# Validate folder exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Directory not found: $MODEL_PATH${NC}"
    exit 1
fi

# Check for model files
PTH_COUNT=$(find "$MODEL_PATH" -maxdepth 1 -name "*.pth" -o -name "*.onnx" 2>/dev/null | wc -l)
if [ "$PTH_COUNT" -eq 0 ]; then
    echo -e "${RED}Error: No .pth or .onnx model files found in $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}=== Uploading Voice Model to S3 ===${NC}"
echo "Model Name: $MODEL_NAME"
echo "Source Path: $MODEL_PATH"
echo "Destination: s3://$BUCKET/$PREFIX/$MODEL_NAME/"
echo ""

# List files to upload
echo -e "${YELLOW}Files to upload:${NC}"
ls -la "$MODEL_PATH"
echo ""

# Confirm
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Upload using MinIO client in container
echo -e "\n${YELLOW}Uploading to MinIO...${NC}"

docker exec -i morphvox-minio-init sh -c "
    mc alias set myminio http://minio:9000 \$MINIO_ACCESS_KEY \$MINIO_SECRET_KEY 2>/dev/null
    mc mb myminio/$BUCKET/$PREFIX/$MODEL_NAME --ignore-existing 2>/dev/null || true
" 

# Copy files into container and upload
for file in "$MODEL_PATH"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "  Uploading: $filename"
        docker cp "$file" morphvox-minio-init:/tmp/
        docker exec morphvox-minio-init mc cp "/tmp/$filename" "myminio/$BUCKET/$PREFIX/$MODEL_NAME/"
        docker exec morphvox-minio-init rm "/tmp/$filename"
    fi
done

echo -e "\n${GREEN}✓ Upload complete!${NC}"

# Sync to database
echo -e "\n${YELLOW}Syncing models to database...${NC}"
docker exec morphvox-api php artisan voice-models:sync --storage=s3

echo -e "\n${GREEN}✓ Done! Model '$MODEL_NAME' is now available.${NC}"
echo ""
echo "To verify:"
echo "  curl https://morphvox.net/api/voice-models/$MODEL_NAME"
