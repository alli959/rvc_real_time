#!/bin/bash
set -e

echo "=== Fixing Storage Migration ==="
echo ""

# Source and target paths
SOURCE_MODELS="/home/alexanderg/rvc_real_time/services/voice-engine/assets/models"
TARGET_MODELS="/home/alexanderg/rvc_real_time/storage/models"
TARGET_DATA="/home/alexanderg/rvc_real_time/storage/data"
TARGET_LOGS="/home/alexanderg/rvc_real_time/storage/logs"

echo "Step 1: Cleaning up incorrectly placed files in storage/models root..."
# Remove .pth and .index files from the root of storage/models
find "$TARGET_MODELS" -maxdepth 1 -type f \( -name "*.pth" -o -name "*.index" \) -delete
echo "✓ Cleaned up root files"

echo ""
echo "Step 2: Migrating model files to correct subdirectories..."
# For each model directory in the source
for model_dir in "$SOURCE_MODELS"/*; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        
        # Skip if it's a backup directory
        if [[ "$model_name" == *.backup ]]; then
            continue
        fi
        
        # Create target directory if it doesn't exist
        mkdir -p "$TARGET_MODELS/$model_name"
        
        # Copy all files from source to target
        if [ "$(ls -A "$model_dir")" ]; then
            cp -v "$model_dir"/* "$TARGET_MODELS/$model_name/" 2>/dev/null || true
            echo "  ✓ Migrated $model_name"
        fi
    fi
done

echo ""
echo "Step 3: Creating output directories..."
mkdir -p "$TARGET_DATA/outputs"
mkdir -p "$TARGET_DATA/uploads"
mkdir -p "$TARGET_DATA/infra/redis"
mkdir -p "$TARGET_DATA/infra/mariadb"
echo "✓ Created output directories"

echo ""
echo "Step 4: Fixing permissions..."
chmod -R 755 "$TARGET_MODELS"
chmod -R 777 "$TARGET_LOGS/redis"
chmod -R 777 "$TARGET_DATA/infra/redis"
chmod -R 755 "$TARGET_DATA/outputs"
chmod -R 755 "$TARGET_DATA/uploads"
echo "✓ Fixed permissions"

echo ""
echo "Step 5: Verifying migration..."
MODEL_COUNT=$(find "$TARGET_MODELS" -mindepth 2 -name "*.pth" | wc -l)
echo "  Found $MODEL_COUNT model files (.pth) in subdirectories"

INDEX_COUNT=$(find "$TARGET_MODELS" -mindepth 2 -name "*.index" | wc -l)
echo "  Found $INDEX_COUNT index files (.index) in subdirectories"

echo ""
echo "=== Migration Complete ==="
echo ""
echo "Summary:"
echo "  - Models: $MODEL_COUNT .pth files"
echo "  - Indexes: $INDEX_COUNT .index files"
echo "  - Outputs directory: $TARGET_DATA/outputs"
echo "  - Uploads directory: $TARGET_DATA/uploads"
echo "  - Logs directories: Ready"
