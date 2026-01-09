#!/bin/bash
# Upload voice models to S3/MinIO
# Usage: ./upload_models_s3.sh <models_directory>

set -e

MODELS_DIR="${1:-./models}"
BUCKET="morphvox"
PREFIX="models"

# MinIO configuration (can be overridden by environment)
MINIO_ENDPOINT="${MINIO_ENDPOINT:-http://localhost:9000}"
MINIO_ACCESS_KEY="${MINIO_ACCESS_KEY:-minioadmin}"
MINIO_SECRET_KEY="${MINIO_SECRET_KEY:-minioadmin}"

echo "📦 MorphVox S3 Model Uploader"
echo "=============================="
echo "Endpoint: $MINIO_ENDPOINT"
echo "Bucket:   $BUCKET"
echo "Prefix:   $PREFIX"
echo "Source:   $MODELS_DIR"
echo ""

# Check if mc (MinIO Client) is installed
if ! command -v mc &> /dev/null; then
    echo "❌ MinIO Client (mc) not found. Installing..."
    
    # Detect OS and install
    if [[ "$OSTYPE" == "darwin"* ]]; then
        brew install minio/stable/mc
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        wget https://dl.min.io/client/mc/release/linux-amd64/mc -O /usr/local/bin/mc
        chmod +x /usr/local/bin/mc
    else
        echo "Please install MinIO Client manually: https://min.io/docs/minio/linux/reference/minio-mc.html"
        exit 1
    fi
fi

# Configure mc alias
echo "🔧 Configuring MinIO alias..."
mc alias set morphvox "$MINIO_ENDPOINT" "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY" --api S3v4

# Check if bucket exists
if ! mc ls morphvox/$BUCKET &> /dev/null; then
    echo "📁 Creating bucket: $BUCKET"
    mc mb morphvox/$BUCKET
fi

# Check models directory
if [ ! -d "$MODELS_DIR" ]; then
    echo "❌ Models directory not found: $MODELS_DIR"
    exit 1
fi

# Upload each model folder
echo ""
echo "📤 Uploading models..."

for model_dir in "$MODELS_DIR"/*/; do
    if [ -d "$model_dir" ]; then
        model_name=$(basename "$model_dir")
        
        # Check if model has .pth file
        pth_file=$(find "$model_dir" -name "*.pth" -o -name "*.onnx" | head -1)
        
        if [ -n "$pth_file" ]; then
            echo "  ⬆️  Uploading: $model_name"
            mc cp --recursive "$model_dir" "morphvox/$BUCKET/$PREFIX/$model_name/"
            echo "     ✅ Done"
        else
            echo "  ⚠️  Skipping $model_name (no .pth or .onnx file found)"
        fi
    fi
done

echo ""
echo "📋 Uploaded models:"
mc ls "morphvox/$BUCKET/$PREFIX/"

echo ""
echo "✅ Upload complete!"
echo ""
echo "Next steps:"
echo "  1. Run: docker exec morphvox-api php artisan voice-models:sync --storage=s3"
echo "  2. Check models at: https://your-domain.com/models"
