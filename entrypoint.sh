#!/bin/bash
set -e

# Ensure offline mode is set
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Simple model verification (no downloads)
verify_models() {
    echo "🔍 Verifying pre-installed models..."
    
    if [ ! -d "/app/models/all-mpnet-base-v2" ] || [ ! -f "/app/models/all-mpnet-base-v2/config.json" ]; then
        echo "❌ Error: SentenceTransformer model not found in image"
        exit 1
    else
        echo "✅ SentenceTransformer model verified"
    fi
    
    if [ ! -d "/app/models/flan-t5-small-openai-feedback" ] || [ ! -f "/app/models/flan-t5-small-openai-feedback/config.json" ]; then
        echo "❌ Error: Flan-T5 model not found in image"
        exit 1
    else
        echo "✅ Flan-T5 model verified"
    fi
    
    echo "🎯 All models ready for offline operation"
}

# Validate input files
echo "🔍 Validating input files..."

if [ ! -f "/app/challenge1b_input.json" ]; then
    echo "❌ Error: challenge1b_input.json not found"
    exit 1
fi

pdf_count=$(find /app/input -name "*.pdf" 2>/dev/null | wc -l)
if [ "$pdf_count" -eq 0 ]; then
    echo "❌ Error: No PDF files found in /app/input/"
    exit 1
fi

echo "✅ Found $pdf_count PDF files to process"

# Verify models are built into image
verify_models

# Run the pipeline in pure offline mode
echo "🚀 Starting offline PDF processing pipeline..."
cd /app
python3 combined.py

echo "✅ Processing complete!"
if [ -f "/app/challenge1b_output.json" ]; then
    echo "📄 Output file generated: challenge1b_output.json"
else
    echo "⚠️  Warning: Output file not found"
fi
