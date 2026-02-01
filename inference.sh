#!/bin/bash
#================================================================
# UltraGen - Quick Inference Script
# Generate high-quality 1080P videos from text prompts
# 
# For 4K generation: Use T3-Video (10x+ faster)
# https://github.com/zhangzjn/T3-Video
#================================================================

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Initialize conda environment

# Check arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <prompt> [checkpoint]"
    echo ""
    echo "Examples:"
    echo "  $0 \"A dog running in a park\""
    echo "  $0 \"A beautiful sunset\" checkpoints/custom_model.ckpt"
    echo ""
    echo "Note: This generates 1080P videos (1920x1088, 81 frames)"
    echo "For 4K generation: Use T3-Video (https://github.com/zhangzjn/T3-Video)"
    echo ""
    exit 1
fi

# Get prompt and optional checkpoint
PROMPT="$1"
CHECKPOINT="${2:-checkpoints/ultragen_1080p.ckpt}"
MODEL_DIR="checkpoints/Wan2.1-T2V-1.3B"
OUTPUT_DIR="outputs"

# Display configuration
echo "========================================"
echo " UltraGen Video Generation"
echo "========================================"
echo "Prompt: $PROMPT"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# Add project to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
python3 tools/inference/generate.py \
  --model_dir "$MODEL_DIR" \
  --checkpoint "$CHECKPOINT" \
  --prompt "$PROMPT" \
  --output_dir "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "✓ Generation complete!"
echo "Video saved to: $OUTPUT_DIR/"
echo "========================================"
