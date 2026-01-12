#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON=${PYTHON:-python3}

IMAGE_DIR="$ROOT_DIR/experiments/data/images/no_helmet"
VIDEO_FILE="$ROOT_DIR/experiments/data/videos/child/20251009102344_02.mp4"
PROMPT_YAML="$ROOT_DIR/experiments/prompt.yaml"
OUTPUT_BASE="$ROOT_DIR/outputs"

mkdir -p "$OUTPUT_BASE"
echo "Using ROOT_DIR=$ROOT_DIR"

run_cmd() {
  echo "\n=== CMD: $* ==="
  if ! $*; then
    echo "Command failed: $*"
    return 1
  fi
}

# # 1) Dry-run on image directory (objects detection)
# run_cmd $PYTHON "$ROOT_DIR/experiments/vlm_model_eval.py" "$IMAGE_DIR" "$PROMPT_YAML" \
#   --prompt-name common_objects_detection --dry-run --output-dir "$OUTPUT_BASE/test_dryrun_images"

# # 2) Dry-run: video sampling by FPS (sample 1 fps fallback)
# run_cmd $PYTHON "$ROOT_DIR/experiments/vlm_model_eval.py" "$VIDEO_FILE" "$PROMPT_YAML" \
#   --prompt-name common_actions_recognition --sample-frame-per-second 1.0 --sample-count 8 --dry-run --output-dir "$OUTPUT_BASE/test_dryrun_video_fps"

# # 3) Dry-run: video uniform sampling by sample-count
# run_cmd $PYTHON "$ROOT_DIR/experiments/vlm_model_eval.py" "$VIDEO_FILE" "$PROMPT_YAML" \
#   --prompt-name common_scenes_classification --video-sampling-strategy uniform --sample-count 5 --dry-run --output-dir "$OUTPUT_BASE/test_dryrun_video_uniform"

# # 4) Dry-run: video sampling by FPS then batch the extracted frames (tests fps sampling + batching)
# run_cmd $PYTHON "$ROOT_DIR/experiments/vlm_model_eval.py" "$VIDEO_FILE" "$PROMPT_YAML" \
#   --prompt-name common_actions_recognition --sample-frame-per-second 2.0 --sample-count 8 \
#   --image-count-per-chat 4 --dry-run --output-dir "$OUTPUT_BASE/test_dryrun_video_fps_batched"

# # 5) Dry-run: image batching behavior (image-count-per-chat)
# run_cmd $PYTHON "$ROOT_DIR/experiments/vlm_model_eval.py" "$IMAGE_DIR" "$PROMPT_YAML" \
#   --prompt-name common_objects_detection --image-count-per-chat 3 --dry-run --output-dir "$OUTPUT_BASE/test_dryrun_images_batch3"

# # 6) Dry-run: image resize test
# run_cmd $PYTHON "$ROOT_DIR/experiments/vlm_model_eval.py" "$IMAGE_DIR" "$PROMPT_YAML" \
#   --prompt-name common_objects_detection --resize 800x600 --dry-run --output-dir "$OUTPUT_BASE/test_dryrun_images_resize"

# # 7) Dry-run: image crop test
# run_cmd $PYTHON "$ROOT_DIR/experiments/vlm_model_eval.py" "$IMAGE_DIR" "$PROMPT_YAML" \
#   --prompt-name common_objects_detection --crop 224x224 --dry-run --output-dir "$OUTPUT_BASE/test_dryrun_images_crop"

# # List generated outputs
# echo "\n=== Outputs under $OUTPUT_BASE ==="
# ls -l "$OUTPUT_BASE" || true

# echo "All tests completed. Check outputs and logs in $OUTPUT_BASE and logs/ for details." 

# --------------------------------------------------
# Non-dry-run safe test (small, sequential)
# --------------------------------------------------
MODEL_NAME="${MODEL_NAME:-}"  # optionally set externally
SMALL_TEST_DIR="$OUTPUT_BASE/non_dryrun_small"
mkdir -p "$SMALL_TEST_DIR"

# Prepare small subset (first 6 images) to avoid flooding local model
TMP_INPUT_DIR="$SMALL_TEST_DIR/input_subset"
rm -rf "$TMP_INPUT_DIR"
mkdir -p "$TMP_INPUT_DIR"
count=0
for img in "$IMAGE_DIR"/*.jpg "$IMAGE_DIR"/*.jpeg "$IMAGE_DIR"/*.png; do
  if [ -f "$img" ]; then
    cp "$img" "$TMP_INPUT_DIR/"
    count=$((count+1))
  fi
  if [ "$count" -ge 6 ]; then
    break
  fi
done

if [ "$count" -eq 0 ]; then
  echo "No images found for non-dry-run test in $IMAGE_DIR"
else
  echo "Running non-dry-run small test with $count images (sequential, batched)"
  # Run the evaluation without --dry-run. Use small image-count-per-chat and wait between runs.
  CMD=("$PYTHON" "$ROOT_DIR/experiments/vlm_model_eval.py" "$TMP_INPUT_DIR" "$PROMPT_YAML" --prompt-name common_objects_detection --image-count-per-chat 2 --output-dir "$SMALL_TEST_DIR")
  if [ -n "$MODEL_NAME" ]; then
    CMD+=(--model "$MODEL_NAME")
  fi
  # Execute and wait (script itself runs sequentially per batch)
  echo "\n=== NON-DRY-RUN CMD: ${CMD[*]} ==="
  "${CMD[@]}"
  echo "Non-dry-run test finished. Waiting 5s to avoid overloading local model..."
  sleep 5
fi

# --------------------------------------------------
# Non-dry-run: video FPS sampling then evaluate (sequential, small)
# --------------------------------------------------
if [ -f "$VIDEO_FILE" ]; then
  VIDEO_TEST_DIR="$OUTPUT_BASE/non_dryrun_video_fps"
  mkdir -p "$VIDEO_TEST_DIR"
  echo "Running non-dry-run video FPS sampling test on $VIDEO_FILE"

  # Small sampling to avoid overloading local model: sample 2 FPS, up to 8 frames
  VID_CMD=("$PYTHON" "$ROOT_DIR/experiments/vlm_model_eval.py" "$VIDEO_FILE" "$PROMPT_YAML"
    --prompt-name common_actions_recognition
    --sample-frame-per-second 2.0
    --sample-count 8
    --image-count-per-chat 2
    --output-dir "$VIDEO_TEST_DIR")

  # optional model name and runtime timeout from env
  if [ -n "${MODEL_NAME:-}" ]; then
    VID_CMD+=(--model "$MODEL_NAME")
  fi
  if [ -n "${RUNTIME_TIMEOUT:-}" ]; then
    VID_CMD+=(--runtime-timeout "$RUNTIME_TIMEOUT")
  fi

  echo "\n=== NON-DRY-RUN VIDEO CMD: ${VID_CMD[*]} ==="
  "${VID_CMD[@]}"
  echo "Video non-dry-run finished. Outputs:"
  ls -l "$VIDEO_TEST_DIR" || true
else
  echo "Video file not found: $VIDEO_FILE — skipping video non-dry-run test."
fi
