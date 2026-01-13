import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from time import perf_counter

import yaml
import re
import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from chat_agent import ChatAgent
from model_config import ModelConfig
from runtime_config import RuntimeConfig

LOG_DIR = Path(__file__).parent.parent / "logs"
logger = logging.getLogger("vlm_model_eval")

try:  # Optional dependency for video frame extraction and image transforms
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - best effort optional module
    cv2 = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpg", ".mpeg"}

API_KEY = "sh-no-need-key"  # 在此填写你的 API Key
API_BASE_URL = "http://192.168.123.19:8080/v1"  # 在此填写你的 API Base URL
DEFAULT_MODEL = None  # 多模态模型


if API_KEY is None:
    API_KEY = os.getenv("OPENAI_API_KEY")
if API_BASE_URL is None:
    API_BASE_URL = os.getenv("OPENAI_API_BASE_URL")

if API_KEY is None or API_BASE_URL is None or DEFAULT_MODEL is None:
    try:
        cfg_path = Path(__file__).parent.parent / "config" / "default_model_config.yaml"
        if cfg_path.exists():
            cfg = ModelConfig.from_yaml(str(cfg_path))
            if API_KEY is None:
                API_KEY = cfg.api_key
            if API_BASE_URL is None:
                API_BASE_URL = cfg.api_base_url
            if DEFAULT_MODEL is None:
                DEFAULT_MODEL = cfg.model
    except Exception:
        pass


@dataclass
class MediaPreparationResult:
    media_type: str
    image_paths: List[Path]
    video_paths: List[Path]
    metadata: Dict[str, object]
    using_video_input: bool = False


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:  # pragma: no cover - argument validation
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:  # pragma: no cover - argument validation
        raise argparse.ArgumentTypeError("value must be a number") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def parse_size_arg(value: str) -> Tuple[int, int]:
    separators = ["x", "X", ",", "*"]
    for sep in separators:
        if sep in value:
            left, right = value.split(sep, 1)
            break
    else:
        raise argparse.ArgumentTypeError("size must use WxH format")
    try:
        width = int(left.strip())
        height = int(right.strip())
    except ValueError as exc:  # pragma: no cover - argument validation
        raise argparse.ArgumentTypeError("size components must be integers") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("size must be greater than zero")
    return width, height


def sanitize_for_filename(value: str) -> str:
    sanitized = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value]
    result = "".join(sanitized).strip("_")
    return result or "item"


def configure_logging(run_id: str) -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"vlm_model_eval_{run_id}.log"

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    logger.info("日志初始化完成，输出文件: %s", log_path)
    return log_path


async def fetch_available_model(api_base_url: str, api_key: Optional[str], timeout_seconds: int) -> Optional[str]:
    """Query the model listing endpoint and return a model id if available."""
    try:
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{api_base_url.rstrip('/')}/models"
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            # Expecting {"data": [{"id": "..."}, ...]} or similar
            models = data.get("data") if isinstance(data, dict) else None
            if isinstance(models, list) and models:
                first = models[0]
                if isinstance(first, dict):
                    # common fields: 'id' or 'model'
                    return first.get("id") or first.get("model")
                if isinstance(first, str):
                    return first
    except Exception:
        return None
    return None


def load_prompts(prompt_path: Path) -> Dict[str, Dict[str, Optional[str]]]:
    if not prompt_path.exists():
        raise FileNotFoundError(f"prompt file not found: {prompt_path}")
    with open(prompt_path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)

    if not isinstance(payload, dict):
        raise ValueError("prompt file must contain a mapping of prompt names to definitions")

    prompts: Dict[str, Dict[str, Optional[str]]] = {}
    for name, entry in payload.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"prompt '{name}' must be a mapping with keys 'user_prompt' and optional 'system_prompt'"
            )

        user_prompt = entry.get("user_prompt")
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            raise ValueError(f"prompt '{name}' must define a non-empty 'user_prompt'")

        system_prompt = entry.get("system_prompt")
        if system_prompt is not None and not isinstance(system_prompt, str):
            raise ValueError(f"prompt '{name}' system_prompt must be a string if provided")

        prompts[name] = {
            "user_prompt": user_prompt,
            "system_prompt": system_prompt,
        }

    return prompts


def select_indices(total: int, strategy: str, sample_count: int) -> List[int]:
    if total <= 0:
        return []
    if strategy == "all" or sample_count >= total:
        return list(range(total))
    if strategy == "first":
        return list(range(min(sample_count, total)))
    # uniform strategy
    step = total / float(sample_count)
    indices = []
    for i in range(sample_count):
        position = int(round(i * step))
        position = max(0, min(position, total - 1))
        if position not in indices:
            indices.append(position)
    while len(indices) < sample_count and len(indices) < total:
        candidate = len(indices)
        if candidate not in indices:
            indices.append(candidate)
    return sorted(indices)


def collect_image_files(directory: Path) -> List[Path]:
    return sorted(
        [item for item in directory.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS]
    )


def sample_video_frames(
    video_path: Path,
    working_dir: Path,
    strategy: str,
    sample_count: int,
    sample_frame_per_second: Optional[float] = None,
    resize: Optional[Tuple[int, int]] = None,
    crop: Optional[Tuple[int, int]] = None,
) -> MediaPreparationResult:
    if cv2 is None:
        raise RuntimeError(
            "opencv-python is required for video processing. Install it to enable video sampling."
        )

    working_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if total_frames <= 0:
        capture.release()
        raise RuntimeError(f"video contains no readable frames: {video_path}")

    if (
        sample_frame_per_second is not None
        and sample_frame_per_second > 0
        and video_fps > 0
    ):
        frame_interval = max(int(round(video_fps / sample_frame_per_second)), 1)
        requested_indices = list(range(0, total_frames, frame_interval))
        if requested_indices and requested_indices[-1] != total_frames - 1:
            requested_indices.append(total_frames - 1)
    else:
        requested_indices = select_indices(total_frames, strategy, sample_count)

    if not requested_indices:
        requested_indices = [0]

    requested_indices = sorted(set(requested_indices))
    saved_paths: List[Path] = []
    captured_indices: List[int] = []

    for frame_index in requested_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            continue
        frame_name = f"{video_path.stem}_frame_{frame_index:06d}.jpg"
        output_path = working_dir / frame_name
        if cv2.imwrite(str(output_path), frame):
            saved_paths.append(output_path)
            captured_indices.append(frame_index)

    capture.release()

    if not saved_paths:
        raise RuntimeError("no frames were extracted from the video")

    metadata: Dict[str, object] = {
        "media_type": "video_frames",
        "total_frames": total_frames,
        "requested_indices": requested_indices,
        "captured_indices": captured_indices,
        "working_directory": str(working_dir),
    }
    metadata["strategy"] = strategy
    metadata["video_fps"] = video_fps
    if (
        sample_frame_per_second is not None
        and sample_frame_per_second > 0
        and video_fps > 0
    ):
        metadata["sample_frame_per_second"] = sample_frame_per_second
        if len(requested_indices) > 1:
            metadata["frame_interval_frames"] = requested_indices[1] - requested_indices[0]
        else:
            metadata["frame_interval_frames"] = total_frames - 1
        metadata["sampling_mode"] = "fps"
    else:
        metadata["sampling_mode"] = "strategy"
    if resize:
        metadata["target_resize"] = {"width": resize[0], "height": resize[1]}
    if crop:
        metadata["target_crop"] = {"width": crop[0], "height": crop[1]}
    return MediaPreparationResult(
        media_type="video_frames",
        image_paths=saved_paths,
        video_paths=[video_path],
        metadata=metadata,
    )


def transform_image(
    image_path: Path,
    output_dir: Path,
    resize: Optional[Tuple[int, int]],
    crop: Optional[Tuple[int, int]],
    index: int,
) -> Path:
    if resize is None and crop is None:
        return image_path

    if cv2 is None:
        raise RuntimeError("opencv-python is required for image resize/crop operations")

    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to load image for processing: {image_path}")

    processed = image

    if resize is not None:
        width, height = resize
        if width <= 0 or height <= 0:
            raise ValueError("resize dimensions must be positive integers")
        processed = cv2.resize(processed, (width, height), interpolation=cv2.INTER_AREA)

    if crop is not None:
        crop_width, crop_height = crop
        if crop_width <= 0 or crop_height <= 0:
            raise ValueError("crop dimensions must be positive integers")
        current_height, current_width = processed.shape[:2]
        if crop_width > current_width or crop_height > current_height:
            raise ValueError(
                f"crop size ({crop_width}, {crop_height}) exceeds image dimensions ({current_width}, {current_height})"
            )
        x_start = max((current_width - crop_width) // 2, 0)
        y_start = max((current_height - crop_height) // 2, 0)
        x_end = x_start + crop_width
        y_end = y_start + crop_height
        processed = processed[y_start:y_end, x_start:x_end]

    suffix = image_path.suffix if image_path.suffix else ".jpg"
    output_name = f"{image_path.stem}_proc_{index:04d}{suffix}"
    output_path = output_dir / output_name

    if not cv2.imwrite(str(output_path), processed):
        raise RuntimeError(f"failed to write processed image: {output_path}")

    logger.debug("图像处理完成: %s -> %s", image_path, output_path)
    return output_path


def apply_image_transforms(
    image_paths: List[Path],
    working_dir: Path,
    resize: Optional[Tuple[int, int]],
    crop: Optional[Tuple[int, int]],
    label: str,
) -> List[Path]:
    if not image_paths or (resize is None and crop is None):
        return image_paths

    target_dir = working_dir / "transformed_images" / label
    logger.info(
        "对 %d 张图像应用缩放/裁剪，输出目录: %s",
        len(image_paths),
        target_dir,
    )
    processed_paths: List[Path] = []
    for idx, path in enumerate(image_paths):
        processed_path = transform_image(path, target_dir, resize, crop, idx)
        processed_paths.append(processed_path)
    return processed_paths


def preprocess_media(
    data_paths: List[Path],
    working_dir: Path,
    video_strategy: str,
    sample_count: int,
    sample_frame_per_second: Optional[float] = None,
    resize: Optional[Tuple[int, int]] = None,
    crop: Optional[Tuple[int, int]] = None,
    use_video: bool = False,
) -> MediaPreparationResult:
    if not data_paths:
        raise ValueError("no input data paths provided")

    if len(data_paths) == 1:
        data_path = data_paths[0]
        if not data_path.exists():
            raise FileNotFoundError(f"input data path not found: {data_path}")

        if data_path.is_dir():
            images = collect_image_files(data_path)
            if not images:
                raise ValueError(f"no image files found in directory: {data_path}")
            metadata: Dict[str, object] = {
                "media_type": "image_sequence",
                "total_images": len(images),
                "directory": str(data_path),
            }
            if resize:
                metadata["target_resize"] = {"width": resize[0], "height": resize[1]}
            if crop:
                metadata["target_crop"] = {"width": crop[0], "height": crop[1]}
            transformed = apply_image_transforms(
                images,
                working_dir,
                resize,
                crop,
                "directory",
            )
            if transformed != images:
                metadata["transform_applied"] = True
                metadata["transformed_from"] = [str(p) for p in images]
                images = transformed
            metadata["paths_processed"] = [str(p) for p in images]
            return MediaPreparationResult(
                media_type="image_sequence",
                image_paths=images,
                video_paths=[],
                metadata=metadata,
            )

        suffix = data_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            metadata = {
                "media_type": "single_image",
                "file": str(data_path),
            }
            if resize:
                metadata["target_resize"] = {"width": resize[0], "height": resize[1]}
            if crop:
                metadata["target_crop"] = {"width": crop[0], "height": crop[1]}
            transformed = apply_image_transforms(
                [data_path],
                working_dir,
                resize,
                crop,
                "single",
            )
            result_paths = transformed if transformed else [data_path]
            if result_paths[0] != data_path:
                metadata["transform_applied"] = True
                metadata["transformed_from"] = [str(data_path)]
            metadata["paths_processed"] = [str(p) for p in result_paths]
            return MediaPreparationResult(
                media_type="single_image",
                image_paths=result_paths,
                video_paths=[],
                metadata=metadata,
            )

        if suffix in VIDEO_EXTENSIONS:
            if use_video:
                metadata = {
                    "media_type": "video",
                    "file": str(data_path),
                    "strategy": "direct",
                }
                if resize:
                    metadata["target_resize"] = {"width": resize[0], "height": resize[1]}
                if crop:
                    metadata["target_crop"] = {"width": crop[0], "height": crop[1]}
                if resize or crop:
                    logger.warning("当前以完整视频模式传输，缩放/裁剪参数不会被应用。")
                return MediaPreparationResult(
                    media_type="video",
                    image_paths=[],
                    video_paths=[data_path],
                    metadata=metadata,
                    using_video_input=True,
                )

            frames_result = sample_video_frames(
                video_path=data_path,
                working_dir=working_dir,
                strategy=video_strategy,
                sample_count=sample_count,
                sample_frame_per_second=sample_frame_per_second,
                resize=resize,
                crop=crop,
            )
            if resize or crop:
                try:
                    transformed_frames = apply_image_transforms(
                        frames_result.image_paths,
                        working_dir,
                        resize,
                        crop,
                        "video_frames",
                    )
                    if transformed_frames != frames_result.image_paths:
                        frames_result.metadata["transform_applied"] = True
                        frames_result.metadata["transformed_from"] = [str(p) for p in frames_result.image_paths]
                        frames_result.image_paths = transformed_frames
                    frames_result.metadata["paths_processed"] = [str(p) for p in frames_result.image_paths]
                except Exception as exc:
                    logger.exception("视频帧图像处理失败: %s", exc)
                    raise
            else:
                frames_result.metadata["paths_processed"] = [str(p) for p in frames_result.image_paths]
            return frames_result

        raise ValueError(f"unsupported data type for path: {data_path}")

    images: List[Path] = []
    for path in data_paths:
        if not path.exists():
            raise FileNotFoundError(f"input data path not found: {path}")
        if path.is_dir():
            raise ValueError("directories cannot be mixed with multiple explicit paths")
        suffix = path.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            raise ValueError(f"only image files are allowed when passing multiple paths: {path}")
        images.append(path)

    metadata = {
        "media_type": "image_sequence",
        "total_images": len(images),
        "source": "multiple_paths",
        "paths": [str(p) for p in images],
    }
    if resize:
        metadata["target_resize"] = {"width": resize[0], "height": resize[1]}
    if crop:
        metadata["target_crop"] = {"width": crop[0], "height": crop[1]}

    transformed = apply_image_transforms(
        images,
        working_dir,
        resize,
        crop,
        "multiple",
    )
    if transformed != images:
        metadata["transform_applied"] = True
        metadata["transformed_from"] = [str(p) for p in images]
        images = transformed
    metadata["paths_processed"] = [str(p) for p in images]

    return MediaPreparationResult(
        media_type="image_sequence",
        image_paths=images,
        video_paths=[],
        metadata=metadata,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate VLM responses for images or video inputs")
    parser.add_argument("data_path", nargs="+", help="Path(s) to media: video file, image file(s), or directory")
    parser.add_argument("prompt_path", help="Path to prompt YAML file with system/user prompt definitions")
    parser.add_argument("--prompt-name", default="default", help="Name of the prompt to use from the JSON file")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name to use for ChatAgent")
    parser.add_argument("--video-sampling-strategy", choices=["uniform"], default="uniform", help="Sampling strategy for extracting frames from video inputs (currently only uniform)")
    parser.add_argument("--sample-count", type=positive_int, default=8, help="Fallback number of frames to sample when frame extraction is used and sample-frame-per-second is not provided")
    parser.add_argument("--sample-frame-per-second", type=positive_float, default=None, help="Target sampling frame rate when extracting frames from video inputs")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for the model")
    parser.add_argument("--resize", type=parse_size_arg, default=None, help="Optional resize target in WxH format (reserved for future processing)")
    parser.add_argument("--crop", type=parse_size_arg, default=None, help="Optional crop size in WxH format (reserved for future processing)")
    parser.add_argument("--output-dir", default=None, help="Directory to store result JSON and processed frames (defaults to project outputs dir)")
    parser.add_argument("--dry-run", action="store_true", help="Prepare inputs and prompts without calling the model")
    parser.add_argument("--use-video", action="store_true", help="Skip frame extraction and send video directly to the model when input is a video")
    parser.add_argument("--image-count-per-chat", type=positive_int, default=None, help="Maximum number of images to send in each chat call when processing image sequences")
    parser.add_argument("--runtime-timeout", type=positive_int, default=60, help="Runtime HTTP request timeout in seconds for contacting the model server")
    return parser


def build_model_config(model_name: str, temperature: float) -> ModelConfig:
    config_kwargs: Dict[str, object] = {
        "model": model_name,
        "temperature": temperature,
    }
    if API_KEY:
        config_kwargs["api_key"] = API_KEY
    if API_BASE_URL:
        config_kwargs["api_base_url"] = API_BASE_URL
    if DEFAULT_MODEL:
        config_kwargs["model"] = model_name
    return ModelConfig(**config_kwargs)


def ensure_output_directory(root: Optional[str]) -> Path:
    if root:
        directory = Path(root)
    else:
        directory = Path(__file__).parent.parent / "outputs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_result(
    output_dir: Path,
    model_config: ModelConfig,
    data_path: Path,
    timestamp: str,
    prompt_path: Path,
    prompt_name: str,
    system_prompt: Optional[str],
    user_prompt: str,
    used_default_system_prompt: bool,
    preprocess_result: MediaPreparationResult,
    conversation_records: List[Dict[str, object]],
    strategy: str,
    sample_count: int,
    image_count_per_chat: Optional[int],
    resize: Optional[Tuple[int, int]],
    crop: Optional[Tuple[int, int]],
    working_dir: Path,
) -> Path:
    model_name = model_config.model or "model"
    model_slug = sanitize_for_filename(model_name)
    if data_path.is_file():
        data_identifier = data_path.stem
    else:
        data_identifier = data_path.name
    data_slug = sanitize_for_filename(data_identifier)
    prompt_slug = sanitize_for_filename(prompt_name)
    filename = f"{model_slug}_{data_slug}_{prompt_slug}_{timestamp}.json"
    output_path = output_dir / filename

    resize_payload = None
    if resize:
        resize_payload = {"width": resize[0], "height": resize[1]}
    crop_payload = None
    if crop:
        crop_payload = {"width": crop[0], "height": crop[1]}

    model_config_dict = model_config.to_dict()
    model_config_dict.pop("api_key", None)
    api_base_url_value = model_config_dict.get("api_base_url")

    prepared_item_count = len(preprocess_result.image_paths)
    if preprocess_result.using_video_input:
        prepared_item_count += len(preprocess_result.video_paths)

    if preprocess_result.using_video_input:
        applied_strategy = "direct"
        applied_sample_count = None
    elif preprocess_result.media_type == "video_frames":
        if preprocess_result.metadata.get("sampling_mode") == "fps":
            applied_strategy = "fps"
            applied_sample_count = None
        else:
            applied_strategy = strategy
            applied_sample_count = sample_count
    else:
        applied_strategy = "n/a"
        applied_sample_count = None

    total_elapsed_seconds = None
    total_latency_seconds = None
    if conversation_records:
        total_elapsed_seconds = conversation_records[-1].get("elapsed_since_start_seconds")
        total_latency_seconds = sum(
            float(record.get("latency_seconds", 0.0)) for record in conversation_records
        )

    # Build compact representations to avoid repeating long lists (especially for video frames)
    prepared_image_paths_value = [str(p) for p in preprocess_result.image_paths]
    input_paths_value = [str(p) for p in preprocess_result.image_paths] + [str(p) for p in preprocess_result.video_paths]

    metadata_copy = dict(preprocess_result.metadata or {})
    if preprocess_result.media_type == "video_frames":
        total_frames = len(prepared_image_paths_value)
        examples = prepared_image_paths_value[: min(5, total_frames)]
        prepared_image_paths_value = {"frame_count": total_frames, "examples": examples}
        # shrink metadata paths_processed if present
        if "paths_processed" in metadata_copy:
            metadata_copy["paths_processed"] = examples
        # for input paths use only examples plus original video paths
        input_paths_value = examples + [str(p) for p in preprocess_result.video_paths]

    payload = {
        "run_id": timestamp,
        "created_at_utc": datetime.utcnow().isoformat() + "Z",
        "model": model_config_dict,
        "prompt": {
            "file": str(prompt_path),
            "name": prompt_name,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "used_default_system_prompt": used_default_system_prompt,
        },
        "input": {
            "input_paths": input_paths_value,
            "data_path": str(data_path),
            "media_type": preprocess_result.media_type,
            "prepared_item_count": prepared_item_count,
            "prepared_image_paths": prepared_image_paths_value,
            "original_video_paths": [str(p) for p in preprocess_result.video_paths],
            "sampling_strategy": applied_strategy,
            "sample_count": applied_sample_count,
            "image_count_per_chat": image_count_per_chat,
            "resize": resize_payload,
            "crop": crop_payload,
            "working_directory": str(working_dir),
            "metadata": metadata_copy,
            "using_video_input": preprocess_result.using_video_input,
        },
        "response": {
            "total_calls": len(conversation_records),
            "total_latency_seconds": total_latency_seconds,
            "total_elapsed_since_start_seconds": total_elapsed_seconds,
            "conversations": conversation_records,
        },
        "api_info": {
            "base_url": api_base_url_value,
        },
    }

    payload["test_evaluation"] = {}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("结果已保存: %s", output_path)
    return output_path


async def main(args: argparse.Namespace) -> None:
    # Create runtime config early so we can use its timeout for model discovery
    runtime_config = RuntimeConfig(enable_logging=False, timeout=args.runtime_timeout)

    # Determine model to use: prefer CLI arg, then DEFAULT_MODEL, otherwise try to discover from API
    model_name = args.model or DEFAULT_MODEL
    if not model_name:
        discovered = await fetch_available_model(API_BASE_URL, API_KEY, runtime_config.timeout)
        if discovered:
            model_name = discovered

    model_config = build_model_config(model_name, args.temperature)

    data_paths = [Path(p).expanduser().resolve() for p in args.data_path]
    prompt_path = Path(args.prompt_path).expanduser().resolve()
    prompts = load_prompts(prompt_path)
    if args.prompt_name not in prompts:
        available = ", ".join(sorted(prompts)) or "<empty>"
        raise KeyError(f"prompt '{args.prompt_name}' not found. available: {available}")

    default_entry = prompts.get("default")
    default_system_prompt = default_entry.get("system_prompt") if default_entry else None

    prompt_entry = prompts[args.prompt_name]
    user_prompt = prompt_entry["user_prompt"]
    system_prompt = prompt_entry.get("system_prompt")
    used_default_system_prompt = False

    if system_prompt is None:
        if default_system_prompt is None:
            raise ValueError(
                f"prompt '{args.prompt_name}' does not define system_prompt and no default system_prompt is available"
            )
        system_prompt = default_system_prompt
        used_default_system_prompt = args.prompt_name != "default"

    logger.debug(
        "已加载prompt，使用的system prompt来源: %s",
        "default" if used_default_system_prompt else args.prompt_name,
    )

    output_dir = ensure_output_directory(args.output_dir)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    working_dir = output_dir / "processed" / run_timestamp
    working_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(run_timestamp)

    # Log startup configuration and resolved model information
    logger.info("开始新的一次评估，prompt=%s，输入数量=%d", args.prompt_name, len(data_paths))
    logger.info("配置: api_base_url=%s, timeout=%ds, image_count_per_chat=%s, sample_count=%d, sample_fps=%s, use_video=%s",
                API_BASE_URL, runtime_config.timeout, str(args.image_count_per_chat), args.sample_count, str(args.sample_frame_per_second), args.use_video)
    resolved_model = model_config.model or "<none>"
    logger.info("使用模型: %s", resolved_model)
    logger.debug("使用的输入路径: %s", ", ".join(str(p) for p in data_paths))

    if not os.getenv("OPENAI_API_KEY") and API_KEY is None:
        logger.warning("未配置 OPENAI_API_KEY，可能导致模型调用失败")

    preprocess_result = preprocess_media(
        data_paths=data_paths,
        working_dir=working_dir,
        video_strategy=args.video_sampling_strategy,
        sample_count=args.sample_count,
        sample_frame_per_second=args.sample_frame_per_second,
        resize=args.resize,
        crop=args.crop,
        use_video=args.use_video,
    )
    logger.info(
        "媒体预处理完成: 类型=%s, 图片数量=%d, 视频数量=%d",
        preprocess_result.media_type,
        len(preprocess_result.image_paths),
        len(preprocess_result.video_paths),
    )
    if args.dry_run:
        logger.info("Dry run 模式，不会调用模型。")
        logger.info("dry-run complete. prepared media summary:")
        if preprocess_result.media_type == "video_frames":
            total_frames = len(preprocess_result.image_paths)
            examples = preprocess_result.image_paths[: min(5, total_frames)]
            logger.info(" - extracted frames: %d (examples: %s)", total_frames, ", ".join(str(p.name) for p in examples))
        else:
            for path in preprocess_result.image_paths:
                logger.info(" - image: %s", path)
        for path in preprocess_result.video_paths:
            logger.info(" - video: %s", path)
        logger.info("system prompt:")
        logger.info("%s", system_prompt or "<none>")
        logger.info("user prompt:")
        logger.info("%s", user_prompt)
        return

    conversation_records: List[Dict[str, object]] = []

    async with ChatAgent(model_config, runtime_config) as agent:
        if system_prompt:
            agent.set_system_prompt(system_prompt)

        conversation_batches: List[Dict[str, List[str]]] = []
        if preprocess_result.using_video_input and preprocess_result.video_paths:
            conversation_batches.append(
                {
                    "image_paths": [],
                    "video_paths": [str(p) for p in preprocess_result.video_paths],
                }
            )
        else:
            all_images = [str(p) for p in preprocess_result.image_paths]
            chunk_size = args.image_count_per_chat or len(all_images)
            if chunk_size is None or chunk_size <= 0:
                chunk_size = len(all_images)
            if not chunk_size:
                chunk_size = 1

            if all_images:
                for start_index in range(0, len(all_images), chunk_size):
                    chunk = all_images[start_index : start_index + chunk_size]
                    conversation_batches.append(
                        {
                            "image_paths": chunk,
                            "video_paths": [str(p) for p in preprocess_result.video_paths],
                        }
                    )
            elif preprocess_result.video_paths:
                # Fallback: no extracted frames but still retain video reference
                conversation_batches.append(
                    {
                        "image_paths": [],
                        "video_paths": [str(p) for p in preprocess_result.video_paths],
                    }
                )

        if not conversation_batches:
            raise ValueError("没有可用于模型输入的媒体内容")

        overall_start = perf_counter()

        for idx, batch in enumerate(conversation_batches, start=1):
            if idx > 1:
                agent.clear_history(keep_system=True)

            chat_kwargs: Dict[str, object] = {}
            if batch["image_paths"]:
                chat_kwargs["image_paths"] = batch["image_paths"]
            if batch["video_paths"] and preprocess_result.using_video_input:
                chat_kwargs["video_paths"] = batch["video_paths"]

            # Log which images/frames are used in this batch (compact for video frames)
            if preprocess_result.media_type == "video_frames" and batch.get("image_paths"):
                # try to extract frame indices for concise logging
                frame_indices = []
                for p in batch["image_paths"]:
                    m = re.search(r"frame_(\d{6})", str(p))
                    if m:
                        try:
                            frame_indices.append(int(m.group(1)))
                        except Exception:
                            frame_indices.append(str(p))
                    else:
                        frame_indices.append(str(Path(p).name))
                logger.info("开始第 %d 轮对话: 帧数量=%d, 帧索引=%s, 视频数=%d", idx, len(frame_indices), frame_indices, len(batch["video_paths"]))
            else:
                logger.info(
                    "开始第 %d 轮对话: 图像数=%d, 视频数=%d",
                    idx,
                    len(batch["image_paths"]),
                    len(batch["video_paths"]),
                )

            start_time = perf_counter()
            try:
                response_text = await agent.chat(user_prompt, **chat_kwargs)
            except Exception as exc:
                logger.exception("第 %d 轮对话失败: %s", idx, exc)
                raise

            latency_seconds = perf_counter() - start_time
            elapsed_seconds = perf_counter() - overall_start

            logger.info(
                "第 %d 轮对话完成，耗时 %.3f 秒，累计 %.3f 秒",
                idx,
                latency_seconds,
                elapsed_seconds,
            )

            logger.info(
                "[conversation %d] model response:\n%s",
                idx,
                response_text,
            )
            logger.info(
                "[conversation %d] vlm latency: %.3fs (elapsed %.3fs)",
                idx,
                latency_seconds,
                elapsed_seconds,
            )

            # Compact representation for recording inputs when using video frames
            recorded_image_input = batch["image_paths"]
            if preprocess_result.media_type == "video_frames" and batch.get("image_paths"):
                compact_indices = []
                for p in batch["image_paths"]:
                    m = re.search(r"frame_(\d{6})", str(p))
                    if m:
                        try:
                            compact_indices.append(int(m.group(1)))
                        except Exception:
                            compact_indices.append(str(p))
                    else:
                        compact_indices.append(str(Path(p).name))
                recorded_image_input = compact_indices

            conversation_records.append(
                {
                    "conversation_index": idx,
                    "input_image_paths": recorded_image_input,
                    "input_video_paths": batch["video_paths"],
                    "response_text": response_text,
                    "latency_seconds": latency_seconds,
                    "elapsed_since_start_seconds": elapsed_seconds,
                }
            )

            # 插入明显间隔，便于阅读日志与分段记录（仅在还有下一轮时输出）
            if idx < len(conversation_batches):
                logger.info("%s", "=" * 80)
                logger.info("准备进入下一轮对话（%d/%d）", idx + 1, len(conversation_batches))

        if conversation_records:
            total_elapsed = float(conversation_records[-1]["elapsed_since_start_seconds"])
            total_latency = sum(float(record["latency_seconds"]) for record in conversation_records)
            logger.info(
                "所有对话完成，总调用数=%d，总模型耗时=%.3f 秒，总耗时=%.3f 秒",
                len(conversation_records),
                total_latency,
                total_elapsed,
            )

    result_path = save_result(
        output_dir=output_dir,
        model_config=model_config,
        data_path=data_paths[0],
        timestamp=run_timestamp,
        prompt_path=prompt_path,
        prompt_name=args.prompt_name,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        used_default_system_prompt=used_default_system_prompt,
        preprocess_result=preprocess_result,
        conversation_records=conversation_records,
        strategy=args.video_sampling_strategy,
        sample_count=args.sample_count,
        image_count_per_chat=args.image_count_per_chat,
        resize=args.resize,
        crop=args.crop,
        working_dir=working_dir,
    )

    logger.info("result saved to: %s", result_path)
    logger.info("评估流程完成")


def warn_missing_api_key():
    if not os.getenv("OPENAI_API_KEY") and API_KEY is None:
        message = "Warning: API Key not configured. 请在脚本中设置 API_KEY 或配置 OPENAI_API_KEY 环境变量"
        if not logger.handlers:
            logging.basicConfig(level=logging.INFO)
        logger.warning(message)


if __name__ == "__main__":
    parser = build_argument_parser()
    arguments = parser.parse_args()
    try:
        warn_missing_api_key()
        asyncio.run(main(arguments))
    except KeyboardInterrupt:
        logger.info("用户中断执行")
    except Exception as exc:
        logger.exception("脚本执行失败: %s", exc)