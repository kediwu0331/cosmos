import subprocess

import numpy as np
from PIL import Image, ImageOps
from IPython.display import display
from pathlib import Path
import os

try:
    import imageio_ffmpeg
except ImportError as exc:
    raise RuntimeError("Install imageio-ffmpeg in this notebook kernel: pip install imageio-ffmpeg") from exc

COSMOS3_ACTION_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "action"
DROID_ASSET_ROOT = COSMOS3_ACTION_ROOT / "assets" / "droid_lerobot_example"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
CAMERA_VIDEO_PATHS = {
    "observation/wrist_image_left": DROID_ASSET_ROOT / "videos" / "observation.image.wrist_image_left" / "chunk-000" / "file-000.mp4",
    "observation/exterior_image_1_left": DROID_ASSET_ROOT / "videos" / "observation.image.exterior_image_1_left" / "chunk-000" / "file-000.mp4",
    "observation/exterior_image_2_left": DROID_ASSET_ROOT / "videos" / "observation.image.exterior_image_2_left" / "chunk-000" / "file-000.mp4",
}
for key, video_path in CAMERA_VIDEO_PATHS.items():
    assert video_path.exists(), f"missing {key}: {video_path}"


def extract_first_frame(video_path: Path, out_path: Path) -> Path:
    if not out_path.exists() or out_path.stat().st_mtime < video_path.stat().st_mtime:
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-i", str(video_path), "-frames:v", "1", str(out_path)],
            check=True,
        )
    return out_path


frame_paths = {
    key: extract_first_frame(video_path, COSMOS3_INPUT_DIR / f"policy_{key.split('/')[-1]}.png")
    for key, video_path in CAMERA_VIDEO_PATHS.items()
}
frames = {key: Image.open(path).convert("RGB") for key, path in frame_paths.items()}

# DROID policy uses a concatenated multi-view frame for the /v1/videos path.
target_w, target_h = 640, 540
top_h = target_h // 2
bottom_h = target_h - top_h
half_w = target_w // 2
wrist = ImageOps.fit(frames["observation/wrist_image_left"], (target_w, top_h), method=Image.Resampling.BICUBIC)
left = ImageOps.fit(frames["observation/exterior_image_1_left"], (half_w, bottom_h), method=Image.Resampling.BICUBIC)
right = ImageOps.fit(frames["observation/exterior_image_2_left"], (half_w, bottom_h), method=Image.Resampling.BICUBIC)
policy_image = Image.new("RGB", (target_w, target_h))
policy_image.paste(wrist, (0, 0))
policy_image.paste(left, (0, top_h))
policy_image.paste(right, (half_w, top_h))
policy_image_path = COSMOS3_INPUT_DIR / "droid_policy_first_frame.png"
policy_image.save(policy_image_path)

policy_prompt = os.environ.get(
    "COSMOS3_POLICY_PROMPT",
    "Pick up the object and place it in the target container.",
)
print("policy prompt:", policy_prompt)
print("video API conditioning image:", policy_image_path, policy_image.size)



# send requests
import requests
import json
import time

ACTION_VIDEO_RES_SIZE_INFO = {
    "480": {
        "1,1": (640, 640),
        "4,3": (736, 544),
        "3,4": (544, 736),
        "16,9": (832, 480),
        "9,16": (480, 832),
    }
}
SGLANG_BASE_URL = "http://127.0.0.1:30000"


def closest_action_size(height: int, width: int, resolution: str = "480") -> tuple[int, int]:
    input_ratio = height / width
    candidates = ACTION_VIDEO_RES_SIZE_INFO[resolution].values()
    return min(candidates, key=lambda size: abs(input_ratio - size[1] / size[0]))


def submit_policy_video() -> dict:
    run_dir = COSMOS3_POLICY_OUTPUT_DIR / "video_api"
    run_dir.mkdir(parents=True, exist_ok=True)

    input_width, input_height = Image.open(policy_image_path).size
    target_width, target_height = closest_action_size(input_height, input_width)
    extra_params = {
        "action_mode": "policy",
        "domain_name": "droid_lerobot",
        "raw_action_dim": 10,
        "action_view_point": "concat_view",
        "guardrails": False,
    }
    form = {
        "model": "Cosmos3-Nano",
        "prompt": policy_prompt,
        "num_frames": 61,
        "fps": 15,
        "size": f"{target_width}x{target_height}",
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 5.0,
        "seed": 0,
        "extra_params": json.dumps(extra_params),
    }

    with policy_image_path.open("rb") as image_file:
        response = requests.post(
            f"{SGLANG_BASE_URL}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (policy_image_path.name, image_file, "image/png")},
            timeout=120,
        )
    if not response.ok:
        (run_dir / "error_response.txt").write_text(response.text)
        print("SGLang request failed:", response.status_code)
        print(response.text)
        print("form:", json.dumps(form, indent=2))
        response.raise_for_status()

    initial = response.json()
    (run_dir / "response.json").write_text(json.dumps(initial, indent=2))

    while True:
        response = requests.get(f"{SGLANG_BASE_URL}/v1/videos/{initial['id']}", timeout=30)
        response.raise_for_status()
        final = response.json()
        (run_dir / "final.json").write_text(json.dumps(final, indent=2))
        print(initial["id"], final.get("status"), f"{final.get('progress', 0)}%")
        if final.get("status") == "completed":
            break
        if final.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(json.dumps(final, indent=2))
        time.sleep(2)

    action = final.get("action")
    if not action or "data" not in action:
        raise RuntimeError(f"SGLang response did not include action data: {json.dumps(final, indent=2)}")
    (run_dir / "action.json").write_text(json.dumps(action, indent=2))
    sample_outputs = {"outputs": [{"content": {"action": action["data"]}}]}
    (run_dir / "sample_outputs.json").write_text(json.dumps(sample_outputs, indent=2))

    content_response = requests.get(f"{SGLANG_BASE_URL}/v1/videos/{initial['id']}/content", timeout=300)
    content_response.raise_for_status()
    video_path = run_dir / "policy_rollout.mp4"
    if content_response.content:
        video_path.write_bytes(content_response.content)
        print("saved", video_path)
    else:
        video_path = None
        print("video content endpoint returned an empty body")

    print("saved", run_dir / "action.json")
    print("action shape:", action.get("shape"), "dtype:", action.get("dtype"), "domain_id:", action.get("domain_id"))
    return {"initial": initial, "final": final, "run_dir": run_dir, "video_path": video_path, "action": action}


policy_video_result = submit_policy_video()
