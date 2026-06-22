import json
import os
from pathlib import Path
from IPython.display import Image, display

COSMOS3_AUDIOVISUAL_ROOT = COSMOS_ROOT / "cookbooks" / "cosmos3" / "generator" / "audiovisual"
COSMOS3_AUDIOVISUAL_OUTPUT_ROOT = Path(
    os.environ.get("COSMOS3_AUDIOVISUAL_OUTPUT_ROOT", COSMOS3_AUDIOVISUAL_ROOT / "outputs" / "notebooks")
).resolve()

DEFAULT_SGLANG_BASE_URL = os.environ.get("COSMOS3_SGLANG_BASE_URL", "http://localhost:30000")
SGLANG_ENDPOINTS = {
    "Cosmos3-Nano": os.environ.get("COSMOS3_SGLANG_NANO_BASE_URL", DEFAULT_SGLANG_BASE_URL),
    "Cosmos3-Super": os.environ.get("COSMOS3_SGLANG_SUPER_BASE_URL", DEFAULT_SGLANG_BASE_URL),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

FIXED_SAMPLING = {
    "num_steps": 35,
    "guidance": 6.0,
    "shift": 10.0,
    "fps": 24,
    "num_frames": 189,
    "resolution": "720",
    "aspect_ratio": "16,9",
    "seed": 0,
}

# All asset paths are repo-relative under cookbooks/cosmos3/generator/audiovisual.
# Model and sound choices live in this manifest; folders are organized only by modality.
ASSET_SETS = {
    "t2i": {
        "model": "Cosmos3-Nano",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "t2i_super": {
        "model": "Cosmos3-Super",
        "mode": "text2image",
        "prompt": "assets/prompts/text2image/robot_draping.json",
        "enable_sound": False,
    },
    "t2v_nano_noaudio": {
        "model": "Cosmos3-Nano",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "enable_sound": False,
    },
    "t2vs": {
        "model": "Cosmos3-Nano",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_pouring_water_audio.json",
        "enable_sound": True,
    },
    "i2v_nano_noaudio": {
        "model": "Cosmos3-Nano",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
    "i2vs": {
        "model": "Cosmos3-Nano",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/coastal_road_audio.json",
        "image": "assets/images/image2video/coastal_road_audio.jpg",
        "enable_sound": True,
    },
    "t2v_super_noaudio": {
        "model": "Cosmos3-Super",
        "mode": "text2video",
        "prompt": "assets/prompts/text2video/robot_kitchen.json",
        "enable_sound": False,
    },
    "i2v_super_noaudio": {
        "model": "Cosmos3-Super",
        "mode": "image2video",
        "prompt": "assets/prompts/image2video/car_driving.json",
        "image": "assets/images/image2video/car_driving.jpg",
        "enable_sound": False,
    },
}


def asset_path(relative_path: str) -> Path:
    path = COSMOS3_AUDIOVISUAL_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(path)
    return path.resolve()


def compact_json_file(path: Path) -> str:
    return json.dumps(json.loads(path.read_text()), ensure_ascii=True, separators=(",", ":"))


def normalize_negative_prompt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        parts = []
        for v in value.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                parts.extend(str(x) for x in v)
        return "\n".join(parts)
    return str(value)


def payload_dimensions(payload: dict) -> tuple[int, int]:
    if payload.get("resolution") == "720" and payload.get("aspect_ratio") == "16,9":
        return 720, 1280
    if payload.get("resolution") == "256" and payload.get("aspect_ratio") == "16,9":
        return 192, 320
    raise ValueError(f"Unsupported payload resolution/aspect ratio: {payload.get('resolution')} {payload.get('aspect_ratio')}")


def resolve_payload_path(payload_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (payload_path.parent / path).resolve()


def create_payload(use_case: str, *, backend: str) -> tuple[Path, Path, str]:
    spec = ASSET_SETS[use_case]
    payload_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / "payloads" / use_case
    output_dir = Path(os.environ["COSMOS3_AUDIOVISUAL_OUTPUT_ROOT"]) / backend / use_case
    payload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = asset_path(spec["prompt"])
    negative_prompt = ""
    if spec["mode"] != "text2image":
        negative_prompt_path = asset_path(f"assets/negative_prompts/{spec['mode']}/neg_prompt.json")
        negative_prompt = normalize_negative_prompt(json.loads(negative_prompt_path.read_text()))
    payload_path = payload_dir / f"{use_case}.json"
    payload = {
        "model_mode": spec["mode"],
        "name": use_case,
        "prompt": compact_json_file(prompt_path),
        "negative_prompt": negative_prompt,
        "enable_sound": spec["enable_sound"],
        **FIXED_SAMPLING,
    }
    if spec["mode"] == "image2video":
        image_path = asset_path(spec["image"])
        payload["vision_path"] = os.path.relpath(image_path, payload_path.parent)

    payload_path.write_text(json.dumps(payload, indent=2) + "\n")

    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_INPUT"] = str(payload_path)
    os.environ[f"COSMOS3_{backend.upper()}_{use_case.upper()}_OUTPUT"] = str(output_dir)

    print(f"model:   {spec['model']}")
    print(f"payload: {payload_path}")
    print(f"output:  {output_dir}")
    print(f"prompt:  {prompt_path.relative_to(COSMOS_ROOT)}")
    if "vision_path" in payload:
        image_display_path = resolve_payload_path(payload_path, payload["vision_path"])
        print(f"image:   {image_display_path.relative_to(COSMOS_ROOT)}")
        display(Image(filename=str(image_display_path), width=420))
    print(json.dumps({k: payload[k] for k in ["model_mode", "name", "enable_sound", "num_steps", "guidance", "shift", "fps", "num_frames", "resolution", "aspect_ratio", "seed"]}, indent=2))
    return payload_path, output_dir, spec["model"]


import base64
import html
import json
import os
import subprocess
import time
from pathlib import Path
from IPython.display import HTML, display


def api_root_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"
    return normalized


def video_api_url(base_url: str) -> str:
    return f"{api_root_url(base_url)}/videos"


def image_api_url(base_url: str) -> str:
    return f"{api_root_url(base_url)}/images/generations"


def build_sglang_video_form(payload: dict) -> dict[str, str]:
    height, width = payload_dimensions(payload)
    extra_params = {
        "use_resolution_template": False,
        "use_duration_template": False,
        "guardrails": True,
    }
    form = {
        "prompt": payload["prompt"],
        "negative_prompt": payload["negative_prompt"],
        "size": f"{width}x{height}",
        "num_frames": str(payload["num_frames"]),
        "fps": str(payload["fps"]),
        "num_inference_steps": str(payload["num_steps"]),
        "guidance_scale": str(payload["guidance"]),
        "flow_shift": str(payload["shift"]),
        "seed": str(payload["seed"]),
        "extra_params": json.dumps(extra_params, separators=(",", ":")),
    }
    if payload["enable_sound"]:
        form["generate_sound"] = "true"
        form["sound_duration"] = f"{payload['num_frames'] / payload['fps']:.3f}"
    return form


def build_sglang_image_body(payload: dict) -> dict:
    height, width = payload_dimensions(payload)
    return {
        "prompt": payload["prompt"],
        "negative_prompt": payload.get("negative_prompt", ""),
        "size": f"{width}x{height}",
        "n": 1,
        "num_inference_steps": payload["num_steps"],
        "guidance_scale": payload["guidance"],
        "flow_shift": payload["shift"],
        "seed": payload["seed"],
        "response_format": "b64_json",
        "extra_params": {
            "use_resolution_template": False,
            "guardrails": True,
        },
    }


def post_video(*, payload_path: Path, payload: dict, output_path: Path, model: str) -> None:
    url = video_api_url(SGLANG_ENDPOINTS[model])
    api_key = None
    tmp_path = Path(f"{output_path}.tmp")
    error_path = Path(f"{output_path}.error.txt")
    if tmp_path.exists():
        tmp_path.unlink()
    if error_path.exists():
        error_path.unlink()

    cmd = [
        "curl",
        "-sS",
        "--fail-with-body",
        "-X",
        "POST",
        url,
        "-H",
        "Accept: video/mp4",
    ]
    if api_key is not None:
        cmd += ["-H", f"Authorization: Bearer {api_key}"]

    for key, value in build_sglang_video_form(payload).items():
        cmd += ["--form-string", f"{key}={value}"]

    if payload["model_mode"] == "image2video":
        image_path = resolve_payload_path(payload_path, payload["vision_path"])
        cmd += ["-F", f"input_reference=@{image_path}"]

    if payload["model_mode"] == "video2video":
        video_path = resolve_payload_path(payload_path, payload["vision_path"])
        cmd += ["-F", f"video_reference=@{video_path};type=video/mp4"]

    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_path.write_text((result.stdout or "") + (result.stderr or ""))
        raise RuntimeError(f"sglang request failed with exit code {result.returncode}; see {error_path}")

    initial = json.loads(result.stdout)
    video_id = initial["id"]

    while True:
        poll = subprocess.run(
            ["curl", "-sS", "--fail-with-body", f"{url}/{video_id}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if poll.returncode != 0:
            error_path.write_text((poll.stdout or "") + (poll.stderr or ""))
            raise RuntimeError(f"SGLang video poll failed; see {error_path}")

        final = json.loads(poll.stdout)
        if final.get("status") == "completed":
            break
        if final.get("status") in {"failed", "cancelled"}:
            error_path.write_text(json.dumps(final, indent=2))
            raise RuntimeError(f"SGLang video failed; see {error_path}")
        time.sleep(5)

    download = subprocess.run(
        ["curl", "-sS", "--fail-with-body", f"{url}/{video_id}/content", "-o", str(tmp_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if download.returncode != 0:
        error_path.write_text(json.dumps(final, indent=2) + "\n" + (download.stderr or ""))
        raise RuntimeError(f"SGLang video download failed; see {error_path}")

    tmp_path.replace(output_path)


def post_image(*, payload: dict, output_path: Path, model: str) -> None:
    url = image_api_url(SGLANG_ENDPOINTS[model])
    api_key = None
    tmp_path = Path(f"{output_path}.tmp")
    error_path = Path(f"{output_path}.error.txt")
    if tmp_path.exists():
        tmp_path.unlink()
    if error_path.exists():
        error_path.unlink()

    cmd = [
        "curl",
        "-sS",
        "--fail-with-body",
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
    ]
    if api_key is not None:
        cmd += ["-H", f"Authorization: Bearer {api_key}"]
    cmd += ["-d", json.dumps(build_sglang_image_body(payload), separators=(",", ":"))]

    result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        error_path.write_text((result.stdout or "") + (result.stderr or ""))
        raise RuntimeError(f"sglang image request failed with exit code {result.returncode}; see {error_path}")
    try:
        response = json.loads(result.stdout)
        b64_json = response["data"][0]["b64_json"]
        tmp_path.write_bytes(base64.b64decode(b64_json))
    except Exception as exc:
        error_path.write_text((result.stdout or "") + (result.stderr or ""))
        raise RuntimeError(f"Could not decode sglang image response; see {error_path}") from exc
    tmp_path.replace(output_path)


def run_sglang_payload(payload_path: Path, output_dir: str | Path, *, model: str) -> Path:
    payload_path = Path(payload_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(payload_path.read_text())
    output_ext = ".png" if payload["model_mode"] == "text2image" else ".mp4"
    output_path = output_dir / f"{payload['name']}{output_ext}"
    endpoint = image_api_url(SGLANG_ENDPOINTS[model]) if payload["model_mode"] == "text2image" else video_api_url(SGLANG_ENDPOINTS[model])
    print("endpoint:", endpoint)
    print("payload:", payload_path)
    print("output:", output_path)
    if payload["model_mode"] == "image2video":
        print("input image:", resolve_payload_path(payload_path, payload["vision_path"]))
    t0 = time.time()
    if payload["model_mode"] == "text2image":
        post_image(payload=payload, output_path=output_path, model=model)
    else:
        post_video(payload_path=payload_path, payload=payload, output_path=output_path, model=model)
    print(f"wrote {output_path} in {time.time() - t0:.1f}s")
    return output_path


def display_video(path: Path, *, width: int = 720) -> None:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    label = html.escape(str(path))
    markup = f"""
<video controls playsinline preload="metadata" width="{width}" style="max-width: 100%; background: #000;">
  <source src="data:video/mp4;base64,{data}" type="video/mp4">
</video>
<div style="font-family: monospace; font-size: 12px; margin-top: 4px;">{label}</div>
"""
    display(HTML(markup))


def view_run(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    videos = [
        path
        for path in sorted(output_dir.rglob("*.mp4"))
        if not path.name.endswith(("_preview.mp4", "_browser.mp4"))
    ]
    images = sorted(output_dir.rglob("*.png"))
    if not videos and not images:
        print(f"No generated media found under {output_dir}")
        return
    for src in videos:
        print(f"source: {src} ({src.stat().st_size // 1024} KB)")
        display_video(src)
    for src in images:
        print(f"source: {src} ({src.stat().st_size // 1024} KB)")
        display(Image(filename=str(src), width=720))


if __name__ == "__main__":
    # nano t2i
    t2i_payload, t2i_output, t2i_model = create_payload("t2i", backend="sglang")
    run_sglang_payload(t2i_payload, t2i_output, model="Cosmos3-Nano")
    view_run(t2i_output)

    # nano t2v_noaudio
    t2v_nano_noaudio_payload, t2v_nano_noaudio_output, t2v_nano_noaudio_model = create_payload("t2v_nano_noaudio", backend="sglang")
    run_sglang_payload(t2v_nano_noaudio_payload, t2v_nano_noaudio_output, model="Cosmos3-Nano")
    view_run(t2v_nano_noaudio_output)

    # nano t2vs
    t2vs_payload, t2vs_output, t2vs_model = create_payload("t2vs", backend="sglang")
    run_sglang_payload(t2vs_payload, t2vs_output, model="Cosmos3-Nano")
    view_run(t2vs_output)

    # nano i2v_noaudio
    i2v_nano_noaudio_payload, i2v_nano_noaudio_output, i2v_nano_noaudio_model = create_payload("i2v_nano_noaudio", backend="sglang")
    run_sglang_payload(i2v_nano_noaudio_payload, i2v_nano_noaudio_output, model="Cosmos3-Nano")
    view_run(i2v_nano_noaudio_output)

    # nano i2vs
    i2vs_payload, i2vs_output, i2vs_model = create_payload("i2vs", backend="sglang")
    run_sglang_payload(i2vs_payload, i2vs_output, model="Cosmos3-Nano")
    view_run(i2vs_output)
