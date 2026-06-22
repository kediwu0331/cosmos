from pathlib import Path
import os
import json
import struct
import requests
import time

def video_wh(path):
    with open(path, 'rb') as f:
        data = f.read(512 * 1024)
    idx = data.find(b'tkhd')
    if idx < 0:
        raise ValueError(f"no tkhd box in {path}")
    w = struct.unpack('>I', data[idx+80:idx+84])[0] >> 16
    h = struct.unpack('>I', data[idx+84:idx+88])[0] >> 16
    return w, h

# Local inputs, relative to the cosmos repo root.
input_videos = {
    "av_inverse_0": "cookbooks/cosmos3/generator/action/assets/videos/av_0.mp4",
    "av_inverse_1": "cookbooks/cosmos3/generator/action/assets/videos/av_1.mp4",
}

def resolve_input(rel_path: str) -> str:
    path = (COSMOS_ROOT / rel_path).resolve()
    assert path.exists(), f"missing input: {path}"
    return str(path)

records = []

for name, video_rel in input_videos.items():
    path = resolve_input(video_rel)
    w, h = video_wh(path)
    records.append({
        "action_chunk_size": 60,
        "domain_name": "av",
        "fps": 10,
        "width": w,
        "height": h,
        "view_point": "ego_view",
        "model_mode": "inverse_dynamics",
        "name": name,
        "prompt": "You are an autonomous vehicle planning system.",
        "seed": 0,
        "vision_path": path,
    })

COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)
id_input_path = COSMOS3_INPUT_DIR / "action_inverse_dynamics_av_custom.jsonl"
id_input_path.write_text("".join(json.dumps(r) + "\n" for r in records))
id_output_dir = COSMOS3_OUTPUT_ROOT / "action_inverse_dynamics_av_custom"

# The bash inference cell can only see the environment, so export the paths it needs.
os.environ["COSMOS3_ID_INPUT"] = str(id_input_path)
os.environ["COSMOS3_ID_OUTPUT"] = str(id_output_dir)

print("wrote spec:", id_input_path)
print("runs:", list(input_videos))
print(id_input_path.read_text())



# send inference request
BASE_URL = "http://127.0.0.1:30000"
def check_server() -> None:
    response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
    response.raise_for_status()
    print(response.json())


def submit_inverse_dynamics(record: dict) -> dict:
    run_dir = id_output_dir / record["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    video_path = Path(record["vision_path"])
    extra_params = {
        "action_mode": "inverse_dynamics",
        "domain_name": record["domain_name"],
        "action_view_point": record["view_point"],
        "raw_action_dim": 9,
        "guardrails": False,
    }
    form = {
        "prompt": record["prompt"],
        "num_frames": record["action_chunk_size"] + 1,
        "fps": record["fps"],
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": record["seed"],
        "extra_params": json.dumps(extra_params),
    }

    with video_path.open("rb") as video_file:
        response = requests.post(
            f"{BASE_URL}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"video_reference": (video_path.name, video_file, "video/mp4")},
            timeout=120,
        )
    response.raise_for_status()
    initial = response.json()
    (run_dir / "response.json").write_text(json.dumps(initial, indent=2))

    while True:
        response = requests.get(f"{BASE_URL}/v1/videos/{initial['id']}", timeout=30)
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
        raise RuntimeError(f"response did not include action data: {json.dumps(final, indent=2)}")
    (run_dir / "action.json").write_text(json.dumps(action, indent=2))

    sample_outputs = {"outputs": [{"content": {"action": action["data"]}}]}
    (run_dir / "sample_outputs.json").write_text(json.dumps(sample_outputs, indent=2))

    print("saved", run_dir / "sample_outputs.json")
    print("action shape:", action.get("shape"), "dtype:", action.get("dtype"))
    return {"record": record, "initial": initial, "final": final, "run_dir": run_dir, "action": action}


check_server()
results = []
for record in records:
    print(f"\nSubmitting {record['name']}")
    results.append(submit_inverse_dynamics(record))
