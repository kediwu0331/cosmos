from pathlib import Path
import os
import json

def resolve_input(rel_path: str) -> str:
    path = (COSMOS_ROOT / rel_path).resolve()
    assert path.exists(), f"missing input: {path}"
    return str(path)

# Local AV inputs, relative to the cosmos repo root.
av_input_image = "cookbooks/cosmos3/generator/action/assets/images/av_0.jpg"
av_input_actions = {
    "av_forward": "cookbooks/cosmos3/generator/action/assets/actions/av_traj_forward.json",
    "av_left": "cookbooks/cosmos3/generator/action/assets/actions/av_traj_left.json",
    "av_right": "cookbooks/cosmos3/generator/action/assets/actions/av_traj_right.json",
}

av_vision_path = resolve_input(av_input_image)
av_records = [
    {
        "action_chunk_size": 60,
        "action_path": resolve_input(action_rel),
        "domain_name": "av",
        "fps": 10,
        "image_size": 480,
        "view_point": "ego_view",
        "model_mode": "forward_dynamics",
        "name": name,
        "prompt": "You are an autonomous vehicle planning system.",
        "seed": 0,
        "vision_path": av_vision_path,
    }
    for name, action_rel in av_input_actions.items()
]

COSMOS3_INPUT_DIR.mkdir(parents=True, exist_ok=True)
av_fd_input_path = COSMOS3_INPUT_DIR / "action_forward_dynamics_av_custom.jsonl"
av_fd_input_path.write_text("".join(json.dumps(r) + "\n" for r in av_records))
av_fd_output_dir = COSMOS3_OUTPUT_ROOT / "action_forward_dynamics_av_custom"

os.environ["COSMOS3_AV_FD_INPUT"] = str(av_fd_input_path)
os.environ["COSMOS3_AV_FD_OUTPUT"] = str(av_fd_output_dir)

print("wrote AV spec:", av_fd_input_path)
print("AV runs:", list(av_input_actions))
print(av_fd_input_path.read_text())




####### plot #######
#import sys
#import json
#import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.collections import LineCollection
#from mpl_toolkits.mplot3d.art3d import Line3DCollection
#
## The notebook kernel may differ from the framework venv, so put the repo on the
## path before importing `cosmos_framework`.
#COSMOS3_REPO = "/sgl-workspace/cosmos-framework"
#if str(COSMOS3_REPO) not in sys.path:
#    sys.path.insert(0, str(COSMOS3_REPO))
#from cosmos_framework.data.vfm.action.pose_utils import pose_rel_to_abs
#
## frustum: apex + image-rectangle corners (camera +Z forward), and their edges
#_FRUSTUM = np.array([[0, 0, 0], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
#_EDGES = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
#
#
#def visualize_pose(poses_abs, *, n_frustums=20, scale_frac=0.03, aspect=16 / 9,
#                   fov_deg=60.0, vertical_exaggeration=1.0, cmap="turbo",
#                   title=None, save_path=None, show=True):
#    """3D camera trajectory (with frustums) + a top-down bird's-eye view."""
#    poses_abs = np.asarray(poses_abs)
#    pos = poses_abs[:, :3, 3]
#    fwd = poses_abs[:, :3, 2]
#    T = len(pos)
#    colors = plt.get_cmap(cmap)(np.arange(T) / max(T - 1, 1))
#    scale = max(np.ptp(pos, axis=0).max() * scale_frac, 1e-3)
#    step = max(1, T // max(n_frustums, 1))
#    xzy = [0, 2, 1]
#
#    fig = plt.figure(figsize=(14, 6))
#
#    ax = fig.add_subplot(1, 2, 1, projection="3d")
#    path = pos[:, xzy]
#    ax.plot(*path.T, color="0.6", lw=1.0, alpha=0.7)
#    lines, lcolors, allpts = [], [], [path]
#    for i in range(0, T, step):
#        cw = ((_FRUSTUM * [aspect, 1, 1] * scale * np.tan(np.radians(fov_deg) / 2))
#              @ poses_abs[i, :3, :3].T + poses_abs[i, :3, 3])[:, xzy]
#        allpts.append(cw)
#        lines += [[cw[a], cw[b]] for a, b in _EDGES]
#        lcolors += [colors[i]] * len(_EDGES)
#    ax.add_collection3d(Line3DCollection(lines, colors=lcolors, linewidths=1.2))
#    ax.scatter(*path[0], color="lime", s=80, edgecolor="k", label="first frame", zorder=5)
#    ax.scatter(*path[-1], color="red", s=80, edgecolor="k", label="last frame", zorder=5)
#    rng = np.clip(np.ptp(np.concatenate(allpts), axis=0), 1e-9, None)
#    ax.set_box_aspect((rng[0], rng[1], rng[2] * vertical_exaggeration))
#    ax.set_xlabel("X (m)", labelpad=12)
#    ax.set_ylabel("Z forward (m)", labelpad=12)
#    ax.set_zlabel("Y up (m)", labelpad=10)
#    ax.set_zticks([])
#    ax.set_title(title or f"Camera trajectory + frustums ({T} frames)")
#    ax.legend(loc="upper left")
#    ax.view_init(elev=22, azim=-70)
#
#    ax2 = fig.add_subplot(1, 2, 2)
#    seg = np.stack([pos[:-1, [0, 2]], pos[1:, [0, 2]]], axis=1)
#    lc = LineCollection(seg, cmap=cmap, norm=plt.Normalize(0, T - 1), linewidth=2.5)
#    lc.set_array(np.arange(T - 1))
#    ax2.add_collection(lc)
#    ax2.quiver(pos[::step, 0], pos[::step, 2], fwd[::step, 0], fwd[::step, 2],
#               color=colors[::step], angles="xy", width=0.005, scale=22, zorder=3)
#    ax2.scatter(*pos[0, [0, 2]], color="lime", s=80, edgecolor="k", label="first frame", zorder=5)
#    ax2.scatter(*pos[-1, [0, 2]], color="red", s=80, edgecolor="k", label="last frame", zorder=5)
#    ax2.set_xlabel("X (m)")
#    ax2.set_ylabel("Z forward (m)")
#    ax2.set_title("Top-down (bird's-eye view)")
#    ax2.set_aspect("equal", adjustable="datalim")
#    ax2.autoscale_view()
#    ax2.legend()
#    fig.colorbar(lc, ax=ax2, label="frame index")
#
#    plt.tight_layout(w_pad=6)
#    if save_path:
#        fig.savefig(save_path, dpi=120, bbox_inches="tight")
#        print("saved", save_path)
#    if show:
#        plt.show()
#
#
#for record in av_records:
#    name = record["name"]
#    with open(record["action_path"]) as f:
#        poses_rel = np.array(json.load(f))
#
#    # AV action convention: rot6d rotation, backward_framewise, translation_scale = 1.35.
#    poses_abs = pose_rel_to_abs(
#        poses_rel,
#        rotation_format="rot6d",
#        pose_convention="backward_framewise",
#        translation_scale=1.35,
#    )
#    print(name, poses_rel.shape, poses_abs.shape)
#    visualize_pose(poses_abs, title=f"{name}: camera trajectory + frustums ({len(poses_abs)} frames)", save_path=f"/sgl-workspace/data/cosmos3/trajectory_{name}.jpg", show=False)





# send requests to inference server
import json
import mimetypes
import time
from pathlib import Path

from PIL import Image

try:
    import requests
except ImportError as exc:
    raise RuntimeError("Install requests in this notebook kernel: pip install requests") from exc

BASE_URL = "http://127.0.0.1:30000"
def check_server(timeout_s: int = 600, interval_s: int = 10) -> None:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
            response.raise_for_status()
            print(response.json())
            return
        except requests.RequestException as exc:
            last_error = exc
            print(f"Waiting for vLLM server at {BASE_URL}: {exc}")
            time.sleep(interval_s)
    raise RuntimeError(
        f"server did not become ready at {BASE_URL} within {timeout_s}s. "
    ) from last_error


def submit_forward_dynamics(record: dict, fd_output_dir: Path) -> dict:
    run_dir = fd_output_dir / record["name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    vision_path = Path(record["vision_path"])
    input_width, input_height = Image.open(vision_path).size
    mime_type = mimetypes.guess_type(vision_path.name)[0] or "application/octet-stream"
    extra_params = {
        "action_mode": "forward_dynamics",
        "domain_name": record["domain_name"],
        "action_view_point": record["view_point"],
        "action": json.loads(Path(record["action_path"]).read_text()),
        "guardrails": False,
    }
    prompt = str(record.get("prompt") or "").strip() or "A robot manipulates an object."
    form = {
        "prompt": prompt,
        "num_frames": record["action_chunk_size"] + 1,
        "fps": record["fps"],
        "size": f"{input_width}x{input_height}",
        "num_inference_steps": 30,
        "guidance_scale": 1.0,
        "flow_shift": 10.0,
        "seed": record["seed"],
        "extra_params": json.dumps(extra_params),
    }

    with vision_path.open("rb") as image_file:
        response = requests.post(
            f"{BASE_URL}/v1/videos",
            data={key: str(value) for key, value in form.items()},
            files={"input_reference": (vision_path.name, image_file, mime_type)},
            timeout=120,
        )
    if not response.ok:
        (run_dir / "error_response.txt").write_text(response.text)
        print("vLLM request failed:", response.status_code)
        print(response.text)
        print("form:", json.dumps(form, indent=2))
        print("extra_params keys:", sorted(extra_params))
        print("action shape:", [len(extra_params["action"]), len(extra_params["action"][0]) if extra_params["action"] else 0])
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

    response = requests.get(f"{BASE_URL}/v1/videos/{initial['id']}/content", timeout=300)
    response.raise_for_status()
    video_path = run_dir / "vision.mp4"
    video_path.write_bytes(response.content)

    action = final.get("action")
    if action is not None:
        (run_dir / "action.json").write_text(json.dumps(action, indent=2))

    print("saved", video_path)
    if action is not None:
        print("action shape:", action.get("shape"), "dtype:", action.get("dtype"))
    return {"record": record, "initial": initial, "final": final, "run_dir": run_dir, "video_path": video_path, "action": action}


check_server()
av_results = []
for record in av_records:
    print(f"\nSubmitting {record['name']}")
    av_results.append(submit_forward_dynamics(record, av_fd_output_dir))
