<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: OpenMDW-1.1 -->

# Cosmos3 SGLang curl Request Samples

Copy-pasteable `curl` requests for every Cosmos3 modality against a running SGLang server, mirroring the request payloads used by the notebooks in this cookbook:

- `audiovisual/run_with_sglang.ipynb` — T2I, T2V, T2VS, I2V, I2VS, V2V
- `action/run_fd_with_sglang.ipynb`, `run_id_with_sglang.ipynb`, `run_policy_with_sglang.ipynb` — action forward dynamics / inverse dynamics / policy
- `transfer/run_video_transfer_with_sglang.ipynb` — transfer edge / segmentation

Requirements on the client machine: `curl`, `jq` (used to embed JSON prompt files and action arrays), and `ffmpeg` (only for building the policy conditioning image).

## 0. Common Setup

Run everything from a `cosmos` repo checkout. Start the server first (see the notebooks for full Docker commands), e.g.:

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v "$PWD:$PWD" \
  -p 30000:30000 \
  --ipc=host \
  lmsysorg/sglang:dev \
  sglang serve --model-path nvidia/Cosmos3-Nano --host 0.0.0.0
```

Then set the shell variables used by all samples below:

```bash
export BASE_URL=http://localhost:30000
export COSMOS_ROOT=$PWD                 # cosmos repo root
AV_ROOT=$COSMOS_ROOT/cookbooks/cosmos3/generator/audiovisual
ACTION_ROOT=$COSMOS_ROOT/cookbooks/cosmos3/generator/action
TRANSFER_ROOT=$COSMOS_ROOT/cookbooks/cosmos3/generator/transfer

# sanity check — should return model metadata
curl -sS "$BASE_URL/v1/models"
```

Notes:

- T2I uses `POST /v1/images/generations` (synchronous JSON, base64 image in the response). Action inverse dynamics and policy use the synchronous `POST /v1/actions/generations` API (the predicted action array comes back directly in the response). All other modalities use the async `POST /v1/videos` API — the POST returns a job `id`; poll it and download the content (see section on polling at the bottom).
- Guardrails: the audiovisual samples send `"guardrails": true` (requires the gated `nvidia/Cosmos-1.0-Guardrail` HF repo and `HF_TOKEN` in the container); the action and transfer samples send `"guardrails": false`, matching the notebooks. Flip the flag inside `extra_params` as needed.
- For **transfer**, `control_path` is a filesystem path read *by the server*, so the repo must be mounted into the container at the same path (the `-v "$PWD:$PWD"` mount above).
- If the server needs auth, add `-H "Authorization: Bearer $COSMOS3_SGLANG_API_KEY"` to each request.

## 1. T2I — Text to Image (`/v1/images/generations`)

```bash
jq -n \
  --arg prompt "$(jq -c . "$AV_ROOT/assets/prompts/text2image/robot_draping.json")" \
  '{
    prompt: $prompt,
    negative_prompt: "",
    size: "1280x720",
    n: 1,
    num_inference_steps: 35,
    guidance_scale: 6.0,
    flow_shift: 10.0,
    seed: 0,
    response_format: "b64_json",
    extra_params: {use_resolution_template: false, guardrails: true}
  }' \
| curl -sS --fail-with-body -X POST "$BASE_URL/v1/images/generations" \
    -H 'Content-Type: application/json' -d @- \
| jq -r '.data[0].b64_json' | base64 -d > t2i.png
```

## 2. T2V — Text to Video, no audio (`/v1/videos`)

```bash
curl -sS --fail-with-body -X POST "$BASE_URL/v1/videos" \
  -H 'Accept: video/mp4' \
  --form-string "prompt=$(jq -c . "$AV_ROOT/assets/prompts/text2video/robot_kitchen.json")" \
  --form-string "negative_prompt=$(jq -c . "$AV_ROOT/assets/negative_prompts/text2video/neg_prompt.json")" \
  --form-string "size=1280x720" \
  --form-string "num_frames=189" \
  --form-string "fps=24" \
  --form-string "num_inference_steps=35" \
  --form-string "guidance_scale=6.0" \
  --form-string "flow_shift=10.0" \
  --form-string "seed=0" \
  --form-string 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}'
```

## 3. T2VS — Text to Video with Sound

Same as T2V but with the audio prompt and the two sound fields (`sound_duration` = `num_frames / fps` = 189/24 = 7.875):

```bash
curl -sS --fail-with-body -X POST "$BASE_URL/v1/videos" \
  -H 'Accept: video/mp4' \
  --form-string "prompt=$(jq -c . "$AV_ROOT/assets/prompts/text2video/robot_pouring_water_audio.json")" \
  --form-string "negative_prompt=$(jq -c . "$AV_ROOT/assets/negative_prompts/text2video/neg_prompt.json")" \
  --form-string "size=1280x720" \
  --form-string "num_frames=189" \
  --form-string "fps=24" \
  --form-string "num_inference_steps=35" \
  --form-string "guidance_scale=6.0" \
  --form-string "flow_shift=10.0" \
  --form-string "seed=0" \
  --form-string "generate_sound=true" \
  --form-string "sound_duration=7.875" \
  --form-string 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}'
```

## 4. I2V — Image to Video, no audio

The conditioning image is uploaded as the `input_reference` file part:

```bash
curl -sS --fail-with-body -X POST "$BASE_URL/v1/videos" \
  -H 'Accept: video/mp4' \
  --form-string "prompt=$(jq -c . "$AV_ROOT/assets/prompts/image2video/car_driving.json")" \
  --form-string "negative_prompt=$(jq -c . "$AV_ROOT/assets/negative_prompts/image2video/neg_prompt.json")" \
  --form-string "size=1280x720" \
  --form-string "num_frames=189" \
  --form-string "fps=24" \
  --form-string "num_inference_steps=35" \
  --form-string "guidance_scale=6.0" \
  --form-string "flow_shift=10.0" \
  --form-string "seed=0" \
  --form-string 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}' \
  -F "input_reference=@$AV_ROOT/assets/images/image2video/car_driving.jpg"
```

## 5. I2VS — Image to Video with Sound

```bash
curl -sS --fail-with-body -X POST "$BASE_URL/v1/videos" \
  -H 'Accept: video/mp4' \
  --form-string "prompt=$(jq -c . "$AV_ROOT/assets/prompts/image2video/coastal_road_audio.json")" \
  --form-string "negative_prompt=$(jq -c . "$AV_ROOT/assets/negative_prompts/image2video/neg_prompt.json")" \
  --form-string "size=1280x720" \
  --form-string "num_frames=189" \
  --form-string "fps=24" \
  --form-string "num_inference_steps=35" \
  --form-string "guidance_scale=6.0" \
  --form-string "flow_shift=10.0" \
  --form-string "seed=0" \
  --form-string "generate_sound=true" \
  --form-string "sound_duration=7.875" \
  --form-string 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}' \
  -F "input_reference=@$AV_ROOT/assets/images/image2video/coastal_road_audio.jpg"
```

## 6. V2V — Video to Video (with sound)

The conditioning video is uploaded as the `video_reference` file part. Drop the two sound fields for a silent run:

```bash
curl -sS --fail-with-body -X POST "$BASE_URL/v1/videos" \
  -H 'Accept: video/mp4' \
  --form-string "prompt=$(jq -c . "$AV_ROOT/assets/prompts/image2video/car_driving.json")" \
  --form-string "negative_prompt=$(jq -c . "$AV_ROOT/assets/negative_prompts/image2video/neg_prompt.json")" \
  --form-string "size=1280x720" \
  --form-string "num_frames=189" \
  --form-string "fps=24" \
  --form-string "num_inference_steps=35" \
  --form-string "guidance_scale=6.0" \
  --form-string "flow_shift=10.0" \
  --form-string "seed=0" \
  --form-string "generate_sound=true" \
  --form-string "sound_duration=7.875" \
  --form-string 'extra_params={"use_resolution_template":false,"use_duration_template":false,"guardrails":true}' \
  -F "video_reference=@$AV_ROOT/assets/videos/car_driving_plain.mp4;type=video/mp4"
```

## 7. Action — Forward Dynamics (AV)

First frame image + an ego-trajectory action array embedded in `extra_params`. The input image `av_0.jpg` is 832x480, and `num_frames` = action_chunk_size + 1 = 61. Swap `av_traj_forward.json` for `av_traj_left.json` / `av_traj_right.json` for the other trajectories:

```bash
curl -sS --fail-with-body -X POST "$BASE_URL/v1/videos" \
  --form-string "prompt=You are an autonomous vehicle planning system." \
  --form-string "num_frames=61" \
  --form-string "fps=10" \
  --form-string "size=832x480" \
  --form-string "num_inference_steps=30" \
  --form-string "guidance_scale=1.0" \
  --form-string "flow_shift=10.0" \
  --form-string "seed=0" \
  --form-string "extra_params=$(jq -cn --slurpfile act "$ACTION_ROOT/assets/actions/av_traj_forward.json" \
    '{action_mode:"forward_dynamics",domain_name:"av",action_view_point:"ego_view",action:$act[0],guardrails:false}')" \
  -F "input_reference=@$ACTION_ROOT/assets/images/av_0.jpg"
```

## 8. Action — Inverse Dynamics (AV) (`/v1/actions/generations`)

Input is a video (uploaded as `input_reference`). This endpoint is synchronous — the predicted action trajectory comes back directly in the response under `.data[0].action` (no job to poll, no video content). Note there is no `size` field here:

```bash
curl -sS --fail-with-body -X POST "$BASE_URL/v1/actions/generations" \
  --form-string "prompt=You are an autonomous vehicle planning system." \
  --form-string "num_frames=61" \
  --form-string "fps=10" \
  --form-string "num_inference_steps=30" \
  --form-string "guidance_scale=1.0" \
  --form-string "flow_shift=10.0" \
  --form-string "seed=0" \
  --form-string 'extra_params={"action_mode":"inverse_dynamics","domain_name":"av","action_view_point":"ego_view","raw_action_dim":9,"guardrails":false}' \
  -F "input_reference=@$ACTION_ROOT/assets/videos/av_0.mp4;type=video/mp4" \
| jq '.data[0].action | {shape, dtype, values}' > id_action.json
```

## 9. Action — Policy (DROID)

Policy conditions on a single concatenated multi-view frame (wrist camera on top, the two exterior cameras below, 640x540). Build it once from the checked-in DROID sample with ffmpeg:

```bash
DROID=$ACTION_ROOT/assets/droid_lerobot_example/videos
ffmpeg -y -loglevel error \
  -i "$DROID/observation.image.wrist_image_left/chunk-000/file-000.mp4" \
  -i "$DROID/observation.image.exterior_image_1_left/chunk-000/file-000.mp4" \
  -i "$DROID/observation.image.exterior_image_2_left/chunk-000/file-000.mp4" \
  -filter_complex "\
[0:v]select='eq(n,0)',scale=640:270:force_original_aspect_ratio=increase,crop=640:270[w];\
[1:v]select='eq(n,0)',scale=320:270:force_original_aspect_ratio=increase,crop=320:270[l];\
[2:v]select='eq(n,0)',scale=320:270:force_original_aspect_ratio=increase,crop=320:270[r];\
[l][r]hstack[b];[w][b]vstack[out]" \
  -map "[out]" -frames:v 1 droid_policy_first_frame.png
```

Then submit to the synchronous `/v1/actions/generations` endpoint. Notes on the fields, matching `run_policy_with_sglang.ipynb` on `main`:

- `height`/`width` (736x544, the action-resolution bucket closest to the 640x540 input) instead of a `size` string — this endpoint does not parse `size` and would fall back to an 832x480 default that center-crops the multi-view composite.
- The prompt is the plain task instruction; the server wraps it into the structured action JSON itself.
- `action_cinematography_framing` describes the concatenated multi-view layout.
- The response is synchronous: the predicted DROID action chunk (16x8: 7 joints + 1 gripper) is at `.data[0].action.values`.

```bash
curl -sS --fail-with-body -X POST "$BASE_URL/v1/actions/generations" \
  --form-string "prompt=Pick up the object and place it in the target container." \
  --form-string "num_frames=17" \
  --form-string "fps=15" \
  --form-string "height=544" \
  --form-string "width=736" \
  --form-string "num_inference_steps=30" \
  --form-string "guidance_scale=1.0" \
  --form-string "flow_shift=5.0" \
  --form-string "seed=0" \
  --form-string 'extra_params={"action_mode":"policy","domain_name":"droid_lerobot","raw_action_dim":8,"action_view_point":"concat_view","action_cinematography_framing":"The top row is from the wrist-mounted camera. The bottom row contains two horizontally concatenated third-person perspective views of the scene from opposite sides, with the robot visible.","guardrails":false}' \
  -F "input_reference=@droid_policy_first_frame.png" \
| tee policy_response.json | jq '.data[0].action.values' > policy_action.json
```
## 10. Transfer — Edge (Canny)

Transfer requests are JSON bodies (not multipart). `control_path` must be readable by the server process, so `$COSMOS_ROOT` has to be mounted into the container at the same path:

```bash
jq -n \
  --arg prompt "$(jq -c . "$TRANSFER_ROOT/assets/edge/prompt.json")" \
  --arg neg "$(jq -c . "$TRANSFER_ROOT/assets/negative_prompt.json")" \
  --arg extra "$(jq -cn --arg cp "$TRANSFER_ROOT/assets/edge/control_edge.mp4" '{
      use_resolution_template: false,
      use_duration_template: false,
      guardrails: false,
      control_path: $cp,
      preset_edge_threshold: "medium",
      control_hint: "edge",
      resolution: "720",
      control_guidance: 1.5,
      num_video_frames_per_chunk: 121,
      num_conditional_frames: 1,
      num_first_chunk_conditional_frames: 0,
      share_vision_temporal_positions: true,
      max_frames: 121
    }')" \
  '{
    prompt: $prompt,
    negative_prompt: $neg,
    size: "1280x720",
    num_frames: "121",
    fps: "30",
    num_inference_steps: "50",
    guidance_scale: "3.0",
    flow_shift: "10.0",
    seed: "2026",
    extra_params: $extra
  }' \
| curl -sS --fail-with-body -X POST "$BASE_URL/v1/videos" \
    -H 'Content-Type: application/json' -H 'Accept: video/mp4' -d @-
```

## 11. Transfer — Segmentation

Same shape as edge, with the seg control video, `control_hint: "seg"`, `control_guidance: 2.0`, and no edge threshold:

```bash
jq -n \
  --arg prompt "$(jq -c . "$TRANSFER_ROOT/assets/seg/prompt.json")" \
  --arg neg "$(jq -c . "$TRANSFER_ROOT/assets/negative_prompt.json")" \
  --arg extra "$(jq -cn --arg cp "$TRANSFER_ROOT/assets/seg/control_seg.mp4" '{
      use_resolution_template: false,
      use_duration_template: false,
      guardrails: false,
      control_path: $cp,
      control_hint: "seg",
      resolution: "720",
      control_guidance: 2.0,
      num_video_frames_per_chunk: 121,
      num_conditional_frames: 1,
      num_first_chunk_conditional_frames: 0,
      share_vision_temporal_positions: true,
      max_frames: 121
    }')" \
  '{
    prompt: $prompt,
    negative_prompt: $neg,
    size: "1280x720",
    num_frames: "121",
    fps: "30",
    num_inference_steps: "50",
    guidance_scale: "3.0",
    flow_shift: "10.0",
    seed: "2026",
    extra_params: $extra
  }' \
| curl -sS --fail-with-body -X POST "$BASE_URL/v1/videos" \
    -H 'Content-Type: application/json' -H 'Accept: video/mp4' -d @-
```

## 12. Polling and Downloading (`/v1/videos` jobs)

Every `POST /v1/videos` returns JSON like `{"id": "...", "status": "queued", ...}`. Poll until `status` is `completed`, then fetch the mp4:

```bash
VIDEO_ID=<id from the POST response>

# poll status (repeat until "completed"; "failed"/"cancelled" mean the job died)
curl -sS "$BASE_URL/v1/videos/$VIDEO_ID" | jq '{status, progress}'

# download the generated video
curl -sS "$BASE_URL/v1/videos/$VIDEO_ID/content" -o output.mp4

# forward-dynamics jobs may also report action metadata in the status JSON
curl -sS "$BASE_URL/v1/videos/$VIDEO_ID" | jq '.action'
```

Or as a one-shot helper:

```bash
poll_and_download () {
  local id=$1 out=$2
  while :; do
    local status=$(curl -sS "$BASE_URL/v1/videos/$id" | jq -r .status)
    echo "$id: $status"
    [ "$status" = completed ] && break
    case "$status" in failed|cancelled) return 1;; esac
    sleep 5
  done
  curl -sS "$BASE_URL/v1/videos/$id/content" -o "$out"
}
# usage: poll_and_download "$VIDEO_ID" output.mp4
```
