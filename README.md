# IndexTTS OpenAI-TTS API

**English** · [中文](./README.zh.md)

An [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech)-compatible HTTP service wrapping [IndexTTS2](https://github.com/index-tts/index-tts) — Bilibili's emotionally expressive, duration-controlled auto-regressive zero-shot TTS — with voice cloning driven by files dropped into a mounted directory and four emotion-control modes.

## Features

- **OpenAI TTS compatible** — `POST /v1/audio/speech` with the same request shape as the OpenAI SDK
- **Zero-shot voice cloning** — each voice is a `xxx.wav` + `xxx.txt` pair in a mounted directory; the `.wav` is used as the speaker reference and the stem becomes the voice id
- **Emotion control** — extra `POST /v1/audio/emotion` endpoint exposes IndexTTS2's three emotion modes: reference audio, 8-dim emotion vector, or natural-language description (routed through the built-in Qwen classifier)
- **2 images** — `cuda` and `cpu`
- **Model weights mounted, not baked** — `/checkpoints` is a writable volume; auxiliary models (MaskGCT, CAM++, BigVGAN) are fetched on first run and cached under `checkpoints/hf_cache`
- **Multiple output formats** — `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`

## Available images

| Image | Device |
|---|---|
| `ghcr.io/seancheung/indextts-openai-tts-api:cuda-latest` | CUDA 12.8 |
| `ghcr.io/seancheung/indextts-openai-tts-api:latest`      | CPU |

Images are built for `linux/amd64`.

## Quick start

### 1. Prepare the checkpoints directory

Download the IndexTTS-2 weights into a local `checkpoints/` directory.

```bash
# via huggingface-cli
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

# or via modelscope
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

The directory must contain `config.yaml` plus the weight files (`gpt.pth`, `s2mel.pth`, `bpe.model`, `wav2vec2bert_stats.pt`, `feat1.pt`, `feat2.pt`, `qwen0.6bemo4-merge/`, …). Total size is roughly 10–15 GB.

### 2. Prepare the voices directory

```
voices/
├── alice.wav     # reference audio, mono, 16kHz+, ~3-20s recommended
├── alice.txt     # UTF-8 text: the transcript of alice.wav (human-readable only)
├── bob.wav
└── bob.txt
```

**Rules**: a voice is valid only when both files with the same stem exist; the stem is the voice id; unpaired or extra files are ignored. IndexTTS2's inference does not require the transcript, but the `.txt` is kept for human review and appears in `/v1/audio/voices`.

### 3. Run the container

GPU (recommended):

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/checkpoints:/checkpoints \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  ghcr.io/seancheung/indextts-openai-tts-api:cuda-latest
```

CPU:

```bash
docker run --rm -p 8000:8000 \
  -v $PWD/checkpoints:/checkpoints \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  ghcr.io/seancheung/indextts-openai-tts-api:latest
```

> **`/checkpoints` must be writable.** IndexTTS2 downloads a few auxiliary models (MaskGCT semantic codec, CAM++ speaker encoder, BigVGAN vocoder) into `checkpoints/hf_cache` on first start. Do not mount it as `:ro`.

> **GPU prerequisites**: NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on Linux. On Windows use Docker Desktop + WSL2 + NVIDIA Windows driver; no host CUDA toolkit required. IndexTTS2 needs roughly 10–15 GB VRAM with fp16 (more with fp32).

### 4. docker-compose

See [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml).

## API usage

The service listens on port `8000` by default. Output audio is mono 22.05 kHz.

### GET `/v1/audio/voices`

List all usable voices.

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "Hello, this is a reference audio sample."
    }
  ]
}
```

### GET `/v1/audio/voices/preview?id={id}`

Returns the raw reference wav (`audio/wav`), suitable for a browser `<audio>` element.

### POST `/v1/audio/speech`

OpenAI TTS-compatible endpoint — zero-shot voice cloning. IndexTTS2 uses the voice's `.wav` as the speaker reference; the emotion of the generated speech matches the speaker reference.

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "indextts",
    "input": "Hello world, this is a test.",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

Request fields:

| Field | Type | Description |
|---|---|---|
| `model` | string | Accepted but ignored (for OpenAI SDK compatibility) |
| `input` | string | Text to synthesize, up to 8000 characters |
| `voice` | string | Voice id — must match an entry from `/v1/audio/voices` |
| `response_format` | string | `mp3` (default) / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | Accepted for OpenAI SDK compatibility but **ignored** — IndexTTS2 has no speed control |
| `generation` | object | Optional sampling overrides (see below) |

Common `generation` keys (all optional; falling back to the server defaults when omitted): `do_sample`, `top_p`, `top_k`, `temperature`, `length_penalty`, `num_beams`, `repetition_penalty`, `max_mel_tokens`, `max_text_tokens_per_segment`, `interval_silence`.

### POST `/v1/audio/emotion`

Emotion-conditioned cloning. Exactly one of `emotion_voice`, `emotion_vector`, `emotion_text` must be provided; the chosen source drives IndexTTS2's emotion path while `voice` still supplies the timbre.

**Mode A — emotion reference audio** (use another voice's `.wav` as the emotion prompt):

```bash
curl -s http://localhost:8000/v1/audio/emotion \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Speak this line with the same mood as the reference.",
    "voice": "alice",
    "emotion_voice": "sad_sample",
    "emotion_alpha": 0.9,
    "response_format": "mp3"
  }' \
  -o out_emo_audio.mp3
```

**Mode B — 8-dim emotion vector** (`[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`, each in `[0, 1]`):

```bash
curl -s http://localhost:8000/v1/audio/emotion \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "I cannot believe this happened.",
    "voice": "alice",
    "emotion_vector": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    "response_format": "mp3"
  }' \
  -o out_emo_vec.mp3
```

**Mode C — natural-language description** (Qwen classifier infers the vector from `emotion_text`):

```bash
curl -s http://localhost:8000/v1/audio/emotion \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Run, they are coming for us!",
    "voice": "alice",
    "emotion_text": "panicked and terrified",
    "emotion_alpha": 0.6,
    "response_format": "mp3"
  }' \
  -o out_emo_text.mp3
```

Request fields:

| Field | Type | Description |
|---|---|---|
| `input` | string | Text to synthesize |
| `voice` | string | Speaker voice id (timbre reference) |
| `emotion_voice` | string | Optional — a voice id whose `.wav` is used as the emotion reference audio |
| `emotion_vector` | float[8] | Optional — `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`, each in `[0, 1]` |
| `emotion_text` | string | Optional — natural-language emotion description |
| `emotion_alpha` | float | Blend weight against the speaker timbre, `0.0 - 1.0` (default `1.0`). Lower values keep more of the speaker's native prosody |
| `use_random` | bool | Enable stochastic emotion sampling (may reduce timbre fidelity) |
| `response_format` | string | Same as `/speech` |
| `generation` | object | Same optional sampling overrides as `/speech` |

### Using the OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="indextts",
    voice="alice",
    input="Hello world",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

The emotion endpoint is not part of the OpenAI schema — call it directly via `requests` / `httpx` or use `extra_body={...}` with a custom client.

### GET `/healthz`

Returns `status`, `model` (checkpoint dir), `device`, and `sample_rate` for health checks.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `INDEXTTS_MODEL_DIR` | `/checkpoints` | Directory with `config.yaml` and IndexTTS-2 weights (must be writable — auxiliary models are downloaded into `hf_cache/` inside it) |
| `INDEXTTS_CFG_PATH` | `${MODEL_DIR}/config.yaml` | Override the config file path |
| `INDEXTTS_VOICES_DIR` | `/voices` | Voices directory |
| `INDEXTTS_DEVICE` | `auto` | `auto` → CUDA > MPS > CPU. Or `cuda` / `mps` / `cpu` |
| `INDEXTTS_CUDA_INDEX` | `0` | Selects `cuda:N` when device is `cuda` or `auto` |
| `INDEXTTS_CACHE_DIR` | — | When set, seeds `HF_HOME` and `HF_HUB_CACHE` before model load |
| `INDEXTTS_USE_FP16` | `false` | Half-precision (CUDA only; ignored on CPU/MPS) |
| `INDEXTTS_USE_CUDA_KERNEL` | `false` | Enable BigVGAN fused CUDA activation kernels (CUDA only) |
| `INDEXTTS_USE_DEEPSPEED` | `false` | Enable DeepSpeed acceleration (CUDA only; DeepSpeed is not installed in the shipped image — build a custom image if you need it) |
| `INDEXTTS_USE_TORCH_COMPILE` | `false` | Enable `torch.compile` for the GPT stack |
| `INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT` | `120` | Internal sentence-splitting threshold |
| `INDEXTTS_INTERVAL_SILENCE` | `200` | Silence (ms) inserted between segments |
| `INDEXTTS_DO_SAMPLE` | `true` | Default sampling mode |
| `INDEXTTS_TOP_P` | `0.8` | Default nucleus sampling threshold |
| `INDEXTTS_TOP_K` | `30` | Default top-k |
| `INDEXTTS_TEMPERATURE` | `0.8` | Default sampling temperature |
| `INDEXTTS_LENGTH_PENALTY` | `0.0` | Default length penalty |
| `INDEXTTS_NUM_BEAMS` | `3` | Default beam width |
| `INDEXTTS_REPETITION_PENALTY` | `10.0` | Default repetition penalty |
| `INDEXTTS_MAX_MEL_TOKENS` | `1500` | Default max mel tokens per segment |
| `MAX_INPUT_CHARS` | `8000` | Upper bound for the `input` field |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## Building images locally

Initialize the submodule first (the workflow does this automatically).

```bash
git submodule update --init --recursive

# CUDA image
docker buildx build -f docker/Dockerfile.cuda \
  -t indextts-openai-tts-api:cuda .

# CPU image
docker buildx build -f docker/Dockerfile.cpu \
  -t indextts-openai-tts-api:cpu .
```

## Caveats

- **`speed` is a no-op.** IndexTTS2 has no native speed control, but the field is kept in the schema so that the OpenAI Python SDK's default request body (which always sends `speed=1.0`) does not 422. Post-process the returned audio if you need tempo control.
- **No built-in OpenAI voice names** (`alloy`, `echo`, `fable`, …). IndexTTS2 is zero-shot; drop `alloy.wav` + `alloy.txt` into `voices/` to bind the name.
- **Concurrency**: a single IndexTTS2 instance is not thread-safe; the service serializes inference with an asyncio Lock. Scale out with more containers behind a load balancer.
- **Long text**: requests whose `input` exceeds `MAX_INPUT_CHARS` (default 8000) return 413. IndexTTS2 splits long text into segments internally (`max_text_tokens_per_segment`).
- **First request is slow**: IndexTTS2 caches the processed speaker/emotion reference. Repeated requests against the same voice reuse the cache and are 10–100× faster than the first call.
- **Streaming is not supported** on the HTTP layer — the endpoint returns the complete audio when generation finishes. (IndexTTS2 supports `stream_return` internally; exposing it here is future work.)
- **DeepSpeed is not installed in the default images** because of its size and build fragility. Set `INDEXTTS_USE_DEEPSPEED=true` only after extending the image to `pip install deepspeed==0.17.1`.
- **CUDA 12.8+ host driver required** for the CUDA image. Older drivers will fail at import time inside the container.
- **Output is 22.05 kHz mono**; `pcm` is raw s16le at 22050 Hz.
- **No built-in auth** — deploy behind a reverse proxy (Nginx, Cloudflare, etc.) if you need token-based access control.
- **IndexTTS v1 is not exposed.** Only IndexTTS-2 is supported; the legacy v1 API is not wrapped by this service.

## Project layout

```
.
├── index-tts/                  # read-only submodule, never modified
├── app/                        # FastAPI application
│   ├── server.py
│   ├── engine.py               # IndexTTS2 wrapper + inference entrypoints
│   ├── voices.py               # voices directory scanner
│   ├── audio.py                # multi-format encoder
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml        # cuda + cpu matrix build
└── README.md
```

## Acknowledgements

Built on top of [index-tts/index-tts](https://github.com/index-tts/index-tts). See the upstream repository for model licenses and citation.
