# IndexTTS OpenAI-TTS API

[English](./README.md) · **中文**

一个 [OpenAI TTS](https://platform.openai.com/docs/api-reference/audio/createSpeech) 兼容的 HTTP 服务，封装了 [IndexTTS2](https://github.com/index-tts/index-tts)——B 站开源的情感可控、时长可控自回归零样本 TTS 模型；支持从挂载目录零样本克隆音色，并提供四种情感控制模式。

## 特性

- **OpenAI TTS 兼容**：`POST /v1/audio/speech`，请求体格式与 OpenAI SDK 一致
- **零样本音色克隆**：挂载 `voices/` 目录下的 `xxx.wav` + `xxx.txt` 对，`.wav` 作为说话人参考，文件名（不含后缀）即音色 id
- **情感控制**：额外提供 `POST /v1/audio/emotion`，暴露 IndexTTS2 的三种情感模式——情感参考音频、8 维情感向量、自然语言情感描述（内置 Qwen 分类器推断向量）
- **2 个镜像**：`cuda` 与 `cpu`
- **模型权重挂载而非打包**：`/checkpoints` 为可写卷，附属模型（MaskGCT、CAM++、BigVGAN）首次启动时下载并缓存到 `checkpoints/hf_cache`
- **多种输出格式**：`mp3`、`opus`、`aac`、`flac`、`wav`、`pcm`

## 可用镜像

| 镜像 | 设备 |
|---|---|
| `ghcr.io/seancheung/indextts-openai-tts-api:cuda-latest` | CUDA 12.8 |
| `ghcr.io/seancheung/indextts-openai-tts-api:latest`      | CPU |

镜像仅构建 `linux/amd64`。

## 快速开始

### 1. 准备 checkpoints 目录

将 IndexTTS-2 权重下载到本地 `checkpoints/` 目录。

```bash
# 使用 huggingface-cli
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints

# 或使用 modelscope
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

该目录必须包含 `config.yaml` 以及权重文件（`gpt.pth`、`s2mel.pth`、`bpe.model`、`wav2vec2bert_stats.pt`、`feat1.pt`、`feat2.pt`、`qwen0.6bemo4-merge/` 等）。整体体积约 10–15 GB。

### 2. 准备音色目录

```
voices/
├── alice.wav     # 参考音频，单声道，16kHz 以上，推荐 3-20 秒
├── alice.txt     # UTF-8 纯文本，alice.wav 的转录（仅供人类阅读）
├── bob.wav
└── bob.txt
```

**规则**：必须同时存在同名的 `.wav` 和 `.txt` 才会被识别为有效音色；文件名（不含后缀）即音色 id；多余或缺对的文件会被忽略。IndexTTS2 推理本身不使用 `.txt`，保留此文件便于人类校对，也会在 `/v1/audio/voices` 中返回。

### 3. 运行容器

GPU 版本（推荐）：

```bash
docker run --rm -p 8000:8000 --gpus all \
  -v $PWD/checkpoints:/checkpoints \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  ghcr.io/seancheung/indextts-openai-tts-api:cuda-latest
```

CPU 版本：

```bash
docker run --rm -p 8000:8000 \
  -v $PWD/checkpoints:/checkpoints \
  -v $PWD/voices:/voices:ro \
  -v $PWD/hf_cache:/root/.cache/huggingface \
  ghcr.io/seancheung/indextts-openai-tts-api:latest
```

> **`/checkpoints` 必须可写**。IndexTTS2 会在首次启动时下载若干附属模型（MaskGCT 语义编解码、CAM++ 说话人编码、BigVGAN 声码器）到 `checkpoints/hf_cache`。不要加 `:ro`。

> **GPU 要求**：宿主机需安装 NVIDIA 驱动与 [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。Windows 需 Docker Desktop + WSL2 + NVIDIA Windows 驱动。IndexTTS2 在 fp16 下约需 10–15 GB 显存（fp32 更多）。

### 4. docker-compose

参考 [`docker/docker-compose.example.yml`](./docker/docker-compose.example.yml)。

## API 用法

服务默认监听 `8000` 端口。输出音频为单声道 22.05 kHz。

### GET `/v1/audio/voices`

列出所有可用音色。

```bash
curl -s http://localhost:8000/v1/audio/voices | jq
```

返回：

```json
{
  "object": "list",
  "data": [
    {
      "id": "alice",
      "preview_url": "http://localhost:8000/v1/audio/voices/preview?id=alice",
      "prompt_text": "你好，这是一段参考音频。"
    }
  ]
}
```

### GET `/v1/audio/voices/preview?id={id}`

返回参考音频本体（`audio/wav`），可用于浏览器 `<audio>` 试听。

### POST `/v1/audio/speech`

OpenAI TTS 兼容接口——零样本音色克隆。IndexTTS2 将音色的 `.wav` 作为说话人参考，生成音频的情绪与参考音频保持一致。

```bash
curl -s http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "indextts",
    "input": "你好世界，这是一段测试语音。",
    "voice": "alice",
    "response_format": "mp3"
  }' \
  -o out.mp3
```

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 接受但忽略（为了与 OpenAI SDK 兼容） |
| `input` | string | 要合成的文本，最长 8000 字符 |
| `voice` | string | 音色 id，必须匹配 `/v1/audio/voices` 中的某一项 |
| `response_format` | string | `mp3`（默认） / `opus` / `aac` / `flac` / `wav` / `pcm` |
| `speed` | float | 为 OpenAI SDK 兼容而保留，**实际忽略**——IndexTTS2 无语速控制 |
| `generation` | object | 可选的采样超参数覆盖（见下） |

`generation` 常用字段（全部可选，未传则用服务端默认值）：`do_sample`、`top_p`、`top_k`、`temperature`、`length_penalty`、`num_beams`、`repetition_penalty`、`max_mel_tokens`、`max_text_tokens_per_segment`、`interval_silence`。

### POST `/v1/audio/emotion`

情感受控的克隆合成。`emotion_voice`、`emotion_vector`、`emotion_text` 三者**必须且只能选一个**：所选字段驱动 IndexTTS2 的情感路径，`voice` 仍然提供音色。

**模式 A — 情感参考音频**（把另一个音色的 `.wav` 作为情感 prompt）：

```bash
curl -s http://localhost:8000/v1/audio/emotion \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "以和参考音频相同的情绪朗读这段话。",
    "voice": "alice",
    "emotion_voice": "sad_sample",
    "emotion_alpha": 0.9,
    "response_format": "mp3"
  }' \
  -o out_emo_audio.mp3
```

**模式 B — 8 维情感向量**（顺序为 `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`，每项取值 `[0, 1]`）：

```bash
curl -s http://localhost:8000/v1/audio/emotion \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "我真的不敢相信发生了这种事。",
    "voice": "alice",
    "emotion_vector": [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0],
    "response_format": "mp3"
  }' \
  -o out_emo_vec.mp3
```

**模式 C — 自然语言描述**（Qwen 分类器从 `emotion_text` 推断向量）：

```bash
curl -s http://localhost:8000/v1/audio/emotion \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "快跑！他们追上来了！",
    "voice": "alice",
    "emotion_text": "恐慌、惊惧",
    "emotion_alpha": 0.6,
    "response_format": "mp3"
  }' \
  -o out_emo_text.mp3
```

请求字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `input` | string | 要合成的文本 |
| `voice` | string | 说话人音色 id（音色参考） |
| `emotion_voice` | string | 可选——作为情感参考音频的音色 id（仅使用其 `.wav`） |
| `emotion_vector` | float[8] | 可选——`[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`，每项 `[0, 1]` |
| `emotion_text` | string | 可选——自然语言情感描述 |
| `emotion_alpha` | float | 情感与音色混合权重，`0.0 - 1.0`（默认 `1.0`），越低越保留说话人原本的韵律 |
| `use_random` | bool | 启用情感随机采样（可能降低音色相似度） |
| `response_format` | string | 同 `/speech` |
| `generation` | object | 同 `/speech` 的采样超参数覆盖 |

### 使用 OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-noop")

with client.audio.speech.with_streaming_response.create(
    model="indextts",
    voice="alice",
    input="你好世界",
    response_format="mp3",
) as resp:
    resp.stream_to_file("out.mp3")
```

`/v1/audio/emotion` 不属于 OpenAI schema，请用 `requests` / `httpx` 等直接调用，或通过 `extra_body={...}` 配合自定义 client。

### GET `/healthz`

返回 `status`、`model`（checkpoint 目录）、`device`、`sample_rate`，用于健康检查。

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `INDEXTTS_MODEL_DIR` | `/checkpoints` | 包含 `config.yaml` 与 IndexTTS-2 权重的目录（必须可写——附属模型会下载到其 `hf_cache/` 子目录） |
| `INDEXTTS_CFG_PATH` | `${MODEL_DIR}/config.yaml` | 显式覆盖 config 路径 |
| `INDEXTTS_VOICES_DIR` | `/voices` | 音色目录 |
| `INDEXTTS_DEVICE` | `auto` | `auto` 按 CUDA > MPS > CPU 优先级。也可强制 `cuda` / `mps` / `cpu` |
| `INDEXTTS_CUDA_INDEX` | `0` | `cuda` / `auto` 时选择的 `cuda:N` |
| `INDEXTTS_CACHE_DIR` | — | 设置后会在加载模型前写入 `HF_HOME` 和 `HF_HUB_CACHE` |
| `INDEXTTS_USE_FP16` | `false` | 半精度（仅 CUDA，CPU/MPS 上被忽略） |
| `INDEXTTS_USE_CUDA_KERNEL` | `false` | 启用 BigVGAN 融合 CUDA 激活核心（仅 CUDA） |
| `INDEXTTS_USE_DEEPSPEED` | `false` | 启用 DeepSpeed 加速（仅 CUDA；镜像**未预装** DeepSpeed，需自建镜像加装） |
| `INDEXTTS_USE_TORCH_COMPILE` | `false` | 启用 `torch.compile` 优化 GPT |
| `INDEXTTS_MAX_TEXT_TOKENS_PER_SEGMENT` | `120` | 分句阈值 |
| `INDEXTTS_INTERVAL_SILENCE` | `200` | 段间静音时长（毫秒） |
| `INDEXTTS_DO_SAMPLE` | `true` | 默认采样模式 |
| `INDEXTTS_TOP_P` | `0.8` | 默认 nucleus 采样 |
| `INDEXTTS_TOP_K` | `30` | 默认 top-k |
| `INDEXTTS_TEMPERATURE` | `0.8` | 默认采样温度 |
| `INDEXTTS_LENGTH_PENALTY` | `0.0` | 默认长度惩罚 |
| `INDEXTTS_NUM_BEAMS` | `3` | 默认 beam 宽度 |
| `INDEXTTS_REPETITION_PENALTY` | `10.0` | 默认重复惩罚 |
| `INDEXTTS_MAX_MEL_TOKENS` | `1500` | 默认单段最大 mel token 数 |
| `MAX_INPUT_CHARS` | `8000` | `input` 字段上限 |
| `DEFAULT_RESPONSE_FORMAT` | `mp3` | |
| `HOST` | `0.0.0.0` | |
| `PORT` | `8000` | |
| `LOG_LEVEL` | `info` | |

## 本地构建镜像

构建前需先初始化 submodule（workflow 已处理）。

```bash
git submodule update --init --recursive

# CUDA 镜像
docker buildx build -f docker/Dockerfile.cuda \
  -t indextts-openai-tts-api:cuda .

# CPU 镜像
docker buildx build -f docker/Dockerfile.cpu \
  -t indextts-openai-tts-api:cpu .
```

## 局限 / 注意事项

- **`speed` 字段是 no-op**：IndexTTS2 无原生语速控制；保留该字段只为让 OpenAI Python SDK 的默认请求体（总会带 `speed=1.0`）不被 422。若需调速请对返回音频做后处理。
- **不做 OpenAI 固定音色名映射**（`alloy`、`echo`、`fable` 等）。IndexTTS2 本身是零样本；若想通过这些名字调用稳定的声音，直接在 `voices/` 放同名 `.wav` + `.txt` 即可。
- **并发**：IndexTTS2 单实例非线程安全，服务内部用 asyncio Lock 串行化。并发请求依赖横向扩容（多容器 + 负载均衡）。
- **长文本**：超过 `MAX_INPUT_CHARS`（默认 8000）返回 413。IndexTTS2 内部通过 `max_text_tokens_per_segment` 分段处理。
- **首次请求较慢**：IndexTTS2 对处理后的 speaker/emotion 参考做了缓存。同一音色后续请求会复用缓存，速度比首次快 10–100 倍。
- **不支持 HTTP 层流式返回**：生成完成后一次性返回。（IndexTTS2 本身支持 `stream_return`，服务层目前未暴露。）
- **默认镜像未安装 DeepSpeed**（体积大、构建易失败）。设 `INDEXTTS_USE_DEEPSPEED=true` 前需在自定义镜像中 `pip install deepspeed==0.17.1`。
- **CUDA 镜像要求宿主机 CUDA 12.8+ 驱动**，旧驱动会在容器内 import 时报错。
- **输出固定为 22.05 kHz 单声道**；`pcm` 为 22050 Hz s16le 裸数据。
- **无内置鉴权**：如需 token 访问控制，请在反向代理层（Nginx、Cloudflare 等）做。
- **不暴露 IndexTTS v1**：本服务仅封装 IndexTTS-2，不适配历史遗留的 v1 API。

## 目录结构

```
.
├── index-tts/                  # 只读 submodule，不修改
├── app/                        # FastAPI 应用
│   ├── server.py
│   ├── engine.py               # IndexTTS2 封装 + 推理入口
│   ├── voices.py               # 音色扫描
│   ├── audio.py                # 多格式编码
│   ├── config.py
│   └── schemas.py
├── docker/
│   ├── Dockerfile.cuda
│   ├── Dockerfile.cpu
│   ├── requirements.api.txt
│   ├── entrypoint.sh
│   └── docker-compose.example.yml
├── .github/workflows/
│   └── build-images.yml        # cuda + cpu 矩阵构建
└── README.md
```

## 致谢

基于 [index-tts/index-tts](https://github.com/index-tts/index-tts)。模型许可与引用见上游仓库。
