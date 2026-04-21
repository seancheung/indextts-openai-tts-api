from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, List

import numpy as np

log = logging.getLogger(__name__)

_DEFAULT_INDEXTTS_PATHS = (
    os.environ.get("INDEXTTS_REPO_DIR", "/opt/index-tts"),
)


def _ensure_sys_path() -> None:
    for base in _DEFAULT_INDEXTTS_PATHS:
        if not base:
            continue
        if Path(base).exists() and base not in sys.path:
            sys.path.insert(0, base)


_ensure_sys_path()


def _resolve_device(settings) -> str:
    import torch

    if torch.cuda.is_available():
        return f"cuda:{settings.indextts_cuda_index}"
    return "cpu"


def _resolve_model_dir(settings) -> str:
    model = settings.indextts_model

    if os.path.isdir(model):
        log.info("using local IndexTTS model dir: %s", model)
        return model

    from huggingface_hub import snapshot_download

    log.info("downloading IndexTTS model snapshot: %s", model)
    local_dir = snapshot_download(
        repo_id=model,
        cache_dir=settings.indextts_cache_dir or None,
    )
    log.info("model snapshot ready at %s", local_dir)
    return local_dir


def _restore_hf_cache_env() -> None:
    """IndexTTS' upstream `infer_v2.py` force-sets `HF_HUB_CACHE` to a relative
    `./checkpoints/hf_cache` path at import time, which silently redirects
    auxiliary model downloads away from the user's mounted cache. Re-point it
    back at the standard HuggingFace cache layout."""

    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")


class TTSEngine:
    def __init__(self, settings):
        self.settings = settings

        if settings.indextts_cache_dir:
            os.environ.setdefault("HF_HOME", settings.indextts_cache_dir)
            os.environ.setdefault("HF_HUB_CACHE", settings.indextts_cache_dir)

        self.device = _resolve_device(settings)
        model_dir = _resolve_model_dir(settings)

        from indextts.infer_v2 import IndexTTS2

        _restore_hf_cache_env()
        cfg_path = str(Path(model_dir) / "config.yaml")
        if not Path(cfg_path).exists():
            raise RuntimeError(
                f"config file {cfg_path} not found inside the IndexTTS model snapshot"
            )

        is_cuda = self.device.startswith("cuda")
        use_fp16 = bool(settings.indextts_use_fp16 and is_cuda)
        use_cuda_kernel = bool(settings.indextts_use_cuda_kernel and is_cuda)
        use_deepspeed = bool(settings.indextts_use_deepspeed and is_cuda)

        log.info(
            "loading IndexTTS2 model_dir=%s device=%s fp16=%s cuda_kernel=%s deepspeed=%s",
            model_dir,
            self.device,
            use_fp16,
            use_cuda_kernel,
            use_deepspeed,
        )
        self.model = IndexTTS2(
            cfg_path=cfg_path,
            model_dir=model_dir,
            use_fp16=use_fp16,
            device=self.device,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed,
            use_torch_compile=settings.indextts_use_torch_compile,
        )
        self.model_dir = model_dir
        self.sample_rate = 22050
        self._lock = asyncio.Lock()

    def _gen_kwargs(self, params) -> dict:
        s = self.settings
        g = params
        return dict(
            do_sample=_pick(g, "do_sample", s.indextts_do_sample),
            top_p=_pick(g, "top_p", s.indextts_top_p),
            top_k=_pick(g, "top_k", s.indextts_top_k),
            temperature=_pick(g, "temperature", s.indextts_temperature),
            length_penalty=_pick(g, "length_penalty", s.indextts_length_penalty),
            num_beams=_pick(g, "num_beams", s.indextts_num_beams),
            repetition_penalty=_pick(g, "repetition_penalty", s.indextts_repetition_penalty),
            max_mel_tokens=_pick(g, "max_mel_tokens", s.indextts_max_mel_tokens),
        )

    def _infer_kwargs(self, params) -> dict:
        s = self.settings
        g = params
        return dict(
            max_text_tokens_per_segment=_pick(
                g, "max_text_tokens_per_segment", s.indextts_max_text_tokens_per_segment
            ),
            interval_silence=_pick(
                g, "interval_silence", s.indextts_interval_silence
            ),
        )

    def _run_infer(self, **infer_kwargs) -> np.ndarray:
        result = self.model.infer(output_path=None, **infer_kwargs)
        return _result_to_float32(result, expected_sr=self.sample_rate)

    async def _synthesize(self, **infer_kwargs) -> np.ndarray:
        async with self._lock:
            return await asyncio.to_thread(self._run_infer, **infer_kwargs)

    # ------------------------------------------------------------------
    # inference entrypoints
    # ------------------------------------------------------------------
    async def synthesize_clone(
        self,
        text: str,
        *,
        spk_wav: str,
        params=None,
    ) -> np.ndarray:
        return await self._synthesize(
            spk_audio_prompt=spk_wav,
            text=text,
            **self._infer_kwargs(params),
            **self._gen_kwargs(params),
        )

    async def synthesize_emotion_audio(
        self,
        text: str,
        *,
        spk_wav: str,
        emo_wav: str,
        emo_alpha: float,
        params=None,
    ) -> np.ndarray:
        return await self._synthesize(
            spk_audio_prompt=spk_wav,
            text=text,
            emo_audio_prompt=emo_wav,
            emo_alpha=emo_alpha,
            **self._infer_kwargs(params),
            **self._gen_kwargs(params),
        )

    async def synthesize_emotion_vector(
        self,
        text: str,
        *,
        spk_wav: str,
        emo_vector: List[float],
        use_random: bool,
        params=None,
    ) -> np.ndarray:
        return await self._synthesize(
            spk_audio_prompt=spk_wav,
            text=text,
            emo_vector=list(emo_vector),
            use_random=use_random,
            **self._infer_kwargs(params),
            **self._gen_kwargs(params),
        )

    async def synthesize_emotion_text(
        self,
        text: str,
        *,
        spk_wav: str,
        emo_text: str,
        emo_alpha: float,
        use_random: bool,
        params=None,
    ) -> np.ndarray:
        return await self._synthesize(
            spk_audio_prompt=spk_wav,
            text=text,
            use_emo_text=True,
            emo_text=emo_text,
            emo_alpha=emo_alpha,
            use_random=use_random,
            **self._infer_kwargs(params),
            **self._gen_kwargs(params),
        )


def _pick(params, name: str, default: Any) -> Any:
    if params is None:
        return default
    value = getattr(params, name, None)
    return default if value is None else value


def _result_to_float32(result: Any, *, expected_sr: int) -> np.ndarray:
    """Convert IndexTTS2.infer output to a 1-D float32 array in [-1, 1]."""
    if result is None:
        raise RuntimeError("inference produced no audio")

    if isinstance(result, tuple) and len(result) == 2:
        sr, wav = result
        if sr and int(sr) != expected_sr:
            log.warning("sample_rate mismatch: got %s expected %s", sr, expected_sr)
    else:
        wav = result

    arr = np.asarray(wav)
    if arr.ndim == 2:
        arr = arr.squeeze()
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        scale = float(max(abs(info.min), info.max))
        arr = arr.astype(np.float32) / scale
    else:
        arr = arr.astype(np.float32, copy=False)

    return np.ascontiguousarray(np.clip(arr, -1.0, 1.0))
