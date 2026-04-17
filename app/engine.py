from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional

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


class TTSEngine:
    def __init__(self, settings):
        from indextts.infer_v2 import IndexTTS2

        self.settings = settings
        self.device = settings.resolved_device

        if settings.indextts_cache_dir:
            os.environ.setdefault("HF_HOME", settings.indextts_cache_dir)
            os.environ.setdefault("HF_HUB_CACHE", settings.indextts_cache_dir)

        self._validate_model_dir()

        is_cuda = self.device.startswith("cuda")
        use_fp16 = bool(settings.indextts_use_fp16 and is_cuda)
        use_cuda_kernel = bool(settings.indextts_use_cuda_kernel and is_cuda)
        use_deepspeed = bool(settings.indextts_use_deepspeed and is_cuda)

        log.info(
            "loading IndexTTS2 model_dir=%s cfg=%s device=%s fp16=%s cuda_kernel=%s deepspeed=%s",
            settings.indextts_model_dir,
            settings.resolved_cfg_path,
            self.device,
            use_fp16,
            use_cuda_kernel,
            use_deepspeed,
        )
        self.model = IndexTTS2(
            cfg_path=settings.resolved_cfg_path,
            model_dir=settings.indextts_model_dir,
            use_fp16=use_fp16,
            device=self.device,
            use_cuda_kernel=use_cuda_kernel,
            use_deepspeed=use_deepspeed,
            use_torch_compile=settings.indextts_use_torch_compile,
        )
        self.sample_rate = 22050
        self._lock = asyncio.Lock()

    def _validate_model_dir(self) -> None:
        p = Path(self.settings.indextts_model_dir)
        if not p.exists() or not p.is_dir():
            raise RuntimeError(
                f"model_dir {p} not found; mount a checkpoint directory with "
                "config.yaml and the required IndexTTS-2 weight files."
            )
        cfg = Path(self.settings.resolved_cfg_path)
        if not cfg.exists():
            raise RuntimeError(
                f"config file {cfg} not found; expected inside the IndexTTS-2 checkpoint directory."
            )

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
