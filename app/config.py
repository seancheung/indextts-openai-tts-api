from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", case_sensitive=False, extra="ignore")

    indextts_model_dir: str = Field(default="/checkpoints")
    indextts_cfg_path: Optional[str] = Field(default=None)
    indextts_voices_dir: str = Field(default="/voices")
    indextts_device: Literal["auto", "cuda", "mps", "cpu"] = Field(default="auto")
    indextts_cuda_index: int = Field(default=0)
    indextts_cache_dir: Optional[str] = Field(default=None)

    indextts_use_fp16: bool = Field(default=False)
    indextts_use_cuda_kernel: bool = Field(default=False)
    indextts_use_deepspeed: bool = Field(default=False)
    indextts_use_torch_compile: bool = Field(default=False)

    indextts_max_text_tokens_per_segment: int = Field(default=120, ge=20, le=600)
    indextts_interval_silence: int = Field(default=200, ge=0, le=2000)

    indextts_do_sample: bool = Field(default=True)
    indextts_top_p: float = Field(default=0.8, ge=0.0, le=1.0)
    indextts_top_k: int = Field(default=30, ge=0)
    indextts_temperature: float = Field(default=0.8, gt=0.0, le=5.0)
    indextts_length_penalty: float = Field(default=0.0)
    indextts_num_beams: int = Field(default=3, ge=1, le=10)
    indextts_repetition_penalty: float = Field(default=10.0, ge=0.0)
    indextts_max_mel_tokens: int = Field(default=1500, ge=50, le=3000)

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="info")
    max_input_chars: int = Field(default=8000)
    default_response_format: Literal[
        "mp3", "opus", "aac", "flac", "wav", "pcm"
    ] = Field(default="mp3")

    @property
    def voices_path(self) -> Path:
        return Path(self.indextts_voices_dir)

    @property
    def resolved_cfg_path(self) -> str:
        if self.indextts_cfg_path:
            return self.indextts_cfg_path
        return str(Path(self.indextts_model_dir) / "config.yaml")

    @property
    def resolved_device(self) -> str:
        import torch

        if self.indextts_device == "auto":
            if torch.cuda.is_available():
                return f"cuda:{self.indextts_cuda_index}"
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return "mps"
            return "cpu"
        if self.indextts_device == "cuda":
            return f"cuda:{self.indextts_cuda_index}"
        return self.indextts_device


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
