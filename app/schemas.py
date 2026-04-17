from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, conlist, model_validator


ResponseFormat = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class GenerationParams(BaseModel):
    """Per-request overrides for IndexTTS2 sampling hyperparameters."""

    do_sample: Optional[bool] = Field(default=None)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    temperature: Optional[float] = Field(default=None, gt=0.0, le=5.0)
    length_penalty: Optional[float] = Field(default=None)
    num_beams: Optional[int] = Field(default=None, ge=1, le=10)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.0)
    max_mel_tokens: Optional[int] = Field(default=None, ge=50, le=3000)
    max_text_tokens_per_segment: Optional[int] = Field(default=None, ge=20, le=600)
    interval_silence: Optional[int] = Field(default=None, ge=0, le=2000)


class SpeechRequest(BaseModel):
    """OpenAI-compatible `/v1/audio/speech` request — zero-shot voice cloning."""

    model: Optional[str] = Field(default=None, description="Accepted for OpenAI compatibility; ignored.")
    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(..., description="Voice id matching a file pair in the voices directory.")
    response_format: ResponseFormat = Field(default="mp3")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Accepted for OpenAI compatibility; ignored (IndexTTS2 has no speed control).")
    generation: Optional[GenerationParams] = Field(default=None)


class EmotionRequest(BaseModel):
    """`/v1/audio/emotion` request — emotion-conditioned cloning.

    Exactly one of `emotion_voice`, `emotion_vector`, `emotion_text` must be provided.
    """

    input: str = Field(..., description="Text to synthesize.")
    voice: str = Field(..., description="Speaker voice id (timbre reference).")
    response_format: ResponseFormat = Field(default="mp3")

    emotion_voice: Optional[str] = Field(
        default=None,
        description="Voice id used as an emotion reference audio. The .txt is ignored.",
    )
    emotion_vector: Optional[conlist(float, min_length=8, max_length=8)] = Field(
        default=None,
        description="8-dim emotion weights: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]. Each value in [0,1].",
    )
    emotion_text: Optional[str] = Field(
        default=None,
        description="Natural-language emotion description; converted to an emotion vector by the built-in Qwen classifier.",
    )
    emotion_alpha: float = Field(default=1.0, ge=0.0, le=1.0, description="Blend weight against the speaker timbre (higher = stronger emotion).")
    use_random: bool = Field(default=False, description="Randomize emotion sampling; may reduce timbre fidelity.")

    generation: Optional[GenerationParams] = Field(default=None)

    @model_validator(mode="after")
    def _one_of_emotion(self):
        sources = [self.emotion_voice, self.emotion_vector, self.emotion_text]
        set_count = sum(x is not None for x in sources)
        if set_count != 1:
            raise ValueError(
                "exactly one of emotion_voice, emotion_vector, emotion_text must be set"
            )
        if self.emotion_vector is not None:
            for v in self.emotion_vector:
                if v < 0.0 or v > 1.0:
                    raise ValueError("each emotion_vector element must be in [0,1]")
        return self


class VoiceInfo(BaseModel):
    id: str
    preview_url: str
    prompt_text: str


class VoiceList(BaseModel):
    object: Literal["list"] = "list"
    data: list[VoiceInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    model: str
    device: Optional[str] = None
    sample_rate: Optional[int] = None
