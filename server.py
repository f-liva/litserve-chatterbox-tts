import base64
import io
import os
import re
import tempfile
from contextlib import contextmanager
from typing import Optional, Tuple

import requests
import torch
import torchaudio as ta
from fastapi.responses import Response
from litserve import LitAPI, LitServer
from pydantic import BaseModel, Field, field_validator

MODEL_TYPES = ("original", "turbo", "multilingual")


@contextmanager
def patched_torch_load(device):
    """Context manager that patches torch.load to use map_location.

    This fixes RuntimeError when loading CUDA checkpoints on CPU-only machines.
    The chatterbox library doesn't pass map_location to torch.load, so we patch it.
    Works correctly on both CPU and GPU systems.
    """
    original_load = torch.load

    def patched_load(*args, **kwargs):
        if "map_location" not in kwargs:
            kwargs["map_location"] = device
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


def load_model(model_type: str, device: str):
    """Load the appropriate Chatterbox model based on model_type."""
    if model_type == "turbo":
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        return ChatterboxTurboTTS.from_pretrained(device=device)
    elif model_type == "multilingual":
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        return ChatterboxMultilingualTTS.from_pretrained(device=device)
    else:
        from chatterbox.tts import ChatterboxTTS

        return ChatterboxTTS.from_pretrained(device=device)


class TTSRequest(BaseModel):
    text: str = Field(
        ..., min_length=1, max_length=500, description="Input text to synthesize"
    )
    audio_prompt: Optional[str] = Field(
        None, description="Base64 audio, URL, or file path"
    )
    exaggeration: float = Field(0.5, ge=0.0, le=1.0)
    cfg: float = Field(0.5, ge=0.0, le=1.0)
    temperature: float = Field(0.8, ge=0.0, le=1.0)
    language_id: Optional[str] = Field(
        None, description="Language code for multilingual model (e.g. 'fr', 'zh', 'it')"
    )

    @field_validator("audio_prompt")
    def validate_audio_prompt(cls, v):
        if v is None:
            return v

        is_url = re.match(r"^https?://", v)
        is_base64 = (
            re.match(r"^[A-Za-z0-9+/=]+\Z", v) and len(v) > 100
        )  # Basic base64 check

        if is_url or is_base64:
            return v

        raise ValueError("audio_prompt must be a base64 string or valid http/https URL")

    def get_audio_tempfile(self) -> Optional[str]:
        if self.audio_prompt is None:
            return None

        if re.match(r"^https?://", self.audio_prompt):
            # Download from URL
            resp = requests.get(self.audio_prompt)
            resp.raise_for_status()
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            tmp_file.write(resp.content)
            tmp_file.close()
            return tmp_file.name

        if (
            re.match(r"^[A-Za-z0-9+/=]+\Z", self.audio_prompt)
            and len(self.audio_prompt) > 100
        ):
            # Base64 string
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            padded = self.audio_prompt + "=" * (-len(self.audio_prompt) % 4)
            decoded = base64.b64decode(padded)
            tmp_file.write(decoded)
            tmp_file.close()
            return tmp_file.name

        # Assume local file path
        return self.audio_prompt


class ChatterboxTTSAPI(LitAPI):
    """LitServe API for Chatterbox TTS models.

    Supports original, turbo, and multilingual models via MODEL_TYPE env var.
    """

    def setup(self, device):
        """Initialize the Chatterbox TTS model."""
        self.model_type = os.environ.get("MODEL_TYPE", "original").lower()
        if self.model_type not in MODEL_TYPES:
            raise ValueError(
                f"Invalid MODEL_TYPE '{self.model_type}'. Must be one of: {MODEL_TYPES}"
            )
        with patched_torch_load(device):
            self.model = load_model(self.model_type, device)
        self.temp_files = []

    def decode_request(self, request: TTSRequest) -> dict:
        """Decode request using TTSRequest model."""
        audio_prompt_path = request.get_audio_tempfile()

        # Track temp files for cleanup
        if audio_prompt_path and audio_prompt_path != request.audio_prompt:
            self.temp_files.append(audio_prompt_path)

        return {
            "text": request.text,
            "audio_prompt_path": audio_prompt_path,
            "exaggeration": request.exaggeration,
            "cfg": request.cfg,
            "temperature": request.temperature,
            "language_id": request.language_id,
        }

    def predict(self, inputs: dict) -> bytes:
        """Generate speech audio using the loaded Chatterbox model."""
        try:
            kwargs = {"audio_prompt_path": inputs["audio_prompt_path"]}

            if self.model_type == "original":
                kwargs["exaggeration"] = inputs["exaggeration"]
                kwargs["cfg_weight"] = inputs["cfg"]
                kwargs["temperature"] = inputs["temperature"]
            elif self.model_type == "multilingual" and inputs["language_id"]:
                kwargs["language_id"] = inputs["language_id"]

            wav = self.model.generate(inputs["text"], **kwargs)

            buffer = io.BytesIO()
            ta.save(buffer, wav, self.model.sr, format="wav")
            return buffer.getvalue()
        finally:
            self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except OSError:
                pass
        self.temp_files.clear()

    def encode_response(self, output: bytes) -> Response:
        """Package the generated audio data into a response."""
        return Response(
            content=output,
            headers={
                "Content-Type": "audio/wav",
                "Content-Disposition": "attachment; filename=generated_speech.wav",
            },
        )


if __name__ == "__main__":
    # Determine accelerator: use DEVICE env var if set, otherwise auto-detect
    device_env = os.environ.get("DEVICE", "").lower()
    if device_env == "cpu":
        accelerator = "cpu"
    elif device_env in ("cuda", "gpu"):
        accelerator = "cuda"
    else:
        accelerator = "auto"

    api = ChatterboxTTSAPI(api_path="/speech")
    server = LitServer(api, accelerator=accelerator, timeout=100)
    server.run(port=8000)
