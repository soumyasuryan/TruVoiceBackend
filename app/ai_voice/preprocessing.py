import io
import logging
from typing import Union
import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
DEFAULT_NUM_SAMPLES = 64600  # AASIST training waveform length (64,600 samples ~ 4.0375 sec)


def preprocess_audio_waveform(
    audio_input: Union[str, bytes, np.ndarray, torch.Tensor],
    sample_rate: int = TARGET_SAMPLE_RATE,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    target_num_samples: int = DEFAULT_NUM_SAMPLES,
) -> torch.Tensor:
    """
    Preprocesses audio from various input types (file path, raw bytes, numpy array, or torch tensor)
    into a standardized float32 tensor of shape (1, target_num_samples) at target_sample_rate (16,000 Hz).
    """
    waveform_np: np.ndarray | None = None

    if isinstance(audio_input, str):
        # File path
        waveform_np, sr = librosa.load(audio_input, sr=target_sample_rate, mono=True)
    elif isinstance(audio_input, bytes):
        # Bytes buffer
        buffer = io.BytesIO(audio_input)
        waveform_np, sr = librosa.load(buffer, sr=target_sample_rate, mono=True)
    elif isinstance(audio_input, torch.Tensor):
        tensor = audio_input.detach().cpu().float()
        if tensor.ndim > 1:
            tensor = tensor.mean(dim=0)  # Convert multi-channel to mono
        waveform_np = tensor.numpy()
        sr = sample_rate
    elif isinstance(audio_input, np.ndarray):
        waveform_np = audio_input.astype(np.float32)
        if waveform_np.ndim > 1:
            waveform_np = np.mean(waveform_np, axis=0)  # Convert stereo/multi-channel to mono
        sr = sample_rate
    else:
        raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

    # Ensure float32 dtype and normalized [-1.0, 1.0] range
    if waveform_np.dtype != np.float32:
        waveform_np = waveform_np.astype(np.float32)

    # Normalize integer PCM if max amplitude exceeds 1.0
    max_val = np.max(np.abs(waveform_np))
    if max_val > 1.0:
        waveform_np = waveform_np / max_val

    # Resample if current sample rate differs from target sample rate
    if sr != target_sample_rate and len(waveform_np) > 0:
        waveform_np = librosa.resample(y=waveform_np, orig_sr=sr, target_sr=target_sample_rate)

    # Handle short or long audio lengths (padding / repeating / cropping)
    cur_len = len(waveform_np)
    if cur_len == 0:
        waveform_np = np.zeros(target_num_samples, dtype=np.float32)
    elif cur_len < target_num_samples:
        # Pad by repeating signal or zero padding if very short
        num_repeats = int(np.ceil(target_num_samples / cur_len))
        waveform_np = np.tile(waveform_np, num_repeats)[:target_num_samples]
    elif cur_len > target_num_samples:
        # Crop to target length (take center slice for best representation)
        start_idx = (cur_len - target_num_samples) // 2
        waveform_np = waveform_np[start_idx : start_idx + target_num_samples]

    tensor_out = torch.from_numpy(waveform_np).float().unsqueeze(0)  # Shape (1, target_num_samples)
    return tensor_out
