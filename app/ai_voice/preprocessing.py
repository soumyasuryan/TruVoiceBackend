import io
import logging
from typing import Tuple, Union
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
) -> Tuple[torch.Tensor, float]:
    """
    Preprocesses audio from various input types (file path, raw bytes, numpy array, or torch tensor)
    into a standardized float32 tensor of shape (1, target_num_samples) at target_sample_rate (16,000 Hz).

    Returns:
        (tensor_out, rms_energy)
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

    # Resample if current sample rate differs from target sample rate
    if sr != target_sample_rate and len(waveform_np) > 0:
        waveform_np = librosa.resample(y=waveform_np, orig_sr=sr, target_sr=target_sample_rate)

    # Calculate RMS energy of raw audio content BEFORE padding/cropping
    rms_energy = float(np.sqrt(np.mean(waveform_np**2))) if len(waveform_np) > 0 else 0.0

    # Normalize integer PCM if max amplitude exceeds 1.0
    max_val = np.max(np.abs(waveform_np)) if len(waveform_np) > 0 else 0.0
    if max_val > 1.0:
        waveform_np = waveform_np / max_val

    # Apply 5ms boundary fading to avoid artificial step-discontinuities
    cur_len = len(waveform_np)
    if cur_len > 100:
        fade_len = min(int(target_sample_rate * 0.005), cur_len // 4)
        fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)
        waveform_np[:fade_len] *= fade_in
        waveform_np[-fade_len:] *= fade_out

    # Handle short or long audio lengths (smooth zero-padding or center cropping)
    if cur_len == 0:
        padded_np = np.zeros(target_num_samples, dtype=np.float32)
    elif cur_len < target_num_samples:
        # Zero-pad short audio to target length to prevent boundary phase glitches
        padded_np = np.zeros(target_num_samples, dtype=np.float32)
        padded_np[:cur_len] = waveform_np
    else:
        # Crop to target length (take center slice for best representation)
        start_idx = (cur_len - target_num_samples) // 2
        padded_np = waveform_np[start_idx : start_idx + target_num_samples]

    tensor_out = torch.from_numpy(padded_np).float().unsqueeze(0)  # Shape (1, target_num_samples)
    return tensor_out, rms_energy
