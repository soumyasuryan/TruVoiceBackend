import logging
import os
from typing import Any, Dict, Union
import numpy as np
import torch
from app.ai_voice.aasist_model import Model as AASISTModel
from app.ai_voice.preprocessing import preprocess_audio_waveform
from app.config import settings

logger = logging.getLogger(__name__)


class AASISTDetector:
    """
    Dedicated AASIST PyTorch Deepfake / AI-Voice Detector abstraction.
    Handles checkpoint loading, device allocation, preprocessing, silence detection, and temperature-calibrated inference.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        threshold: float = None,
        temperature: float = None,
        min_rms: float = None,
    ):
        if not os.path.exists(model_path):
            error_msg = f"AASIST model checkpoint not found: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.threshold = float(threshold if threshold is not None else settings.AASIST_THRESHOLD)
        self.temperature = float(temperature if temperature is not None else settings.AASIST_TEMPERATURE)
        self.min_rms = float(min_rms if min_rms is not None else settings.AASIST_MIN_RMS)
        self.model_path = model_path

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            error_msg = f"Failed to load AASIST checkpoint file at '{model_path}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint or "model_config" not in checkpoint:
            error_msg = f"Invalid AASIST checkpoint format at '{model_path}'. Required keys 'model_state_dict' and 'model_config'."
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.config = checkpoint["model_config"]
        self.sample_rate = checkpoint.get("sample_rate", 16000)
        self.num_samples = self.config.get("nb_samp", 64600)

        # Instantiate PyTorch model and load state_dict
        self.model = AASISTModel(self.config)
        try:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if missing_keys or unexpected_keys:
                logger.warning(f"AASIST load_state_dict warnings - Missing: {missing_keys}, Unexpected: {unexpected_keys}")
        except Exception as e:
            error_msg = f"State dict mismatch loading AASIST model: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

        self.model.to(self.device)
        self.model.eval()

        logger.info(f"AASIST model loaded successfully from {model_path}")
        logger.info(f"AASIST device: {self.device}")
        logger.info(f"AASIST sample rate: {self.sample_rate}")
        logger.info(f"AASIST input samples: {self.num_samples}")
        logger.info(f"AASIST threshold: {self.threshold}")
        logger.info(f"AASIST temperature: {self.temperature}")
        logger.info(f"AASIST min RMS: {self.min_rms}")

    def predict(
        self,
        audio_input: Union[str, bytes, np.ndarray, torch.Tensor],
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """
        Executes AASIST deepfake voice detection on input audio.
        Returns dictionary containing spoof_probability, bonafide_probability, and prediction.
        """
        waveform, rms_energy = preprocess_audio_waveform(
            audio_input=audio_input,
            sample_rate=sample_rate,
            target_sample_rate=self.sample_rate,
            target_num_samples=self.num_samples,
        )

        # Silence / Low Energy Gate: If clip contains no active voice speech, return bonafide (safe)
        if rms_energy < self.min_rms:
            logger.debug(f"Audio RMS energy ({rms_energy:.6f}) below minimum threshold ({self.min_rms}); marking bonafide.")
            return {
                "spoof_probability": 0.0,
                "bonafide_probability": 1.0,
                "prediction": "bonafide",
                "rms_energy": rms_energy,
            }

        waveform = waveform.to(self.device)

        with torch.inference_mode():
            _, logits = self.model(waveform)
            # Temperature scaling on logits to prevent extreme probability saturation on live audio
            scaled_logits = logits / self.temperature
            probabilities = torch.softmax(scaled_logits, dim=1)
            bonafide_prob = float(probabilities[0, 0].item())
            spoof_prob = float(probabilities[0, 1].item())

        prediction = "spoof" if spoof_prob >= self.threshold else "bonafide"

        return {
            "spoof_probability": spoof_prob,
            "bonafide_probability": bonafide_prob,
            "prediction": prediction,
            "rms_energy": rms_energy,
        }
