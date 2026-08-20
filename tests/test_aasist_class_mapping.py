import unittest
from unittest.mock import Mock, patch
import torch
import numpy as np
from app.ai_voice.aasist_detector import AASISTDetector


class TestAASISTClassMapping(unittest.TestCase):
    """
    Focused unit test suite verifying AASIST class mapping semantics:
    Class 0 = SPOOF / AI-GENERATED
    Class 1 = BONAFIDE / HUMAN
    """

    @patch("app.ai_voice.aasist_detector.torch.load")
    @patch("app.ai_voice.aasist_detector.AASISTModel")
    @patch("os.path.exists", return_value=True)
    def test_class_1_dominant_means_human_voice(self, mock_exists, mock_model_cls, mock_torch_load):
        # Mock checkpoint structure
        mock_torch_load.return_value = {
            "model_config": {"nb_samp": 64600},
            "model_state_dict": {},
            "sample_rate": 16000,
        }

        # Mock PyTorch model forward pass returning logits where Class 1 is overwhelmingly larger [-10.0, 10.0]
        mock_model_inst = Mock()
        mock_model_inst.load_state_dict.return_value = ([], [])
        mock_model_inst.return_value = (None, torch.tensor([[-10.0, 10.0]]))
        mock_model_cls.return_value = mock_model_inst

        detector = AASISTDetector(model_path="dummy_path.pth", threshold=0.5, temperature=1.0, min_rms=0.0)

        # Run prediction on a dummy signal
        dummy_audio = np.ones(64600, dtype=np.float32) * 0.1
        result = detector.predict(dummy_audio)

        # Class 1 dominant -> Human probability ≈ 1.0, AI probability ≈ 0.0, Prediction = bonafide
        self.assertAlmostEqual(result["bonafide_probability"], 1.0, places=4)
        self.assertAlmostEqual(result["spoof_probability"], 0.0, places=4)
        self.assertEqual(result["prediction"], "bonafide")
        self.assertLess(result["spoof_probability"], detector.threshold)

    @patch("app.ai_voice.aasist_detector.torch.load")
    @patch("app.ai_voice.aasist_detector.AASISTModel")
    @patch("os.path.exists", return_value=True)
    def test_class_0_dominant_means_ai_voice(self, mock_exists, mock_model_cls, mock_torch_load):
        # Mock checkpoint structure
        mock_torch_load.return_value = {
            "model_config": {"nb_samp": 64600},
            "model_state_dict": {},
            "sample_rate": 16000,
        }

        # Mock PyTorch model forward pass returning logits where Class 0 is overwhelmingly larger [10.0, -10.0]
        mock_model_inst = Mock()
        mock_model_inst.load_state_dict.return_value = ([], [])
        mock_model_inst.return_value = (None, torch.tensor([[10.0, -10.0]]))
        mock_model_cls.return_value = mock_model_inst

        detector = AASISTDetector(model_path="dummy_path.pth", threshold=0.5, temperature=1.0, min_rms=0.0)

        # Run prediction on a dummy signal
        dummy_audio = np.ones(64600, dtype=np.float32) * 0.1
        result = detector.predict(dummy_audio)

        # Class 0 dominant -> AI probability ≈ 1.0, Human probability ≈ 0.0, Prediction = spoof
        self.assertAlmostEqual(result["spoof_probability"], 1.0, places=4)
        self.assertAlmostEqual(result["bonafide_probability"], 0.0, places=4)
        self.assertEqual(result["prediction"], "spoof")
        self.assertGreaterEqual(result["spoof_probability"], detector.threshold)


if __name__ == "__main__":
    unittest.main()
