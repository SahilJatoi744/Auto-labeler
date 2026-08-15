# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.intelligence import get_intelligence_service
from app.services.model_manager import get_model_manager


class ModelIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.manager = get_model_manager()
        self.intelligence = get_intelligence_service()

    def test_reports_optional_local_integration_status(self):
        status = self.manager.get_advanced_model_status()

        self.assertIn("sam3", status)
        self.assertIn("grounding_dino", status)
        self.assertIn("dinov3", status)
        self.assertEqual(status["sam3"]["weight_file"], "sam3.pt")
        self.assertIn("ready", status["grounding_dino"])
        self.assertIn("install_hint", status["dinov3"])

    def test_model_profiles_emit_runtime_configs_for_real_adapters(self):
        sam3_config = self.intelligence.runtime_config_for_profile("sam3_concept")
        grounding_config = self.intelligence.runtime_config_for_profile("grounding_dino_15")
        dinov3_config = self.intelligence.runtime_config_for_profile("dinov3_quality")

        self.assertEqual(sam3_config["preferred_runtime"], "sam3")
        self.assertTrue(sam3_config["fallback_profile"])
        self.assertEqual(grounding_config["preferred_runtime"], "grounding_dino")
        self.assertEqual(dinov3_config["preferred_runtime"], "dinov3")
        self.assertTrue(dinov3_config["quality_only"])

    def test_optional_sam3_adapter_fails_safely_without_local_weight(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)

        try:
            annotations = self.manager.detect_and_segment_sam3(
                image,
                classes=["person"],
                conf_threshold=0.25,
                allow_download=False,
            )
        except RuntimeError as exc:
            self.assertIn("SAM3", str(exc))
        else:
            self.assertIsInstance(annotations, list)

    def test_prepare_sam3_reports_actionable_state(self):
        result = self.manager.prepare_advanced_model("sam3", allow_download=False)

        self.assertEqual(result["model_key"], "sam3")
        self.assertIn(result["status"], {"ready", "blocked"})
        self.assertIn("message", result)
        self.assertIn("next_steps", result)


if __name__ == "__main__":
    unittest.main()
