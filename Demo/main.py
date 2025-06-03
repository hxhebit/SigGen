#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Signal Generation Demo

Description:
    This script loads two images, extracts their features using a pre-trained model,
    performs feature fusion, and saves the generated image.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.utils import save_image

# Constants
DEFAULT_IMAGE_SIZE = (128, 128)
FEATURE_SPLIT_INDEX = 512
PATHS = {
    "person": "person1_wave/[Person1]_Click.png",
    "gesture": "person1_wave/Person2_[Wave].png",
    "model": "checkpoint.pth",
    "output": "person1_wave/[Person1]-[Wave].png"  
}
# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImageProcessor:
    """Handles image loading and preprocessing"""

    @staticmethod
    def load_image(path: str) -> Image.Image:
        """Load and validate image"""
        try:
            img = Image.open(path).convert('RGB')
            logger.info(f"Successfully loaded image: {path}")
            return img
        except Exception as e:
            logger.error(f"Failed to load image {path}: {str(e)}")
            raise

    @staticmethod
    def get_transform() -> T.Compose:
        """Create image transformation pipeline"""
        return T.Compose([
            T.Resize(DEFAULT_IMAGE_SIZE),
            T.ToTensor(),
        ])

    @staticmethod
    def preprocess(img: Image.Image, transform: T.Compose) -> torch.Tensor:
        """Apply transformations and move to GPU"""
        return transform(img).unsqueeze(0).cuda()


class FeatureFusion:
    """Handles model inference and feature manipulation"""

    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.model.eval()

    @staticmethod
    def _load_model(path: str) -> torch.nn.Module:
        """Load and validate PyTorch model"""
        try:
            model = torch.load(path).cuda()
            logger.info(f"Successfully loaded model from {path}")
            return model
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def extract_features(self, img_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run model inference"""
        with torch.no_grad():
            _, z, z_padding = self.model(img_tensor)
        return z, z_padding

    @staticmethod
    def split_features(
        features: torch.Tensor, 
        padding: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Split features and padding tensors"""
        return (
            (features[:, :, :FEATURE_SPLIT_INDEX], features[:, :, FEATURE_SPLIT_INDEX:]),
            (padding[:, :, :FEATURE_SPLIT_INDEX], padding[:, :, FEATURE_SPLIT_INDEX:])
        )

    def fuse_features(
        self,
        features_a: Tuple[torch.Tensor, torch.Tensor],
        features_b: Tuple[torch.Tensor, torch.Tensor],
        padding_a: Tuple[torch.Tensor, torch.Tensor],
        padding_b: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Combine features"""
        (a1, a2), (a1_pad, a2_pad) = features_a, padding_a
        (b1, b2), (b1_pad, b2_pad) = features_b, padding_b

        # Only create A2B1 combination
        a2b1 = torch.cat((b1, a2), dim=2)
        a2b1_pad = torch.cat((b1_pad, a2_pad), dim=2)

        return self.model.decoder_combine(a2b1, a2b1_pad)


def main():
    """Main execution pipeline"""
    try:
        logger.info("Starting feature fusion pipeline")

        # Initialize components
        processor = ImageProcessor()
        fusion = FeatureFusion(PATHS["model"])
        transform = processor.get_transform()

        # Load and preprocess images
        images = {
            "A": processor.preprocess(
                processor.load_image(PATHS["gesture"]),
                transform
            ),
            "B": processor.preprocess(
                processor.load_image(PATHS["person"]),
                transform
            )
        }

        # Feature extraction
        features = {
            "A": fusion.extract_features(images["A"]),
            "B": fusion.extract_features(images["B"])
        }

        # Feature splitting
        split_features = {
            "A": fusion.split_features(*features["A"]),
            "B": fusion.split_features(*features["B"])
        }

        # Generate feature combination
        generated_image = fusion.fuse_features(
            split_features["A"][0], split_features["B"][0],
            split_features["A"][1], split_features["B"][1]
        )

        # Save result
        save_image(generated_image, PATHS["output"])
        logger.info(f"Saved output image: {PATHS['output']}")

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()