# ==========================================
# Created by Sahil Jatoi (SJ)
# AutoLabeler - AI Image Dataset Labeling
# ==========================================

"""
Configuration module for AutoLabeler backend.
Centralizes all settings and environment variables.
"""

from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # App Info
    APP_NAME: str = "AutoLabeler"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Server Settings
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = Field(default=Path("uploads"))
    OUTPUT_DIR: Path = Field(default=Path("outputs"))
    MODELS_DIR: Path = Field(default=Path("models"))
    LOGS_DIR: Path = Field(default=Path("logs"))
    
    # Processing Settings
    BATCH_SIZE: int = Field(default=16, env="BATCH_SIZE")
    MAX_WORKERS: int = Field(default=4, env="MAX_WORKERS")
    MAX_IMAGE_SIZE: int = Field(default=2048, env="MAX_IMAGE_SIZE")
    SUPPORTED_FORMATS: List[str] = Field(default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"])
    
    # GPU Settings
    USE_GPU: bool = Field(default=True, env="USE_GPU")
    GPU_DEVICE: int = Field(default=0, env="GPU_DEVICE")
    MIXED_PRECISION: bool = Field(default=True, env="MIXED_PRECISION")
    
    # Model Settings
    DEFAULT_CONFIDENCE_THRESHOLD: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    NMS_IOU_THRESHOLD: float = Field(default=0.45, env="NMS_IOU_THRESHOLD")
    MAX_DETECTIONS_PER_IMAGE: int = Field(default=300, env="MAX_DETECTIONS_PER_IMAGE")
    
    # Model Weights URLs
    # Legacy models (kept for backward compatibility)
    YOLOV8_MODEL: str = Field(default="yolov8x-seg.pt")
    SAM_MODEL: str = Field(default="sam_vit_h.pth")
    SAM_MODEL_URL: str = Field(default="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
    
    # YOLOv26 Models (Mappings to robust supported models)
    YOLOV26_MODEL: str = Field(default="yolov8x-seg.pt")  # Using YOLOv8x-seg as high-quality fallback
    YOLOV26_DETECT_MODEL: str = Field(default="yolov8x.pt")
    
    # SAM2 Models (Native Ultralytics Support)
    SAM2_MODEL: str = Field(default="sam2_l.pt")
    SAM2_MODEL_URL: str = Field(default="https://github.com/ultralytics/assets/releases/download/v8.2.0/sam2_l.pt")

    # Advanced optional local integrations. These are disabled unless the
    # required packages and model files/cache are available locally.
    ALLOW_MODEL_DOWNLOADS: bool = Field(default=False, env="ALLOW_MODEL_DOWNLOADS")
    SAM3_MODEL: str = Field(default="sam3.pt", env="SAM3_MODEL")
    GROUNDING_DINO_MODEL_ID: str = Field(default="IDEA-Research/grounding-dino-base", env="GROUNDING_DINO_MODEL_ID")
    DINOV3_MODEL_ID: str = Field(default="facebook/dinov3-vits16-pretrain-lvd1689m", env="DINOV3_MODEL_ID")
    
    # Labeling Settings
    MIN_CONFIDENCE_FOR_AUTO_ACCEPT: float = 0.85
    UNCERTAINTY_THRESHOLD_LOW: float = 0.3
    UNCERTAINTY_THRESHOLD_HIGH: float = 0.7
    
    # Default class alias map — maps user-friendly group names to COCO model class names.
    # This is used as a fallback when per-job aliases are not configured.
    # Override via environment variable or .env file (JSON string).
    DEFAULT_ALIAS_MAP: dict = Field(default={
        "vehicle": ["car", "bus", "truck", "motorcycle", "train"],
        "person": ["person", "human", "pedestrian"],
        "building": ["house", "building", "skyscraper"],
        "plant": ["tree", "flower", "grass", "bush"],
        "animal": ["dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"],
        "furniture": ["chair", "sofa", "bed", "dining table", "toilet"],
        "appliance": ["tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator"],
        "drivable area": ["road", "street", "asphalt", "highway", "driveway"],
        "sidewalk": ["pavement", "walkway", "side walk", "footpath"],
        "alley": ["alleyway", "narrow street", "backstreet", "passage"]
    })
    
    # Export Settings
    DEFAULT_EXPORT_FORMAT: str = "coco"
    DEFAULT_SPLIT_RATIOS: dict = Field(default={"train": 0.7, "val": 0.15, "test": 0.15})
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.UPLOAD_DIR = self.BASE_DIR / self.UPLOAD_DIR
        self.OUTPUT_DIR = self.BASE_DIR / self.OUTPUT_DIR
        self.MODELS_DIR = self.BASE_DIR / self.MODELS_DIR
        self.LOGS_DIR = self.BASE_DIR / self.LOGS_DIR
        
        for dir_path in [self.UPLOAD_DIR, self.OUTPUT_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings
