# app/core/model_loader.py

import torch
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)


def load_yolo_model(model_path: str):
    """
    Load YOLO model with PyTorch 2.6+ compatibility fix
    
    Args:
        model_path: Path to the YOLO model file
        
    Returns:
        YOLO model instance
    """
    try:
        # Fix for PyTorch 2.6+ weights_only=True default
        # This is required for ultralytics models
        if hasattr(torch.serialization, 'add_safe_globals'):
            from ultralytics.nn.tasks import DetectionModel, SegmentationModel, ClassificationModel
            torch.serialization.add_safe_globals([
                DetectionModel,
                SegmentationModel, 
                ClassificationModel
            ])
            logger.info("PyTorch safe globals configured for YOLO models")
        
        # Load the model
        model = YOLO(model_path)
        logger.info(f"YOLO model loaded successfully from: {model_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Failed to load YOLO model from {model_path}: {str(e)}")
        raise


def check_model_compatibility():
    """Check PyTorch version and model compatibility"""
    import sys
    
    pytorch_version = torch.__version__
    python_version = sys.version
    
    logger.info(f"PyTorch version: {pytorch_version}")
    logger.info(f"Python version: {python_version}")
    
    # Check if weights_only is the issue
    if hasattr(torch, 'load'):
        import inspect
        sig = inspect.signature(torch.load)
        if 'weights_only' in sig.parameters:
            default = sig.parameters['weights_only'].default
            logger.info(f"torch.load weights_only default: {default}")
    
    return True