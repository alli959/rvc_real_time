"""
Model Manager Module - Handles RVC model loading and inference
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages RVC model loading and inference"""
    
    def __init__(self, model_dir: str = "assets/models"):
        """
        Initialize model manager
        
        Args:
            model_dir: Directory containing RVC models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_model: Optional[torch.nn.Module] = None
        self.model_name: Optional[str] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
    
    def load_model(self, model_path: str) -> bool:
        """
        Load an RVC model from file
        
        Args:
            model_path: Path to model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_file = self.model_dir / model_path
            
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                return False
            
            # Load model checkpoint
            checkpoint = torch.load(model_file, map_location=self.device)
            
            # For demonstration, we'll use a placeholder model structure
            # In production, this would load the actual RVC model architecture
            self.current_model = self._create_model_from_checkpoint(checkpoint)
            self.model_name = model_path
            
            logger.info(f"Successfully loaded model: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _create_model_from_checkpoint(self, checkpoint: Dict[str, Any]) -> torch.nn.Module:
        """
        Create model from checkpoint
        
        Args:
            checkpoint: Model checkpoint dictionary
            
        Returns:
            Model instance
        """
        # Placeholder for actual RVC model architecture
        # In production, this would instantiate the proper model class
        class PlaceholderRVCModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv1d(1, 1, kernel_size=3, padding=1)
            
            def forward(self, x):
                # Simple passthrough for demonstration
                return x
        
        model = PlaceholderRVCModel()
        
        # Load state dict if available
        if 'model_state_dict' in checkpoint:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                logger.warning("Could not load model state dict, using random initialization")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def infer(self, audio_features: np.ndarray) -> np.ndarray:
        """
        Perform inference on audio features
        
        Args:
            audio_features: Input audio features
            
        Returns:
            Converted audio features
        """
        if self.current_model is None:
            logger.warning("No model loaded, returning input as-is")
            return audio_features
        
        try:
            # Convert to tensor
            input_tensor = torch.from_numpy(audio_features).float()
            
            # Add batch dimension if needed
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
            elif input_tensor.dim() == 2:
                input_tensor = input_tensor.unsqueeze(0)
            
            input_tensor = input_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                output_tensor = self.current_model(input_tensor)
            
            # Convert back to numpy
            output = output_tensor.cpu().numpy().squeeze()
            
            return output
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            return audio_features
    
    def list_available_models(self) -> list:
        """
        List all available models in the model directory
        
        Returns:
            List of model filenames
        """
        model_files = []
        for ext in ['*.pth', '*.pt', '*.ckpt']:
            model_files.extend(self.model_dir.glob(ext))
        
        return [f.name for f in model_files]
    
    def unload_model(self):
        """Unload current model and free memory"""
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            self.model_name = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded")
