"""Custom exceptions for Voice Engine."""


class VoiceEngineError(Exception):
    """Base exception for Voice Engine errors."""
    pass


class ModelNotFoundError(VoiceEngineError):
    """Raised when a model file cannot be found."""
    
    def __init__(self, model_path: str, message: str = None):
        self.model_path = model_path
        self.message = message or f"Model not found: {model_path}"
        super().__init__(self.message)


class ModelLoadError(VoiceEngineError):
    """Raised when a model fails to load."""
    
    def __init__(self, model_name: str, reason: str = None):
        self.model_name = model_name
        self.reason = reason
        message = f"Failed to load model '{model_name}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class AudioProcessingError(VoiceEngineError):
    """Raised when audio processing fails."""
    
    def __init__(self, operation: str, reason: str = None):
        self.operation = operation
        self.reason = reason
        message = f"Audio processing failed during '{operation}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class TTSError(VoiceEngineError):
    """Raised when TTS generation fails."""
    
    def __init__(self, engine: str, reason: str = None):
        self.engine = engine
        self.reason = reason
        message = f"TTS generation failed using '{engine}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class ConversionError(VoiceEngineError):
    """Raised when voice conversion fails."""
    
    def __init__(self, reason: str = None):
        self.reason = reason
        message = "Voice conversion failed"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class SeparationError(VoiceEngineError):
    """Raised when vocal/instrumental separation fails."""
    
    def __init__(self, model: str, reason: str = None):
        self.model = model
        self.reason = reason
        message = f"Vocal separation failed using '{model}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class YouTubeError(VoiceEngineError):
    """Raised when YouTube operations fail."""
    
    def __init__(self, operation: str, reason: str = None):
        self.operation = operation
        self.reason = reason
        message = f"YouTube {operation} failed"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class TrainingError(VoiceEngineError):
    """Raised when training operations fail."""
    
    def __init__(self, stage: str, reason: str = None):
        self.stage = stage
        self.reason = reason
        message = f"Training failed at stage '{stage}'"
        if reason:
            message += f": {reason}"
        super().__init__(message)


class ValidationError(VoiceEngineError):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"Validation error for '{field}': {reason}")
