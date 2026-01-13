"""YouTube request/response models."""

from typing import Optional, List
from pydantic import BaseModel, Field


class YouTubeSearchRequest(BaseModel):
    """YouTube search request."""
    query: str = Field(..., description="Search query", max_length=500)
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results to return")


class YouTubeSearchResult(BaseModel):
    """Single YouTube search result."""
    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    channel: str = Field(..., description="Channel name")
    duration: Optional[str] = Field(default=None, description="Video duration")
    thumbnail: Optional[str] = Field(default=None, description="Thumbnail URL")
    view_count: Optional[int] = Field(default=None, description="View count")


class YouTubeSearchResponse(BaseModel):
    """YouTube search response."""
    results: List[YouTubeSearchResult] = Field(..., description="Search results")
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Number of results returned")


class YouTubeDownloadRequest(BaseModel):
    """YouTube download request."""
    video_id: str = Field(..., description="YouTube video ID")
    format: str = Field(default="wav", description="Output audio format")
    quality: str = Field(default="best", description="Audio quality: 'best', '192k', '128k'")


class YouTubeDownloadResponse(BaseModel):
    """YouTube download response."""
    audio: str = Field(..., description="Base64 encoded audio data")
    sample_rate: int = Field(..., description="Audio sample rate")
    format: str = Field(..., description="Audio format")
    video_id: str = Field(..., description="YouTube video ID")
    title: Optional[str] = Field(default=None, description="Video title")
    duration: Optional[float] = Field(default=None, description="Duration in seconds")


class YouTubeVideoInfo(BaseModel):
    """YouTube video information."""
    video_id: str = Field(..., description="YouTube video ID")
    title: str = Field(..., description="Video title")
    channel: str = Field(..., description="Channel name")
    duration: float = Field(..., description="Duration in seconds")
    thumbnail: Optional[str] = Field(default=None, description="Thumbnail URL")
    description: Optional[str] = Field(default=None, description="Video description")
