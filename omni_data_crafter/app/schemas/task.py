from datetime import datetime

from pydantic import BaseModel, Field


class Task(BaseModel):
    id: str

    status: str = "pending"
    message: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    to_process_after: datetime = Field(default_factory=datetime.utcnow)
    processing_at: datetime | None = None
    completed_at: datetime | None = None


class CutDetectionTask(Task):
    video_path: str

    fps: float | None = None
    total_frames: int | None = None
    width: int | None = None
    height: int | None = None
    cuts: list[tuple[int, int]] | None = None


class CutDetectionTaskResponse(BaseModel):
    task: CutDetectionTask | None


class CutDetectionTaskCreation(BaseModel):
    video_path: str


class CutDetectionCompletion(BaseModel):
    id: str
    status: str
    message: str | None = None
    fps: float | None = None
    total_frames: int | None = None
    width: int | None = None
    height: int | None = None
    cuts: list[tuple[int, int]] | None = None
