from datetime import datetime, timedelta
from typing import Optional

from beanie import Document, Indexed, UpdateResponse, PydanticObjectId
from pydantic import Field

from ..schemas.task import Task, TaskCreation, TaskCompletion


class TaskModel(Document):
    video_path: str

    fps: float | None = None
    total_frames: int | None = None
    width: int | None = None
    height: int | None = None

    status: Indexed(str) = "pending"
    message: str | None = None
    created_at: Indexed(datetime) = Field(default_factory=datetime.utcnow)
    to_process_after: Indexed(datetime) = Field(default_factory=datetime.utcnow)
    processing_at: Indexed(datetime) | None = None
    completed_at: datetime | None = None

    @classmethod
    def from_task_creation_schema(cls, task: TaskCreation) -> "TaskModel":
        return cls(video_path=task.video_path)

    @classmethod
    async def create_task(cls, task: TaskCreation) -> "TaskModel":
        task_doc = cls.from_task_creation_schema(task)
        await task_doc.insert()
        return task_doc

    def to_task_schema(self) -> Task:
        return Task(
            id=str(self.id),
            video_path=self.video_path,
            fps=self.fps,
            total_frames=self.total_frames,
            width=self.width,
            height=self.height,
            status=self.status,
            message=self.message,
            created_at=self.created_at,
            to_process_after=self.to_process_after,
            processing_at=self.processing_at,
            completed_at=self.completed_at,
        )

    @classmethod
    async def get_by_id(cls, task_id: str | PydanticObjectId) -> Optional["TaskModel"]:
        if isinstance(task_id, str):
            task_id = PydanticObjectId(task_id)

        return await cls.find_one(cls.id == task_id)

    @classmethod
    async def get_pending_task(cls) -> Optional["TaskModel"]:
        found_task = (
            await cls.find(cls.status == "pending", cls.to_process_after < datetime.utcnow())
                .sort(+cls.to_process_after)
                .first_or_none()
        )

        if found_task is not None:
            task = (
                await cls.find_one(cls.id == found_task.id, cls.status == "pending")
                    .set(
                        {cls.status: "processing", cls.processing_at: datetime.utcnow()},
                        response_type=UpdateResponse.NEW_DOCUMENT,
                    )
            )
            # check if this task was not taken by another worker
            if task is None:
                return await cls.get_pending_task()
        else:
            task = None

        return task

    @classmethod
    async def requeue(cls, task_id: str | PydanticObjectId) -> None:
        if isinstance(task_id, str):
            task_id = PydanticObjectId(task_id)

        {
            await cls.find_one(cls.id == task_id, cls.status == "processing")
                .set({
                    cls.status: "pending",
                    cls.to_process_after: datetime.utcnow() + timedelta(minutes=5),
                    cls.processing_at: None,
                })
        }

    @classmethod
    async def clear_expired_tasks(cls) -> None:
        (
            await cls.find(
                cls.status == "processing",
                cls.processing_at < datetime.utcnow() - timedelta(minutes=15),
            )
                .set({
                    cls.status: "pending",
                    cls.to_process_after: datetime.utcnow() + timedelta(minutes=5),
                    cls.processing_at: None,
                })
        )

    @classmethod
    async def complete(cls, task: TaskCompletion) -> None:
        if isinstance(task.id, str):
            task_id = PydanticObjectId(task.id)
        else:
            task_id = task.id

        assert task.status in ("completed", "failed")

        (
            await cls.find_one(cls.id == task_id, cls.status == "processing")
                .set({
                    cls.status: task.status,
                    cls.message: task.message,
                    cls.completed_at: datetime.utcnow(),
                    cls.fps: task.fps,
                    cls.total_frames: task.total_frames,
                    cls.width: task.width,
                    cls.height: task.height,
                })
        )
