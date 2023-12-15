from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI

from .db import init_db
from .middlewares.correlation import CorrelationMiddleware
from .models.task import CutDetectionTaskModel
from .logging import setup_logging
from .schemas.task import (
    CutDetectionTaskResponse, CutDetectionTaskCreation, CutDetectionCompletion
)
from .settings import settings

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

logger: "BoundLogger" = structlog.get_logger()


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    logger.info("Database connecting...")
    await init_db()
    logger.info("Database connected.")
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(CorrelationMiddleware)


def get_task_model_by_type(task_type: str) -> type[CutDetectionTaskModel] | None:
    if task_type == "cut_detection":
        return CutDetectionTaskModel
    else:
        return None


@app.post("/task", response_model=CutDetectionTaskResponse)
async def create_task(task_type: str, token: str, task: CutDetectionTaskCreation):
    if token != settings.worker_token:
        return {"task": None}

    TaskModel = get_task_model_by_type(task_type)

    if TaskModel is not None:
        task_doc = await TaskModel.create_task(task)
    else:
        task_doc = None

    if task_doc is not None:
        task = task_doc.to_task_schema()
    else:
        task = None

    print("task", task)

    return CutDetectionTaskResponse(task=task)


@app.get("/task", response_model=CutDetectionTaskResponse)
async def get_pending_task(task_type: str, token: str):
    if token != settings.worker_token:
        return {"task": None}

    TaskModel = get_task_model_by_type(task_type)

    if TaskModel is not None:
        task_doc = await TaskModel.get_pending_task()
    else:
        task_doc = None

    if task_doc is not None:
        task = task_doc.to_task_schema()
    else:
        task = None

    return CutDetectionTaskResponse(task=task)


@app.get("/tasks/{task_id}/requeue")
async def requeue_task(task_type: str, token: str, task_id: str):
    if token != settings.worker_token:
        return {"task": None}

    TaskModel = get_task_model_by_type(task_type)

    await TaskModel.requeue(task_id)

    return {"task": "done"}


@app.put("/tasks/{task_id}")
async def complete_task(
    task_type: str,
    token: str,
    task_id: str,
    task: CutDetectionCompletion,
):
    if token != settings.worker_token or task_id != task.id:
        return {"task": None}

    TaskModel = get_task_model_by_type(task_type)

    await TaskModel.complete(task)

    return {"task": "done"}


@app.get("/ping")
async def ping():
    return {"ping": "pong"}
