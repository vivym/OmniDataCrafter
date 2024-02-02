from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import structlog
from fastapi import FastAPI

from .db import init_db
from .middlewares.correlation import CorrelationMiddleware
from .models.task import TaskModel
from .logging import setup_logging
from .schemas.task import TaskResponse, TaskCreation, TaskCompletion
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


@app.post("/task", response_model=TaskResponse)
async def create_task(token: str, task: TaskCreation):
    if token != settings.worker_token:
        return {"task": None}

    if TaskModel is not None:
        task_doc = await TaskModel.create_task(task)
    else:
        task_doc = None

    if task_doc is not None:
        task = task_doc.to_task_schema()
    else:
        task = None

    print("task", task)

    return TaskResponse(task=task)


@app.get("/task", response_model=TaskResponse)
async def get_pending_task(token: str):
    if token != settings.worker_token:
        return {"task": None}

    if TaskModel is not None:
        task_doc = await TaskModel.get_pending_task()
    else:
        task_doc = None

    if task_doc is not None:
        task = task_doc.to_task_schema()
    else:
        task = None

    return TaskResponse(task=task)


@app.get("/tasks/{task_id}/requeue")
async def requeue_task(token: str, task_id: str):
    if token != settings.worker_token:
        return {"task": None}

    await TaskModel.requeue(task_id)

    return {"task": "done"}


@app.put("/tasks/{task_id}")
async def complete_task(
    token: str,
    task_id: str,
    task: TaskCompletion,
):
    if token != settings.worker_token or task_id != task.id:
        return {"task": None}

    await TaskModel.complete(task)

    return {"task": "done"}


@app.get("/ping")
async def ping():
    return {"ping": "pong"}
