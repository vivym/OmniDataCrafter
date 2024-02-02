import argparse
import io
import time
import traceback
import queue
import signal
import pickle
import multiprocessing as mp
from bisect import bisect_left
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from pathlib import Path

import requests
import structlog
import torch
import lz4.frame as lz4

from omni_data_crafter.app.logging import setup_logging
from omni_data_crafter.app.schemas.task import Task, TaskCompletion
from omni_data_crafter.app.settings import settings

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

logger: "BoundLogger" = structlog.get_logger()


def worker_fn(
    process_name: str,
    device_idx: int,
    task_queue: queue.Queue[Task],
    result_queue: queue.Queue[TaskCompletion],
) -> None:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_idx)

    from omni_data_crafter.pipeline import Pipeline

    setup_logging()
    logger: "BoundLogger" = structlog.get_logger(
        process_name=process_name,
        device_idx=device_idx,
    )
    pipe = Pipeline(
        batch_size=256,
        device_id=device_idx,
        raft_model_path="./weights/raft_things.safetensors",
        ocr_model_path="./weights/pp-ocr-v4-det-fp16.engine",
        det_model_path="./weights/yolov8m-fp16.engine",
    )
    pipe.start()

    while True:
        task = task_queue.get()
        if task is None:
            break

        logger.debug(
            "Processing task...",
            task_id=task.id,
            video_path=task.video_path,
        )

        try:
            pipe(task.id, task.video_path, result_queue)
        except Exception as e:
            str_io = io.StringIO()
            traceback.print_exc(file=str_io)
            result_queue.put(
                TaskCompletion(
                    id=task.id,
                    status="failed",
                    message=str_io.getvalue(),
                )
            )

    pipe.close()
    logger.info("Stopped.")


def daemon_fn(
    process_name: str,
    device_idx: int,
    controller_url: str,
    num_tasks_per_device: int,
    task_queue: queue.Queue[Task],
    result_queue: queue.Queue[TaskCompletion],
    item_speed_queue: queue.Queue[int],
    frame_speed_queue: queue.Queue[float],
    is_running,
    is_closing,
    is_all_workers_stopped,
    total_items,
    total_frames,
    failed_items,
) -> None:
    setup_logging()
    logger: "BoundLogger" = structlog.get_logger(
        process_name=process_name,
        device_idx=device_idx,
    )

    sess = requests.Session()
    fetch_url = controller_url + "/task"

    def request_task() -> None:
        if not is_running.value or is_closing.value:
            return

        logger.debug("Requesting task...")
        rsp = sess.get(
            fetch_url,
            params={"token": settings.worker_token},
        )
        rsp.raise_for_status()
        task = rsp.json()["task"]
        if task is not None:
            task_queue.put(Task(**task))

    for _ in range(num_tasks_per_device * 2):
        request_task()

    all_results = []

    root_path = Path(f"./data/omni-data-crafter/{device_idx:04d}")
    if not root_path.exists():
        root_path.mkdir(parents=True)
    file_paths = list(root_path.glob("*.pkl.lz4"))
    file_paths.sort()
    if len(file_paths) == 0:
        save_iter = 0
    else:
        save_iter = int(file_paths[-1].name.split(".")[0]) + 1
    print("save_iter", save_iter)

    while True:
        try:
            result = result_queue.get(timeout=1.0)
        except queue.Empty:
            if is_all_workers_stopped.value:
                break
            else:
                if task_queue.qsize() < num_tasks_per_device:
                    request_task()

                continue

        if isinstance(result, TaskCompletion):
            logger.debug(
                "Sending result...",
                task_id=result.id,
                task_status=result.status,
                message=result.message,
                fps=result.fps,
                total_frames=result.total_frames,
                width=result.width,
                height=result.height,
            )

            if result.status == "failed":
                print(result)

            rsp = sess.put(
                controller_url + f"/tasks/{result.id}",
                params={"token": settings.worker_token},
                json=result.model_dump(mode="json"),
            )
            rsp.raise_for_status()

            if result.status == "completed":
                total_items.value += 1
                total_frames.value += result.total_frames
                now = datetime.utcnow()
                item_speed_queue.put((now, 1))
                frame_speed_queue.put((now, result.total_frames))
            else:
                failed_items.value += 1
        elif isinstance(result, dict):
            all_results.append(result)

            if len(all_results) >= 10000:
                with open(root_path / f"{save_iter:08d}.pkl.lz4", "wb") as f:
                    f.write(lz4.compress(pickle.dumps(all_results)))
                all_results = []
                save_iter += 1
        else:
            raise ValueError("Unknown result type.", result, type(result))

    if len(all_results) > 0:
        with open(root_path / f"{save_iter:08d}.pkl.lz4", "wb") as f:
            f.write(lz4.compress(pickle.dumps(all_results)))
        all_results = []
        save_iter += 1

    logger.info("Stopped.")


class Worker:
    def __init__(
        self,
        controller_url: str,
        device_idxs: list[int],
        num_tasks_per_device: int,
    ):
        self.controller_url = controller_url
        self.device_idxs = device_idxs
        self.num_tasks_per_device = num_tasks_per_device

        self.task_queues: list[queue.Queue[Task]] | None = None
        self.result_queues: list[queue.Queue[TaskCompletion]] | None = None
        self.item_speed_queues: list[queue.Queue[int]] | None = None
        self.frame_speed_queues: list[queue.Queue[float]] | None = None

        self.item_speed_lists: list[list[tuple[datetime, int]]] = []
        self.frame_speed_lists: list[list[tuple[datetime, float]]] = []

        self.workers: list[mp.Process] | None = None
        self.daemons: list[mp.Process] | None = None

        self.is_running = mp.Value("b", True)
        self.is_closing = mp.Value("b", False)
        self.is_all_workers_stopped = mp.Value("b", False)
        self.total_items = mp.Value("L", 0)
        self.total_frames = mp.Value("L", 0)
        self.failed_items = mp.Value("L", 0)

        self.manager = mp.Manager()

    def start(self):
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.task_queues = [self.manager.Queue() for _ in self.device_idxs]
        self.result_queues = [self.manager.Queue() for _ in self.device_idxs]
        self.item_speed_queues = [self.manager.Queue() for _ in self.device_idxs]
        self.frame_speed_queues = [self.manager.Queue() for _ in self.device_idxs]

        self.item_speed_lists = [[] for _ in self.device_idxs]
        self.frame_speed_lists = [[] for _ in self.device_idxs]

        self.workers = [
            mp.Process(
                target=worker_fn,
                args=(
                    "worker-{}-{}".format(device_idx, i),
                    device_idx,
                    self.task_queues[idx],
                    self.result_queues[idx],
                ),
            )
            for idx, device_idx in enumerate(self.device_idxs)
            for i in range(self.num_tasks_per_device)
        ]

        self.daemons = [
            mp.Process(
                target=daemon_fn,
                args=(
                    "daemon-{}".format(device_idx),
                    device_idx,
                    self.controller_url,
                    self.num_tasks_per_device,
                    self.task_queues[idx],
                    self.result_queues[idx],
                    self.item_speed_queues[idx],
                    self.frame_speed_queues[idx],
                    self.is_running,
                    self.is_closing,
                    self.is_all_workers_stopped,
                    self.total_items,
                    self.total_frames,
                    self.failed_items,
                ),
            )
            for idx, device_idx in enumerate(self.device_idxs)
        ]

        for w in self.workers:
            w.start()

        for d in self.daemons:
            d.start()

        signal.signal(signal.SIGINT, original_sigint_handler)

    def stop(self):
        logger.info("Force stopping...")

        self.is_closing.value = True
        for queue in self.task_queues:
            for _ in range(self.num_tasks_per_device):
                queue.put(None)

        for w in self.workers:
            w.join()

        logger.info("All workers stopped.")

        self.is_all_workers_stopped.value = True

        for d in self.daemons:
            d.join()

        logger.info("All daemons stopped.")

    def stats(self):
        results = {}

        for idx, device_idx in enumerate(self.device_idxs):
            item_speed_queue = self.item_speed_queues[idx]
            item_speed_list = self.item_speed_lists[idx]

            queue_size = item_speed_queue.qsize()
            for _ in range(queue_size):
                item_speed_list.append(item_speed_queue.get())

            # remove old items
            if len(item_speed_list) > 0:
                index = bisect_left(item_speed_list, (datetime.utcnow() - timedelta(minutes=10), 0))
                item_speed_list = item_speed_list[index:]
                self.item_speed_lists[idx] = item_speed_list

            if len(item_speed_list) > 2:
                duration = (item_speed_list[-1][0] - item_speed_list[0][0]).total_seconds()
            else:
                logger.warning("Not enough item speed data.")
                duration = 1.0
            items_per_second = sum([x[1] for x in item_speed_list]) / duration

            frame_speed_queue = self.frame_speed_queues[idx]
            frame_speed_list = self.frame_speed_lists[idx]

            queue_size = frame_speed_queue.qsize()
            for _ in range(queue_size):
                frame_speed_list.append(frame_speed_queue.get())

            # remove old items
            if len(frame_speed_list) > 0:
                index = bisect_left(frame_speed_list, (datetime.utcnow() - timedelta(minutes=10), 0))
                frame_speed_list = frame_speed_list[index:]
                self.frame_speed_lists[idx] = frame_speed_list

            if len(frame_speed_list) > 2:
                duration = (frame_speed_list[-1][0] - frame_speed_list[0][0]).total_seconds()
            else:
                logger.warning("Not enough frame speed data.")
                duration = 1.0
            frames_per_second = sum([x[1] for x in frame_speed_list]) / duration
            results[device_idx] = {
                "items_per_second": items_per_second,
                "frames_per_second": frames_per_second,
            }

        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, default="http://127.0.0.1:12800")
    parser.add_argument("--devices", type=str, default="all")
    parser.add_argument("--ntask-per-device", type=int, default=1)
    args = parser.parse_args()

    setup_logging()

    if args.devices == "all":
        device_idxs = list(range(torch.cuda.device_count()))
    else:
        device_idxs = [int(idx.strip()) for idx in args.devices.split(",")]

    mp.set_start_method("spawn")

    worker = Worker(
        controller_url=args.controller,
        device_idxs=device_idxs,
        num_tasks_per_device=args.ntask_per_device,
    )
    worker.start()

    try:
        while worker.is_running.value:
            stats = worker.stats()

            msg = "\n" + "#" * 50
            msg += "\nStats:\n"

            items_per_second = 0.
            frames_per_second = 0.
            for device_idx in device_idxs:
                msg += "\tDevice #{}: {:.2f} items/s, {:.2f} frames/s\n".format(
                    device_idx,
                    stats[device_idx]["items_per_second"],
                    stats[device_idx]["frames_per_second"],
                )
                items_per_second += stats[device_idx]["items_per_second"]
                frames_per_second += stats[device_idx]["frames_per_second"]

            msg += "Total: {:.2f} items/s, {:.2f} frames/s, {:d} items, {:d} frames, failed: {:d} items\n".format(
                items_per_second,
                frames_per_second,
                worker.total_items.value,
                worker.total_frames.value,
                worker.failed_items.value,
            )

            logger.info(msg)

            time.sleep(5)
    except KeyboardInterrupt:
        ...
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
