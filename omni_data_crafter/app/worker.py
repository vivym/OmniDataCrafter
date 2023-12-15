import argparse
import io
import time
import traceback
import queue
import signal
import multiprocessing as mp
from bisect import bisect_left
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import requests
import structlog
import torch

from omni_data_crafter.io.vpf_stream import VPFStream
from omni_data_crafter.ops.cut_detectors import ContentDetector
from .logging import setup_logging
from .schemas.task import CutDetectionTask, CutDetectionCompletion
from .settings import settings

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

logger: "BoundLogger" = structlog.get_logger()


def cut_detection_worker_fn(
    process_name: str,
    device_idx: int,
    task_queue: queue.Queue[CutDetectionTask],
    result_queue: queue.Queue[CutDetectionCompletion],
) -> None:
    setup_logging()
    logger: "BoundLogger" = structlog.get_logger(
        process_name=process_name,
        device_idx=device_idx,
    )

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
            stream = VPFStream(task.video_path, device_idx)
            detector = ContentDetector()

            cuts = []
            last_cut_idx = 0
            for frame_idx, frame in enumerate(stream.decode(pixel_format="hsv")):
                cut_idx, _ = detector.process_frame(frame_idx, frame)
                if cut_idx is not None:
                    cuts.append((last_cut_idx, cut_idx))
                    last_cut_idx = cut_idx

                if frame_idx % 5 == 0:
                    ...

            if frame_idx != last_cut_idx:
                cuts.append((last_cut_idx, frame_idx))

            result_queue.put(
                CutDetectionCompletion(
                    id=task.id,
                    status="completed",
                    fps=stream.fps,
                    total_frames=stream.total_frames,
                    width=stream.width,
                    height=stream.height,
                    cuts=cuts,
                )
            )
        except Exception as e:
            str_io = io.StringIO()
            traceback.print_exc(file=str_io)
            result_queue.put(
                CutDetectionCompletion(
                    id=task.id,
                    status="failed",
                    message=str_io.getvalue(),
                )
            )

    logger.info("Stopped.")


def cut_detection_daemon_fn(
    process_name: str,
    device_idx: int,
    controller_url: str,
    num_tasks_per_device: int,
    task_queue: queue.Queue[CutDetectionTask],
    result_queue: queue.Queue[CutDetectionCompletion],
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
            params={
                "task_type": "cut_detection",
                "token": settings.worker_token
            },
        )
        rsp.raise_for_status()
        task = rsp.json()["task"]
        if task is not None:
            task_queue.put(CutDetectionTask(**task))

    for _ in range(num_tasks_per_device * 2):
        request_task()

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

        logger.debug(
            "Sending result...",
            task_id=result.id,
            task_status=result.status,
            message=result.message,
            fps=result.fps,
            total_frames=result.total_frames,
            width=result.width,
            height=result.height,
            cuts=result.cuts,
        )

        rsp = sess.put(
            controller_url + f"/tasks/{result.id}",
            params={
                "task_type": "cut_detection",
                "token": settings.worker_token
            },
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

    logger.info("Stopped.")


class Worker:
    def __init__(
        self,
        controller_url: str,
        task_type: str,
        device_idxs: list[int],
        num_tasks_per_device: int,
    ):
        self.controller_url = controller_url
        self.task_type = task_type
        self.device_idxs = device_idxs
        self.num_tasks_per_device = num_tasks_per_device

        self.task_queues: list[queue.Queue[CutDetectionTask]] | None = None
        self.result_queues: list[queue.Queue[CutDetectionCompletion]] | None = None
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

        if self.task_type == "cut_detection":
            worker_fn = cut_detection_worker_fn
            daemon_fn = cut_detection_daemon_fn
        else:
            raise ValueError("Unknown task type: {}".format(self.task_type))

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
    parser.add_argument("--task", type=str, default="cut_detection")
    parser.add_argument("--devices", type=str, default="all")
    parser.add_argument("--ntask-per-device", type=int, default=1)
    args = parser.parse_args()

    setup_logging()

    if args.devices == "all":
        device_idxs = list(range(torch.cuda.device_count()))
    else:
        device_idxs = [int(idx.strip()) for idx in args.devices.split(",")]

    worker = Worker(
        controller_url=args.controller,
        task_type=args.task,
        device_idxs=device_idxs,
        num_tasks_per_device=args.ntask_per_device,
    )
    worker.start()

    try:
        while worker.is_running.value:
            stats = worker.stats()

            msg = "\nStats:\n"

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

            msg += "Total: {:.2f} items/s, {:.2f} frames/s, {:d} items, {:d} frames\n".format(
                items_per_second,
                frames_per_second,
                worker.total_items.value,
                worker.total_frames.value,
            )

            logger.info(msg)

            time.sleep(5)
    except KeyboardInterrupt:
        ...
    finally:
        worker.stop()


if __name__ == "__main__":
    main()
