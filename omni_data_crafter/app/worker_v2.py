import multiprocessing as mp

import torch
import torch.multiprocessing as torch_mp


class WorkerProcess(mp.Process):
    ...


def video_decoding_worker_fn():
    ...


def clip_worker_fn():
    ...


def optical_flow_worker_fn():
    ...


def cut_detection_worker_fn():
    ...


def main():
    mp.set_start_method("spawn")




if __name__ == "__main__":
    main()
