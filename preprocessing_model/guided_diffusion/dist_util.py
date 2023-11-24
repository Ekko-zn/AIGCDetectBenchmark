"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf

import torch as th
import torch.distributed as dist
from . import logger
# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3




def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    chunk_size = 2 ** 30  # Use a large chunk size

    # Load the entire file into memory
    with open(path, "rb") as f:
        data = f.read()

    num_chunks = len(data) // chunk_size
    if len(data) % chunk_size:
        num_chunks += 1

    for i in range(0, len(data), chunk_size):
        chunk_data = data[i : i + chunk_size]
        # Use `th.load` to load the checkpoint
        model_state_dict = th.load(io.BytesIO(chunk_data), **kwargs)

    return model_state_dict


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
