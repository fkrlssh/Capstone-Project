"""
AirSim RPC coordination for multi-agent runs.

Six agents each opening their own client and calling simGetImages concurrently
often triggers SimpleFlight "API call was not received" hover. Use one shared
client + serialized calls + a small image concurrency cap.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, List, Optional

import airsim

from config import CFG

_rpc_lock = threading.RLock()
_shared_client: Optional[airsim.MultirotorClient] = None
_image_sem: Optional[threading.Semaphore] = None


def _image_semaphore() -> threading.Semaphore:
    global _image_sem
    if _image_sem is None:
        n = max(1, int(getattr(CFG, "RPC_IMAGE_MAX_CONCURRENT", 1) or 1))
        _image_sem = threading.Semaphore(n)
    return _image_sem


def reset_shared_client() -> None:
    """Drop shared client (e.g. after AirSim simReset)."""
    global _shared_client, _image_sem
    with _rpc_lock:
        _shared_client = None
        _image_sem = None


def get_client(confirm: bool = False) -> airsim.MultirotorClient:
    if not getattr(CFG, "RPC_SHARED_CLIENT_ENABLED", True):
        client = airsim.MultirotorClient()
        if confirm:
            client.confirmConnection()
        return client

    global _shared_client
    with _rpc_lock:
        if _shared_client is None:
            _shared_client = airsim.MultirotorClient()
            if confirm or getattr(CFG, "RPC_CONFIRM_ON_FIRST_CONNECT", True):
                _shared_client.confirmConnection()
        return _shared_client


def rpc_call(func: Callable[..., Any], *args, **kwargs) -> Any:
    if not getattr(CFG, "RPC_SERIALIZE_CALLS", True):
        return func(*args, **kwargs)
    with _rpc_lock:
        return func(*args, **kwargs)


def sim_get_images(
    client: airsim.MultirotorClient,
    requests: List[airsim.ImageRequest],
    vehicle_name: str = "",
) -> Any:
    max_concurrent = int(getattr(CFG, "RPC_IMAGE_MAX_CONCURRENT", 1) or 0)
    if max_concurrent <= 0:
        return rpc_call(client.simGetImages, requests, vehicle_name=vehicle_name)

    with _image_semaphore():
        return rpc_call(client.simGetImages, requests, vehicle_name=vehicle_name)
