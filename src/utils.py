import psutil
import torch


def get_system_available_memory() -> int:
    return psutil.virtual_memory().available


def get_system_total_memory() -> int:
    return psutil.virtual_memory().total


def get_device_allocated_memory(device: torch.device) -> int:
    if device.type == "mps":
        return torch.mps.driver_allocated_memory()

    raise ValueError(f"Unsupported device: {device}")


def get_device_memory_limit(device: torch.device) -> int:
    if device.type == "mps":
        return torch.mps.recommended_max_memory()

    raise ValueError(f"Unsupported device: {device}")


def synchronize_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
        return

    raise ValueError(f"Unsupported device: {device}")


def empty_device_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
        return

    raise ValueError(f"Unsupported device: {device}")
