"""
Device Utilities

This module provides utility functions for Vulkan device management.
"""

from typing import Any, Dict, List, Optional, Tuple

import vllm_vulkan


def get_device_properties(device_idx: int = 0) -> Dict[str, Any]:
    """
    Get all properties for a Vulkan device.

    Args:
        device_idx: Device index

    Returns:
        Dictionary of device properties including:
        - name: Device name
        - vendor: Vendor name
        - memory_mb: Total memory in MB
        - device_type: Device type (discrete, integrated, etc.)
        - api_version: Vulkan API version
        - driver_version: Driver version
        - supports_fp16: FP16 support
        - supports_int8: INT8 support
        - max_compute_work_group_count: Max work group counts [x, y, z]
        - max_compute_work_group_size: Max work group sizes [x, y, z]
    """
    try:
        return vllm_vulkan.get_device_info(device_idx)
    except Exception as e:
        return {
            "name": "Unknown",
            "vendor": "Unknown",
            "memory_mb": 0,
            "device_type": "unknown",
            "api_version": "0.0.0",
            "driver_version": "0.0.0",
            "supports_fp16": False,
            "supports_int8": False,
            "max_compute_work_group_count": [0, 0, 0],
            "max_compute_work_group_size": [0, 0, 0],
            "error": str(e),
        }


def get_device_memory_info(device_idx: int = 0) -> Tuple[int, int]:
    """
    Get memory information for a device.

    Args:
        device_idx: Device index

    Returns:
        Tuple of (used_bytes, total_bytes)
    """
    try:
        return vllm_vulkan.get_memory_info(device_idx)
    except Exception:
        return (0, 0)


def get_device_compute_capability(device_idx: int = 0) -> Tuple[int, int]:
    """
    Get compute capability for a device.

    For Vulkan, this returns the API version as a (major, minor) tuple.

    Args:
        device_idx: Device index

    Returns:
        Tuple of (major, minor) version
    """
    try:
        props = get_device_properties(device_idx)
        api_version = props.get("api_version", "1.0.0")
        parts = api_version.split(".")
        if len(parts) >= 2:
            return (int(parts[0]), int(parts[1]))
    except Exception:
        pass
    return (1, 0)


def is_device_available(device_idx: int = 0) -> bool:
    """
    Check if a device is available.

    Args:
        device_idx: Device index

    Returns:
        True if device is available
    """
    try:
        device_count = vllm_vulkan.get_device_count()
        return device_idx < device_count
    except Exception:
        return False


def synchronize_device(device_idx: Optional[int] = None) -> None:
    """
    Synchronize a device (wait for all operations to complete).

    Args:
        device_idx: Device index (None for all devices)
    """
    vllm_vulkan.synchronize()


def get_available_memory(device_idx: int = 0) -> int:
    """
    Get available memory on a device in bytes.

    Args:
        device_idx: Device index

    Returns:
        Available memory in bytes
    """
    used, total = get_device_memory_info(device_idx)
    return total - used


def get_device_count() -> int:
    """
    Get the number of available Vulkan devices.

    Returns:
        Number of devices
    """
    return vllm_vulkan.get_device_count()


def list_devices() -> List[Dict[str, Any]]:
    """
    List all available Vulkan devices with their properties.

    Returns:
        List of device property dictionaries
    """
    devices = []
    count = get_device_count()
    for i in range(count):
        props = get_device_properties(i)
        props["index"] = i
        devices.append(props)
    return devices


def get_best_device() -> int:
    """
    Get the index of the best available device.

    The "best" device is determined by:
    1. Preferring discrete GPUs over integrated
    2. Preferring more memory
    3. Preferring newer API versions

    Returns:
        Index of the best device, or 0 if no devices
    """
    devices = list_devices()
    if not devices:
        return 0

    # Sort by criteria
    def device_score(d: Dict[str, Any]) -> Tuple[int, int, str]:
        # Discrete GPUs get higher priority
        type_score = 1 if d.get("device_type") == "discrete" else 0
        # More memory is better
        memory_score = d.get("memory_mb", 0)
        # Newer API version is better
        api_version = d.get("api_version", "0.0.0")
        return (type_score, memory_score, api_version)

    sorted_devices = sorted(devices, key=device_score, reverse=True)
    return sorted_devices[0].get("index", 0)


def set_current_device(device_idx: int) -> None:
    """
    Set the current device for operations.

    Note: In Vulkan, device selection is typically done at operation
    time rather than globally, but this can be used as a hint.

    Args:
        device_idx: Device index to use
    """
    if not is_device_available(device_idx):
        raise ValueError(f"Device {device_idx} is not available")
    # In real implementation, this would set a thread-local device context


def get_device_name(device_idx: int = 0) -> str:
    """
    Get the name of a device.

    Args:
        device_idx: Device index

    Returns:
        Device name string
    """
    props = get_device_properties(device_idx)
    return props.get("name", f"Vulkan Device {device_idx}")


def supports_dtype(device_idx: int, dtype: str) -> bool:
    """
    Check if a device supports a given data type.

    Args:
        device_idx: Device index
        dtype: Data type string (e.g., "f16", "f32", "q4_0")

    Returns:
        True if supported
    """
    props = get_device_properties(device_idx)

    dtype_lower = dtype.lower()

    # F32 is always supported
    if dtype_lower in ("f32", "float32"):
        return True

    # F16 support
    if dtype_lower in ("f16", "float16"):
        return props.get("supports_fp16", False)

    # INT8 support
    if dtype_lower in ("i8", "int8", "q8_0", "q8_1"):
        return props.get("supports_int8", False)

    # Quantized types are generally supported if the device works
    if dtype_lower.startswith("q"):
        return True

    return False
