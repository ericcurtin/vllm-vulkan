# SPDX-License-Identifier: Apache-2.0
"""Subprocess-bridged Vulkan context — fallback for processes where direct
in-process `VulkanContext` creation fails.

## Background

On some hardware/process combinations (confirmed: NVIDIA GB10, real vLLM
`EngineCore` worker processes specifically), creating a `VulkanContext`
directly inside the worker process fails with::

    vkCreateDevice: Initialization of an object has failed

every single time, with no retry count or delay resolving it, while the
*exact same* `VulkanContext(device_idx)` call reliably succeeds:

  - in an interactive/standalone Python process,
  - in a `multiprocessing`-spawned child of that same worker process
    (confirmed empirically: spawning a throwaway child *from inside* the
    failing `EngineCore` process and creating the context there succeeds
    every time, even though the parent's own subsequent attempt still
    fails),
  - regardless of NUMA memory policy, CPU thread count/affinity, model
    weights being loaded, `torch`/`vllm` import order, `fork` vs `spawn`,
    or Vulkan device/instance API version (all ruled out by direct testing).

The root cause of why the *specific* worker process's own address space is
incompatible with Vulkan device creation was not pinned down (plausibly an
NVIDIA driver/ICD quirk related to this process's accumulated memory-
mapping or shared-library-loading state — vLLM's CPU worker process ends up
with hundreds of shared libraries and several GB of tensors mapped in by
the time this runs), but the *workaround* is straightforward and reliable:
create the real `VulkanContext` in a fresh child process instead, and proxy
calls to it over a pipe.

## Usage

`create_vulkan_context_with_fallback(device_idx)` tries direct creation
first (so this adds zero overhead on any system/process where direct
creation already works, including every case this project's existing test
suite runs under) and only falls back to `RemoteVulkanContext` if that
raises.

## Cost

Every `execute_batch`/`upload_tensor`/`available_shaders` call now pays an
extra pipe round-trip (send + recv) on top of the real GPU dispatch's own
cost, since the actual `VulkanContext` lives in a different process.
Weight uploads happen once per weight (already cached by
`vulkan_ops._get_or_upload_weight`) so this only matters for the
per-forward-call `execute_batch` dispatches. Measured directly: still a net
win over the CPU fallback at the shapes `vulkan_ops.linear()` already
gates GPU dispatch on (see its module docstring's measurements), but the
margin is real and much narrower than direct in-process dispatch — see
`RemoteVulkanContext`'s own docstring.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from vllm_vulkan._rs import VulkanContext

logger = logging.getLogger(__name__)

_CMD_UPLOAD_TENSOR = 1
_CMD_AVAILABLE_SHADERS = 2
_CMD_EXECUTE_BATCH = 3
_CMD_SHUTDOWN = 4


def _subprocess_main(device_idx: int, conn: Connection) -> None:
    """Entry point for the persistent Vulkan-owning child process.

    Creates the real `VulkanContext` once, reports success/failure back to
    the parent over `conn`, then services requests until told to shut down
    or the pipe is closed. Never raises out of this function on a per-
    request basis — request-level errors are sent back as
    `("error", message)` so the parent can raise them in its own caller's
    context instead of killing this persistent process.
    """
    try:
        from vllm_vulkan._rs import VulkanContext  # noqa: PLC0415

        ctx = VulkanContext(device_idx)
    except Exception as exc:  # noqa: BLE001
        conn.send(("init_error", str(exc)))
        return

    conn.send(("init_ok", None))

    # tensor_id -> GpuTensor. GpuTensor objects hold raw Vulkan/driver
    # handles that are only valid in the process that created them, so
    # only integer IDs (not the objects themselves) ever cross `conn`.
    tensors: dict[int, Any] = {}
    next_id = 0

    while True:
        try:
            request = conn.recv()
        except (EOFError, OSError):
            break
        if request is None:
            break

        cmd, payload = request
        try:
            if cmd == _CMD_SHUTDOWN:
                break
            if cmd == _CMD_UPLOAD_TENSOR:
                gpu_tensor = ctx.upload_tensor(payload)
                tid = next_id
                next_id += 1
                tensors[tid] = gpu_tensor
                conn.send(("ok", tid))
            elif cmd == _CMD_AVAILABLE_SHADERS:
                conn.send(("ok", ctx.available_shaders()))
            elif cmd == _CMD_EXECUTE_BATCH:
                resolved_ops = []
                for shader, bindings, out_sizes, pc, workgroups, barrier in payload:
                    resolved_bindings = []
                    for kind, value in bindings:
                        if kind == "tensor":
                            resolved_bindings.append(tensors[value])
                        else:
                            resolved_bindings.append(value)
                    resolved_ops.append(
                        (shader, resolved_bindings, out_sizes, pc, workgroups, barrier)
                    )
                conn.send(("ok", ctx.execute_batch(resolved_ops)))
            else:
                conn.send(("error", f"unknown command {cmd}"))
        except Exception as exc:  # noqa: BLE001
            try:
                conn.send(("error", str(exc)))
            except Exception:  # noqa: BLE001
                # Parent already gone; nothing more to do.
                break


class _RemoteGpuTensor:
    """Stand-in for a real `GpuTensor` returned by `RemoteVulkanContext.upload_tensor`.

    Holds only the integer ID the subprocess uses to look up the real
    `GpuTensor` in its own `tensors` dict — never the tensor data or a
    cross-process handle to it.
    """

    __slots__ = ("tensor_id",)

    def __init__(self, tensor_id: int) -> None:
        self.tensor_id = tensor_id

    def __repr__(self) -> str:
        return f"RemoteGpuTensor(id={self.tensor_id})"


class RemoteVulkanContext:
    """Drop-in replacement for `vllm_vulkan._rs.VulkanContext` that proxies
    every call to a persistent child process holding the real context.

    Only implements the subset of `VulkanContext`'s interface that
    `vulkan_ops.py`'s Linear-dispatch hot path actually uses
    (`upload_tensor`, `available_shaders`, `execute_batch`) — sufficient to
    unblock GPU dispatch for `linear()`. `execute_chained`/`alloc_activation`/
    `update_activation` (used only by `attention.py`'s opportunistic Vulkan
    KV-cache mirroring, and by `rms_norm_then_linear`, which is currently
    unused dead code — see that function's doc comment) are intentionally
    not implemented here: attention already has its own fallback to
    CPU_ATTN if Vulkan is unavailable, so it degrades safely rather than
    needing every method proxied for correctness.
    """

    def __init__(self, device_idx: int = 0, init_timeout: float = 60.0) -> None:
        ctx = mp.get_context("spawn")
        self._parent_conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(
            target=_subprocess_main,
            args=(device_idx, child_conn),
            daemon=True,
            name="vllm-vulkan-bridge",
        )
        self._proc.start()
        child_conn.close()  # parent doesn't need its own copy of the child's end

        if not self._parent_conn.poll(init_timeout):
            self._cleanup()
            raise RuntimeError(
                f"Vulkan bridge subprocess did not respond within {init_timeout}s"
            )
        status, message = self._parent_conn.recv()
        if status != "init_ok":
            self._cleanup()
            raise RuntimeError(f"Vulkan bridge subprocess failed to init: {message}")

        logger.info(
            "Direct in-process VulkanContext creation failed; using a "
            "subprocess-bridged Vulkan context instead (pid=%d). GPU "
            "dispatch still works, with added IPC overhead per call.",
            self._proc.pid,
        )

    def _cleanup(self) -> None:
        try:
            self._parent_conn.close()
        except Exception:  # noqa: BLE001
            pass
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)

    def _call(self, cmd: int, payload: object) -> object:
        self._parent_conn.send((cmd, payload))
        status, result = self._parent_conn.recv()
        if status == "error":
            raise RuntimeError(f"Vulkan bridge request failed: {result}")
        return result

    def upload_tensor(self, data: bytes) -> _RemoteGpuTensor:
        tensor_id = self._call(_CMD_UPLOAD_TENSOR, bytes(data))
        return _RemoteGpuTensor(tensor_id)  # type: ignore[arg-type]

    def available_shaders(self) -> list[str]:
        return self._call(_CMD_AVAILABLE_SHADERS, None)  # type: ignore[return-value]

    def execute_batch(self, ops: list[tuple]) -> list[list[bytes]]:
        wire_ops = []
        for shader, bindings, out_sizes, pc, workgroups, barrier in ops:
            wire_bindings = []
            for binding in bindings:
                if isinstance(binding, _RemoteGpuTensor):
                    wire_bindings.append(("tensor", binding.tensor_id))
                else:
                    wire_bindings.append(("bytes", bytes(binding)))
            wire_ops.append((shader, wire_bindings, out_sizes, pc, workgroups, barrier))
        return self._call(_CMD_EXECUTE_BATCH, wire_ops)  # type: ignore[return-value]

    def close(self) -> None:
        try:
            self._parent_conn.send((_CMD_SHUTDOWN, None))
        except Exception:  # noqa: BLE001
            pass
        self._cleanup()

    def __del__(self) -> None:
        self.close()


def create_vulkan_context_with_fallback(
    device_idx: int = 0,
) -> VulkanContext | RemoteVulkanContext:
    """Create a `VulkanContext`, falling back to a subprocess-bridged one
    if direct in-process creation fails *and* the caller has opted in via
    `VLLM_VULKAN_ALLOW_SUBPROCESS_BRIDGE=1`.

    This is the only place that should construct the "first" context for a
    process — see this module's docstring for why the fallback exists.
    Direct creation is always tried first, so this is a zero-overhead,
    zero-behavior-change no-op on any system/process where it already
    works (which is everywhere this project's test suite runs).

    The subprocess bridge is opt-in, not automatic, despite being fully
    correct and functional: measured directly on this project's reference
    hardware (NVIDIA GB10), every `execute_batch` call now pays a pipe
    round-trip on top of the real GPU dispatch, and vLLM's CPU backend's
    own oneDNN-packed GEMM path (`ops.create_onednn_mm`/`onednn_mm` --
    what `orig()`'s CPU fallback in `model_runner.py`'s `vk_forward`
    already uses) turned out to be fast enough that the *end-to-end*
    serving benchmark got *slower* with the bridge enabled by default
    (roughly 2x, across every concurrency level tested) than simply
    falling back to that CPU path, even though GPU dispatch is itself
    correct and, in isolated microbenchmarks, faster than the CPU path
    for *some* (not all) of this model's Linear weight shapes. Silently
    enabling a "fix" that regresses the common case by default would be
    the wrong trade-off; set `VLLM_VULKAN_ALLOW_SUBPROCESS_BRIDGE=1` to
    opt into it anyway (e.g. for larger models/shapes where the balance
    may favor GPU dispatch, or future work reducing the bridge's
    per-call IPC overhead).
    """
    from vllm_vulkan._rs import VulkanContext  # noqa: PLC0415

    try:
        return VulkanContext(device_idx)
    except Exception as exc:  # noqa: BLE001
        if os.environ.get("VLLM_VULKAN_ALLOW_SUBPROCESS_BRIDGE") != "1":
            logger.warning(
                "Direct VulkanContext(%d) creation failed (%s). A "
                "subprocess-bridged fallback is available and fully "
                "correct, but is opt-in (not automatic) since it measured "
                "slower than this platform's CPU fallback in end-to-end "
                "serving benchmarks on this project's reference hardware "
                "-- set VLLM_VULKAN_ALLOW_SUBPROCESS_BRIDGE=1 to enable it "
                "anyway. Continuing without Vulkan GPU dispatch for this "
                "process (falls back to CPU, same as before this context "
                "creation was ever attempted).",
                device_idx,
                exc,
            )
            raise
        logger.warning(
            "Direct VulkanContext(%d) creation failed (%s); falling back to "
            "a subprocess-bridged Vulkan context (VLLM_VULKAN_ALLOW_SUBPROCESS_BRIDGE=1).",
            device_idx,
            exc,
        )
        return RemoteVulkanContext(device_idx)
