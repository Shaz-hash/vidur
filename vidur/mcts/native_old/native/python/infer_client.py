from __future__ import annotations

import multiprocessing as mp
import os
import socket
import struct
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from ...DNN.infer import NUM_ACTIONS_ADVERSARY, NUM_ACTIONS_CONTROLLER
from ...DNN.models import V_MIN, V_STEP
from ...DNN.types import ModelInputs

try:
    from ... import mcts_native as _mcts_native
except Exception:  # pragma: no cover - optional compiled extension
    _mcts_native = None


DEFAULT_AUTHKEY = b"mcts_infer"

_CMD_PING = 1
_CMD_SHUTDOWN = 2
_CMD_LOAD_MODELS = 3
_CMD_INFER = 4


def _parse_addr(addr: str) -> Tuple[str, int]:
    s = str(addr).strip()
    if ":" not in s:
        raise ValueError(f"infer_service_addr must be host:port, got: {addr!r}")
    host, port_s = s.rsplit(":", 1)
    return host, int(port_s)


def _send_frame(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack("<I", len(payload)))
    if payload:
        sock.sendall(payload)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    if n <= 0:
        return b""
    chunks = bytearray()
    remaining = int(n)
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("connection closed while reading response")
        chunks.extend(chunk)
        remaining -= len(chunk)
    return bytes(chunks)


def _recv_frame(sock: socket.socket) -> bytes:
    hdr = _recv_exact(sock, 4)
    (n,) = struct.unpack("<I", hdr)
    return _recv_exact(sock, int(n))


def _pack_str(s: str) -> bytes:
    b = str(s).encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _parse_error_payload(payload: bytes) -> RuntimeError:
    if len(payload) < 5:
        return RuntimeError("infer service error: malformed error payload")
    (n,) = struct.unpack_from("<I", payload, 1)
    if len(payload) < 5 + int(n):
        return RuntimeError("infer service error: truncated error payload")
    msg = payload[5 : 5 + int(n)].decode("utf-8", errors="replace")
    return RuntimeError(f"infer service error: {msg}")


@dataclass(frozen=True)
class TorchScriptModelPaths:
    controller_path: str
    adversary_path: str
    model_version: int


class _ManagedCppInferService:
    def __init__(self, proc: subprocess.Popen) -> None:
        self._proc = proc

    def terminate(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()

    def join(self, timeout: Optional[float] = None) -> None:
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return

    @property
    def pid(self) -> int:
        return int(self._proc.pid)


class TorchScriptInferClient:
    """
    Client for the C++ TorchScript inference service in infer_service_main.cpp.
    """

    def __init__(
        self,
        *,
        addr: str,
        authkey: bytes = DEFAULT_AUTHKEY,  # kept for compatibility
        connect_timeout_s: float = 2.0,
        request_timeout_s: float = 30.0,
    ) -> None:
        del authkey
        self._host, self._port = _parse_addr(addr)
        self._connect_timeout_s = float(connect_timeout_s)
        self._request_timeout_s = float(request_timeout_s)
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()

    def _ensure_sock(self) -> socket.socket:
        if self._sock is None:
            s = socket.create_connection(
                (self._host, self._port),
                timeout=self._connect_timeout_s,
            )
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(self._request_timeout_s)
            self._sock = s
        return self._sock

    def _close_sock(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _request(self, payload: bytes) -> bytes:
        with self._lock:
            last_exc: Optional[Exception] = None
            for _ in range(2):
                try:
                    sock = self._ensure_sock()
                    _send_frame(sock, payload)
                    return _recv_frame(sock)
                except Exception as exc:
                    last_exc = exc
                    self._close_sock()
            raise RuntimeError(f"infer service request failed: {last_exc!r}")

    def ping(self) -> bool:
        resp = self._request(bytes([_CMD_PING]))
        if not resp:
            raise RuntimeError("infer service ping: empty response")
        status = resp[0]
        if status != 0:
            raise _parse_error_payload(resp)
        return True

    def shutdown(self) -> None:
        resp = self._request(bytes([_CMD_SHUTDOWN]))
        if not resp:
            raise RuntimeError("infer service shutdown: empty response")
        status = resp[0]
        if status != 0:
            raise _parse_error_payload(resp)

    def close(self) -> None:
        with self._lock:
            self._close_sock()

    def ensure_models_loaded(self, paths: TorchScriptModelPaths) -> None:
        payload = bytearray()
        payload.append(_CMD_LOAD_MODELS)
        payload.extend(struct.pack("<i", int(paths.model_version)))
        payload.extend(_pack_str(str(paths.controller_path)))
        payload.extend(_pack_str(str(paths.adversary_path)))

        resp = self._request(bytes(payload))
        if not resp:
            raise RuntimeError("infer service load_models: empty response")
        status = resp[0]
        if status != 0:
            raise _parse_error_payload(resp)

    def infer_from_inputs(
        self,
        *,
        inputs: ModelInputs,
        player: str,
        model_version: int,
    ) -> Tuple[float, list[float]]:
        player_s = str(player)
        if player_s not in {"controller", "adversary"}:
            raise ValueError(f"invalid player={player!r}")
        player_code = 0 if player_s == "controller" else 1
        action_len = NUM_ACTIONS_CONTROLLER if player_s == "controller" else NUM_ACTIONS_ADVERSARY

        req_features_t = (
            inputs.req_features.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .view(-1)
        )
        global_features_t = (
            inputs.global_features.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .view(-1)
        )
        if req_features_t.numel() != 60:
            raise ValueError(
                f"req_features must flatten to 60 values (got {int(req_features_t.numel())})"
            )
        if global_features_t.numel() != 9:
            raise ValueError(
                f"global_features must flatten to 9 values (got {int(global_features_t.numel())})"
            )

        req_mask = inputs.req_mask
        if req_mask is None:
            req_mask_t = torch.ones((20,), dtype=torch.uint8)
        else:
            req_mask_t = req_mask.detach().to(device="cpu", dtype=torch.uint8).contiguous().view(-1)
        if req_mask_t.numel() != 20:
            raise ValueError(f"req_mask must flatten to 20 values (got {int(req_mask_t.numel())})")

        action_mask = inputs.action_mask
        if action_mask is None:
            action_mask_t = torch.ones((action_len,), dtype=torch.uint8)
        else:
            action_mask_t = action_mask.detach().to(device="cpu", dtype=torch.uint8).contiguous().view(-1)
        if action_mask_t.numel() != int(action_len):
            raise ValueError(
                f"action_mask must flatten to {action_len} values (got {int(action_mask_t.numel())})"
            )

        payload = bytearray()
        payload.append(_CMD_INFER)
        payload.extend(struct.pack("<iBH", int(model_version), int(player_code), int(action_len)))
        payload.extend(req_features_t.numpy().tobytes())
        payload.extend(global_features_t.numpy().tobytes())
        payload.extend(req_mask_t.numpy().tobytes())
        payload.extend(action_mask_t.numpy().tobytes())

        resp = self._request(bytes(payload))
        if not resp:
            raise RuntimeError("infer service infer: empty response")
        status = resp[0]
        if status != 0:
            raise _parse_error_payload(resp)
        if len(resp) < 7:
            raise RuntimeError("infer service infer: malformed success payload")

        value, pri_len = struct.unpack_from("<fH", resp, 1)
        expected = 1 + 4 + 2 + (int(pri_len) * 4)
        if len(resp) != expected:
            raise RuntimeError(
                f"infer service infer: invalid payload length (got {len(resp)}, expected {expected})"
            )
        priors = list(struct.unpack_from(f"<{int(pri_len)}f", resp, 7))
        return float(value), [float(x) for x in priors]


class TorchScriptServiceModelAdapter:
    """
    Adapts infer service to the current model interface used by mctsDNN:
      infer_from_inputs(inputs, player, device=...) -> (value, priors)
    """

    def __init__(
        self,
        *,
        client: TorchScriptInferClient,
        model_version: int,
        fallback_model: Optional[Any] = None,
        fallback_to_python: bool = True,
    ) -> None:
        self._client = client
        self._model_version = int(model_version)
        self._fallback_model = fallback_model
        self._fallback_to_python = bool(fallback_to_python)
        self._dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def model_version(self) -> int:
        return int(self._model_version)

    def set_model_version(self, model_version: int) -> None:
        self._model_version = int(model_version)

    def parameters(self):
        # mctsDNN checks next(model.parameters()).device
        yield self._dummy_param

    def infer_from_inputs(self, inputs: ModelInputs, player: str, *, device: Optional[torch.device] = None):
        del device
        try:
            return self._client.infer_from_inputs(
                inputs=inputs,
                player=str(player),
                model_version=int(self._model_version),
            )
        except Exception:
            if self._fallback_to_python and self._fallback_model is not None:
                return self._fallback_model.infer_from_inputs(inputs, player, device=torch.device("cpu"))
            raise


class NativeTorchScriptModelAdapter:
    """
    Adapter backed by C++ TorchScript runtime (mcts_native.NativeTorchScriptInferRuntime).
    Exposes the same infer_from_inputs interface expected by mctsDNN.
    """

    def __init__(
        self,
        *,
        runtime: Any,
        model_version: int,
        fallback_model: Optional[Any] = None,
        fallback_to_python: bool = True,
    ) -> None:
        self._native_ts_runtime = runtime
        self._native_ts_model_version = int(model_version)
        self._fallback_model = fallback_model
        self._fallback_to_python = bool(fallback_to_python)
        self._dummy_param = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    @property
    def model_version(self) -> int:
        return int(self._native_ts_model_version)

    def set_model_version(self, model_version: int) -> None:
        self._native_ts_model_version = int(model_version)

    def parameters(self):
        yield self._dummy_param

    def infer_from_inputs(self, inputs: ModelInputs, player: str, *, device: Optional[torch.device] = None):
        del device
        try:
            return self._native_ts_runtime.infer_from_inputs(
                inputs,
                str(player),
                int(self._native_ts_model_version),
            )
        except Exception:
            if self._fallback_to_python and self._fallback_model is not None:
                return self._fallback_model.infer_from_inputs(inputs, player, device=torch.device("cpu"))
            raise


def build_native_torchscript_runtime(*, device: str = "cpu", v_min: float = V_MIN, v_step: float = V_STEP):
    if _mcts_native is None or not hasattr(_mcts_native, "NativeTorchScriptInferRuntime"):
        raise RuntimeError("mcts_native.NativeTorchScriptInferRuntime is unavailable")
    return _mcts_native.NativeTorchScriptInferRuntime(str(device), float(v_min), float(v_step))


def build_native_infer_service_runtime(
    *,
    addr: str,
    v_min: float = V_MIN,
    v_step: float = V_STEP,
    connect_timeout_ms: int = 2000,
    request_timeout_ms: int = 30000,
):
    if _mcts_native is None or not hasattr(_mcts_native, "NativeInferServiceRuntime"):
        raise RuntimeError("mcts_native.NativeInferServiceRuntime is unavailable")
    return _mcts_native.NativeInferServiceRuntime(
        str(addr),
        float(v_min),
        float(v_step),
        int(connect_timeout_ms),
        int(request_timeout_ms),
    )


def _server_target(addr: str, device: str, max_batch: int, max_wait_us: int) -> None:
    # Legacy Python service path kept for compatibility.
    from multiprocessing.connection import Listener

    def recv_exact(conn, n: int) -> bytes:
        b = bytearray()
        while len(b) < n:
            chunk = conn.recv_bytes(min(65536, n - len(b)))
            if not chunk:
                break
            b.extend(chunk)
        return bytes(b)

    listener = Listener(_parse_addr(addr), authkey=DEFAULT_AUTHKEY)
    shutdown = False
    while not shutdown:
        conn = listener.accept()
        try:
            while True:
                msg = conn.recv()
                cmd = str(msg.get("cmd", ""))
                if cmd == "ping":
                    conn.send({"ok": True, "pong": True})
                elif cmd == "shutdown":
                    conn.send({"ok": True, "bye": True})
                    shutdown = True
                    break
                else:
                    conn.send(
                        {
                            "ok": False,
                            "error": (
                                "python infer service backend is deprecated; "
                                "use infer_service_impl='cpp'"
                            ),
                        }
                    )
        except EOFError:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
    listener.close()


def _resolve_cpp_infer_service_binary() -> Path:
    env_path = str(os.environ.get("VIDUR_CPP_INFER_SERVICE_BIN", "")).strip()
    candidates: list[Path] = []
    if env_path:
        candidates.append(Path(env_path))

    native_dir = Path(__file__).resolve().parents[1]  # .../mcts/native
    mcts_dir = native_dir.parent
    cwd = Path.cwd()
    candidates.extend(
        [
            native_dir / "build" / "infer_service_main",
            native_dir / "build" / "Release" / "infer_service_main",
            mcts_dir / "infer_service_main",
            cwd / "vidur" / "mcts" / "native" / "build" / "infer_service_main",
            cwd / "vidur" / "vidur" / "mcts" / "native" / "build" / "infer_service_main",
        ]
    )

    for p in candidates:
        if p.is_file():
            return p.resolve()
    raise RuntimeError(
        "C++ infer service binary not found. Build native target 'infer_service_main' "
        "or set VIDUR_CPP_INFER_SERVICE_BIN to the binary path."
    )


def start_torchscript_infer_service(
    *,
    addr: str,
    device: str = "cuda:0",
    max_batch: int = 256,
    max_wait_us: int = 2000,
    impl: str = "cpp",
):
    impl_s = str(impl).strip().lower()
    if impl_s == "python":
        ctx = mp.get_context("spawn")
        p = ctx.Process(
            target=_server_target,
            args=(str(addr), str(device), int(max_batch), int(max_wait_us)),
            daemon=True,
        )
        p.start()
        return p

    if impl_s != "cpp":
        raise ValueError(f"unsupported infer service impl={impl!r}; expected 'cpp' or 'python'")

    bin_path = _resolve_cpp_infer_service_binary()
    cmd = [
        str(bin_path),
        "--addr",
        str(addr),
        "--device",
        str(device),
        "--max-batch",
        str(int(max_batch)),
        "--max-wait-us",
        str(int(max_wait_us)),
    ]
    proc = subprocess.Popen(cmd)
    return _ManagedCppInferService(proc)
