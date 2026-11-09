"""Offline guard: tests not marked `integration` may not open non-local network connections."""

import ipaddress
import socket
from collections.abc import Iterator
from typing import Any

import pytest

_LOCAL_NAMES = {"localhost"}


def _is_local(address: Any) -> bool:
    if not isinstance(address, tuple) or not address:  # AF_UNIX paths and similar
        return True
    host = address[0]
    if isinstance(host, bytes):
        host = host.decode()
    if host in _LOCAL_NAMES or host == "":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def block_network(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    if request.node.get_closest_marker("integration"):
        yield
        return

    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def guarded_connect(self: socket.socket, address: Any) -> None:
        if not _is_local(address):
            raise RuntimeError(f"network access is blocked in offline tests: {address!r}")
        real_connect(self, address)

    def guarded_connect_ex(self: socket.socket, address: Any) -> int:
        if not _is_local(address):
            raise RuntimeError(f"network access is blocked in offline tests: {address!r}")
        return real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    yield
