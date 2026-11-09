import socket

import pytest


def test_remote_connection_is_blocked() -> None:
    with socket.socket() as s, pytest.raises(RuntimeError, match="blocked"):
        s.connect(("93.184.216.34", 80))
