import io
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import DeterministicFakeEmbedding
from pydantic import ValidationError

from app.core.config import Settings
from app.core.logging import APP_LOGGER, configure_logging
from app.main import create_app
from tests.fakes import RelevanceVectorStore, ScriptedChatModel


@pytest.fixture(autouse=True)
def restore_app_logger() -> Iterator[None]:
    logger = logging.getLogger(APP_LOGGER)
    level, handlers = logger.level, list(logger.handlers)
    yield
    logger.setLevel(level)
    logger.handlers = handlers


def app_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger(APP_LOGGER).handlers if h.get_name() == "docuquery-app"]


def test_configure_logging_sets_level_and_one_handler() -> None:
    configure_logging("DEBUG")
    configure_logging("WARNING")
    assert logging.getLogger(APP_LOGGER).level == logging.WARNING
    assert len(app_handlers()) == 1


def test_app_info_logs_are_emitted_with_format() -> None:
    configure_logging("INFO")
    stream = io.StringIO()
    app_handlers()[0].setStream(stream)  # type: ignore[attr-defined]
    logging.getLogger("app.services.qa_service").info("ask request_id=abc status=answered")
    line = stream.getvalue().strip()
    assert line.endswith("INFO app.services.qa_service ask request_id=abc status=answered")


def test_startup_configures_logging(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        log_level="WARNING",
        data_dir=str(tmp_path / "data"),
        index_dir=str(tmp_path / "index"),
    )
    emb = DeterministicFakeEmbedding(size=8)
    app = create_app(
        settings,
        chat_model=ScriptedChatModel(responder=lambda _: "ok", prompts=[]),
        embeddings=emb,
        vector_store=RelevanceVectorStore(emb),
    )
    logging.getLogger(APP_LOGGER).handlers = []
    with TestClient(app):
        assert logging.getLogger(APP_LOGGER).level == logging.WARNING
        assert len(app_handlers()) == 1


def test_request_is_logged_with_duration_and_request_id(caplog: pytest.LogCaptureFixture) -> None:
    client = TestClient(create_app())
    with caplog.at_level(logging.DEBUG, logger="app.api.handlers"):
        client.get("/", headers={"X-Request-ID": "req-1"})
        client.get("/health", headers={"X-Request-ID": "req-2"})
    lines = {
        r.getMessage(): r.levelno for r in caplog.records if "request method=" in r.getMessage()
    }
    root = next(m for m in lines if "request_id=req-1" in m)
    assert "path=/ status=200 duration_ms=" in root and lines[root] == logging.INFO
    health = next(m for m in lines if "request_id=req-2" in m)
    assert lines[health] == logging.DEBUG  # probes stay out of the INFO log


@pytest.mark.parametrize(("value", "expected"), [("debug", "DEBUG"), (" warning ", "WARNING")])
def test_log_level_is_normalized(value: str, expected: str) -> None:
    assert Settings(_env_file=None, log_level=value).log_level == expected  # type: ignore[call-arg]


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        Settings(_env_file=None, log_level="LOUD")  # type: ignore[call-arg]
