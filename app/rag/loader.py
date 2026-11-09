"""Load Markdown manuals from the data directory."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_HEADING = re.compile(r"^#\s+(?P<id>\S+)\s+[–—-]\s+(?P<name>.+?)\s*$")


class DocumentLoadError(ValueError):
    """A manual file is unreadable or does not follow the expected format."""


@dataclass(frozen=True)
class ManualDocument:
    """One manual: file name, parsed product identity and the raw Markdown text."""

    source: str
    product_id: str
    product_name: str
    content: str


def parse_manual(source: str, content: str) -> ManualDocument:
    """Parse the `# <ID> – <name>` heading; raise DocumentLoadError when it is missing."""
    first = next((line for line in content.splitlines() if line.strip()), "")
    match = _HEADING.match(first.strip())
    if match is None:
        raise DocumentLoadError("missing '# <ID> – <name>' title heading")
    return ManualDocument(source, match["id"], match["name"], content)


def load_documents(data_dir: str | Path) -> list[ManualDocument]:
    """Load every `*.md` file in `data_dir`; bad files are logged with the error and skipped."""
    directory = Path(data_dir)
    if not directory.is_dir():
        logger.error("Data directory does not exist: %s", directory)
        return []
    documents: list[ManualDocument] = []
    for path in sorted(directory.glob("*.md")):
        try:
            doc = parse_manual(path.name, path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, DocumentLoadError) as exc:
            logger.error("Skipping %s: %s", path.name, exc)
            continue
        logger.info("Loaded document %s (product %s)", doc.source, doc.product_id)
        documents.append(doc)
    return documents
