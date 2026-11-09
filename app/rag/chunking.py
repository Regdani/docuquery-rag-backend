"""Section-aware chunking: every `##` section is split on its own, never merged with others."""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loader import ManualDocument

_SECTION = re.compile(r"^##\s+(.+?)\s*$")
_INTRO_TITLE = "Introduction"


def split_sections(content: str) -> list[tuple[str, str]]:
    """Return (section title, body) pairs; text before the first `##` is the introduction."""
    sections: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    seen_title = False
    for line in content.splitlines():
        if not seen_title:
            seen_title = bool(line.strip())  # the first non-empty line is the product title
            continue
        match = _SECTION.match(line)
        if match:
            current = (match[1], [])
            sections.append(current)
        elif current is not None:
            current[1].append(line)
        elif line.strip():
            current = (_INTRO_TITLE, [line])
            sections.append(current)
    return [
        (title, "\n".join(lines).strip()) for title, lines in sections if "".join(lines).strip()
    ]


def chunk_document(doc: ManualDocument, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split one manual into chunks carrying source/product/section/chunk ID metadata.

    The chunk text is prefixed with a `<ID> <name> - <section>` line so a chunk is
    self-describing for retrieval; the prefix is not counted against `chunk_size`.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = []
    for section_index, (title, body) in enumerate(split_sections(doc.content)):
        for chunk_index, piece in enumerate(splitter.split_text(body)):
            chunk_id = f"{doc.source}::{section_index}::{chunk_index}"
            chunks.append(
                Document(
                    id=chunk_id,
                    page_content=f"{doc.product_id} {doc.product_name} - {title}\n\n{piece}",
                    metadata={
                        "source": doc.source,
                        "product_id": doc.product_id,
                        "product_name": doc.product_name,
                        "section": title,
                        "chunk_id": chunk_id,
                    },
                )
            )
    return chunks


def chunk_documents(
    docs: list[ManualDocument], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    """Chunk all manuals."""
    return [c for doc in docs for c in chunk_document(doc, chunk_size, chunk_overlap)]
