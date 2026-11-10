"""HTTP routes. They only call the QAService; no RAG or provider logic lives here."""

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Request

from app.api.deps import get_qa_service, require_index_ready, validate_question_length
from app.api.schemas import AskRequest, AskResponse, ErrorResponse, Source
from app.services.qa_service import QAServiceProtocol

router = APIRouter(prefix="/api/v1")

_SRC = {
    "product_id": "KV-200",
    "product_name": "Espresso Coffee Maker",
    "section": "Maintenance",
    "chunk_id": "kv-200-espresso-coffee-maker.md::3::0",
    "score": 0.82,
    "snippet": "Descale the machine every 60 days or after 200 brew cycles.",
}

_REQUEST_EXAMPLES: dict[str, Any] = {
    "successful_question": {
        "summary": "Successful question",
        "value": {"question": "How often should I descale the KV-200 coffee maker?"},
    },
    "follow_up": {
        "summary": "Follow-up with a session key",
        "description": "Send the same session_key with each question of a conversation.",
        "value": {"question": "And what about the KV-300?", "session_key": "demo-session-01"},
    },
    "clarification": {
        "summary": "Unclear question (the assistant asks back)",
        "value": {"question": "How do I clean it?"},
    },
    "product_scoped": {
        "summary": "Question restricted to one product",
        "value": {
            "question": "How do I descale it?",
            "product_id": "KV-200",
        },
    },
}


def _example(summary: str, value: dict[str, Any]) -> dict[str, Any]:
    return {"summary": summary, "value": value}


_RID = "3f2b8c1e9d7a4e6f"


def _error(code: str, detail: Any, request_id: str = _RID) -> dict[str, Any]:
    return {"detail": detail, "code": code, "request_id": request_id}


_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Answer, clarifying question or no-information reply.",
        "content": {
            "application/json": {
                "examples": {
                    "answered": _example(
                        "Answered with sources",
                        {
                            "answer": "Descale the KV-200 every 60 days or after 200 brew cycles.",
                            "status": "answered",
                            "sources": [_SRC],
                            "session_key": None,
                            "request_id": "3f2b8c1e9d7a4e6f8a1b2c3d4e5f6a7b",
                        },
                    ),
                    "clarification_needed": _example(
                        "Clarifying question",
                        {
                            "answer": "Which appliance would you like to clean?",
                            "status": "clarification_needed",
                            "sources": [],
                            "session_key": "demo-session-01",
                            "request_id": "3f2b8c1e9d7a4e6f8a1b2c3d4e5f6a7b",
                        },
                    ),
                    "no_answer": _example(
                        "Nothing relevant found",
                        {
                            "answer": "I could not find relevant information in the manuals.",
                            "status": "no_answer",
                            "sources": [],
                            "session_key": None,
                            "request_id": "3f2b8c1e9d7a4e6f8a1b2c3d4e5f6a7b",
                        },
                    ),
                }
            }
        },
    },
    404: {
        "model": ErrorResponse,
        "description": "The product_id is not in the index.",
        "content": {
            "application/json": {
                "examples": {
                    "product_not_found": _example(
                        "Unknown product",
                        _error(
                            "PRODUCT_NOT_FOUND",
                            "Unknown product_id 'KV-999'; it is not in the document index.",
                        ),
                    )
                }
            }
        },
    },
    422: {
        "model": ErrorResponse,
        "description": (
            "Invalid request: missing or wrong-typed field, malformed JSON, blank or too long "
            "question, bad session_key. Checked before the index state."
        ),
        "content": {
            "application/json": {
                "examples": {
                    "validation": _example(
                        "Blank question",
                        {
                            "detail": "Invalid request: question: "
                            "String should have at least 1 character",
                            "code": "VALIDATION_ERROR",
                            "request_id": _RID,
                            "errors": [
                                {
                                    "field": "question",
                                    "message": "String should have at least 1 character",
                                    "type": "string_too_short",
                                }
                            ],
                        },
                    )
                }
            }
        },
    },
    502: {
        "model": ErrorResponse,
        "description": "Provider unavailable, empty answer, or unexpected pipeline failure.",
        "content": {
            "application/json": {
                "examples": {
                    "llm_unavailable": _example(
                        "Provider unreachable or failing",
                        _error("LLM_UNAVAILABLE", "openai is unreachable (APIConnectionError)"),
                    ),
                    "llm_empty_response": _example(
                        "Empty model answer",
                        _error("LLM_EMPTY_RESPONSE", "The model returned an empty answer."),
                    ),
                    "rag_pipeline_error": _example(
                        "Unexpected failure in the question flow",
                        _error("RAG_PIPELINE_ERROR", "The question could not be processed."),
                    ),
                }
            }
        },
    },
    503: {
        "model": ErrorResponse,
        "description": "The index is not ready, or the provider is still rate limiting.",
        "headers": {
            "Retry-After": {
                "description": "Seconds to wait before retrying (LLM_RATE_LIMITED).",
                "schema": {"type": "integer", "examples": [5]},
            }
        },
        "content": {
            "application/json": {
                "examples": {
                    "index_not_ready": _example(
                        "Index still being built",
                        _error(
                            "INDEX_NOT_READY",
                            "The document index is still being built; retry shortly.",
                        ),
                    ),
                    "llm_rate_limited": _example(
                        "Rate limited after retries (Retry-After header set)",
                        _error("LLM_RATE_LIMITED", "azure rate limit exceeded (RateLimitError)"),
                    ),
                }
            }
        },
    },
    504: {
        "model": ErrorResponse,
        "description": "The model provider did not answer within the timeout.",
        "content": {
            "application/json": {
                "examples": {
                    "timeout": _example(
                        "Provider timeout",
                        _error("LLM_TIMEOUT", "azure request timed out (APITimeoutError)"),
                    )
                }
            }
        },
    },
}


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question about the product manuals",
    responses=_RESPONSES,
)
async def ask(
    request: Request,
    body: Annotated[AskRequest, Body(openapi_examples=_REQUEST_EXAMPLES)],
    service: Annotated[QAServiceProtocol, Depends(get_qa_service)],
) -> AskResponse:
    """Answer a question from the manuals and return the chunks used as sources."""
    validate_question_length(request, body.question)  # 422 before the index guard
    require_index_ready(request)
    request_id: str = request.state.request_id
    result = await service.ask(
        body.question,
        session_key=body.session_key,
        product_id=body.product_id,
        request_id=request_id,
    )
    return AskResponse(
        answer=result.answer,
        status=result.status,
        sources=[Source.model_validate(s) for s in result.sources],
        session_key=result.session_key,
        request_id=request_id,
    )
