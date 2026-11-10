# DocuQuery – RAG backend

## Problem

A large webshop sells household and DIY appliances, and every product comes with a user manual. These manuals run to many pages, so customers and customer support agents struggle to find specific answers in them. The target users are customers before a purchase, who compare product features and technical data; customers after a purchase, who turn to the manual with questions about usage, maintenance or troubleshooting; and customer support agents, who want to give accurate answers quickly. DocuQuery is an HTTP API that the webshop's search box can be built on. Users ask their question in natural language, for example: "How often should I descale the KV-200 coffee maker?" The system answers from the relevant passages of the manuals rather than from a manual as a whole. The answer also names its source, meaning the product and the manual section. This lets users verify the answer without having to read the entire document.

### Example questions

- **Usage:** How do I start the espresso program on the KV-200 coffee maker?
- **Maintenance:** How often should I descale the KV-200 coffee maker?
- **Troubleshooting:** Why is the red light flashing on the drill, and what should I do?
- **Technical data:** What are the maximum speed and the power of the impact drill?
- **Safety:** Can I use the chainsaw in the rain, and what protective equipment do I need?

### Scope and limitations

- Answers come exclusively from the user manuals.
- Price, stock, orders and product recommendations are not part of the application.
- The webshop's search interface (UI) is out of scope; the prototype provides only the API.
- There is no authentication.
- There is no quality measurement (answer quality is not evaluated).
- Conversation history is kept per `session_key`, in memory only, and is lost on restart.

## Sample data

Real manufacturer manuals are copyrighted, so the repository ships its own short manuals in `data/sample/`. All products and brands in them (KV-200, KV-300, DR-450, CS-350, RV-700, WM-800, AP-120) are fictional, and the documents contain no confidential or real manufacturer content. Each manual is a markdown file with the same sections: Safety, Setup, Usage, Maintenance, Troubleshooting and Technical data. The set includes two similar coffee makers (KV-200 and KV-300) with different descaling intervals, so retrieval has to pick the right product.

### Questions the documents can answer

- How do I start the espresso program on the KV-200 coffee maker? (`kv-200-espresso-coffee-maker.md`, Usage)
- How often should I descale the KV-200 coffee maker? (`kv-200-espresso-coffee-maker.md`, Maintenance: every 60 days or 200 brew cycles. The KV-300 drip coffee maker differs: every 30 days.)
- Why is the red light flashing on the DR-450 drill, and what should I do? (`dr-450-cordless-impact-drill.md`, Troubleshooting)
- What are the maximum speed and the power of the DR-450 impact drill? (`dr-450-cordless-impact-drill.md`, Technical data: 2000 rpm, 600 W)
- Can I use the CS-350 chainsaw in the rain, and what protective equipment do I need? (`cs-350-electric-chainsaw.md`, Safety)

### Questions the documents cannot answer

- How much does the KV-200 coffee maker cost?
- Is the WM-800 washing machine in stock?
- How do I descale the KV-500 coffee maker? (this product is not in the set)
- Which smartphones can control the RV-700 robot vacuum? (no manual states this)

## Indexing

There is no upload API. When the application starts, it loads the manuals from `DATA_DIR` (default `data/sample/`), splits them, embeds the chunks and writes them to the vector store. Indexing runs in a background task, so the server starts immediately. Until indexing has succeeded, endpoints that need the index respond with `503` and the code `INDEX_NOT_READY`. If the data directory is missing or empty, the state becomes `failed` with a clear message, the app still starts and the same `503` is returned. The state (`indexing`, `ready` or `failed`, document and chunk counts, embedding model, timestamp, error) is kept on `app.state.index_status`.

The same pipeline runs from the command line:

```bash
uv run python -m app.ingest
```

It prints a summary and exits with 0 on success and 1 (with the error message) when indexing failed. Ollama must be running with the embedding model pulled (`ollama pull nomic-embed-text`) when `EMBEDDING_PROVIDER=ollama`.

Settings: `DATA_DIR`, `INDEX_DIR`, `CHROMA_COLLECTION`, `CHUNK_SIZE` and `CHUNK_OVERLAP` (see `.env.example`).

### Chunking

Each manual is split along its `## ` sections, and every section is split on its own with a recursive character splitter (`CHUNK_SIZE`, `CHUNK_OVERLAP`). A section is a Markdown level-2 heading (`## Maintenance`) and runs until the next one; the heading text becomes the section title, and any text before the first `##` is the introduction. Only exactly two `#` start a section: the `#` title and `###` sub-headings stay inside the surrounding text. Text is never merged across sections, so a chunk answers from exactly one part of one manual (for example "Maintenance" of the KV-200). Every chunk carries metadata: source file, product ID and name (parsed from the `# <ID> – <name>` title) and section title, plus a stable, unique chunk ID. These are the fields the answer later reports as its sources. The chunk text starts with a short `<ID> <name> - <section>` line so similar products (KV-200 and KV-300) stay distinguishable. Files without the title heading are logged and skipped.

### Vector store

Chroma runs embedded and is persisted to `INDEX_DIR` (`./data/index`). It needs no extra service, keeps the setup to a single process and is enough for a manual set of this size. The collection name is `CHROMA_COLLECTION` plus the sanitized embedding model name, so indexes built with different embedding models never mix.

### Startup indexing trade-offs

- A fingerprint (SHA-256 over the sorted file names and contents, the embedding model, the chunk size and the overlap) is stored next to the index. If it is unchanged and all chunks are present, the persisted index is reused and no embedding calls are made. Otherwise the old chunks are deleted and the index is rebuilt without duplicates.
- A first start (or a start after a change) is slower because every chunk is embedded; later starts are fast.
- With several instances, each one indexes on its own (there is no coordination between instances), which costs duplicate embedding work. A shared index built once by a separate job is the next step.

## Conversation history

Every answer returns a `session_key`; sending it with the next question continues the conversation.

- **How it works:** history is a dict of `session_key` to messages, guarded by an `asyncio.Lock`. The history is loaded before the question is processed and saved afterwards (question plus answer). Clarifying questions asked by the assistant are remembered too, so the user's next message is read in that context.
- **Key handling:** without a `session_key` (or with an empty one) the request starts a new conversation and the server generates a random key (32 hex characters), stores the question and answer under it and returns it in the response. With a `session_key` (8-64 characters: letters, digits, `-`, `_`) the stored history is used and extended, and the same key is returned; an unknown key starts a new conversation under that key. A request without a key never sees earlier history.
- **Setting:** `HISTORY_MAX_MESSAGES` (default 20, minimum 1) is the number of most recent messages kept per session; older messages are dropped.
- **Limits:** history lives in process memory only and is lost on restart. Several instances do not share it, so a follow-up routed to another instance starts empty. It is not a LangGraph checkpointer, and the key is not authentication: anyone who knows a key can read and extend that session. The number of sessions is not bounded, and since every request without a key creates one, memory grows with the request count until the restart.
- **Production alternative:** store the transcripts in Redis (with a TTL) or PostgreSQL, behind the same get/save interface, so that all instances share them and they survive restarts.

## Question flow

Every question runs through a LangGraph graph that is compiled once at startup. The state (question, history, hits, answer, sources, status) is created per request, so requests share nothing except the session store.

```mermaid
flowchart TD
    __start__([start]) --> load_history
    load_history --> condense
    condense --> assess
    assess -- clear --> retrieve
    assess -- unclear --> clarify
    retrieve -- relevant hit --> generate
    retrieve -- no hit --> no_answer
    generate --> format
    format --> save_history
    clarify --> save_history
    no_answer --> save_history
    save_history --> __end__([end])
```

- `load_history`: loads the stored messages for the `session_key` (empty for a new conversation).
- `condense`: with history, one LLM call rewrites a follow-up into a standalone question; without history it makes no LLM call.
- `assess`: one LLM call decides whether the question is clear enough to search. If the answer cannot be parsed, the question is treated as clear so retrieval still runs.
- `clarify`: the LLM asks one targeted clarifying question (`status: clarification_needed`); retrieval and generation are skipped.
- `retrieve`: fetches the top `RETRIEVAL_TOP_K` chunks (server setting, 1 to 20; an out-of-range value stops startup) with relevance scores (0 to 1). If at least one score reaches `RETRIEVAL_RELEVANCE_THRESHOLD` the flow continues with `generate`, otherwise with `no_answer`.
- `generate`: answers from the chunks at or above the threshold only, and says so when the context does not contain the answer.
- `no_answer`: a fixed "no relevant information" reply (`status: no_answer`, no sources, no LLM call).
- `format`: sets `status: answered` and the sources (product ID and name, section, source file, chunk ID, relevance score, snippet) of the chunks used, and logs them.
- `save_history`: appends the question and the reply (answer or clarifying question) to the session.

Provider errors (timeout, unavailable, rate limited) leave the graph unchanged as domain errors and nothing is saved to the history; unexpected errors are wrapped in a generic pipeline error (`RAG_PIPELINE_ERROR`, 502) without internal details.

## Ask endpoint

`POST /api/v1/ask` runs the question flow and returns the answer with the chunks it used. The route only calls `QAService`, which invokes the compiled graph; the graph and the service are built once in the application lifespan and live on `app.state`. Swagger UI (`/docs`) shows request examples (successful question, follow-up, clarification, product-scoped question) and response examples for every status and error.

Request body:

| Field | Required | Description |
|---|---|---|
| `question` | yes | The question. Whitespace is trimmed; a blank question is a `422`. |
| `session_key` | no | 8-64 characters (letters, digits, `-`, `_`) linking follow-up questions. Without it (or with an empty string) a new conversation starts and the response returns a generated key. |
| `product_id` | no | Restricts retrieval to one product's manual (for example `KV-200`). Applied as a metadata filter on the shared index; without it, retrieval searches all manuals. An empty string counts as absent. |

Response body: `answer`, `status`, `sources`, `session_key` (the request's key or a newly generated one) and `request_id`. Each source has `product_id`, `product_name`, `section`, `chunk_id`, `score` (0 to 1) and a `snippet` capped at `SNIPPET_MAX_CHARS`.

| `status` | Meaning | `sources` |
|---|---|---|
| `answered` | A grounded answer from relevant chunks | the chunks used |
| `clarification_needed` | The question is unclear; `answer` is a clarifying question | empty |
| `no_answer` | No chunk reached the relevance threshold | empty |

**Unknown product:** a `product_id` that is not in the index returns `404` with the code `PRODUCT_NOT_FOUND` (not `422`, because the request is well formed and the resource it names does not exist). The known IDs come from the index itself; matching ignores case.

**Check order:** the request body is validated first (`422`), then the index guard runs (`503 INDEX_NOT_READY`). A malformed request is wrong whatever the index state, so while indexing is running or has failed an invalid request still gets a `422` and a valid one gets the `503`.

**Invalid vs unclear:** a technically invalid request (missing or wrong-typed field, malformed JSON, blank question, question longer than `MAX_QUESTION_LENGTH`, bad `session_key`) is a `422`. A well-formed but unclear question (for example "My machine doesn't work.") is not an error: it returns `200` with `status: clarification_needed` and the clarifying question as `answer`. With a `session_key` the next question is interpreted together with the stored history.

**Errors** always use the body `{"detail", "code", "request_id"}`; `detail` is a readable string and never contains a stack trace or internal exception text. A `422` adds `errors`, a list of `{"field", "message", "type"}` items; the submitted values are not echoed. Example:

```json
{
  "detail": "Invalid request: question: String should have at least 1 character",
  "code": "VALIDATION_ERROR",
  "request_id": "3f2b8c1e9d7a4e6f",
  "errors": [
    {"field": "question", "message": "String should have at least 1 character", "type": "string_too_short"}
  ]
}
```

| Status | `code` | When |
|---|---|---|
| 404 | `PRODUCT_NOT_FOUND` | `product_id` is not in the index |
| 404 / 405 / other | `NOT_FOUND`, `METHOD_NOT_ALLOWED`, ... | Unknown route, wrong method (same unified body) |
| 422 | `VALIDATION_ERROR` | Invalid request body (see above) |
| 502 | `LLM_UNAVAILABLE` | Provider unreachable or answering with a 5xx error |
| 502 | `LLM_EMPTY_RESPONSE` | The model returned an empty answer |
| 502 | `RAG_PIPELINE_ERROR` | Unexpected failure inside the question flow; generic message. It has its own code so a bug is not reported as the model being unreachable |
| 503 | `INDEX_NOT_READY` | Indexing is running or failed |
| 503 | `LLM_RATE_LIMITED` | Provider still rate limits after the bounded retries; carries a `Retry-After` header |
| 504 | `LLM_TIMEOUT` | Provider did not answer within the timeout |
| 500 | `INTERNAL_ERROR` | Any other unhandled exception; generic message |

`Retry-After` is an integer number of seconds. It is taken from the provider's `Retry-After` header when that is a number; otherwise `RATE_LIMIT_RETRY_AFTER_SECONDS` (default 5) is used.

**Logging:** the application's logs go to stdout as `<time> <LEVEL> <logger> <message>` lines, next to uvicorn's own output. Read them in the terminal that runs uvicorn, or with `docker compose logs -f app` under Docker. `LOG_LEVEL` (`DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`, default `INFO`; any other value stops startup) sets the level. Every request is logged at INFO with method, path, status, duration in milliseconds and `request_id`; health probes are logged at DEBUG so they do not flood the log. Every `4xx` is logged at WARNING with the error code, the affected field names (for `422`) and the `request_id`; the request body is not logged. Every `5xx` and every unexpected exception is logged at ERROR with the stack trace, in the log only. `clarification_needed` and `no_answer` are normal outcomes and are logged at INFO.

**Request ID:** the `X-Request-ID` header is used when the client sends one, otherwise an ID is generated. It is returned in the `X-Request-ID` response header and in `request_id` of success and error bodies, and appears in the log lines of the request together with each source's product, section, chunk ID and score. The question is logged truncated to `LOG_QUESTION_MAX_CHARS`.

Sample requests:

```bash
# A question
curl -s -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How often should I descale the KV-200 coffee maker?"}'

# A follow-up: send the session_key returned by the previous answer (or your own key)
curl -s -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "And the KV-300?", "session_key": "demo-session-01"}'

# Restricted to one product
curl -s -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I descale it?", "product_id": "KV-200"}'

# Unclear question: 200 with status "clarification_needed" and a clarifying question
curl -s -X POST http://localhost:8000/api/v1/ask   -H "Content-Type: application/json"   -d '{"question": "How do I clean it?"}'

# Empty question: 422 with the unified error body (shown above)
curl -s -X POST http://localhost:8000/api/v1/ask   -H "Content-Type: application/json"   -d '{"question": ""}'
```

Health requests are in the next section.

## Health endpoints

Both endpoints are at the root, outside `/api/v1`, are listed in Swagger UI with an example for every state, and are not affected by the index guard of the ask endpoint. Every response carries `X-Request-ID`.

**`GET /health`** (liveness) returns `200` with `{"status": "ok"}` whenever the process is serving, whatever the indexing state. It touches neither the index nor any provider.

```bash
curl -s http://localhost:8000/health
```

**`GET /health/ready`** (readiness) reports the startup indexing state:

| Field | Meaning |
|---|---|
| `state` | `indexing`, `ready` or `failed` |
| `document_count`, `chunk_count` | Size of the index |
| `embedding_model` | Embedding model the index was built with |
| `last_indexed_at` | UTC time (ISO 8601) of the last indexing status change |
| `error` | Reason when `failed`, otherwise `null` |
| `llm` | Only with `check_llm=true` (see below) |

| Condition | Status code |
|---|---|
| `state` is `ready` and `chunk_count >= 1` | `200` |
| `indexing`, `failed`, or `ready` with zero chunks | `503`, with the same body so clients can read the state |

```bash
curl -s -i http://localhost:8000/health/ready
```

**Optional LLM check:** `GET /health/ready?check_llm=true` also sends one minimal request to the configured chat model and reports `"llm": {"status": "ok" | "unreachable" | "timeout"}`. It is off by default because it costs latency and, on cloud providers, a few tokens. It is informational: it never changes the status code or `state`, and a failure is logged at WARNING without exposing exception details. The call is bounded by `HEALTH_LLM_TIMEOUT_SECONDS` (default 5).

```bash
curl -s -i "http://localhost:8000/health/ready?check_llm=true"
```

## Model providers

Three variables choose the models, and all three are required:

| Variable | Values | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `azure`, `gemini`, `ollama` | Provider of the chat model |
| `EMBEDDING_PROVIDER` | `azure`, `gemini`, `ollama` | Provider of the embeddings, chosen independently of `LLM_PROVIDER` |
| `EMBEDDING_MODEL` | model name | Embedding model of `EMBEDDING_PROVIDER`: `nomic-embed-text` (Ollama), `gemini-embedding-001` (Gemini) or the embedding deployment name (Azure) |

There are no defaults and no fallbacks: the embeddings never follow `LLM_PROVIDER`, and a missing or unknown value stops the application (and the ingest command) at startup with an error that names the variable. Because chat and embeddings are always configured separately, a mixed setup changes only these lines. The chat model names stay provider-specific (`AZURE_OPENAI_CHAT_DEPLOYMENT`, `GEMINI_CHAT_MODEL`, `OLLAMA_CHAT_MODEL`). Every external call uses `LLM_TIMEOUT_SECONDS` / `EMBEDDING_TIMEOUT_SECONDS` and `MAX_RETRIES`.

For the cloud providers, vendor failures are translated into three domain errors: a timeout (`LLM_TIMEOUT`, HTTP 504), an unreachable provider or a 5xx answer (`LLM_UNAVAILABLE`, 502) and rate limiting (`LLM_RATE_LIMITED`, 503 with a `Retry-After` header). An HTTP 429 is retried at most `MAX_RETRIES` times with an exponential backoff starting at `RETRY_BACKOFF_SECONDS` before the rate-limit error is raised. This sits on top of the vendor client's own retries, so the worst case is more calls than `MAX_RETRIES` alone suggests. Error messages never contain keys or vendor messages.

Ollama runs locally, so there are no rate limits and no retries. Its failures use the same domain errors (a timeout is `LLM_TIMEOUT`, 504; an unreachable server or a model that is not pulled is `LLM_UNAVAILABLE`, 502), and the message names the base URL and, for a missing model, the `ollama pull <model>` command.

### Configure Azure OpenAI

Azure OpenAI is the primary, production-like mode: a managed service with a stable API, predictable quotas and enterprise data-handling terms. Create a chat deployment and an embedding deployment in your Azure OpenAI resource, then set:

```env
LLM_PROVIDER=azure
EMBEDDING_PROVIDER=azure
EMBEDDING_MODEL=<embedding deployment name>
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_DEPLOYMENT=<chat deployment name>
```

`AZURE_OPENAI_API_VERSION` has a default and may be omitted. The deployment names are the names you gave the deployments, not the model names.

### Configure Google Gemini

Gemini is the cloud mode that can be tried for free: create an API key in Google AI Studio and set:

```env
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001
GOOGLE_API_KEY=<key>
GEMINI_CHAT_MODEL=gemini-3.1-flash-lite
```

Besides the three provider variables only `GOOGLE_API_KEY` is required; `GEMINI_CHAT_MODEL` has a default. `gemini-3.1-flash-lite` is the small, low-latency, low-cost model, which is enough for answering from a few retrieved manual chunks and stays within free-tier rate limits. `gemini-embedding-001` is Google's general-purpose text embedding model. The free tier is rate limited, so a first indexing run may hit 429 answers; they are retried with backoff.

To confirm that the embedding model is available for your key, run the key-gated integration test, which calls `embed_query` with the configured model:

```bash
GOOGLE_API_KEY=<key> uv run pytest -m integration -k gemini
```

You can also list the models your key can use with `GET https://generativelanguage.googleapis.com/v1beta/models?key=<key>`. If the embedding model is not available to you, set `EMBEDDING_MODEL` to another model from that list.

### Configure Ollama (free local mode)

Ollama runs the chat model and the embeddings on your own machine: no account, no key, no cost. `.env.example` is set up for this mode.

1. Install Ollama from https://ollama.com/download (Windows, macOS, Linux).
2. Pull the two default models:

   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

3. Make sure the server is running. The desktop installers start it automatically; otherwise run `ollama serve`. It listens on `http://localhost:11434`.
4. Configure the application (a `.env` copied from `.env.example` already contains these values):

   ```env
   LLM_PROVIDER=ollama
   EMBEDDING_PROVIDER=ollama
   EMBEDDING_MODEL=nomic-embed-text
   OLLAMA_BASE_URL=http://localhost:11434
   OLLAMA_CHAT_MODEL=llama3.2:3b
   ```

5. Build the index and start the application:

   ```bash
   uv sync
   uv run python -m app.ingest
   uv run uvicorn app.main:app --reload
   ```

No key variable is needed. The index collection for this setup is `manuals-ollama-nomic-embed-text`. Local mode is verified at the level of the chat model, the embeddings and the retriever by the integration test below; `POST /api/v1/ask` needs a running Ollama to answer.

If Ollama is not running, ingest prints `Ollama is not reachable at <base url>. Start it with 'ollama serve' or check OLLAMA_BASE_URL` and exits with 1; if a model is missing, the message names it and says `ollama pull <model>`. The application itself still starts, and endpoints that need the index answer `503`.

Integration test against the real server (indexes one sample manual into a temporary directory, retrieves a relevant chunk and makes one chat call):

```bash
uv run pytest -m integration -k ollama
```

It is skipped, not failed, when Ollama is unreachable or a configured model is not pulled.

#### Hardware, speed and quality (estimates)

These figures are rough estimates based on the public model sizes.

- **Memory:** `llama3.2:3b` is about a 2 GB download and needs roughly 3 to 4 GB of free RAM (or VRAM) while loaded; `nomic-embed-text` is about 270 MB and needs well under 1 GB. Plan for about 8 GB of RAM in total, with room for the operating system and the application.
- **CPU or GPU:** both models run on a CPU only. A GPU with 4 GB or more of VRAM (or Apple Silicon) is considerably faster but not required.
- **Speed on a CPU:** expect a few seconds before the first token and roughly 5 to 20 generated tokens per second on a recent laptop CPU, so a short answer takes several seconds to half a minute. Indexing the seven sample manuals should take on the order of tens of seconds to a few minutes on a CPU the first time; later starts reuse the persisted index. Raise `LLM_TIMEOUT_SECONDS` on slow machines, and note that the first request after a pause also loads the model into memory.
- **Quality limits:** a 3B model gives noticeably weaker answers than the cloud models. It can ignore the retrieved context or make up details (hallucination), follows instructions less reliably, handles long context poorly and is clearly weaker in languages other than English. The sample manuals are in English, which is the best case.
- **Concurrency:** a single local server processes few requests at a time; parallel questions queue up. This mode is meant for development, demos and offline use.

### Mixed setup: cloud chat with local embeddings

Chat answers can come from Gemini while the embeddings stay local and free:

```env
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
GOOGLE_API_KEY=<key>
```

Run `ollama pull nomic-embed-text` first. The index collection name contains the embedding model, so changing the embeddings builds a new index instead of mixing vectors: this setup uses `manuals-ollama-nomic-embed-text`, while `EMBEDDING_PROVIDER=gemini` with `EMBEDDING_MODEL=gemini-embedding-001` uses `manuals-gemini-gemini-embedding-001`. The old collection stays in `INDEX_DIR` unused.

## Server address and API docs

The API listens on port `8000` by default:

| URL | What it is |
|---|---|
| http://localhost:8000/docs | Swagger UI with request and response examples |
| http://localhost:8000/redoc | ReDoc view of the same API |
| http://localhost:8000/openapi.json | OpenAPI schema |
| http://localhost:8000/health | Liveness check |
| http://localhost:8000/api/v1/ask | Ask endpoint (`POST`) |

The port depends on how the server is started:

- **Locally** (`uv run uvicorn app.main:app --reload`): uvicorn's default, `127.0.0.1:8000`. Use `--port <port>` to pick another one.
- **Docker:** the container always listens on `8000`; compose publishes it on the host port `APP_PORT` (default `8000`). With `APP_PORT=8080` in `.env` the API is at http://localhost:8080.

Neither mode opens a browser; open the Swagger UI by hand. It loads while startup indexing is still running, but `/api/v1/ask` answers `503 INDEX_NOT_READY` until `GET /health/ready` reports `ready`.

## Run with Docker

The `Dockerfile` builds a `python:3.12-slim` image that installs the locked dependencies with uv (`uv sync --frozen --no-dev`), runs as the non-root user `appuser` and starts `uvicorn app.main:app` on `0.0.0.0:8000`. The sample manuals are baked into the image; the index is not. `docker-compose.yml` defines:

- `app`: the API with Ollama as its provider (local mode); `OLLAMA_BASE_URL` is set to `http://ollama:11434` by compose, overriding `.env`. As the local-mode service it fills in `LLM_PROVIDER=ollama`, `EMBEDDING_PROVIDER=ollama` and `EMBEDDING_MODEL=nomic-embed-text` when `.env` does not set them, so it starts without a `.env`; values from `.env` take precedence.
- `ollama`: the Ollama server (`ollama/ollama`), with the models stored in the named volume `ollama`.
- `app-cloud`: the same image without Ollama, in the `cloud` profile, for Azure OpenAI or Gemini.

The index lives in the named volume `index` (mounted at `INDEX_DIR=/app/data/index`), so it survives restarts and is shared by the `app` and the ingest command. A `.env` file is optional; copy `.env.example` to `.env` to change settings. The health check of the API container calls `GET /health`.

### Local mode (Ollama)

```bash
docker compose up --build -d
docker compose exec ollama ollama pull llama3.2:3b
docker compose exec ollama ollama pull nomic-embed-text
docker compose run --rm app python -m app.ingest
docker compose restart app
curl http://localhost:8000/health
curl -X POST http://localhost:8000/api/v1/ask   -H "Content-Type: application/json"   -d '{"question": "How do I descale the espresso machine?"}'
```

Pull both models before the ingest step and before the first question: without them the startup indexing fails (the API still starts and `/health` answers 200, but `/health/ready` reports `503` and `/api/v1/ask` returns `503`). The pulls are a one-time download of about 2 GB (`llama3.2:3b`) and 270 MB (`nomic-embed-text`) into the `ollama` volume; plan for about 8 GB of RAM and several GB of free disk for the images and models (see the hardware notes above). The app also indexes at startup, so the ingest command only pre-builds the index in the shared volume; the restart makes a running app pick it up. To use an NVIDIA GPU, add a `deploy.resources.reservations.devices` entry to the `ollama` service (requires the NVIDIA Container Toolkit); this is optional and not configured here.

### Cloud mode (Azure OpenAI or Gemini)

Set the three provider variables and the keys in `.env` (names in `.env.example`), for example:

```env
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001
GOOGLE_API_KEY=<key>
```

or the Azure equivalent with the `AZURE_OPENAI_*` variables, then start only the cloud service. `app-cloud` has no provider defaults, so a missing variable stops it at startup:

```bash
docker compose --profile cloud up --build -d app-cloud
docker compose run --rm app-cloud python -m app.ingest
```

Ollama is not started. Both app services publish port 8000, so run only one of them at a time; set `APP_PORT` in `.env` to publish another host port. Stop everything with `docker compose --profile cloud down` (add `-v` to delete the index and model volumes).

## Testing

```bash
uv sync                               # install dependencies
uv run pytest -m "not integration"    # offline suite
uv run pytest -m integration          # integration suite
uv run ruff check .                   # lint
uv run ruff format --check .          # format check
uv run mypy app                       # type check
```

**Offline suite.** It needs no `.env`, no API keys, no Ollama and no network: all model, embedding and vector store calls use test doubles from `tests/fakes.py`, and an autouse fixture in `tests/conftest.py` makes any non-local socket connection fail (local connections stay allowed, so the HTTP test client works). The doubles exist only under `tests/`; `app/` contains real implementations only. It covers these paths, both at graph level (`tests/unit/test_graph.py`) and at HTTP level (`tests/api/`):

- success with sources;
- invalid input (422 in the unified error body);
- no relevant hits, both when every score is below the threshold and when retrieval returns zero documents (status `no_answer`, no generation call, empty sources);
- model error, rate limit and model timeout (502, 503, 504 with `code` and `request_id`);
- embedding failure or timeout during retrieval.

**Integration suite.** It makes real calls to Azure OpenAI, Gemini or a local Ollama server. Each test is skipped, not failed, when its credentials are missing or Ollama is unreachable or lacks a model, so the command ends with exit code 0 on a machine without any of them (see "Configure Ollama" for the local setup).

**CI.** `.github/workflows/ci.yml` runs lint, format check, type check and the offline suite on every push and pull request with Python 3.12 and no secrets.

