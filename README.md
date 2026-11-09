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

It prints a summary and exits with 0 on success and 1 (with the error message) when indexing failed. Ollama must be running with the embedding model pulled (`ollama pull nomic-embed-text`) when `LLM_PROVIDER=ollama`.

Settings: `DATA_DIR`, `INDEX_DIR`, `CHROMA_COLLECTION`, `CHUNK_SIZE` and `CHUNK_OVERLAP` (see `.env.example`).

### Chunking

Each manual is split along its `## ` sections, and every section is split on its own with a recursive character splitter (`CHUNK_SIZE`, `CHUNK_OVERLAP`). Text is never merged across sections, so a chunk answers from exactly one part of one manual (for example "Maintenance" of the KV-200). Every chunk carries metadata: source file, product ID and name (parsed from the `# <ID> – <name>` title) and section title, plus a stable, unique chunk ID. These are the fields the answer later reports as its sources. The chunk text starts with a short `<ID> <name> - <section>` line so similar products (KV-200 and KV-300) stay distinguishable. Files without the title heading are logged and skipped.

### Vector store

Chroma runs embedded and is persisted to `INDEX_DIR` (`./data/index`). It needs no extra service, keeps the setup to a single process and is enough for a manual set of this size. The collection name is `CHROMA_COLLECTION` plus the sanitized embedding model name, so indexes built with different embedding models never mix.

### Startup indexing trade-offs

- A fingerprint (SHA-256 over the sorted file names and contents, the embedding model, the chunk size and the overlap) is stored next to the index. If it is unchanged and all chunks are present, the persisted index is reused and no embedding calls are made. Otherwise the old chunks are deleted and the index is rebuilt without duplicates.
- A first start (or a start after a change) is slower because every chunk is embedded; later starts are fast.
- With several instances, each one indexes on its own (there is no coordination between instances), which costs duplicate embedding work. A shared index built once by a separate job is the next step.

## Model providers

The chat model and the embeddings are chosen with `LLM_PROVIDER` (`azure`, `gemini` or `ollama`, default `ollama`) and `EMBEDDING_PROVIDER` (defaults to `LLM_PROVIDER`). A missing variable or an unknown value stops the application at startup with an error that names the variable; there is never a silent fallback. Every external call uses `LLM_TIMEOUT_SECONDS` / `EMBEDDING_TIMEOUT_SECONDS` and `MAX_RETRIES`.

For the cloud providers, vendor failures are translated into three domain errors: a timeout (`llm_timeout`, HTTP 504), an unreachable provider or a 5xx answer (`llm_unavailable`, 503) and rate limiting (`llm_rate_limited`, 429). An HTTP 429 is retried at most `MAX_RETRIES` times with an exponential backoff starting at `RETRY_BACKOFF_SECONDS` before the rate-limit error is raised. This sits on top of the vendor client's own retries, so the worst case is more calls than `MAX_RETRIES` alone suggests. Error messages never contain keys or vendor messages.

### Configure Azure OpenAI

Azure OpenAI is the primary, production-like mode: a managed service with a stable API, predictable quotas and enterprise data-handling terms. Create a chat deployment and an embedding deployment in your Azure OpenAI resource, then set:

```env
LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_CHAT_DEPLOYMENT=<chat deployment name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding deployment name>
```

`AZURE_OPENAI_API_VERSION` has a default and may be omitted. The deployment names are the names you gave the deployments, not the model names.

### Configure Google Gemini

Gemini is the cloud mode that can be tried for free: create an API key in Google AI Studio and set:

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=<key>
GEMINI_CHAT_MODEL=gemini-3.1-flash-lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

Only `GOOGLE_API_KEY` is required. `gemini-3.1-flash-lite` is the small, low-latency, low-cost model, which is enough for answering from a few retrieved manual chunks and stays within free-tier rate limits. `gemini-embedding-001` is Google's general-purpose text embedding model. The free tier is rate limited, so a first indexing run may hit 429 answers; they are retried with backoff.

Availability of the embedding model: the key-gated integration test calls `embed_query` with `gemini-embedding-001`. It was not executed while this section was written (no key was available), so confirm it with your own key:

```bash
GOOGLE_API_KEY=<key> uv run pytest -m integration -k gemini
```

You can also list the models your key can use with `GET https://generativelanguage.googleapis.com/v1beta/models?key=<key>`. If the embedding model is not available to you, set `GEMINI_EMBEDDING_MODEL` to another model from that list.

### Mixed setup: cloud chat with local embeddings

Chat answers can come from Gemini while the embeddings stay local and free:

```env
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=ollama
GOOGLE_API_KEY=<key>
```

Run `ollama pull nomic-embed-text` first. The index collection name contains the embedding model, so changing the embeddings builds a new index instead of mixing vectors: this setup uses `manuals-ollama-nomic-embed-text`, while `EMBEDDING_PROVIDER=gemini` uses `manuals-gemini-gemini-embedding-001` (names checked with the provider factory). The old collection stays in `INDEX_DIR` unused.

### Integration tests for the cloud providers

```bash
uv run pytest -m integration
```

The Azure and Gemini tests make one real chat call and one embedding call. They are skipped (not failed) when the credentials are not set.
