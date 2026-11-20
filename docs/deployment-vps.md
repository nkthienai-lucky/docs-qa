# VPS deployment

The selected target is Docker Compose on the VPS, not Render. The checked-in
Render template is an optional legacy alternative, not the active deployment.

## Current rollout (2026-11-20)

- `docsqa-vps:cpu` built on the VPS; CPU Torch 2.15.0 and Docling/API/UI
  imports passed with networking disabled. Docker reports 3.23 GB for the image.
- `docsqa-db-1` is running and healthy, with no published database port.
- API startup is **pending runtime provider configuration and approval**. No
  successful `/ready`, provider smoke, document ingestion or end-to-end QA is
  claimed. The existing chatbot services were left unchanged.
- This is private staging, not a certified public production service. Application
  auth/rate limits and remaining DB concurrency/recovery review are still open.

## Footprint experiments (2026-11-20)

These are isolated packaging spikes, not deployed replacements. No parser,
embedding identity or evaluation baseline was changed.

| Variant | Measured footprint | Verification | Limitation |
| --- | --- | --- | --- |
| Current CPU deployment image | Docker reports 3.23 GB; virtualenv 1,929 MiB | Offline Torch/Docling/API/UI imports | Includes unnecessary UI/evaluation/local-reranker dependencies |
| API-only experiment | Docker reports 346 MB (includes uv and Python, excludes mounted application source) | Offline API import and creation of all 9 routes | No PDF parser; not a working upload deployment |
| Docling retained, UI/eval/local-reranker removed | Virtualenv 1,392 MiB; final image not built/measured | Offline Docling import and creation of all 9 routes | Still carries Torch/OpenCV; parsing quality, memory and model assets not exercised |

The current virtualenv's largest directories include Torch 714 MiB, Gradio
185 MiB, PyArrow 163 MiB and combined OpenCV directories about 152 MiB.
Directory byte totals and Docker image reporting use different accounting;
do not subtract them to claim a final image size or filesystem reclamation.

The retained-parser spike exported frozen requirements with `uv export --prune`
for `gradio`, `ragas`, `sentence-transformers`, `langchain-community` and `pytest`.
It retained Docling/Transformers, removed CUDA/NVIDIA/Triton packages and installed
matching CPU Torch as in the existing Dockerfile. The API-only spike additionally
pruned Docling and Transformers. Both need **explicit python-multipart==0.0.33**
from the existing lock: uploads currently receive it transitively via Gradio.
Simply pruning Gradio without adding multipart makes route creation fail.
Do not promote the experimental requirement files without declaring production
extras/dependency groups and verifying them against the lock.

Recommended order:

1. Package an API+Docling extra separately from UI/eval/local inference. Preserve
   current parsing behavior; build and measure the actual image before switching.
2. For substantial savings, assess API-only serving with separately operated
   Docling ingestion, or an explicitly selected native-text parser. The former
   needs an authenticated ingestion protocol; the latter changes extraction and
   requires page/citation/quality regression tests, explicit index provenance and
   a scan/OCR policy. Neither is implemented by these spikes.
3. Do not rewrite this established Python evaluation project merely to reduce
   image size. Future repositories default to Rust, but model/parser assets must
   still be accounted for independently.

No runtime RSS, peak build disk, parser model cache size, PDF parse latency or
end-to-end provider availability was measured in this experiment. Root storage
is the bottleneck (about 3 GB free during the experiments); avoid additional full
image builds there without capacity checks. Experimental files are local to the
VPS under `/data/docsqa/experiments`; they contain no provider credentials.

## Files and storage

Deployment directory: `/data/docsqa/app`. Private environment:
`/data/docsqa/app/.env.vps` (mode 0600; never print, commit or copy into an image).
A fresh DB password is generated on the server. Do not replace it once the DB
has been initialized without an explicit password rotation procedure.

Persistent bind mounts on the VPS `/data` disk:

| Host | Container | Purpose |
| --- | --- | --- |
| `/data/docsqa/postgres` | `/var/lib/postgresql/data` | PostgreSQL 17 |
| `/data/docsqa/uploads` | `/var/data` | PDF uploads in `uploads/` |
| `/data/docsqa/model-cache` | `/home/docsqa/.cache` | Parser/model cache |

Upload/cache parents are owned by UID/GID 10001. The API runs non-root and its
startup script still requires `/var/data` to be an actual mount. Do not loosen
that check. Parser assets are not downloaded at build time; their resource,
network and license requirements must be verified before accepting documents.

## Complete private configuration

Edit `.env.vps` on the server, not in chat. Set the following nine independent
provider values (chat/embedding use OpenAI-compatible APIs, rerank uses the
full-list Cohere-style contract documented in README):

```text
RUNTIME_CHAT_BASE_URL
RUNTIME_CHAT_TOKEN
RUNTIME_CHAT_MODEL
RUNTIME_EMBEDDING_BASE_URL
RUNTIME_EMBEDDING_TOKEN
RUNTIME_EMBEDDING_MODEL
RUNTIME_RERANK_BASE_URL
RUNTIME_RERANK_TOKEN
RUNTIME_RERANK_MODEL
```

Select the embedding query instruction appropriate for the model via
`RUNTIME_EMBEDDING_QUERY_INSTRUCTION`; empty disables the default wrapper.
Retain the generated `DATABASE_URL` pointing at Compose host `db` and the
matching `POSTGRES_PASSWORD`. Never reuse eval judge credentials implicitly.

Only after approving provider costs/privacy and synthetic startup calls, set
`RUNTIME_DEPLOYMENT_APPROVED=true`. Startup immediately makes an embedding
probe. Actual corpus upload requires separate licensing/data-egress approval.

Run commands from `/data/docsqa/app`:

```sh
docker compose --env-file .env.vps -f compose.vps.yaml up -d --wait db
docker compose --env-file .env.vps -f compose.vps.yaml --profile runtime up -d --wait
curl --fail http://127.0.0.1:8001/health
curl --fail http://127.0.0.1:8001/ready
```

`/health` is process-only; `/ready` validates DB/index configuration, not live
chat/rerank service health. Use the explicit synthetic provider smoke only after
network/cost approval. Do not use held-out evaluation as a deployment smoke.

## Private access

The API publishes **only `127.0.0.1:8001`** on the VPS. Use SSH forwarding:

```sh
ssh -N -L 8001:127.0.0.1:8001 -p 2222 ibssh_x3aoof4mo7zmvxhvmv65g6wjwy@ssh.instabox.sh
```

Then access `http://127.0.0.1:8001/docs` locally. Port 8080 belongs to the
existing chatbot; do not change its containers or proxy. No public DocsQA URL
is provisioned. Never change the bind to `0.0.0.0` without authenticated ingress,
rate/size/concurrency controls, TLS, quota/budget limits and a security review.

## Build, updates and rollback

```sh
docker build --progress plain -t docsqa-vps:cpu .
```

The multi-stage image exports frozen application dependencies and substitutes
only Torch's matching CPU distribution; CUDA/NVIDIA/Triton packages are excluded.
Local GPU development continues to use the original `uv.lock`. Verify Torch,
Docling and API imports after every image build. Retain a previous image tag
before an update and recreate only the API to roll back. Do not roll back onto
an incompatible embedding identity or reset tables to fix startup.

The VPS root filesystem is small. Monitor `df -h / /data` and `docker system df`
before builds; Docker layers remain on root. Do not globally prune images/cache
or move the Docker daemon because other services share it.

## Backup and recovery

Before ingest/update, stop the API to keep PDFs and database consistent. Back up
Postgres using `pg_dump` (custom format), then archive the uploads tree and record
the image/model configuration. Store encrypted backups off the VPS; `/data` is
not itself a backup. Restore to a separate database and upload directory, check
embedding identity and readiness, then switch configuration. Never run
`docker compose down -v`, delete bind mounts, or reset an existing index as an
automatic repair. VPS lifetime and `/data` retention remain hosting-provider
contracts, not guarantees made by this repository.
