# Graphiti Memory System

**Give your agent a real long-term memory.**

Graphiti Memory System turns raw chat logs into a living, queryable memory layer for LLM agents.  
Instead of replaying full history every time, agents get a compact, high-signal `memory_packet` with the facts, decisions, tasks, and entities that actually matter.

Result: **better continuity, lower token cost, and sharper responses across sessions**.

---

## Why this project exists

Most agents are brilliant in-session and forgetful between sessions.
This system solves that by continuously compiling conversation history into structured memory and serving it back right before the next interaction.

It is built for practical, production-like workflows (Pi, Claude CLI, and other agent clients) where memory must be:

- **Structured** (not just transcript stuffing)
- **Temporal** (what changed over time)
- **Traceable** (linked to source turns)
- **Retrievable** (fast, relevant, token-efficient)

---

## Architecture at a glance (layered)

1. **Agent Sessions (Input)**  
   Any agent session produces structured turns: messages, tool calls, timestamps, metadata.

2. **Raw Archive (Immutable)**  
   Full session JSON is written append-only (`data/raw`). This is the audit and replay source.

3. **Memory Compiler (Processing Core)**  
   A pipeline converts noisy turns into clean memory events:
   - Normalize
   - Split into semantic episodes
   - Classify episode type
   - Extract structured memory items
   - Resolve entity aliases
   - Detect contradictions / superseded facts
   - Select high-value items only

4. **Graph Core (Graphiti / Neo4j)**  
   Stores temporal memory as entities + relationships + episodes with provenance and status.

5. **Vector Index (Qdrant)**  
   Embeds episodes/facts/entities for semantic recall.

6. **Metadata Store (PostgreSQL)**  
   Tracks session index, entity registry, job queue, fact lifecycle, and system metrics.

7. **Retrieval Service (`/retrieve`)**  
   Runs hybrid retrieval (semantic + graph + temporal) and returns a compact `memory_packet`.

8. **New Agent Session (Output)**  
   Agent starts with curated memory context, not full transcript history.

9. **Live Dashboard (`/dashboard`)**  
   Operational visibility: services, queue health, growth, quality signals, connected agents.

10. **Dev Tooling & Iteration Loop**  
    Local-first stack, testable pipeline modules, and repeatable improvement cycle.

---

## How it works in one flow

`/ingest/turn` (streaming turns) → raw JSON archive → compiler job queue → memory pipeline → graph + vectors + metadata → `/retrieve` builds `memory_packet` → next agent run starts with focused memory.

This is memory as an **always-on background system**, not a one-time prompt trick.

---

## Tech stack

- **FastAPI** (ingest + retrieval + dashboard endpoints)
- **PostgreSQL** (session/job/fact metadata)
- **Neo4j + Graphiti** (temporal graph memory)
- **Qdrant** (vector search)
- **Docker Compose** (local orchestration)
- **LM Studio / local LLM endpoints** (model serving)

---

## Quick start

### 1) Configure environment

```bash
cp .env.example .env
# edit .env with your values
```

### 2) Start the full stack

```bash
./start.sh
```

### 3) Open dashboard

- http://localhost:8100/dashboard

---

## Key endpoints

- `POST /ingest/turn` — ingest one turn in real time
- `POST /ingest/close` — flush session tail
- `POST /ingest` — full session ingest / replay
- `POST /retrieve` — build memory packet for a new run
- `GET /metrics` — dashboard data
- `GET /dashboard` — live status UI
- `GET /healthz` — liveness

---

## Open-source note

This repository is public and intended for experimentation and extension.

`data/raw` is intentionally ignored by git (except `.gitkeep`) so local conversation archives do not get committed.

---

## Vision

We believe agent memory should feel like this:

- **Persistent like a knowledge system**
- **Precise like a retrieval engine**
- **Adaptive like a graph that evolves over time**

If you are building serious agent workflows, this project gives you a strong memory backbone to build on.
