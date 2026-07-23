-- Graphiti Memory System — PostgreSQL schema
-- Applied automatically on first container start via docker-entrypoint-initdb.d

-- Which sessions have been ingested
CREATE TABLE IF NOT EXISTS sessions (
    session_id               TEXT PRIMARY KEY,
    source_agent              TEXT,
    ingested_at               TIMESTAMPTZ,
    status                    TEXT,   -- pending | processing | done | error | pending_close
    episode_count             INT DEFAULT 0,
    -- Cursor for real-time sliding-window processing: the message_id of the
    -- last turn that has already been run through the compiler pipeline.
    -- Lets the compiler worker process only the new tail of turns on each
    -- job instead of reprocessing the whole session every time.
    last_processed_message_id TEXT
);

-- Canonical entity registry
CREATE TABLE IF NOT EXISTS entities (
    canonical_id    TEXT PRIMARY KEY,
    canonical_name  TEXT NOT NULL,
    type            TEXT,
    aliases         TEXT[] DEFAULT '{}',   -- all known surface forms
    first_seen      TIMESTAMPTZ,
    last_seen       TIMESTAMPTZ
);

-- Fact lifecycle
CREATE TABLE IF NOT EXISTS facts (
    fact_id         TEXT PRIMARY KEY,
    entity_ids      TEXT[] DEFAULT '{}',
    type            TEXT,
    status          TEXT DEFAULT 'active',   -- active | outdated | discarded
    confidence      FLOAT,
    session_id      TEXT REFERENCES sessions(session_id),
    source_agent    TEXT,
    created_at      TIMESTAMPTZ DEFAULT now(),
    superseded_by   TEXT    -- fact_id of replacement, nullable
);

-- Compiler job queue
CREATE TABLE IF NOT EXISTS compiler_jobs (
    job_id          SERIAL PRIMARY KEY,
    session_id      TEXT REFERENCES sessions(session_id),
    status          TEXT DEFAULT 'queued',   -- queued | running | done | error
    attempts        INT DEFAULT 0,
    error_message   TEXT,
    -- True when this job was enqueued by POST /ingest/close (session end).
    -- Forces the compiler to process the buffered tail of turns even if
    -- it's smaller than MIN_WINDOW_SIZE, instead of deferring it.
    force_flush     BOOLEAN DEFAULT false,
    created_at      TIMESTAMPTZ DEFAULT now(),
    updated_at      TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_compiler_jobs_status ON compiler_jobs(status);
CREATE INDEX IF NOT EXISTS idx_facts_status ON facts(status);
CREATE INDEX IF NOT EXISTS idx_facts_session ON facts(session_id);

-- Session 7: review queue for Contradiction Checker's `flag_for_review`
-- action (contradiction score 0.6-0.85, per spec Layer 3 Step 6). Prior to
-- this table, `flag_for_review` was handled identically to `auto_update`
-- (old fact marked outdated + supersedes link created immediately) — a
-- known divergence from the spec's literal "flag for review" language,
-- logged in the tech-spec implementation log since session 6. Rows here
-- represent a contradiction the system is *not confident enough* to apply
-- automatically; the new fact is still written normally (per spec: "both
-- versions are kept"), but the old fact is left `active` and unlinked
-- until a human (or a future auto-resolution pass) reviews the pair.
CREATE TABLE IF NOT EXISTS review_queue (
    review_id           SERIAL PRIMARY KEY,
    new_fact_id         TEXT REFERENCES facts(fact_id),
    existing_fact_id     TEXT REFERENCES facts(fact_id),
    contradiction_score FLOAT,
    status              TEXT DEFAULT 'pending',  -- pending | approved | rejected
    created_at           TIMESTAMPTZ DEFAULT now(),
    resolved_at           TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue(status);

-- Dashboard support: compiler worker heartbeat. The compiler has no HTTP
-- port of its own (per docker-compose services table — only `ingest` is
-- exposed), so this is the only reliable signal that the worker *loop* is
-- actually alive and cycling, as opposed to the container merely being in
-- a "running" Docker state (a hung/deadlocked process would still show as
-- "running" in `docker compose ps`). `worker_loop()` upserts its own row
-- on every poll iteration (every POLL_INTERVAL_SECONDS, ~2s) whether or
-- not there was a job to process — the dashboard (`services/ingest`
-- `/metrics`) flags the compiler as down if `last_seen` is older than a
-- few poll intervals.
CREATE TABLE IF NOT EXISTS worker_heartbeat (
    worker_name TEXT PRIMARY KEY,
    last_seen   TIMESTAMPTZ NOT NULL
);

-- Agent integration dashboard support ("Connected Agents" section). One row
-- per `source_agent` value ever seen (see agent-integration.md, decision
-- #3): the `ingest` service upserts this row on every `/ingest/turn`
-- (action='write') and `/retrieve` (action='read') request, using the
-- `source_agent` string already carried in those request bodies — no
-- plugin-side changes needed to produce this data, the server already
-- knows who's calling. `read_tokens_estimate_total` approximates how many
-- tokens a paid agent session "spends" on memory_packet injection, using
-- the same `_APPROX_CHARS_PER_TOKEN` heuristic as `retrieval.py`'s token
-- budget (write-side ingestion runs through the free local LM Studio
-- worker, so only read-side cost is tracked here per the stated business
-- reason in agent-integration.md's "Что делаем и почему" section).
CREATE TABLE IF NOT EXISTS agent_activity (
    agent_name               TEXT PRIMARY KEY,
    last_seen                TIMESTAMPTZ NOT NULL,
    last_action               TEXT,   -- 'write' | 'read'
    turns_written_total       BIGINT DEFAULT 0,
    retrieve_calls_total      BIGINT DEFAULT 0,
    read_tokens_estimate_total BIGINT DEFAULT 0
);
