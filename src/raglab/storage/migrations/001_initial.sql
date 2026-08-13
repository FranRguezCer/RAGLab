CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS collections (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    name text NOT NULL UNIQUE,
    model text NOT NULL,
    dimension integer NOT NULL CHECK (dimension > 0),
    metric text NOT NULL CHECK (metric = 'cosine'),
    chunk_config jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documents (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id uuid NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    source_uri text NOT NULL,
    markdown text NOT NULL,
    content_hash text NOT NULL,
    converter text NOT NULL,
    converter_version text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    fingerprint text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (collection_id, source_uri)
);

CREATE TABLE IF NOT EXISTS chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index integer NOT NULL,
    content text NOT NULL,
    embedding_text text NOT NULL,
    token_count integer NOT NULL CHECK (token_count > 0),
    heading_path jsonb NOT NULL DEFAULT '[]'::jsonb,
    start_line integer,
    end_line integer,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1024) NOT NULL,
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS documents_collection_idx ON documents (collection_id);

CREATE OR REPLACE VIEW collection_stats AS
SELECT
    c.name,
    c.model,
    c.dimension,
    count(DISTINCT d.id) AS document_count,
    count(ch.id) AS chunk_count,
    coalesce(sum(ch.token_count), 0) AS token_count
FROM collections c
LEFT JOIN documents d ON d.collection_id = c.id
LEFT JOIN chunks ch ON ch.document_id = d.id
GROUP BY c.id, c.name, c.model, c.dimension;

