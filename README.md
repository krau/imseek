# imseek

[English](README.md) | [中文](README.zh-CN.md)

Pure Go image search by ORB feature matching.

## Backends

| Backend | Storage | Index |
| ------- | ------- | ----- |
| `local` | SQLite + local files | IVF |
| `pgvector` | PostgreSQL (metadata + vectors) | IVFFlat or HNSW |
| `milvus` | Milvus | BIN_IVF_FLAT |

### local

`local` is the default backend. No external services or build dependencies — works out of the box.

The `local` backend is largely a Go rewrite of [imsearch](https://github.com/lolishinshi/imsearch). Credit to the original authors.

Data directory:

```
data/
├── imseek.db          # metadata, descriptors, index info
├── quantizer.bin      # coarse quantizer
└── invlists.bin       # inverted lists
```

### pgvector

Metadata and vectors are stored entirely in PostgreSQL. Requires the pgvector extension. Supports IVFFlat and HNSW.

IVFFlat builds faster but has lower recall; HNSW is the opposite.

## Build

```bash
# Default (local backend only)
go build -o imseek ./cmd/imseek

# With pgvector support
go build -tags pgvector -o imseek ./cmd/imseek

# With milvus support
go build -tags milvus -o imseek ./cmd/imseek

# All backends
go build -tags 'pgvector milvus' -o imseek ./cmd/imseek
```

## Usage

### local backend

```bash
# 1. Add images
./imseek add /path/to/images

# 2. Train the quantizer
./imseek train

# 3. Build the index
./imseek build

# 4. Search
./imseek search query.jpg

# 5. Start HTTP server
./imseek server --token my-secret
```

### pgvector backend

Requires PostgreSQL with the pgvector extension, or start an instance with Docker:

```bash
docker run -d --name imseek-pg --shm-size=1g \
  -e POSTGRES_USER=imseek -e POSTGRES_PASSWORD=imseek \
  -e POSTGRES_DB=imseek -p 5432:5432 pgvector/pgvector:pg18
```

Add images:

```bash
./imseek --backend pgvector \
  --pgvector.conn_string 'postgres://imseek:imseek@localhost:5432/imseek?sslmode=disable' \
  add /path/to/images
```

Search:

```bash
./imseek --backend pgvector \
  --pgvector.conn_string 'postgres://imseek:imseek@localhost:5432/imseek?sslmode=disable' \
  search query.jpg
```

The pgvector backend needs no manual train/build — indexing is handled automatically when images are added.

## HTTP API

```bash
# Start server
./imseek server --token my-secret

# Add image
curl -H "Authorization: Bearer my-secret" \
  -F "file=@photo.jpg" http://localhost:8000/api/v1/images

# Search
curl -H "Authorization: Bearer my-secret" \
  -F "file=@query.jpg" http://localhost:8000/api/v1/search

# Delete
curl -X DELETE -H "Authorization: Bearer my-secret" \
  http://localhost:8000/api/v1/images/1

# Stats
curl -H "Authorization: Bearer my-secret" \
  http://localhost:8000/api/v1/stats
```

Swagger docs: `http://localhost:8000/swagger`

## Configuration

Use a config file (`imseek.toml`), environment variables (`IMSEEK_*`), or CLI flags. See `imseek.toml.example`.

Key search parameters:

| Parameter | Default | Description |
| --------- | ------- | ----------- |
| `search.distance` | 64 | Max Hamming distance |
| `search.count` | 10 | Number of results |
| `search.k` | 3 | KNN per descriptor |
| `search.nprobe` | 3 | Inverted lists to probe (local) |
| `pgvector.probes` | 10 | Lists to probe (pgvector IVFFlat) |
| `pgvector.ef_search` | 40 | Candidate list size (pgvector HNSW) |

## Acknowledgements

- [imsearch](https://github.com/lolishinshi/imsearch)
- [ORB_SLAM3](https://github.com/raulmur/ORB_SLAM3)
