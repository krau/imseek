package imdb

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"imseek/internal/config"
	"imseek/internal/db"
	"imseek/internal/index"
	"imseek/internal/index/local"
	"imseek/internal/scoring"
)

type OnlineStore interface {
	AddImage(ctx context.Context, hash []byte, path string, descriptors [][]byte) (int64, error)
	// AddImageBatch inserts many images efficiently (single txn + COPY when possible).
	AddImageBatch(ctx context.Context, jobs []BatchItem, codeSize int) (int, error)
	// BeginBulkImport / EndBulkImport bracket large CLI loads (drop/rebuild HNSW).
	BeginBulkImport(ctx context.Context) error
	EndBulkImport(ctx context.Context) error
	DeleteImage(ctx context.Context, id int64) error
	CheckHash(ctx context.Context, hash []byte) (int64, bool, error)
	GetImage(ctx context.Context, id int64) (path string, hash []byte, vectorCount int64, err error)
	ListImages(ctx context.Context, limit, offset int64) ([]OnlineImage, error)
	Count(ctx context.Context) (images, vectors int64, err error)
	SearchDescriptors(ctx context.Context, descriptors [][]byte, k, nprobe int) []index.Neighbor
	GetImagePath(ctx context.Context, id int64) (string, error)
	IsNotFound(err error) bool
}

type OnlineImage struct {
	ID          int64
	Path        string
	Hash        []byte
	VectorCount int64
}

type IMDB struct {
	db        *db.DB // nil when using OnlineStore only (pgvector)
	backend   index.Backend
	online    OnlineStore // non-nil for production pg path
	confDir   string
	scoreType config.ScoreType
	codeSize  int

	// cache of per-image vector-id bounds for fast vector-id -> image-id
	// lookup; populated when cache is enabled. Ordered by Total ascending.
	// Unused when online != nil (neighbors carry ImageID).
	cache  bool
	mu     sync.RWMutex
	bounds []db.VectorBound
}

type Options struct {
	ConfDir   string
	WAL       bool
	Cache     bool
	ScoreType config.ScoreType
	CodeSize  int
	Backend   string
	Milvus    MilvusOptions
	Pgvector  PgvectorOptions
}

type MilvusOptions struct {
	Address    string
	Collection string
}

type PgvectorOptions struct {
	ConnString string
	Table      string
	IndexType  string
	// HNSW parameters (0 = package defaults).
	M              int
	EFConstruction int
	EFSearch       int
	// IVFFlat parameters. Lists=0 auto from row count; Probes is query-time.
	Lists  int
	Probes int
}

func (o Options) WithWAL(wal bool) Options {
	o.WAL = wal
	return o
}

func Open(ctx context.Context, opts Options) (*IMDB, error) {
	if opts.CodeSize == 0 {
		opts.CodeSize = 32
	}

	if opts.Backend == "pgvector" {
		online, backend, err := newPgvectorStore(opts)
		if err != nil {
			return nil, err
		}
		return &IMDB{
			backend:   backend,
			online:    online,
			confDir:   opts.ConfDir,
			scoreType: opts.ScoreType,
			codeSize:  opts.CodeSize,
		}, nil
	}

	if err := os.MkdirAll(opts.ConfDir, 0o755); err != nil {
		return nil, err
	}
	database, err := db.Open(opts.ConfDir, opts.WAL)
	if err != nil {
		return nil, err
	}
	if detected, err := database.GuessCodeSize(ctx); err != nil {
		database.Close()
		return nil, err
	} else if detected > 0 {
		opts.CodeSize = detected
	}
	backend, err := newBackend(opts)
	if err != nil {
		database.Close()
		return nil, err
	}
	m := &IMDB{
		db:        database,
		backend:   backend,
		confDir:   opts.ConfDir,
		scoreType: opts.ScoreType,
		codeSize:  opts.CodeSize,
		cache:     opts.Cache,
	}
	if opts.Cache {
		if err := m.loadVectorBounds(ctx); err != nil {
			backend.Close()
			database.Close()
			return nil, err
		}
	}
	return m, nil
}

func (m *IMDB) Online() bool { return m.online != nil }

// BeginBulkImport prepares the online store for a large import (e.g. drop HNSW).
// No-op for local backends.
func (m *IMDB) BeginBulkImport(ctx context.Context) error {
	if m.online == nil {
		return nil
	}
	return m.online.BeginBulkImport(ctx)
}

// EndBulkImport finishes a bulk import (e.g. rebuild HNSW). No-op for local.
func (m *IMDB) EndBulkImport(ctx context.Context) error {
	if m.online == nil {
		return nil
	}
	return m.online.EndBulkImport(ctx)
}

func newBackend(opts Options) (index.Backend, error) {
	switch opts.Backend {
	case "", "local":
		return local.New(opts.ConfDir, opts.CodeSize), nil
	case "milvus":
		return newMilvusBackend(opts)
	default:
		return nil, fmt.Errorf("unknown index backend %q", opts.Backend)
	}
}

func (m *IMDB) Close() error {
	if m.backend != nil {
		m.backend.Close()
	}
	if m.db != nil {
		return m.db.Close()
	}
	return nil
}

func (m *IMDB) Backend() index.Backend { return m.backend }

func (m *IMDB) DB() *db.DB { return m.db }

func (m *IMDB) loadVectorBounds(ctx context.Context) error {
	bounds, err := m.db.GetAllVectorBounds(ctx)
	if err != nil {
		return err
	}
	m.mu.Lock()
	m.bounds = bounds
	m.mu.Unlock()
	return nil
}

func (m *IMDB) ReloadCache(ctx context.Context) error {
	if !m.cache {
		return nil
	}
	return m.loadVectorBounds(ctx)
}

// Count returns (imageCount, vectorCount).
func (m *IMDB) Count(ctx context.Context) (int64, int64, error) {
	if m.online != nil {
		return m.online.Count(ctx)
	}
	return m.db.GetCount(ctx)
}

// CheckHash returns the id of an existing image with the exact hash.
func (m *IMDB) CheckHash(ctx context.Context, hash []byte) (int64, bool, error) {
	if m.online != nil {
		return m.online.CheckHash(ctx, hash)
	}
	return m.db.CheckImageHash(ctx, hash)
}

func (m *IMDB) AddImage(ctx context.Context, hash []byte, path string, descriptors [][]byte) (int64, error) {
	if m.online != nil {
		return m.online.AddImage(ctx, hash, path, descriptors)
	}
	tx, err := m.db.Begin(ctx)
	if err != nil {
		return 0, err
	}
	id, err := m.db.AddImage(ctx, tx, hash, path)
	if err != nil {
		tx.Rollback()
		return 0, err
	}
	blob := flatten(descriptors)
	if err := m.db.AddVector(ctx, tx, id, blob); err != nil {
		tx.Rollback()
		return 0, err
	}
	if err := m.db.AddVectorStats(ctx, tx, id, int64(len(descriptors))); err != nil {
		tx.Rollback()
		return 0, err
	}
	if err := tx.Commit(); err != nil {
		return 0, err
	}
	return id, nil
}

func (m *IMDB) AddImageBatch(ctx context.Context, jobs []BatchItem) (int, error) {
	if m.online != nil {
		return m.online.AddImageBatch(ctx, jobs, m.codeSize)
	}

	tx, err := m.db.Begin(ctx)
	if err != nil {
		return 0, err
	}
	added := 0
	for _, j := range jobs {
		id, err := m.db.AddImage(ctx, tx, j.Hash, j.Path)
		if err != nil {
			if isUniqueConstraintErr(err) {
				continue // expected duplicate
			}
			tx.Rollback()
			return added, fmt.Errorf("batch add image %q: %w", j.Path, err)
		}
		if err := m.db.AddVector(ctx, tx, id, j.Blob); err != nil {
			tx.Rollback()
			return added, fmt.Errorf("batch add vector for %q: %w", j.Path, err)
		}
		if err := m.db.AddVectorStats(ctx, tx, id, j.Count); err != nil {
			tx.Rollback()
			return added, fmt.Errorf("batch add stats for %q: %w", j.Path, err)
		}
		added++
	}
	if err := tx.Commit(); err != nil {
		return 0, err
	}
	return added, nil
}

type sqliteErrorCode interface {
	error
	Code() int
}

func isUniqueConstraintErr(err error) bool {
	sqliteErr, ok := errors.AsType[sqliteErrorCode](err)
	return ok && sqliteErr.Code() == 19 // sqlite3.CONSTRAINT
}

type BatchItem struct {
	Hash  []byte
	Path  string
	Blob  []byte
	Count int64
}

func (m *IMDB) AddImageRaw(ctx context.Context, hash []byte, path string, blob []byte, count int64) (int64, error) {
	if m.online != nil {
		descs := unflattenN(blob, m.codeSize, int(count))
		return m.online.AddImage(ctx, hash, path, descs)
	}
	tx, err := m.db.Begin(ctx)
	if err != nil {
		return 0, err
	}
	id, err := m.db.AddImage(ctx, tx, hash, path)
	if err != nil {
		tx.Rollback()
		return 0, err
	}
	if err := m.db.AddVector(ctx, tx, id, blob); err != nil {
		tx.Rollback()
		return 0, err
	}
	if err := m.db.AddVectorStats(ctx, tx, id, count); err != nil {
		tx.Rollback()
		return 0, err
	}
	if err := tx.Commit(); err != nil {
		return 0, err
	}
	return id, nil
}

func (m *IMDB) UpdateImagePath(ctx context.Context, id int64, path string) error {
	if m.online != nil {
		return errors.New("UpdateImagePath not supported on online backend")
	}
	return m.db.UpdateImagePath(ctx, id, path)
}

func (m *IMDB) AppendImagePath(ctx context.Context, id int64, path string) (bool, error) {
	if m.online != nil {
		return false, errors.New("AppendImagePath not supported on online backend")
	}
	return m.db.AppendImagePath(ctx, id, path)
}

func flatten(descriptors [][]byte) []byte {
	if len(descriptors) == 0 {
		return nil
	}
	cs := len(descriptors[0])
	out := make([]byte, 0, len(descriptors)*cs)
	for _, d := range descriptors {
		out = append(out, d...)
	}
	return out
}

func unflatten(blob []byte, codeSize int) [][]byte {
	if codeSize <= 0 || len(blob) < codeSize {
		return nil
	}
	n := len(blob) / codeSize
	return unflattenN(blob, codeSize, n)
}

func unflattenN(blob []byte, codeSize, n int) [][]byte {
	if codeSize <= 0 || n <= 0 {
		return nil
	}
	out := make([][]byte, 0, n)
	for i := 0; i < n && (i+1)*codeSize <= len(blob); i++ {
		d := make([]byte, codeSize)
		copy(d, blob[i*codeSize:(i+1)*codeSize])
		out = append(out, d)
	}
	return out
}

func isPGUniqueErr(err error) bool {
	if err == nil {
		return false
	}
	msg := strings.ToLower(err.Error())
	return strings.Contains(msg, "duplicate key") ||
		strings.Contains(msg, "unique constraint")
}

func (m *IMDB) findImageID(ctx context.Context, vectorID uint64) (int64, error) {
	if !m.cache {
		return m.db.GetImageIDByVectorID(ctx, int64(vectorID))
	}
	m.mu.RLock()
	bounds := m.bounds
	m.mu.RUnlock()
	vid := int64(vectorID)
	// first bound whose Total >= vid
	i := sort.Search(len(bounds), func(i int) bool { return bounds[i].Total >= vid })
	if i >= len(bounds) {
		return 0, fmt.Errorf("vector id %d out of range", vectorID)
	}
	b := bounds[i]
	// must fall inside this image's half-open range (Total-Count, Total]
	if vid <= b.Total-b.Count {
		return 0, fmt.Errorf("vector id %d orphan (gap before image %d)", vectorID, b.ImageID)
	}
	return b.ImageID, nil
}

type SearchResult struct {
	Score float32 `json:"score"`
	Path  string  `json:"path"`
}

func (m *IMDB) Search(ctx context.Context, searcher index.Searcher, descriptors [][]byte, opts config.SearchOptions) ([]SearchResult, error) {
	if len(descriptors) == 0 {
		return nil, nil
	}
	var neighbors []index.Neighbor
	if m.online != nil {
		neighbors = m.online.SearchDescriptors(ctx, descriptors, opts.K, opts.NProbe)
	} else if searcher != nil {
		neighbors = searcher.Search(ctx, descriptors, opts.K, opts.NProbe)
	} else {
		return nil, errors.New("search: no searcher")
	}
	return m.processNeighbors(ctx, neighbors, opts)
}

func (m *IMDB) processNeighbors(ctx context.Context, neighbors []index.Neighbor, opts config.SearchOptions) ([]SearchResult, error) {
	maxBits := m.codeSize * 8
	groups := make(map[int64][]float32)
	for _, n := range neighbors {
		if n.Distance > opts.Distance {
			continue
		}
		imgID := n.ImageID
		if imgID == 0 {
			var err error
			imgID, err = m.findImageID(ctx, n.ID)
			if err != nil {
				if !m.cache {
					log.Printf("findImageID: vector %d: %v", n.ID, err)
				}
				continue
			}
		}
		sim := 1.0 - float32(n.Distance)/float32(maxBits)
		groups[imgID] = append(groups[imgID], sim)
	}

	type scored struct {
		id    int64
		score float32
	}
	results := make([]scored, 0, len(groups))
	for id, scores := range groups {
		var s float32
		if opts.ScoreType == config.ScoreCount {
			s = float32(len(scores))
		} else {
			s = 100 * scoring.Wilson(scores)
		}
		results = append(results, scored{id: id, score: s})
	}
	sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })
	if len(results) > opts.Count {
		results = results[:opts.Count]
	}

	out := make([]SearchResult, 0, len(results))
	for _, r := range results {
		path, err := m.imagePath(ctx, r.id)
		if err != nil {
			continue
		}
		out = append(out, SearchResult{Score: r.score, Path: path})
	}
	return out, nil
}

func (m *IMDB) imagePath(ctx context.Context, id int64) (string, error) {
	if m.online != nil {
		return m.online.GetImagePath(ctx, id)
	}
	return m.db.GetImagePath(ctx, id)
}

func (m *IMDB) QuantizerPath() string {
	return filepath.Join(m.confDir, config.QuantizerFile)
}

func (m *IMDB) InvlistsPath() string {
	return filepath.Join(m.confDir, config.InvlistsFile)
}

func (m *IMDB) ConfDir() string { return m.confDir }

func (m *IMDB) DeleteImage(ctx context.Context, id int64) error {
	if m.online != nil {
		return m.online.DeleteImage(ctx, id)
	}

	rng, err := m.db.GetVectorIDRange(ctx, id)
	hasRange := false
	switch {
	case err == nil:
		hasRange = rng.Lo > 0 && rng.Hi >= rng.Lo
	case errors.Is(err, sql.ErrNoRows):
	default:
		return err
	}

	if hasRange && m.backend != nil {
		if err := m.backend.DeleteVectors(ctx, uint64(rng.Lo), uint64(rng.Hi)); err != nil {
			return fmt.Errorf("delete image %d: purge index vectors [%d,%d]: %w", id, rng.Lo, rng.Hi, err)
		}
	}

	if _, err := m.db.DeleteImage(ctx, id); err != nil {
		return err
	}
	if m.cache {
		if err := m.loadVectorBounds(ctx); err != nil {
			return fmt.Errorf("delete image %d: reload cache: %w", id, err)
		}
	}
	return nil
}

func (m *IMDB) ListImages(ctx context.Context, limit, offset int64) ([]db.ImageInfo, error) {
	if m.online != nil {
		rows, err := m.online.ListImages(ctx, limit, offset)
		if err != nil {
			return nil, err
		}
		out := make([]db.ImageInfo, len(rows))
		for i, r := range rows {
			out[i] = db.ImageInfo{
				ID:          r.ID,
				Path:        r.Path,
				Hash:        fmt.Sprintf("%x", r.Hash),
				VectorCount: r.VectorCount,
				Indexed:     true, // online path always searchable
			}
		}
		return out, nil
	}
	return m.db.ListImages(ctx, limit, offset)
}

func (m *IMDB) GetImage(ctx context.Context, id int64) (*db.ImageInfo, error) {
	if m.online != nil {
		path, hash, vc, err := m.online.GetImage(ctx, id)
		if err != nil {
			if m.online.IsNotFound(err) {
				return nil, sql.ErrNoRows
			}
			return nil, err
		}
		return &db.ImageInfo{
			ID:          id,
			Path:        path,
			Hash:        fmt.Sprintf("%x", hash),
			VectorCount: vc,
			Indexed:     true,
		}, nil
	}
	return m.db.GetImage(ctx, id)
}

func (m *IMDB) CountUnindexed(ctx context.Context) (int64, error) {
	if m.online != nil {
		return 0, nil
	}
	return m.db.CountImageUnindexed(ctx)
}
