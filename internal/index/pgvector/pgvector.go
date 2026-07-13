//go:build pgvector

// Build with: go build -tags pgvector ./...
// Requires PostgreSQL with the pgvector extension.
package pgvector

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"strings"
	"sync"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"imseek/internal/index"
)

const (
	IndexHNSW    = "hnsw"
	IndexIVFFlat = "ivfflat"
)

const (
	defaultM              = 16
	defaultEFConstruction = 64
	defaultEFSearch       = 40
	defaultProbes         = 10
	defaultMaxQueryDescs  = 64 // ORB can emit ~500; each is a remote ANN
	defaultMaxConns       = 32
)

type Options struct {
	ConnString string
	// Table is the name prefix for schema objects. Tables created:
	//   {Table}_image, {Table}_vector  and ANN index {Table}_vector_{hnsw|ivfflat}.
	Table string

	// IndexType is "ivfflat" (default, faster build) or "hnsw" (better recall).
	IndexType string

	// HNSW parameters (ignored for ivfflat). Zero = package default.
	M              int
	EFConstruction int
	EFSearch       int

	// IVFFlat parameters (ignored for hnsw). Lists=0 auto-picks from row count.
	// Probes is the query-time list count (like local nprobe).
	Lists  int
	Probes int

	// MaxQueryDescs caps how many query descriptors are sent to ANN.
	// ORB yields ~500/image; each becomes a remote KNN. 0 = default (64).
	MaxQueryDescs int

	// MaxConns overrides the pgx pool size (0 = pgx default).
	MaxConns int32
}

type Store struct {
	pool           *pgxpool.Pool
	prefix         string
	imageTable     string
	vectorTable    string
	annIndex       string // physical index name
	indexType      string // IndexHNSW | IndexIVFFlat
	codeSize       int
	dim            int
	m              int
	efConstruction int
	efSearch       int
	lists          int // 0 = auto at build time
	probes         int
	maxQueryDescs  int

	mu       sync.Mutex
	bulkMode bool
}

func New(ctx context.Context, opts Options, codeSize int) (*Store, error) {
	if opts.ConnString == "" {
		return nil, errors.New("pgvector: conn string is required")
	}
	if opts.Table == "" {
		return nil, errors.New("pgvector: table name is required")
	}
	if codeSize <= 0 {
		return nil, errors.New("pgvector: codeSize must be positive")
	}
	if !isSafeIdent(opts.Table) {
		return nil, fmt.Errorf("pgvector: unsafe table name %q", opts.Table)
	}

	idxType := strings.ToLower(strings.TrimSpace(opts.IndexType))
	if idxType == "" {
		idxType = IndexIVFFlat
	}
	if idxType != IndexHNSW && idxType != IndexIVFFlat {
		return nil, fmt.Errorf("pgvector: unknown index_type %q (want hnsw|ivfflat)", opts.IndexType)
	}

	m := opts.M
	if m <= 0 {
		m = defaultM
	}
	efC := opts.EFConstruction
	if efC <= 0 {
		efC = defaultEFConstruction
	}
	efS := opts.EFSearch
	if efS <= 0 {
		efS = defaultEFSearch
	}
	probes := opts.Probes
	if probes <= 0 {
		probes = defaultProbes
	}
	maxQD := opts.MaxQueryDescs
	if maxQD <= 0 {
		maxQD = defaultMaxQueryDescs
	}

	cfg, err := pgxpool.ParseConfig(opts.ConnString)
	if err != nil {
		return nil, fmt.Errorf("pgvector parse config: %w", err)
	}
	maxConns := opts.MaxConns
	if maxConns <= 0 {
		maxConns = defaultMaxConns
	}
	cfg.MaxConns = maxConns
	cfg.MinConns = min(4, maxConns)

	// Session GUC for query-time ANN quality.
	probeOrEF := probes
	useHNSW := idxType == IndexHNSW
	if useHNSW {
		probeOrEF = efS
	}
	cfg.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		if useHNSW {
			_, err := conn.Exec(ctx, "SET hnsw.ef_search = "+itoa(probeOrEF))
			return err
		}
		_, err := conn.Exec(ctx, "SET ivfflat.probes = "+itoa(probeOrEF))
		return err
	}

	pool, err := pgxpool.NewWithConfig(ctx, cfg)
	if err != nil {
		return nil, fmt.Errorf("pgvector connect: %w", err)
	}

	var ext string
	err = pool.QueryRow(ctx, "SELECT extversion FROM pg_extension WHERE extname = 'vector'").Scan(&ext)
	if err != nil {
		pool.Close()
		return nil, fmt.Errorf("pgvector extension not installed: %w (run: CREATE EXTENSION vector)", err)
	}

	s := &Store{
		pool:           pool,
		prefix:         opts.Table,
		imageTable:     opts.Table + "_image",
		vectorTable:    opts.Table + "_vector",
		annIndex:       opts.Table + "_vector_" + idxType,
		indexType:      idxType,
		codeSize:       codeSize,
		dim:            codeSize * 8,
		m:              m,
		efConstruction: efC,
		efSearch:       efS,
		lists:          opts.Lists,
		probes:         probes,
		maxQueryDescs:  maxQD,
	}
	if err := s.ensureSchema(ctx); err != nil {
		pool.Close()
		return nil, err
	}
	return s, nil
}

func (s *Store) ensureSchema(ctx context.Context) error {
	sql := fmt.Sprintf(`
CREATE TABLE IF NOT EXISTS %s (
    id           bigserial PRIMARY KEY,
    hash         bytea UNIQUE NOT NULL,
    path         text NOT NULL,
    vector_count integer NOT NULL DEFAULT 0
)`, s.imageTable)
	if _, err := s.pool.Exec(ctx, sql); err != nil {
		return fmt.Errorf("pgvector create image table: %w", err)
	}

	// Descriptors: global id, owning image, bit vector.
	// No FK to image for insert speed; we delete vectors explicitly / by image_id.
	sql = fmt.Sprintf(`
CREATE TABLE IF NOT EXISTS %s (
    id       bigserial PRIMARY KEY,
    image_id bigint NOT NULL,
    vec      bit(%d) NOT NULL
)`, s.vectorTable, s.dim)
	if _, err := s.pool.Exec(ctx, sql); err != nil {
		return fmt.Errorf("pgvector create vector table: %w", err)
	}

	sql = fmt.Sprintf(`CREATE INDEX IF NOT EXISTS %s_image_id ON %s (image_id)`,
		s.vectorTable, s.vectorTable)
	if _, err := s.pool.Exec(ctx, sql); err != nil {
		return fmt.Errorf("pgvector create image_id index: %w", err)
	}

	// ANN index is created lazily (EndBulkImport / NewSearcher). Keeping it
	// during bulk insert makes every row update the index and is very slow.
	return s.ensureANN(ctx)
}

func (s *Store) ensureANN(ctx context.Context) error {
	s.mu.Lock()
	bulk := s.bulkMode
	s.mu.Unlock()
	if bulk {
		return nil
	}

	// Fast path: index already exists — do not COUNT(*) or CREATE INDEX.
	if s.indexExists(ctx, s.annIndex) {
		return nil
	}

	// Drop the other index type if present (switching hnsw <-> ivfflat).
	other := s.prefix + "_vector_hnsw"
	if s.indexType == IndexHNSW {
		other = s.prefix + "_vector_ivfflat"
	}
	if s.indexExists(ctx, other) {
		_, _ = s.pool.Exec(ctx, fmt.Sprintf(`DROP INDEX IF EXISTS %s`, other))
	}

	if s.indexType == IndexIVFFlat {
		return s.createIVFFlat(ctx)
	}
	return s.createHNSW(ctx)
}

func (s *Store) indexExists(ctx context.Context, name string) bool {
	var exists bool
	err := s.pool.QueryRow(ctx,
		`SELECT EXISTS (
			SELECT 1 FROM pg_indexes
			WHERE schemaname = current_schema() AND indexname = $1
		)`, name).Scan(&exists)
	return err == nil && exists
}

func (s *Store) createHNSW(ctx context.Context) error {
	sql := fmt.Sprintf(
		`CREATE INDEX IF NOT EXISTS %s ON %s USING hnsw (vec bit_hamming_ops) WITH (m = %d, ef_construction = %d)`,
		s.annIndex, s.vectorTable, s.m, s.efConstruction)
	if _, err := s.pool.Exec(ctx, sql); err != nil {
		return fmt.Errorf("pgvector create hnsw index: %w", err)
	}
	log.Printf("pgvector: HNSW index created (%s)", s.annIndex)
	return nil
}

func (s *Store) createIVFFlat(ctx context.Context) error {
	// Estimate row count cheaply (reltuples); fall back to COUNT only if needed.
	var n float64
	err := s.pool.QueryRow(ctx,
		`SELECT COALESCE(reltuples, 0) FROM pg_class
		 WHERE relname = $1 AND relnamespace = (SELECT oid FROM pg_namespace WHERE nspname = current_schema())`,
		s.vectorTable).Scan(&n)
	if err != nil || n < 1 {
		var cnt int64
		if err := s.pool.QueryRow(ctx,
			fmt.Sprintf(`SELECT COUNT(*) FROM %s`, s.vectorTable)).Scan(&cnt); err != nil {
			return err
		}
		n = float64(cnt)
	}
	if n < 1 {
		// IVFFlat needs data for k-means; skip until rows exist.
		return nil
	}
	lists := s.lists
	if lists <= 0 {
		lists = autoLists(int64(n))
	}
	sql := fmt.Sprintf(
		`CREATE INDEX IF NOT EXISTS %s ON %s USING ivfflat (vec bit_hamming_ops) WITH (lists = %d)`,
		s.annIndex, s.vectorTable, lists)
	if _, err := s.pool.Exec(ctx, sql); err != nil {
		return fmt.Errorf("pgvector create ivfflat index (lists=%d): %w", lists, err)
	}
	log.Printf("pgvector: IVFFlat index created (lists=%d, ~rows=%.0f)", lists, n)
	return nil
}

func autoLists(n int64) int {
	if n <= 0 {
		return 100
	}
	if n < 1_000_000 {
		// rows/1000, clamp
		l := int(n / 1000)
		if l < 10 {
			l = 10
		}
		if l > 1000 {
			l = 1000
		}
		return l
	}
	// sqrt(rows) for larger sets
	l := int(math.Sqrt(float64(n)))
	if l < 1000 {
		l = 1000
	}
	if l > 8192 {
		l = 8192
	}
	return l
}

func (s *Store) dropANN(ctx context.Context) error {
	// Drop both possible index names so bulk import is clean after type switch.
	for _, name := range []string{
		s.annIndex,
		s.prefix + "_vector_hnsw",
		s.prefix + "_vector_ivfflat",
	} {
		if _, err := s.pool.Exec(ctx, fmt.Sprintf(`DROP INDEX IF EXISTS %s`, name)); err != nil {
			return fmt.Errorf("pgvector drop index %s: %w", name, err)
		}
	}
	return nil
}

func (s *Store) BeginBulkImport(ctx context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.bulkMode {
		return nil
	}
	if err := s.dropANN(ctx); err != nil {
		return err
	}
	_, _ = s.pool.Exec(ctx, `SET synchronous_commit = off`)
	s.bulkMode = true
	return nil
}

func (s *Store) EndBulkImport(ctx context.Context) error {
	s.mu.Lock()
	s.bulkMode = false
	s.mu.Unlock()

	attempts := []struct {
		workers string
		mem     string
	}{
		{"7", "4GB"},
		{"4", "2GB"},
		{"2", "1GB"},
		{"0", "512MB"},
		{"0", "64MB"},
	}
	var err error
	for _, a := range attempts {
		_, _ = s.pool.Exec(ctx, `SET max_parallel_maintenance_workers = `+a.workers)
		_, _ = s.pool.Exec(ctx, `SET maintenance_work_mem = '`+a.mem+`'`)
		log.Printf("pgvector: building %s (workers=%s, maintenance_work_mem=%s)",
			s.indexType, a.workers, a.mem)
		err = s.ensureANN(ctx)
		if err == nil {
			break
		}
		log.Printf("pgvector: %s build attempt failed: %v", s.indexType, err)
		_ = s.dropANN(ctx)
	}
	_, _ = s.pool.Exec(ctx, `RESET maintenance_work_mem`)
	_, _ = s.pool.Exec(ctx, `RESET max_parallel_maintenance_workers`)
	return err
}

func isShmError(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	return contains(msg, "shared memory") ||
		contains(msg, "No space left on device") ||
		contains(msg, "53100")
}

func (s *Store) InBulkImport() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.bulkMode
}

func (s *Store) Close() error {
	if s.pool != nil {
		s.pool.Close()
	}
	return nil
}

type ImageInfo struct {
	ID          int64
	Path        string
	Hash        []byte
	VectorCount int64
}

func (s *Store) AddImage(ctx context.Context, hash []byte, path string, descriptors [][]byte) (int64, error) {
	if len(descriptors) == 0 {
		return 0, errors.New("pgvector: no descriptors")
	}
	for i, d := range descriptors {
		if len(d) != s.codeSize {
			return 0, fmt.Errorf("pgvector: descriptor[%d] length %d, want %d", i, len(d), s.codeSize)
		}
	}

	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback(ctx)

	var id int64
	err = tx.QueryRow(ctx,
		fmt.Sprintf(`INSERT INTO %s (hash, path, vector_count) VALUES ($1, $2, $3) RETURNING id`, s.imageTable),
		hash, path, len(descriptors)).Scan(&id)
	if err != nil {
		return 0, fmt.Errorf("pgvector insert image: %w", err)
	}

	if err := s.copyVectors(ctx, tx, id, descriptors); err != nil {
		return 0, err
	}
	if err := tx.Commit(ctx); err != nil {
		return 0, err
	}
	return id, nil
}

type ImageBatchItem struct {
	Hash        []byte
	Path        string
	Descriptors [][]byte
}

func (s *Store) AddImageBatch(ctx context.Context, items []ImageBatchItem) (int, error) {
	if len(items) == 0 {
		return 0, nil
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return 0, err
	}
	defer tx.Rollback(ctx)

	type pending struct {
		id    int64
		descs [][]byte
	}
	pendingRows := make([]pending, 0, len(items))
	added := 0
	totalVecs := 0

	imgSQL := fmt.Sprintf(
		`INSERT INTO %s (hash, path, vector_count) VALUES ($1, $2, $3) RETURNING id`,
		s.imageTable)

	for _, it := range items {
		if len(it.Descriptors) == 0 {
			continue
		}
		var id int64
		err := tx.QueryRow(ctx, imgSQL, it.Hash, it.Path, len(it.Descriptors)).Scan(&id)
		if err != nil {
			if isUniqueViolation(err) {
				continue
			}
			return added, fmt.Errorf("pgvector insert image %q: %w", it.Path, err)
		}
		pendingRows = append(pendingRows, pending{id: id, descs: it.Descriptors})
		added++
		totalVecs += len(it.Descriptors)
	}

	if totalVecs > 0 {
		rows := make([][]any, 0, totalVecs)
		for _, p := range pendingRows {
			for _, d := range p.descs {
				if len(d) != s.codeSize {
					return added, fmt.Errorf("pgvector: bad descriptor length %d want %d", len(d), s.codeSize)
				}
				rows = append(rows, []any{p.id, bitString(d)})
			}
		}
		_, err = tx.CopyFrom(ctx,
			pgx.Identifier{s.vectorTable},
			[]string{"image_id", "vec"},
			pgx.CopyFromRows(rows))
		if err != nil {
			return 0, fmt.Errorf("pgvector copy vectors: %w", err)
		}
	}

	if err := tx.Commit(ctx); err != nil {
		return 0, err
	}
	return added, nil
}

func (s *Store) copyVectors(ctx context.Context, tx pgx.Tx, imageID int64, descriptors [][]byte) error {
	rows := make([][]any, len(descriptors))
	for i, d := range descriptors {
		rows[i] = []any{imageID, bitString(d)}
	}
	_, err := tx.CopyFrom(ctx,
		pgx.Identifier{s.vectorTable},
		[]string{"image_id", "vec"},
		pgx.CopyFromRows(rows))
	if err != nil {
		return fmt.Errorf("pgvector copy vectors: %w", err)
	}
	return nil
}

func isUniqueViolation(err error) bool {
	if err == nil {
		return false
	}
	// pgx/pgconn: SQLSTATE 23505
	var pe interface{ SQLState() string }
	if errors.As(err, &pe) && pe.SQLState() == "23505" {
		return true
	}
	msg := err.Error()
	return contains(msg, "duplicate key") || contains(msg, "unique constraint")
}

func contains(s, sub string) bool {
	return len(sub) == 0 || (len(s) >= len(sub) && indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

func (s *Store) DeleteImage(ctx context.Context, id int64) error {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return err
	}
	defer tx.Rollback(ctx)
	if _, err := tx.Exec(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE image_id = $1`, s.vectorTable), id); err != nil {
		return fmt.Errorf("pgvector delete vectors: %w", err)
	}
	tag, err := tx.Exec(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE id = $1`, s.imageTable), id)
	if err != nil {
		return fmt.Errorf("pgvector delete image: %w", err)
	}
	if tag.RowsAffected() == 0 {
		return errNoRows
	}
	return tx.Commit(ctx)
}

var errNoRows = errors.New("pgvector: image not found")

func IsNotFound(err error) bool {
	return errors.Is(err, errNoRows)
}

func (s *Store) CheckHash(ctx context.Context, hash []byte) (int64, bool, error) {
	var id int64
	err := s.pool.QueryRow(ctx,
		fmt.Sprintf(`SELECT id FROM %s WHERE hash = $1`, s.imageTable), hash).Scan(&id)
	if errors.Is(err, pgx.ErrNoRows) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, err
	}
	return id, true, nil
}

func (s *Store) GetImage(ctx context.Context, id int64) (*ImageInfo, error) {
	var img ImageInfo
	err := s.pool.QueryRow(ctx,
		fmt.Sprintf(`SELECT id, path, hash, vector_count FROM %s WHERE id = $1`, s.imageTable), id).
		Scan(&img.ID, &img.Path, &img.Hash, &img.VectorCount)
	if errors.Is(err, pgx.ErrNoRows) {
		return nil, errNoRows
	}
	if err != nil {
		return nil, err
	}
	return &img, nil
}

func (s *Store) ListImages(ctx context.Context, limit, offset int64) ([]ImageInfo, error) {
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	rows, err := s.pool.Query(ctx,
		fmt.Sprintf(`SELECT id, path, hash, vector_count FROM %s ORDER BY id ASC LIMIT $1 OFFSET $2`, s.imageTable),
		limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ImageInfo
	for rows.Next() {
		var img ImageInfo
		if err := rows.Scan(&img.ID, &img.Path, &img.Hash, &img.VectorCount); err != nil {
			return nil, err
		}
		out = append(out, img)
	}
	return out, rows.Err()
}

func (s *Store) Count(ctx context.Context) (images, vectors int64, err error) {
	err = s.pool.QueryRow(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM %s`, s.imageTable)).Scan(&images)
	if err != nil {
		return 0, 0, err
	}
	err = s.pool.QueryRow(ctx, fmt.Sprintf(`SELECT COUNT(*) FROM %s`, s.vectorTable)).Scan(&vectors)
	return images, vectors, err
}

func (s *Store) SearchDescriptors(ctx context.Context, descriptors [][]byte, k, nprobe int) []index.Neighbor {
	if len(descriptors) == 0 || k <= 0 {
		return nil
	}
	// Cap query descriptors: each is a remote ANN against the full table.
	// Evenly subsample so spatial coverage is preserved.
	descriptors = subsampleDescs(descriptors, s.maxQueryDescs)

	sql := s.searchSQL()
	// Query-time quality: HNSW uses ef_search; IVFFlat uses probes.
	qParam := s.probes
	if s.indexType == IndexHNSW {
		qParam = max(k*4, s.efSearch)
	} else if nprobe > 0 {
		qParam = nprobe
	}

	// Parallelize across most of the pool (leave a couple free for other ops).
	maxConns := int(s.pool.Config().MaxConns)
	if maxConns < 1 {
		maxConns = int(defaultMaxConns)
	}
	numWorkers := maxConns - 2
	if numWorkers < 1 {
		numWorkers = 1
	}
	if numWorkers > len(descriptors) {
		numWorkers = len(descriptors)
	}
	chunkSize := (len(descriptors) + numWorkers - 1) / numWorkers

	var mu sync.Mutex
	out := make([]index.Neighbor, 0, len(descriptors)*k)
	var wg sync.WaitGroup
	for i := 0; i < numWorkers && i*chunkSize < len(descriptors); i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > len(descriptors) {
			end = len(descriptors)
		}
		wg.Add(1)
		go func(chunk [][]byte) {
			defer wg.Done()
			local := s.queryBatch(ctx, sql, chunk, k, qParam)
			mu.Lock()
			out = append(out, local...)
			mu.Unlock()
		}(descriptors[start:end])
	}
	wg.Wait()
	return out
}

func subsampleDescs(descs [][]byte, max int) [][]byte {
	if max <= 0 || len(descs) <= max {
		return descs
	}
	out := make([][]byte, max)
	// stride sampling: indices 0, step, 2*step, ... covering the full set
	for i := 0; i < max; i++ {
		out[i] = descs[i*len(descs)/max]
	}
	return out
}

func (s *Store) searchSQL() string {
	// Return image_id so scoring needs no reverse map.
	return fmt.Sprintf(
		`SELECT id, image_id, (vec <~> $1::bit(%d))::int AS dist
		 FROM %s ORDER BY vec <~> $1::bit(%d) LIMIT $2`,
		s.dim, s.vectorTable, s.dim)
}

func (s *Store) queryBatch(ctx context.Context, sql string, descs [][]byte, k, qParam int) []index.Neighbor {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		log.Printf("pgvector: acquire conn: %v", err)
		return nil
	}
	defer conn.Release()

	if reset := s.setQueryGUC(ctx, conn.Conn(), qParam); reset != nil {
		defer reset()
	}

	batch := &pgx.Batch{}
	nQueued := 0
	for _, d := range descs {
		if len(d) != s.codeSize {
			log.Printf("pgvector: query descriptor length %d, want %d", len(d), s.codeSize)
			continue
		}
		batch.Queue(sql, bitString(d), k)
		nQueued++
	}
	if nQueued == 0 {
		return nil
	}
	br := conn.Conn().SendBatch(ctx, batch)
	defer br.Close()

	out := make([]index.Neighbor, 0, nQueued*k)
	for range nQueued {
		rows, err := br.Query()
		if err != nil {
			log.Printf("pgvector: batch query: %v", err)
			break
		}
		for rows.Next() {
			var vid, imgID int64
			var dist int
			if err := rows.Scan(&vid, &imgID, &dist); err != nil {
				log.Printf("pgvector: scan: %v", err)
				continue
			}
			out = append(out, index.Neighbor{
				Distance: uint32(dist),
				ID:       uint64(vid),
				ImageID:  imgID,
			})
		}
		rows.Close()
	}
	return out
}

func (s *Store) setQueryGUC(ctx context.Context, conn *pgx.Conn, qParam int) func() {
	if s.indexType == IndexHNSW {
		if qParam <= s.efSearch {
			return nil
		}
		if _, err := conn.Exec(ctx, "SET hnsw.ef_search = "+itoa(qParam)); err != nil {
			log.Printf("pgvector: set ef_search: %v", err)
			return nil
		}
		return func() { _, _ = conn.Exec(ctx, "RESET hnsw.ef_search") }
	}
	if qParam <= s.probes {
		return nil
	}
	if _, err := conn.Exec(ctx, "SET ivfflat.probes = "+itoa(qParam)); err != nil {
		log.Printf("pgvector: set probes: %v", err)
		return nil
	}
	return func() { _, _ = conn.Exec(ctx, "RESET ivfflat.probes") }
}

func (s *Store) queryOne(ctx context.Context, sql string, desc []byte, k, qParam int, out *[]index.Neighbor) {
	if len(desc) != s.codeSize {
		log.Printf("pgvector: query descriptor length %d, want %d", len(desc), s.codeSize)
		return
	}
	scan := func(rows pgx.Rows) {
		for rows.Next() {
			var vid, imgID int64
			var dist int
			if err := rows.Scan(&vid, &imgID, &dist); err != nil {
				log.Printf("pgvector: scan: %v", err)
				continue
			}
			*out = append(*out, index.Neighbor{
				Distance: uint32(dist),
				ID:       uint64(vid),
				ImageID:  imgID,
			})
		}
		if err := rows.Err(); err != nil {
			log.Printf("pgvector: rows: %v", err)
		}
	}

	// Use pool default session GUC when qParam is at floor.
	floor := s.probes
	if s.indexType == IndexHNSW {
		floor = s.efSearch
	}
	if qParam <= floor {
		rows, err := s.pool.Query(ctx, sql, bitString(desc), k)
		if err != nil {
			log.Printf("pgvector: query: %v", err)
			return
		}
		defer rows.Close()
		scan(rows)
		return
	}
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		log.Printf("pgvector: acquire conn: %v", err)
		return
	}
	defer conn.Release()
	if reset := s.setQueryGUC(ctx, conn.Conn(), qParam); reset != nil {
		defer reset()
	}
	rows, err := conn.Conn().Query(ctx, sql, bitString(desc), k)
	if err != nil {
		log.Printf("pgvector: query: %v", err)
		return
	}
	defer rows.Close()
	scan(rows)
}

func (s *Store) GetImagePath(ctx context.Context, id int64) (string, error) {
	var path string
	err := s.pool.QueryRow(ctx,
		fmt.Sprintf(`SELECT path FROM %s WHERE id = $1`, s.imageTable), id).Scan(&path)
	if errors.Is(err, pgx.ErrNoRows) {
		return "", errNoRows
	}
	return path, err
}

func (s *Store) Trained(ctx context.Context) bool {
	return s.tableExists(ctx, s.vectorTable)
}

func (s *Store) tableExists(ctx context.Context, name string) bool {
	var exists bool
	err := s.pool.QueryRow(ctx,
		`SELECT EXISTS (
			SELECT 1 FROM information_schema.tables
			WHERE table_schema = current_schema() AND table_name = $1
		)`, name).Scan(&exists)
	return err == nil && exists
}

func (s *Store) Train(ctx context.Context, _ [][]byte, _ index.TrainOptions) (index.TrainResult, error) {
	if err := s.ensureSchema(ctx); err != nil {
		return index.TrainResult{}, err
	}
	return index.TrainResult{}, nil
}

func (s *Store) NewBuilder(ctx context.Context) (index.Builder, error) {
	if !s.Trained(ctx) {
		if err := s.ensureSchema(ctx); err != nil {
			return nil, err
		}
	}
	return &noopBuilder{}, nil
}

func (s *Store) NewSearcher(ctx context.Context, _ int) (index.Searcher, error) {
	if !s.Trained(ctx) {
		if err := s.ensureSchema(ctx); err != nil {
			return nil, err
		}
	}
	// Ensure ANN index exists (no-op if already present or in bulk mode).
	if err := s.ensureANN(ctx); err != nil {
		return nil, err
	}
	return &searcher{s: s}, nil
}

func (s *Store) DeleteVectors(ctx context.Context, lo, hi uint64) error {
	if lo > hi {
		return nil
	}
	_, err := s.pool.Exec(ctx,
		fmt.Sprintf(`DELETE FROM %s WHERE id >= $1 AND id <= $2`, s.vectorTable),
		int64(lo), int64(hi))
	return err
}

func (s *Store) Online() bool { return true }

type searcher struct{ s *Store }

func (sr *searcher) Search(ctx context.Context, descriptors [][]byte, k, nprobe int) []index.Neighbor {
	return sr.s.SearchDescriptors(ctx, descriptors, k, nprobe)
}

func (sr *searcher) Close() error { return nil }

type noopBuilder struct{}

func (n *noopBuilder) Add(context.Context, [][]byte, []uint64) error { return nil }
func (n *noopBuilder) Flush(context.Context) error                   { return nil }
func (n *noopBuilder) Commit(context.Context) error                  { return nil }
func (n *noopBuilder) Close() error                                  { return nil }

// bitLookupTable maps each byte to its 8-char binary form (MSB first).
var bitLookupTable = func() [256][8]byte {
	var t [256][8]byte
	for i := range 256 {
		for j := 7; j >= 0; j-- {
			if i&(1<<j) != 0 {
				t[i][7-j] = '1'
			} else {
				t[i][7-j] = '0'
			}
		}
	}
	return t
}()

// bitBufPool reuses bit-string buffers (codeSize*8 bytes, typically 256).
var bitBufPool = sync.Pool{
	New: func() any {
		b := make([]byte, 256)
		return &b
	},
}

func bitString(b []byte) string {
	need := len(b) * 8
	bp := bitBufPool.Get().(*[]byte)
	buf := *bp
	if cap(buf) < need {
		buf = make([]byte, need)
	} else {
		buf = buf[:need]
	}
	for i, byt := range b {
		copy(buf[i*8:], bitLookupTable[byt][:])
	}
	// string copy; return buffer to pool
	out := string(buf)
	*bp = buf
	bitBufPool.Put(bp)
	return out
}

func isSafeIdent(s string) bool {
	for _, r := range s {
		if !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' ||
			r >= '0' && r <= '9' || r == '_') {
			return false
		}
	}
	return len(s) > 0
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
