package imdb

import (
	"context"
	"errors"
	"fmt"

	"imseek/internal/index"
)

var errNotTrained = errors.New("index backend not trained: run `train` first")

// ExportVectors returns up to count descriptor codes for quantizer training,
// sampled evenly across the whole dataset so the training set is
// representative (rather than biased toward the first few images).
func (m *IMDB) ExportVectors(ctx context.Context, count int) ([][]byte, error) {
	imgCount, totalVec, err := m.db.GetCount(ctx)
	if err != nil {
		return nil, err
	}
	if imgCount == 0 || totalVec == 0 {
		return nil, nil
	}
	// If the whole dataset fits within count, just take everything.
	if int64(count) >= totalVec {
		return m.exportSequential(ctx, count)
	}
	// Otherwise subsample: keep roughly `count/totalVec` of each image's
	// descriptors, spread across all images by iterating in id order.
	keepRatio := float64(count) / float64(totalVec)
	var out [][]byte
	const batch = 500
	var offset int64
	for len(out) < count {
		rows, err := m.db.GetVectors(ctx, batch, offset)
		if err != nil {
			return nil, err
		}
		if len(rows) == 0 {
			break
		}
		for _, r := range rows {
			n := len(r.Vector) / m.codeSize
			// deterministic stride sampling within this image
			step := 1
			if keepRatio > 0 {
				step = int(1.0 / keepRatio)
			}
			if step < 1 {
				step = 1
			}
			for i := 0; i < n; i += step {
				code := make([]byte, m.codeSize)
				copy(code, r.Vector[i*m.codeSize:(i+1)*m.codeSize])
				out = append(out, code)
				if len(out) >= count {
					return out, nil
				}
			}
		}
		offset += batch
	}
	return out, nil
}

func (m *IMDB) exportSequential(ctx context.Context, count int) ([][]byte, error) {
	var out [][]byte
	const batch = 1000
	var offset int64
	for len(out) < count {
		rows, err := m.db.GetVectors(ctx, batch, offset)
		if err != nil {
			return nil, err
		}
		if len(rows) == 0 {
			break
		}
		for _, r := range rows {
			for i := 0; i+m.codeSize <= len(r.Vector); i += m.codeSize {
				code := make([]byte, m.codeSize)
				copy(code, r.Vector[i:i+m.codeSize])
				out = append(out, code)
				if len(out) >= count {
					return out, nil
				}
			}
		}
		offset += batch
	}
	return out, nil
}

type BuildOptions struct {
	BatchSize int
}

// vectorsPerSubIndex bounds how many descriptors are held in memory before a
// builder Flush (durable checkpoint). This keeps build memory bounded
// regardless of dataset size. ~5M codes ≈ a few hundred MB.
const vectorsPerSubIndex = 5_000_000

// dbReadChunk is how many image rows are pulled from SQLite per query, bounding
// transient DB-read memory.
const dbReadChunk = 500

func (m *IMDB) TrainIndex(ctx context.Context, samples [][]byte, opts index.TrainOptions) (index.TrainResult, error) {
	return m.backend.Train(ctx, samples, opts)
}

// BuildIndex builds (or extends) the index from unindexed vectors via the
// backend builder. It requires the backend to be trained.
//
// On the online (pgvector) path this is a no-op: AddImage already wrote vectors.
func (m *IMDB) BuildIndex(ctx context.Context, opts BuildOptions) error {
	if m.online != nil {
		return nil
	}
	if !m.backend.Trained(ctx) {
		return errNotTrained
	}

	vecCap := vectorsPerSubIndex
	if opts.BatchSize > 0 {
		vecCap = opts.BatchSize
	}

	builder, err := m.backend.NewBuilder(ctx)
	if err != nil {
		return fmt.Errorf("build: new builder: %w", err)
	}
	defer builder.Close()

	var batchImageIDs []int64
	var batchVecCount int

	flush := func() error {
		if batchVecCount == 0 {
			return nil
		}
		if err := builder.Flush(ctx); err != nil {
			return fmt.Errorf("build: flush: %w", err)
		}
		if err := m.db.SetIndexedBatch(ctx, batchImageIDs); err != nil {
			return fmt.Errorf("build: mark indexed: %w", err)
		}
		batchImageIDs = batchImageIDs[:0]
		batchVecCount = 0
		return nil
	}

	for {
		if ctx.Err() != nil {
			return fmt.Errorf("build: cancelled: %w", ctx.Err())
		}
		rows, err := m.db.GetVectorsUnindexed(ctx, int64(dbReadChunk), int64(len(batchImageIDs)))
		if err != nil {
			return fmt.Errorf("build: read unindexed vectors: %w", err)
		}
		if len(rows) == 0 {
			break
		}
		for _, r := range rows {
			batchImageIDs = append(batchImageIDs, r.ID)
			n := len(r.Vector) / m.codeSize
			features := make([][]byte, n)
			ids := make([]uint64, n)
			for i := range n {
				features[i] = r.Vector[i*m.codeSize : (i+1)*m.codeSize]
				ids[i] = uint64(r.TotalVectorCount) - uint64(i)
			}
			if err := builder.Add(ctx, features, ids); err != nil {
				return fmt.Errorf("build: add vectors: %w", err)
			}
			batchVecCount += n
		}
		if batchVecCount >= vecCap {
			if err := flush(); err != nil {
				return err
			}
		}
	}
	if err := flush(); err != nil {
		return err
	}

	if err := builder.Commit(ctx); err != nil {
		return fmt.Errorf("build: commit: %w", err)
	}
	return nil
}

// OpenIndex opens the built index for querying via the backend.
// On the online path returns a searcher that delegates to PostgreSQL.
func (m *IMDB) OpenIndex(ctx context.Context, threads int) (index.Searcher, func() error, error) {
	s, err := m.backend.NewSearcher(ctx, threads)
	if err != nil {
		return nil, nil, fmt.Errorf("open index: %w", err)
	}
	return s, s.Close, nil
}

// ClearCache deletes stored descriptor blobs. If all is true, deletes every
// vector; otherwise only vectors already indexed. No-op on the online path
// (descriptors live only in PostgreSQL).
func (m *IMDB) ClearCache(ctx context.Context, all bool) error {
	if m.online != nil {
		return nil
	}
	if all {
		return m.db.DeleteVectorsAll(ctx)
	}
	return m.db.DeleteVectors(ctx)
}
