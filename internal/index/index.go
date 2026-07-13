package index

import "context"

// Neighbor is a search hit. ImageID, when non-zero, is the owning image id
// already resolved by the backend (pgvector online path).
type Neighbor struct {
	Distance uint32
	ID       uint64
	ImageID  int64 // 0 = unknown; resolve via vector-id map
}

type TrainOptions struct {
	NList    int
	MaxIter  int
	Init     string // "random" | "kmeans-plus-plus"
	TwoLevel bool
}

type TrainResult struct {
	Centroids int
	Imbalance float64
}

// Builder ingests vectors. Flush is a durable checkpoint; Commit finalizes.
type Builder interface {
	Add(ctx context.Context, codes [][]byte, ids []uint64) error
	Flush(ctx context.Context) error
	Commit(ctx context.Context) error
	Close() error
}

type Searcher interface {
	Search(ctx context.Context, descriptors [][]byte, k, nprobe int) []Neighbor
	Close() error
}

type Backend interface {
	Train(ctx context.Context, samples [][]byte, opts TrainOptions) (TrainResult, error)
	Trained(ctx context.Context) bool
	NewBuilder(ctx context.Context) (Builder, error)
	NewSearcher(ctx context.Context, threads int) (Searcher, error)
	// DeleteVectors removes ids in [lo, hi]. lo>hi is a no-op.
	DeleteVectors(ctx context.Context, lo, hi uint64) error
	Close() error
}
