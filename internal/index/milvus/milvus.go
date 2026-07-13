//go:build milvus

// Package milvus implements the vector-index Backend interface using a Milvus
// server with BIN_IVF_FLAT / HAMMING over binary (256-bit ORB) descriptors.
//
// Build with: go build -tags milvus ./...
package milvus

import (
	"context"
	"errors"
	"fmt"

	"github.com/milvus-io/milvus/client/v2/column"
	"github.com/milvus-io/milvus/client/v2/entity"
	milvusindex "github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"

	"imseek/internal/index"
)

const (
	idField  = "id"
	vecField = "vector"
)

type Options struct {
	Address    string
	Collection string
}

type Backend struct {
	client     *milvusclient.Client
	collection string
	dim        int // dimension in bits (= codeSize*8)
}

// New connects to the Milvus server and returns the backend.
// codeSize is the descriptor length in bytes (32 for ORB => dim 256).
func New(ctx context.Context, opts Options, codeSize int) (*Backend, error) {
	if opts.Address == "" {
		return nil, errors.New("milvus: address is required")
	}
	if opts.Collection == "" {
		return nil, errors.New("milvus: collection name is required")
	}
	if codeSize <= 0 {
		return nil, errors.New("milvus: codeSize must be positive")
	}
	c, err := milvusclient.New(ctx, &milvusclient.ClientConfig{
		Address: opts.Address,
	})
	if err != nil {
		return nil, fmt.Errorf("milvus connect: %w", err)
	}
	return &Backend{
		client:     c,
		collection: opts.Collection,
		dim:        codeSize * 8,
	}, nil
}

func (b *Backend) Close() error {
	if b.client != nil {
		return b.client.Close(context.Background())
	}
	return nil
}

func (b *Backend) Trained(ctx context.Context) bool {
	ok, err := b.client.HasCollection(ctx,
		milvusclient.NewHasCollectionOption(b.collection))
	return err == nil && ok
}

func (b *Backend) Train(ctx context.Context, _ [][]byte, opts index.TrainOptions) (index.TrainResult, error) {
	if opts.NList <= 0 {
		return index.TrainResult{}, errors.New("milvus: nlist must be positive")
	}

	// Drop existing collection so re-training starts clean.
	if ok, _ := b.client.HasCollection(ctx, milvusclient.NewHasCollectionOption(b.collection)); ok {
		if err := b.client.DropCollection(ctx, milvusclient.NewDropCollectionOption(b.collection)); err != nil {
			return index.TrainResult{}, fmt.Errorf("milvus drop: %w", err)
		}
	}

	schema := entity.NewSchema().WithName(b.collection).
		WithField(entity.NewField().
			WithName(idField).
			WithDataType(entity.FieldTypeInt64).
			WithIsPrimaryKey(true)).
		WithField(entity.NewField().
			WithName(vecField).
			WithDataType(entity.FieldTypeBinaryVector).
			WithDim(int64(b.dim)))

	if err := b.client.CreateCollection(ctx,
		milvusclient.NewCreateCollectionOption(b.collection, schema)); err != nil {
		return index.TrainResult{}, fmt.Errorf("milvus create collection: %w", err)
	}

	task, err := b.client.CreateIndex(ctx, milvusclient.NewCreateIndexOption(
		b.collection, vecField,
		milvusindex.NewBinIvfFlatIndex(entity.HAMMING, opts.NList),
	))
	if err != nil {
		return index.TrainResult{}, fmt.Errorf("milvus create index: %w", err)
	}
	if err := task.Await(ctx); err != nil {
		return index.TrainResult{}, fmt.Errorf("milvus build index: %w", err)
	}

	return index.TrainResult{Centroids: opts.NList}, nil
}

func (b *Backend) NewBuilder(ctx context.Context) (index.Builder, error) {
	if !b.Trained(ctx) {
		return nil, errors.New("milvus: collection not trained")
	}
	return &builder{b: b}, nil
}

func (b *Backend) NewSearcher(ctx context.Context, _ int) (index.Searcher, error) {
	if !b.Trained(ctx) {
		return nil, errors.New("milvus: collection not trained")
	}
	task, err := b.client.LoadCollection(ctx,
		milvusclient.NewLoadCollectionOption(b.collection))
	if err != nil {
		return nil, fmt.Errorf("milvus load: %w", err)
	}
	if err := task.Await(ctx); err != nil {
		return nil, fmt.Errorf("milvus load await: %w", err)
	}
	return &searcher{b: b}, nil
}

func (b *Backend) DeleteVectors(ctx context.Context, lo, hi uint64) error {
	if lo > hi {
		return nil
	}
	if !b.Trained(ctx) {
		return nil
	}
	expr := fmt.Sprintf("%s >= %d && %s <= %d", idField, lo, idField, hi)
	_, err := b.client.Delete(ctx, milvusclient.NewDeleteOption(b.collection).WithExpr(expr))
	if err != nil {
		return fmt.Errorf("milvus delete: %w", err)
	}
	return nil
}

type builder struct {
	b        *Backend
	bufCodes [][]byte
	bufIDs   []int64
}

const insertBatchSize = 50_000

func (bl *builder) Add(ctx context.Context, codes [][]byte, ids []uint64) error {
	if len(codes) == 0 {
		return nil
	}
	for i, code := range codes {
		bl.bufCodes = append(bl.bufCodes, code)
		bl.bufIDs = append(bl.bufIDs, int64(ids[i]))
	}
	for len(bl.bufCodes) >= insertBatchSize {
		if err := bl.insertBatch(ctx, bl.bufCodes[:insertBatchSize], bl.bufIDs[:insertBatchSize]); err != nil {
			return err
		}
		bl.bufCodes = append(bl.bufCodes[:0], bl.bufCodes[insertBatchSize:]...)
		bl.bufIDs = append(bl.bufIDs[:0], bl.bufIDs[insertBatchSize:]...)
	}
	return nil
}

func (bl *builder) insertBatch(ctx context.Context, codes [][]byte, ids []int64) error {
	_, err := bl.b.client.Insert(ctx,
		milvusclient.NewColumnBasedInsertOption(bl.b.collection).
			WithInt64Column(idField, ids).
			WithColumns(column.NewColumnBinaryVector(vecField, bl.b.dim, codes)))
	return err
}

func (bl *builder) Flush(ctx context.Context) error {
	if len(bl.bufCodes) == 0 {
		return nil
	}
	codes := bl.bufCodes
	ids := bl.bufIDs
	bl.bufCodes = nil
	bl.bufIDs = nil
	return bl.insertBatch(ctx, codes, ids)
}

func (bl *builder) Commit(ctx context.Context) error {
	if err := bl.Flush(ctx); err != nil {
		return err
	}
	task, err := bl.b.client.Flush(ctx,
		milvusclient.NewFlushOption(bl.b.collection))
	if err != nil {
		return err
	}
	return task.Await(ctx)
}

func (bl *builder) Close() error { return nil }

type searcher struct {
	b *Backend
}

func (s *searcher) Search(ctx context.Context, descriptors [][]byte, k, nprobe int) []index.Neighbor {
	if len(descriptors) == 0 || k <= 0 {
		return nil
	}

	// Batch all descriptors into a single Search request to avoid per-descriptor
	// gRPC round-trips. Milvus returns one ResultSet per query vector.
	qvecs := make([]entity.Vector, len(descriptors))
	for i, desc := range descriptors {
		qvecs[i] = entity.BinaryVector(desc)
	}
	ann := milvusindex.NewIvfAnnParam(nprobe)
	res, err := s.b.client.Search(ctx, milvusclient.NewSearchOption(
		s.b.collection, k, qvecs).
		WithANNSField(vecField).
		WithAnnParam(ann))
	if err != nil {
		return nil
	}
	out := make([]index.Neighbor, 0, len(res)*k)
	for _, rs := range res {
		for i := 0; i < rs.Len(); i++ {
			id, err := rs.IDs.GetAsInt64(i)
			if err != nil {
				continue
			}
			out = append(out, index.Neighbor{
				Distance: uint32(rs.Scores[i]),
				ID:       uint64(id),
			})
		}
	}
	return out
}

func (s *searcher) Close() error { return nil }
