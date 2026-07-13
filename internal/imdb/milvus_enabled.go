//go:build milvus

package imdb

import (
	"context"
	"fmt"

	"imseek/internal/index"
	"imseek/internal/index/milvus"
)

// newMilvusBackend constructs the Milvus-backed vector index. Only available
// when built with the `milvus` build tag.
func newMilvusBackend(opts Options) (index.Backend, error) {
	b, err := milvus.New(context.Background(), milvus.Options{
		Address:    opts.Milvus.Address,
		Collection: opts.Milvus.Collection,
	}, opts.CodeSize)
	if err != nil {
		return nil, fmt.Errorf("milvus backend: %w", err)
	}
	return b, nil
}
