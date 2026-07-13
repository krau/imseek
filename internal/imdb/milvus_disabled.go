//go:build !milvus

package imdb

import (
	"fmt"

	"imseek/internal/index"
)

// newMilvusBackend returns an error when the binary was not built with the
// `milvus` build tag.
func newMilvusBackend(_ Options) (index.Backend, error) {
	return nil, fmt.Errorf("milvus backend not compiled in: rebuild with -tags milvus")
}
