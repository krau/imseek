//go:build !pgvector

package imdb

import (
	"fmt"

	"imseek/internal/index"
)

func newPgvectorStore(_ Options) (OnlineStore, index.Backend, error) {
	return nil, nil, fmt.Errorf("pgvector backend not compiled in: rebuild with -tags pgvector")
}
