//go:build pgvector

package imdb

import (
	"context"
	"fmt"

	"imseek/internal/index"
	"imseek/internal/index/pgvector"
)

type pgOnline struct {
	s *pgvector.Store
}

func (p *pgOnline) AddImage(ctx context.Context, hash []byte, path string, descriptors [][]byte) (int64, error) {
	return p.s.AddImage(ctx, hash, path, descriptors)
}

func (p *pgOnline) AddImageBatch(ctx context.Context, jobs []BatchItem, codeSize int) (int, error) {
	items := make([]pgvector.ImageBatchItem, 0, len(jobs))
	for _, j := range jobs {
		n := int(j.Count)
		if n <= 0 {
			n = len(j.Blob) / codeSize
		}
		descs := unflattenN(j.Blob, codeSize, n)
		if len(descs) == 0 {
			continue
		}
		items = append(items, pgvector.ImageBatchItem{
			Hash:        j.Hash,
			Path:        j.Path,
			Descriptors: descs,
		})
	}
	return p.s.AddImageBatch(ctx, items)
}

func (p *pgOnline) BeginBulkImport(ctx context.Context) error {
	return p.s.BeginBulkImport(ctx)
}

func (p *pgOnline) EndBulkImport(ctx context.Context) error {
	return p.s.EndBulkImport(ctx)
}

func (p *pgOnline) DeleteImage(ctx context.Context, id int64) error {
	return p.s.DeleteImage(ctx, id)
}

func (p *pgOnline) CheckHash(ctx context.Context, hash []byte) (int64, bool, error) {
	return p.s.CheckHash(ctx, hash)
}

func (p *pgOnline) GetImage(ctx context.Context, id int64) (string, []byte, int64, error) {
	img, err := p.s.GetImage(ctx, id)
	if err != nil {
		return "", nil, 0, err
	}
	return img.Path, img.Hash, img.VectorCount, nil
}

func (p *pgOnline) ListImages(ctx context.Context, limit, offset int64) ([]OnlineImage, error) {
	rows, err := p.s.ListImages(ctx, limit, offset)
	if err != nil {
		return nil, err
	}
	out := make([]OnlineImage, len(rows))
	for i, r := range rows {
		out[i] = OnlineImage{ID: r.ID, Path: r.Path, Hash: r.Hash, VectorCount: r.VectorCount}
	}
	return out, nil
}

func (p *pgOnline) Count(ctx context.Context) (int64, int64, error) {
	return p.s.Count(ctx)
}

func (p *pgOnline) SearchDescriptors(ctx context.Context, descriptors [][]byte, k, nprobe int) []index.Neighbor {
	return p.s.SearchDescriptors(ctx, descriptors, k, nprobe)
}

func (p *pgOnline) GetImagePath(ctx context.Context, id int64) (string, error) {
	return p.s.GetImagePath(ctx, id)
}

func (p *pgOnline) IsNotFound(err error) bool {
	return pgvector.IsNotFound(err)
}

func newPgvectorStore(opts Options) (OnlineStore, index.Backend, error) {
	s, err := pgvector.New(context.Background(), pgvector.Options{
		ConnString:     opts.Pgvector.ConnString,
		Table:          opts.Pgvector.Table,
		IndexType:      opts.Pgvector.IndexType,
		M:              opts.Pgvector.M,
		EFConstruction: opts.Pgvector.EFConstruction,
		EFSearch:       opts.Pgvector.EFSearch,
		Lists:          opts.Pgvector.Lists,
		Probes:         opts.Pgvector.Probes,
	}, opts.CodeSize)
	if err != nil {
		return nil, nil, fmt.Errorf("pgvector store: %w", err)
	}
	return &pgOnline{s: s}, s, nil
}
