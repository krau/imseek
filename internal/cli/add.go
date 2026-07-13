package cli

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"
	"github.com/zeebo/blake3"

	"imseek/internal/imdb"
)

type addOpts struct {
	xf           extractorFlags
	suffix       string
	minKeypoints int
}

func newAddCmd() *cobra.Command {
	o := &addOpts{}
	cmd := &cobra.Command{
		Use:   "add DIR",
		Short: "Extract features from images in DIR and add to the database",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runAdd(cmd.Context(), o, args[0])
		},
	}
	o.xf.register(cmd)
	cmd.Flags().StringVarP(&o.suffix, "suffix", "s", "jpg,png,webp", "comma-separated image extensions to scan")
	cmd.Flags().IntVarP(&o.minKeypoints, "min-keypoints", "m", 4, "skip images with no more than this many keypoints")
	return cmd
}

// insertJob is the result of processing one image: hash + descriptors.
// Raw image bytes are already freed by this point.
type insertJob struct {
	path string
	hash [32]byte
	desc [][]byte
}

func runAdd(ctx context.Context, o *addOpts, dir string) error {
	suffixRe, err := suffixRegexp(o.suffix)
	if err != nil {
		return err
	}

	m, err := imdb.Open(ctx, dbOptions(codeSize(), false, 0))
	if err != nil {
		return err
	}
	defer m.Close()

	// Bulk import: drop the ANN index during import, rebuild once at the end.
	if m.Online() {
		log.Info("bulk import: ANN index paused, will rebuild on completion")
		if err := m.BeginBulkImport(ctx); err != nil {
			return err
		}
		defer func() {
			log.Info("rebuilding ANN index...")
			if err := m.EndBulkImport(ctx); err != nil {
				log.Error("ANN index rebuild failed", "err", err)
			} else {
				log.Info("ANN index ready")
			}
		}()
	}

	ext, err := o.xf.newExtractor()
	if err != nil {
		return err
	}
	defer ext.Close()

	// Two-stage pipeline:
	//   Stage 1 (N workers): read file → hash → dedup → extract → insertJob
	//   Stage 2 (1 writer):  insertJob → store (batched)
	// Raw image bytes live only inside Stage 1 workers, keeping memory bounded.
	pathCh := make(chan string, runtime.NumCPU()*2)
	insertCh := make(chan insertJob, runtime.NumCPU()*2)

	var added, skipped atomic.Int64

	// Stage 1: process files in parallel.
	var procWg sync.WaitGroup
	workers := min(runtime.NumCPU(), 16)
	for range workers {
		procWg.Go(func() {
			for path := range pathCh {
				data, err := os.ReadFile(path)
				if err != nil {
					log.Warn("read failed", "path", path, "err", err)
					skipped.Add(1)
					continue
				}

				hash := blake3.Sum256(data)

				_, ok, err := m.CheckHash(ctx, hash[:])
				if err != nil {
					log.Warn("dedup check failed", "path", path, "err", err)
					skipped.Add(1)
					continue
				}
				if ok {
					skipped.Add(1)
					continue
				}

				desc, err := ext.DetectBytes(data)
				if err != nil {
					log.Warn("feature extraction failed", "path", path, "err", err)
					skipped.Add(1)
					continue
				}
				if len(desc) <= o.minKeypoints {
					skipped.Add(1)
					continue
				}

				select {
				case insertCh <- insertJob{path: path, hash: hash, desc: desc}:
				case <-ctx.Done():
					return
				}
			}
		})
	}

	// Stage 2: single-writer batched insert.
	batchSize := 16
	if m.Online() {
		batchSize = 64
	}
	batch := make([]imdb.BatchItem, 0, batchSize)
	flush := func() {
		if len(batch) == 0 {
			return
		}
		n, err := m.AddImageBatch(ctx, batch)
		if err != nil {
			// Batch failed; retry each item individually.
			for i := 0; i < len(batch); i++ {
				item := batch[i]
				if _, err := m.AddImageRaw(ctx, item.Hash, item.Path, item.Blob, item.Count); err != nil {
					skipped.Add(1)
					continue
				}
				added.Add(1)
			}
		} else {
			added.Add(int64(n))
		}
		batch = batch[:0]
	}

	var writeWg sync.WaitGroup
	writeWg.Go(func() {
		for job := range insertCh {
			if _, ok, err := m.CheckHash(ctx, job.hash[:]); err != nil {
				log.Warn("dedup check failed", "path", job.path, "err", err)
				skipped.Add(1)
				continue
			} else if ok {
				skipped.Add(1)
				continue
			}
			batch = append(batch, imdb.BatchItem{
				Hash:  job.hash[:],
				Path:  job.path,
				Blob:  flattenDesc(job.desc),
				Count: int64(len(job.desc)),
			})
			if len(batch) >= batchSize {
				flush()
				n := added.Load()
				if n%100 < int64(batchSize) {
					log.Info("added", "count", n)
				}
			}
		}
		flush()
	})

	go func() {
		defer close(pathCh)
		if err := filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				log.Warn("walk error", "path", path, "err", err)
				return nil
			}
			if d.IsDir() || !suffixRe.MatchString(path) {
				return nil
			}
			select {
			case pathCh <- path:
			case <-ctx.Done():
				return ctx.Err()
			}
			return nil
		}); err != nil && !errors.Is(err, ctx.Err()) {
			log.Error("walk failed", "dir", dir, "err", err)
		}
	}()

	procWg.Wait()
	close(insertCh)
	writeWg.Wait()

	log.Info("add complete", "added", added.Load(), "skipped", skipped.Load())
	return nil
}

func suffixRegexp(suffix string) (*regexp.Regexp, error) {
	parts := strings.Split(suffix, ",")
	for i := range parts {
		parts[i] = regexp.QuoteMeta(strings.TrimSpace(parts[i]))
	}
	return regexp.Compile(`(?i)\.(` + strings.Join(parts, "|") + `)$`)
}

func flattenDesc(descs [][]byte) []byte {
	if len(descs) == 0 {
		return nil
	}
	cs := len(descs[0])
	out := make([]byte, 0, len(descs)*cs)
	for _, d := range descs {
		out = append(out, d...)
	}
	return out
}
