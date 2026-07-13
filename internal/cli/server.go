package cli

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"

	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"

	"imseek/internal/config"
	"imseek/internal/imdb"
	"imseek/internal/index"
	"imseek/internal/server"
)

type serverOpts struct {
	xf        extractorFlags
	scoreType string
	addr      string
	token     string
}

func newServerCmd() *cobra.Command {
	o := &serverOpts{}
	cmd := &cobra.Command{
		Use:   "server",
		Short: "Start the HTTP search service",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runServer(o)
		},
	}
	o.xf.register(cmd)
	cmd.Flags().StringVar(&o.scoreType, "score-type", "wilson", "scoring: wilson|count")
	cmd.Flags().StringVar(&o.addr, "addr", "127.0.0.1:8000", "listen address")
	cmd.Flags().StringVar(&o.token, "token", "", "Bearer auth token (random if empty)")
	return cmd
}

func runServer(o *serverOpts) error {
	ctx := bgContext()
	opts := searchOptions(o.scoreType)

	token := o.token
	if token == "" {
		var err error
		token, err = randomToken(16)
		if err != nil {
			return err
		}
		log.Info("generated random token", "token", token)
	}

	m, err := imdb.Open(ctx, dbOptions(codeSize(), true, opts.ScoreType))
	if err != nil {
		return err
	}
	defer m.Close()

	ext, err := o.xf.newExtractor()
	if err != nil {
		return err
	}
	defer ext.Close()

	var idx index.Searcher
	var closeIdx = func() error { return nil }
	s, c, err := m.OpenIndex(ctx, opts.Threads)
	if err != nil {
		if m.Online() {
			return err
		}
		// Local: allow starting before first train/build.
		log.Warn("index not ready; search returns empty until train+build", "err", err)
		idx = emptySearcher{}
	} else {
		idx, closeIdx = s, c
	}
	defer closeIdx()

	if m.Online() {
		log.Info("online backend: add is immediately searchable (no train/build)")
	}

	app := server.New(&server.Deps{
		DB:        m,
		Index:     idx,
		Extractor: ext,
		Search:    opts,
		Token:     token,
		Threads:   opts.Threads,
	})

	backend := "local"
	if m.Online() {
		backend = "pgvector"
	}
	log.Info("HTTP server starting", "addr", o.addr, "backend", backend, "data_dir", config.DataDir())
	return app.Listen(o.addr)
}

// emptySearcher returns no hits (local server started before train/build).
type emptySearcher struct{}

func (emptySearcher) Search(context.Context, [][]byte, int, int) []index.Neighbor { return nil }
func (emptySearcher) Close() error                                                { return nil }

func randomToken(n int) (string, error) {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("generate random token: %w", err)
	}
	return hex.EncodeToString(b), nil
}
