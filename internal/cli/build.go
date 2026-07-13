package cli

import (
	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"

	"imseek/internal/imdb"
)

func newBuildCmd() *cobra.Command {
	var batchSize int
	cmd := &cobra.Command{
		Use:   "build",
		Short: "Build the index from added features",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			m, err := imdb.Open(ctx, dbOptions(codeSize(), false, 0))
			if err != nil {
				return err
			}
			defer m.Close()

			n, _ := m.CountUnindexed(ctx)
			log.Info("building index", "unindexed", n, "batch", batchSize)
			if err := m.BuildIndex(ctx, imdb.BuildOptions{BatchSize: batchSize}); err != nil {
				return err
			}
			log.Info("index built", "path", m.InvlistsPath())
			return nil
		},
	}
	cmd.Flags().IntVarP(&batchSize, "batch-size", "b", 100000, "images per build batch")
	return cmd
}

func newCleanCmd() *cobra.Command {
	var all, force bool
	cmd := &cobra.Command{
		Use:   "clean",
		Short: "Remove stored feature blobs to shrink the database",
		RunE: func(cmd *cobra.Command, args []string) error {
			ctx := cmd.Context()
			if !force {
				log.Warn("this deletes stored descriptor blobs; pass --force to proceed")
				return nil
			}
			m, err := imdb.Open(ctx, dbOptions(codeSize(), false, 0).WithWAL(false))
			if err != nil {
				return err
			}
			defer m.Close()
			if err := m.ClearCache(ctx, all); err != nil {
				return err
			}
			log.Info("clean complete", "all", all)
			return nil
		},
	}
	cmd.Flags().BoolVar(&all, "all", false, "delete all vectors (otherwise only indexed ones)")
	cmd.Flags().BoolVar(&force, "force", false, "skip confirmation")
	return cmd
}
