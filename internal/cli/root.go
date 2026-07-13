package cli

import (
	"context"
	"fmt"
	"os"

	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"

	"imseek/internal/config"
	"imseek/internal/imdb"
	"imseek/internal/orb"
)

func bgContext() context.Context { return context.Background() }

var configFile string

func NewRootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:          "imseek",
		Short:        "Large-scale image similarity search (ORB + IVF/HNSW)",
		SilenceUsage: true,
		SilenceErrors: true,
		PersistentPreRunE: func(cmd *cobra.Command, args []string) error {
			if os.Getenv("IMSEEK_DEBUG") != "" {
				log.SetLevel(log.DebugLevel)
			}
			if err := config.InitViper(configFile); err != nil {
				return fmt.Errorf("load config: %w", err)
			}
			// Bind each persistent flag to its viper key explicitly.
			flags := cmd.Flags()
			bindMap := map[string]string{
				"data-dir":                 "data_dir",
				"backend":                  "backend",
				"milvus.address":           "milvus.address",
				"milvus.collection":        "milvus.collection",
				"pgvector.conn_string":     "pgvector.conn_string",
				"pgvector.table":           "pgvector.table",
				"pgvector.index_type":      "pgvector.index_type",
				"pgvector.m":               "pgvector.m",
				"pgvector.ef_construction": "pgvector.ef_construction",
				"pgvector.ef_search":       "pgvector.ef_search",
				"pgvector.lists":           "pgvector.lists",
				"pgvector.probes":          "pgvector.probes",
				"search.distance":          "search.distance",
				"search.count":             "search.count",
				"search.k":                 "search.k",
				"search.nprobe":            "search.nprobe",
				"search.threads":           "search.threads",
			}
			for flagName, viperKey := range bindMap {
				if flag := flags.Lookup(flagName); flag != nil {
					viper.BindPFlag(viperKey, flag)
				}
			}
			return nil
		},
	}

	root.PersistentFlags().StringVar(&configFile, "config", config.DefaultConfigFile(),
		"config file path (TOML)")
	root.PersistentFlags().StringP("data-dir", "d", config.DefaultDataDir(),
		"data directory (database, quantizer, index files)")
	root.PersistentFlags().String("backend", "local",
		"index backend: local | milvus | pgvector")
	root.PersistentFlags().String("milvus.address", "localhost:19530",
		"[milvus] server address")
	root.PersistentFlags().String("milvus.collection", "imseek",
		"[milvus] collection name")
	root.PersistentFlags().String("pgvector.conn_string", "",
		"[pgvector] PostgreSQL connection string")
	root.PersistentFlags().String("pgvector.table", "imseek",
		"[pgvector] table name prefix ({table}_image / {table}_vector)")
	root.PersistentFlags().String("pgvector.index_type", "ivfflat",
		"[pgvector] index type: ivfflat (fast build) | hnsw (better recall)")
	root.PersistentFlags().Int("pgvector.m", 16,
		"[pgvector] HNSW m (hnsw only)")
	root.PersistentFlags().Int("pgvector.ef_construction", 64,
		"[pgvector] HNSW ef_construction (hnsw only)")
	root.PersistentFlags().Int("pgvector.ef_search", 40,
		"[pgvector] HNSW ef_search floor (hnsw only)")
	root.PersistentFlags().Int("pgvector.lists", 0,
		"[pgvector] IVFFlat lists (0=auto from row count)")
	root.PersistentFlags().Int("pgvector.probes", 10,
		"[pgvector] IVFFlat probes (lists to probe at query time)")
	root.PersistentFlags().Int("search.distance", 64,
		"max Hamming distance")
	root.PersistentFlags().Int("search.count", 10,
		"number of results to return")
	root.PersistentFlags().Int("search.k", 3,
		"KNN per descriptor")
	root.PersistentFlags().Int("search.nprobe", 3,
		"number of inverted lists to probe")
	root.PersistentFlags().Int("search.threads", 0,
		"search threads (0=auto)")

	root.AddCommand(
		newAddCmd(),
		newTrainCmd(),
		newBuildCmd(),
		newSearchCmd(),
		newServerCmd(),
		newCleanCmd(),
	)
	return root
}

func Execute() {
	if err := NewRootCmd().Execute(); err != nil {
		log.Error("command failed", "err", err)
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func dbOptions(codeSize int, cache bool, scoreType config.ScoreType) imdb.Options {
	return imdb.Options{
		ConfDir:   config.DataDir(),
		WAL:       true,
		Cache:     cache,
		ScoreType: scoreType,
		CodeSize:  codeSize,
		Backend:   viper.GetString("backend"),
		Milvus: imdb.MilvusOptions{
			Address:    viper.GetString("milvus.address"),
			Collection: viper.GetString("milvus.collection"),
		},
		Pgvector: imdb.PgvectorOptions{
			ConnString:     viper.GetString("pgvector.conn_string"),
			Table:          viper.GetString("pgvector.table"),
			IndexType:      viper.GetString("pgvector.index_type"),
			M:              viper.GetInt("pgvector.m"),
			EFConstruction: viper.GetInt("pgvector.ef_construction"),
			EFSearch:       viper.GetInt("pgvector.ef_search"),
			Lists:          viper.GetInt("pgvector.lists"),
			Probes:         viper.GetInt("pgvector.probes"),
		},
	}
}

func searchOptions(scoreType string) config.SearchOptions {
	opts := config.SearchOptionsFromViper()
	if scoreType != "" {
		opts.ScoreType = config.ParseScoreType(scoreType)
	}
	return opts
}

func codeSize() int { return orb.DescriptorBytes }
