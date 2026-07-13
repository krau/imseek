package config

import (
	"log"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/spf13/viper"
)

const (
	DBFile        = "imseek.db"
	QuantizerFile = "quantizer.bin"
	InvlistsFile  = "invlists.bin"
)

func DefaultDataDir() string {
	return "data"
}

func DefaultConfigFile() string {
	return "imseek.toml"
}

func InitViper(configFile string) error {
	v := viper.GetViper()

	// Env var support: IMSEEK_DATA_DIR, IMSEEK_BACKEND, etc.
	v.SetEnvPrefix("IMSEEK")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	// Defaults
	v.SetDefault("data_dir", DefaultDataDir())
	v.SetDefault("backend", "local")
	v.SetDefault("milvus.address", "localhost:19530")
	v.SetDefault("milvus.collection", "imseek")
	v.SetDefault("pgvector.conn_string", "")
	v.SetDefault("pgvector.table", "imseek")
	v.SetDefault("pgvector.index_type", "ivfflat") // ivfflat | hnsw
	v.SetDefault("pgvector.m", 16)
	v.SetDefault("pgvector.ef_construction", 64)
	v.SetDefault("pgvector.ef_search", 40)
	v.SetDefault("pgvector.lists", 0) // 0 = auto from row count
	v.SetDefault("pgvector.probes", 10)
	v.SetDefault("search.distance", 64)
	v.SetDefault("search.count", 10)
	v.SetDefault("search.k", 3)
	v.SetDefault("search.nprobe", 3)
	v.SetDefault("search.threads", runtime.NumCPU())

	// Config file (optional) — missing file is fine, other errors are not.
	v.SetConfigFile(configFile)
	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok && !os.IsNotExist(err) {
			return err
		}
	}
	return nil
}

func DataDir() string {
	dir := viper.GetString("data_dir")
	if dir == "" {
		dir = DefaultDataDir()
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		log.Printf("warn: failed to create data dir %s: %v", dir, err)
	}
	return dir
}

type ScoreType int

const (
	ScoreWilson ScoreType = iota
	ScoreCount
)

func ParseScoreType(s string) ScoreType {
	if s == "count" {
		return ScoreCount
	}
	return ScoreWilson
}

// SearchOptions groups search tuning parameters.
type SearchOptions struct {
	Distance  uint32
	Count     int
	K         int
	NProbe    int
	ScoreType ScoreType
	Threads   int
}

// DefaultSearchOptions returns sensible defaults.
func DefaultSearchOptions() SearchOptions {
	return SearchOptions{
		Distance:  64,
		Count:     10,
		K:         3,
		NProbe:    3,
		ScoreType: ScoreWilson,
		Threads:   runtime.NumCPU(),
	}
}

func SearchOptionsFromViper() SearchOptions {
	d := DefaultSearchOptions()
	if v := viper.GetInt("search.distance"); v > 0 {
		d.Distance = uint32(v)
	}
	if v := viper.GetInt("search.count"); v > 0 {
		d.Count = v
	}
	if v := viper.GetInt("search.k"); v > 0 {
		d.K = v
	}
	if v := viper.GetInt("search.nprobe"); v > 0 {
		d.NProbe = v
	}
	if v := viper.GetInt("search.threads"); v > 0 {
		d.Threads = v
	}
	return d
}

// JoinDataDir joins the data directory with a filename.
func JoinDataDir(name string) string {
	return filepath.Join(DataDir(), name)
}
