package config

import (
	"testing"
)

func TestParseScoreType(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  ScoreType
	}{
		{"wilson explicit", "wilson", ScoreWilson},
		{"count explicit", "count", ScoreCount},
		{"empty defaults to wilson", "", ScoreWilson},
		{"unknown defaults to wilson", "foobar", ScoreWilson},
		{"case sensitive", "Wilson", ScoreWilson},
		{"count uppercase", "Count", ScoreWilson},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ParseScoreType(tt.input)
			if got != tt.want {
				t.Errorf("ParseScoreType(%q) = %d, want %d", tt.input, got, tt.want)
			}
		})
	}
}

func TestDefaultSearchOptions(t *testing.T) {
	opts := DefaultSearchOptions()
	if opts.Distance != 64 {
		t.Errorf("Distance = %d, want 64", opts.Distance)
	}
	if opts.Count != 10 {
		t.Errorf("Count = %d, want 10", opts.Count)
	}
	if opts.K != 3 {
		t.Errorf("K = %d, want 3", opts.K)
	}
	if opts.NProbe != 3 {
		t.Errorf("NProbe = %d, want 3", opts.NProbe)
	}
	if opts.ScoreType != ScoreWilson {
		t.Errorf("ScoreType = %d, want %d", opts.ScoreType, ScoreWilson)
	}
	if opts.Threads <= 0 {
		t.Errorf("Threads = %d, want > 0", opts.Threads)
	}
}

func TestDefaultDataDir(t *testing.T) {
	dir := DefaultDataDir()
	if dir == "" {
		t.Error("DefaultDataDir() returned empty string")
	}
	if dir != "data" {
		t.Errorf("DefaultDataDir() = %q, want %q", dir, "data")
	}
}

func TestDefaultConfigFile(t *testing.T) {
	f := DefaultConfigFile()
	if f != "imseek.toml" {
		t.Errorf("DefaultConfigFile() = %q, want %q", f, "imseek.toml")
	}
}
