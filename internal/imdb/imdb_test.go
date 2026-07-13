package imdb

import (
	"context"
	"math/rand"
	"testing"

	"imseek/internal/config"
	"imseek/internal/index"
)

// TestFullCycle exercises add -> train -> build -> search at the library level
// using synthetic 8-byte descriptors (no feature-extraction model required).
func TestFullCycle(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	const cs = 8

	m, err := Open(ctx, Options{ConfDir: dir, WAL: false, CodeSize: cs})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()

	rng := rand.New(rand.NewSource(1))

	// Create some "prototype" descriptors; each image is a cluster of noisy
	// copies of one prototype so that searching a prototype finds its image.
	const nImages = 40
	prototypes := make([][]byte, nImages)
	for i := range prototypes {
		prototypes[i] = make([]byte, cs)
		rng.Read(prototypes[i])
	}

	noisyCopy := func(p []byte, flips int) []byte {
		v := make([]byte, cs)
		copy(v, p)
		for range flips {
			bit := rng.Intn(cs * 8)
			v[bit/8] ^= 1 << (bit % 8)
		}
		return v
	}

	for i := range nImages {
		descs := make([][]byte, 100)
		for j := range descs {
			descs[j] = noisyCopy(prototypes[i], 1)
		}
		hash := make([]byte, 32)
		rng.Read(hash)
		if _, err := m.AddImage(ctx, hash, "/img/"+string(rune('A'+i))+".jpg", descs); err != nil {
			t.Fatal(err)
		}
	}

	imgCount, vecCount, _ := m.Count(ctx)
	if imgCount != nImages || vecCount != int64(nImages*100) {
		t.Fatalf("count = %d,%d", imgCount, vecCount)
	}

	// Train a quantizer over exported vectors via the backend abstraction.
	export, err := m.ExportVectors(ctx, nImages*100)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := m.TrainIndex(ctx, export, index.TrainOptions{
		NList:   32,
		MaxIter: 20,
		Init:    "kmeans-plus-plus",
	}); err != nil {
		t.Fatal(err)
	}

	// Build the index.
	if err := m.BuildIndex(ctx, BuildOptions{BatchSize: 1000}); err != nil {
		t.Fatal(err)
	}

	// Reopen with cache for search.
	m2, err := Open(ctx, Options{ConfDir: dir, WAL: false, Cache: true, CodeSize: cs})
	if err != nil {
		t.Fatal(err)
	}
	defer m2.Close()

	idx, closeIdx, err := m2.OpenIndex(ctx, 4)
	if err != nil {
		t.Fatal(err)
	}
	defer closeIdx()

	// Query with noisy copies of a known prototype; its image should rank #1.
	target := 7
	query := make([][]byte, 50)
	for i := range query {
		query[i] = noisyCopy(prototypes[target], 1)
	}
	opts := config.DefaultSearchOptions()
	opts.NProbe = 8
	opts.Distance = 8
	opts.Count = 5
	results, err := m2.Search(ctx, idx, query, opts)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 {
		t.Fatal("no results")
	}
	wantPath := "/img/" + string(rune('A'+target)) + ".jpg"
	if results[0].Path != wantPath {
		t.Fatalf("top result = %q (score %f) want %q; all=%+v",
			results[0].Path, results[0].Score, wantPath, results)
	}
	t.Logf("top: %s score=%.2f", results[0].Path, results[0].Score)
}
