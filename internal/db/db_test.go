package db

import (
	"context"
	"testing"
)

func openTest(t *testing.T) *DB {
	t.Helper()
	d, err := Open(t.TempDir(), false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { d.Close() })
	return d
}

func TestAddImageAndVectors(t *testing.T) {
	ctx := context.Background()
	d := openTest(t)

	tx, err := d.Begin(ctx)
	if err != nil {
		t.Fatal(err)
	}
	hash := []byte{1, 2, 3, 4, 5, 6, 7, 8}
	id, err := d.AddImage(ctx, tx, hash, "/img/a.jpg")
	if err != nil {
		t.Fatal(err)
	}
	if id != 1 {
		t.Fatalf("first id = %d want 1", id)
	}
	if err := d.AddVector(ctx, tx, id, make([]byte, 8*10)); err != nil {
		t.Fatal(err)
	}
	if err := d.AddVectorStats(ctx, tx, id, 10); err != nil {
		t.Fatal(err)
	}
	if err := tx.Commit(); err != nil {
		t.Fatal(err)
	}

	// exact hash lookup
	got, ok, err := d.CheckImageHash(ctx, hash)
	if err != nil || !ok || got != id {
		t.Fatalf("CheckImageHash = %d,%v,%v", got, ok, err)
	}
	// missing hash
	_, ok, _ = d.CheckImageHash(ctx, []byte{9, 9})
	if ok {
		t.Fatal("expected miss")
	}

	imgCount, vecCount, err := d.GetCount(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if imgCount != 1 || vecCount != 10 {
		t.Fatalf("count = %d,%d want 1,10", imgCount, vecCount)
	}
}

func TestTotalVectorCountRunning(t *testing.T) {
	ctx := context.Background()
	d := openTest(t)

	counts := []int64{5, 3, 7}
	for i, c := range counts {
		tx, _ := d.Begin(ctx)
		hash := []byte{byte(i + 1), 0, 0, 0}
		id, err := d.AddImage(ctx, tx, hash, "p")
		if err != nil {
			t.Fatal(err)
		}
		d.AddVector(ctx, tx, id, make([]byte, int(c)*8))
		if err := d.AddVectorStats(ctx, tx, id, c); err != nil {
			t.Fatal(err)
		}
		tx.Commit()
	}
	totals, err := d.GetAllTotalVectorCount(ctx)
	if err != nil {
		t.Fatal(err)
	}
	want := []int64{5, 8, 15}
	if len(totals) != 3 {
		t.Fatalf("totals=%v", totals)
	}
	for i := range want {
		if totals[i] != want[i] {
			t.Fatalf("totals=%v want %v", totals, want)
		}
	}

	// GetImageIDByVectorID: vector id 6 falls in image 2 (total 8 >= 6, image1 total 5 < 6)
	imgID, err := d.GetImageIDByVectorID(ctx, 6)
	if err != nil {
		t.Fatal(err)
	}
	if imgID != 2 {
		t.Fatalf("vector 6 -> image %d want 2", imgID)
	}
	// vector id 5 -> image 1 (total 5 >= 5)
	imgID, _ = d.GetImageIDByVectorID(ctx, 5)
	if imgID != 1 {
		t.Fatalf("vector 5 -> image %d want 1", imgID)
	}
}

func TestAppendImagePath(t *testing.T) {
	ctx := context.Background()
	d := openTest(t)
	tx, _ := d.Begin(ctx)
	id, _ := d.AddImage(ctx, tx, []byte{1, 2, 3, 4}, "/a.jpg")
	tx.Commit()

	added, err := d.AppendImagePath(ctx, id, "/b.jpg")
	if err != nil || !added {
		t.Fatalf("append = %v,%v", added, err)
	}
	// duplicate append returns false
	added, _ = d.AppendImagePath(ctx, id, "/a.jpg")
	if added {
		t.Fatal("expected false for duplicate path")
	}
	path, _ := d.GetImagePath(ctx, id)
	if path != "/a.jpg"+PathSeparator+"/b.jpg" {
		t.Fatalf("path = %q", path)
	}
}

func TestIndexedBatchAndUnindexed(t *testing.T) {
	ctx := context.Background()
	d := openTest(t)
	for i := 0; i < 3; i++ {
		tx, _ := d.Begin(ctx)
		id, _ := d.AddImage(ctx, tx, []byte{byte(i + 1), 0, 0, 0}, "p")
		d.AddVector(ctx, tx, id, make([]byte, 8))
		d.AddVectorStats(ctx, tx, id, 1)
		tx.Commit()
	}
	n, _ := d.CountImageUnindexed(ctx)
	if n != 3 {
		t.Fatalf("unindexed=%d want 3", n)
	}
	rows, err := d.GetVectorsUnindexed(ctx, 100, 0)
	if err != nil {
		t.Fatal(err)
	}
	if len(rows) != 3 {
		t.Fatalf("got %d unindexed rows", len(rows))
	}
	if err := d.SetIndexedBatch(ctx, []int64{1, 2}); err != nil {
		t.Fatal(err)
	}
	n, _ = d.CountImageUnindexed(ctx)
	if n != 1 {
		t.Fatalf("unindexed after batch=%d want 1", n)
	}
}
