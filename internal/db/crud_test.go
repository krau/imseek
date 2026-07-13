package db

import (
	"context"
	"database/sql"
	"errors"
	"testing"
)

func TestDeleteImage(t *testing.T) {
	ctx := context.Background()
	d := openTest(t)

	// Add 3 images
	for i := range 3 {
		tx, _ := d.Begin(ctx)
		id, _ := d.AddImage(ctx, tx, []byte{byte(i + 1), 0, 0, 0}, "/img")
		d.AddVector(ctx, tx, id, make([]byte, 32))
		d.AddVectorStats(ctx, tx, id, 1)
		tx.Commit()
	}

	// Delete image 2
	if _, err := d.DeleteImage(ctx, 2); err != nil {
		t.Fatalf("DeleteImage(2): %v", err)
	}

	// Image 2 should be gone
	var path string
	err := d.sql.QueryRowContext(ctx, "SELECT path FROM image WHERE id = 2").Scan(&path)
	if !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("expected ErrNoRows, got %v", err)
	}

	// Vector and stats for image 2 should be gone
	var count int
	d.sql.QueryRowContext(ctx, "SELECT COUNT(*) FROM vector WHERE id = 2").Scan(&count)
	if count != 0 {
		t.Fatalf("vector count for deleted image = %d want 0", count)
	}
	d.sql.QueryRowContext(ctx, "SELECT COUNT(*) FROM vector_stats WHERE id = 2").Scan(&count)
	if count != 0 {
		t.Fatalf("vector_stats count for deleted image = %d want 0", count)
	}

	// Images 1 and 3 should still exist
	images, _ := d.ListImages(ctx, 100, 0)
	if len(images) != 2 {
		t.Fatalf("image count after delete = %d want 2", len(images))
	}

	// Delete non-existent image returns ErrNoRows
	_, err = d.DeleteImage(ctx, 999)
	if !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("DeleteImage(999): expected ErrNoRows, got %v", err)
	}

	// Orphan vector id that belonged to deleted image 2 must not remap to image 3
	_, err = d.GetImageIDByVectorID(ctx, 2)
	if !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("orphan vid 2: expected ErrNoRows, got id/err %v", err)
	}
	// Remaining images still map correctly
	imgID, err := d.GetImageIDByVectorID(ctx, 1)
	if err != nil || imgID != 1 {
		t.Fatalf("vid 1 -> %d err=%v want 1", imgID, err)
	}
	imgID, err = d.GetImageIDByVectorID(ctx, 3)
	if err != nil || imgID != 3 {
		t.Fatalf("vid 3 -> %d err=%v want 3", imgID, err)
	}
}

func TestListImages(t *testing.T) {
	ctx := context.Background()
	d := openTest(t)

	// Add 5 images
	for i := range 5 {
		tx, _ := d.Begin(ctx)
		id, _ := d.AddImage(ctx, tx, []byte{byte(i + 1), 0, 0, 0}, "/img/"+string(rune('a'+i))+".jpg")
		d.AddVector(ctx, tx, id, make([]byte, 32))
		d.AddVectorStats(ctx, tx, id, int64(i+1))
		tx.Commit()
	}

	tests := []struct {
		name       string
		limit      int64
		offset     int64
		wantLen    int
		wantFirst  int64
		wantVecCnt int64
	}{
		{"first page", 3, 0, 3, 1, 1},
		{"second page", 3, 3, 2, 4, 4},
		{"beyond range", 10, 10, 0, 0, 0},
		{"limit clamped to min", 0, 0, 5, 1, 1}, // limit 0 -> default 20
		{"limit clamped to max", 200, 0, 5, 1, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			images, err := d.ListImages(ctx, tt.limit, tt.offset)
			if err != nil {
				t.Fatal(err)
			}
			if len(images) != tt.wantLen {
				t.Fatalf("len = %d want %d", len(images), tt.wantLen)
			}
			if tt.wantLen > 0 {
				if images[0].ID != tt.wantFirst {
					t.Errorf("first ID = %d want %d", images[0].ID, tt.wantFirst)
				}
				if images[0].VectorCount != tt.wantVecCnt {
					t.Errorf("vec count = %d want %d", images[0].VectorCount, tt.wantVecCnt)
				}
				if images[0].Hash == "" {
					t.Error("hash should not be empty")
				}
				if images[0].Indexed {
					t.Error("should be indexed=false by default")
				}
			}
		})
	}
}

func TestListImages_IndexedFlag(t *testing.T) {
	ctx := context.Background()
	d := openTest(t)

	// Add 2 images, mark first as indexed
	for i := range 2 {
		tx, _ := d.Begin(ctx)
		id, _ := d.AddImage(ctx, tx, []byte{byte(i + 1), 0, 0, 0}, "/img")
		d.AddVector(ctx, tx, id, make([]byte, 32))
		d.AddVectorStats(ctx, tx, id, 1)
		tx.Commit()
	}
	d.SetIndexedBatch(ctx, []int64{1})

	images, _ := d.ListImages(ctx, 10, 0)
	if len(images) != 2 {
		t.Fatalf("len = %d want 2", len(images))
	}
	if !images[0].Indexed {
		t.Error("image 1 should be indexed")
	}
	if images[1].Indexed {
		t.Error("image 2 should not be indexed")
	}
}

func TestGetImage(t *testing.T) {
	ctx := context.Background()
	d := openTest(t)

	tx, _ := d.Begin(ctx)
	id, _ := d.AddImage(ctx, tx, []byte{0xAB, 0xCD}, "/test.jpg")
	d.AddVector(ctx, tx, id, make([]byte, 32))
	d.AddVectorStats(ctx, tx, id, 5)
	tx.Commit()

	// Existing image
	img, err := d.GetImage(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if img.ID != id {
		t.Errorf("ID = %d want %d", img.ID, id)
	}
	if img.Path != "/test.jpg" {
		t.Errorf("Path = %q want /test.jpg", img.Path)
	}
	if img.Hash != "ABCD" {
		t.Errorf("Hash = %q want ABCD", img.Hash)
	}
	if img.VectorCount != 5 {
		t.Errorf("VectorCount = %d want 5", img.VectorCount)
	}

	// Non-existent image
	_, err = d.GetImage(ctx, 999)
	if !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("expected ErrNoRows, got %v", err)
	}
}

func TestGetImage_NoStats(t *testing.T) {
	ctx := context.Background()
	d := openTest(t)

	// Add image without vector_stats (LEFT JOIN should handle it)
	tx, _ := d.Begin(ctx)
	id, _ := d.AddImage(ctx, tx, []byte{1, 2}, "/no_stats.jpg")
	tx.Commit()

	img, err := d.GetImage(ctx, id)
	if err != nil {
		t.Fatal(err)
	}
	if img.VectorCount != 0 {
		t.Errorf("VectorCount = %d want 0 (no stats)", img.VectorCount)
	}
	if img.Indexed {
		t.Error("should be indexed=false when no stats exist")
	}
}
