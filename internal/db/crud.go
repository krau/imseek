package db

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strings"
)

func (d *DB) AddImage(ctx context.Context, tx *sql.Tx, hash []byte, path string) (int64, error) {
	var id int64
	err := tx.QueryRowContext(ctx,
		`INSERT INTO image (hash, path) VALUES (?, ?) RETURNING id`, hash, path).Scan(&id)
	return id, err
}

// CheckImageHash returns the id of the image with the exact hash, or (0, false).
func (d *DB) CheckImageHash(ctx context.Context, hash []byte) (int64, bool, error) {
	var id int64
	err := d.sql.QueryRowContext(ctx, `SELECT id FROM image WHERE hash = ?`, hash).Scan(&id)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, false, nil
	}
	if err != nil {
		return 0, false, err
	}
	return id, true, nil
}

// GetImagePath returns the (separator-joined) path of an image.
func (d *DB) GetImagePath(ctx context.Context, id int64) (string, error) {
	var path string
	err := d.sql.QueryRowContext(ctx, `SELECT path FROM image WHERE id = ?`, id).Scan(&path)
	return path, err
}

// UpdateImagePath replaces the path of an image.
func (d *DB) UpdateImagePath(ctx context.Context, id int64, path string) error {
	_, err := d.sql.ExecContext(ctx, `UPDATE image SET path = ? WHERE id = ?`, path, id)
	return err
}

// AppendImagePath appends a path to an image record unless already present.
// It returns false if the path was already present.
func (d *DB) AppendImagePath(ctx context.Context, id int64, path string) (bool, error) {
	old, err := d.GetImagePath(ctx, id)
	if err != nil {
		return false, err
	}
	for _, p := range strings.Split(old, PathSeparator) {
		if p == path {
			return false, nil
		}
	}
	if err := d.UpdateImagePath(ctx, id, old+PathSeparator+path); err != nil {
		return false, err
	}
	return true, nil
}

func (d *DB) AddVector(ctx context.Context, tx *sql.Tx, id int64, vector []byte) error {
	_, err := tx.ExecContext(ctx, `INSERT INTO vector (id, vector) VALUES (?, ?)`, id, vector)
	return err
}

func (d *DB) AddVectorStats(ctx context.Context, tx *sql.Tx, id int64, vectorCount int64) error {
	_, err := tx.ExecContext(ctx,
		`INSERT INTO vector_stats (id, vector_count, total_vector_count)
		 VALUES (?, ?, COALESCE((SELECT MAX(total_vector_count) FROM vector_stats), 0) + ?)`,
		id, vectorCount, vectorCount)
	return err
}

// GetImageIDByVectorID maps a global vector id to its owning image id using
// the half-open range (total-count, total] owned by each image. Orphan ids
// (e.g. from a deleted image whose index vectors were not purged) return
// sql.ErrNoRows rather than being remapped to a neighbor image.
func (d *DB) GetImageIDByVectorID(ctx context.Context, vectorID int64) (int64, error) {
	var id int64
	err := d.sql.QueryRowContext(ctx,
		`SELECT id FROM vector_stats
		 WHERE total_vector_count - vector_count < ? AND total_vector_count >= ?
		 LIMIT 1`,
		vectorID, vectorID).Scan(&id)
	return id, err
}

// VectorIDRange is the inclusive global vector-id span owned by one image.
type VectorIDRange struct {
	Lo int64 // first vector id (inclusive)
	Hi int64 // last vector id (inclusive); Hi-Lo+1 == vector_count
}

func (d *DB) GetVectorIDRange(ctx context.Context, imageID int64) (VectorIDRange, error) {
	var count, total int64
	err := d.sql.QueryRowContext(ctx,
		`SELECT vector_count, total_vector_count FROM vector_stats WHERE id = ?`,
		imageID).Scan(&count, &total)
	if err != nil {
		return VectorIDRange{}, err
	}
	return VectorIDRange{Lo: total - count + 1, Hi: total}, nil
}

func (d *DB) CountImageUnindexed(ctx context.Context) (int64, error) {
	var n int64
	err := d.sql.QueryRowContext(ctx, `SELECT COUNT(*) FROM vector_stats WHERE indexed = 0`).Scan(&n)
	return n, err
}

// VectorRow is a descriptor blob with its running total vector count.
type VectorRow struct {
	ID               int64
	Vector           []byte
	TotalVectorCount int64
}

func (d *DB) GetVectorsUnindexed(ctx context.Context, limit, offset int64) ([]VectorRow, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT v.id, v.vector, s.total_vector_count
		 FROM vector v JOIN vector_stats s ON v.id = s.id
		 WHERE s.indexed = 0 ORDER BY v.id ASC LIMIT ? OFFSET ?`, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanVectorRows(rows)
}

func (d *DB) GetVectors(ctx context.Context, limit, offset int64) ([]VectorRow, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT v.id, v.vector, s.total_vector_count
		 FROM vector v JOIN vector_stats s ON v.id = s.id
		 ORDER BY v.id ASC LIMIT ? OFFSET ?`, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	return scanVectorRows(rows)
}

func scanVectorRows(rows *sql.Rows) ([]VectorRow, error) {
	var out []VectorRow
	for rows.Next() {
		var r VectorRow
		if err := rows.Scan(&r.ID, &r.Vector, &r.TotalVectorCount); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (d *DB) SetIndexedBatch(ctx context.Context, ids []int64) error {
	tx, err := d.sql.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	stmt, err := tx.PrepareContext(ctx, `UPDATE vector_stats SET indexed = 1 WHERE id = ?`)
	if err != nil {
		tx.Rollback()
		return err
	}
	defer stmt.Close()
	for _, id := range ids {
		if _, err := stmt.ExecContext(ctx, id); err != nil {
			tx.Rollback()
			return err
		}
	}
	return tx.Commit()
}

// DeleteVectors removes descriptor blobs of images already indexed and vacuums.
func (d *DB) DeleteVectors(ctx context.Context) error {
	if _, err := d.sql.ExecContext(ctx,
		`DELETE FROM vector WHERE id IN (SELECT id FROM vector_stats WHERE indexed = 1)`); err != nil {
		return err
	}
	_, err := d.sql.ExecContext(ctx, `VACUUM`)
	return err
}

// DeleteVectorsAll removes all descriptor blobs and vacuums.
func (d *DB) DeleteVectorsAll(ctx context.Context) error {
	if _, err := d.sql.ExecContext(ctx, `DELETE FROM vector`); err != nil {
		return err
	}
	_, err := d.sql.ExecContext(ctx, `VACUUM`)
	return err
}

func (d *DB) GetCount(ctx context.Context) (int64, int64, error) {
	var images int64
	if err := d.sql.QueryRowContext(ctx, `SELECT COUNT(*) FROM image`).Scan(&images); err != nil {
		return 0, 0, err
	}
	var total sql.NullInt64
	if err := d.sql.QueryRowContext(ctx,
		`SELECT MAX(total_vector_count) FROM vector_stats`).Scan(&total); err != nil {
		return 0, 0, err
	}
	if !total.Valid {
		return images, 0, nil
	}
	return images, total.Int64, nil
}

// VectorBound is one image's cumulative vector-id bound for cache lookup.
type VectorBound struct {
	ImageID int64
	Total   int64 // inclusive upper bound of this image's vector ids
	Count   int64 // number of descriptors; range is (Total-Count, Total]
}

func (d *DB) GetAllVectorBounds(ctx context.Context) ([]VectorBound, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT id, total_vector_count, vector_count FROM vector_stats ORDER BY total_vector_count ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []VectorBound
	for rows.Next() {
		var b VectorBound
		if err := rows.Scan(&b.ImageID, &b.Total, &b.Count); err != nil {
			return nil, err
		}
		out = append(out, b)
	}
	return out, rows.Err()
}

func (d *DB) GetAllTotalVectorCount(ctx context.Context) ([]int64, error) {
	rows, err := d.sql.QueryContext(ctx,
		`SELECT total_vector_count FROM vector_stats ORDER BY id ASC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []int64
	for rows.Next() {
		var t int64
		if err := rows.Scan(&t); err != nil {
			return nil, err
		}
		out = append(out, t)
	}
	return out, rows.Err()
}

// GuessHashLen returns the byte length of a stored hash, or 0 if empty.
func (d *DB) GuessHashLen(ctx context.Context) (int, error) {
	var hash []byte
	err := d.sql.QueryRowContext(ctx, `SELECT hash FROM image LIMIT 1`).Scan(&hash)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	return len(hash), nil
}

// GuessCodeSize infers the descriptor byte length from a stored vector blob:
// blob length / vector_count. Returns 0 if there are no vectors yet.
func (d *DB) GuessCodeSize(ctx context.Context) (int, error) {
	var blobLen, vecCount int64
	err := d.sql.QueryRowContext(ctx,
		`SELECT length(v.vector), s.vector_count
		 FROM vector v JOIN vector_stats s ON v.id = s.id
		 WHERE s.vector_count > 0 LIMIT 1`).Scan(&blobLen, &vecCount)
	if errors.Is(err, sql.ErrNoRows) {
		return 0, nil
	}
	if err != nil {
		return 0, err
	}
	if vecCount == 0 {
		return 0, nil
	}
	return int(blobLen / vecCount), nil
}

func (d *DB) Begin(ctx context.Context) (*sql.Tx, error) {
	return d.sql.BeginTx(ctx, nil)
}

// DeleteImage removes an image and its associated vectors and stats.
// It returns the global vector-id range that belonged to the image (empty if
// the image had no stats) so callers can purge the vector index.
func (d *DB) DeleteImage(ctx context.Context, id int64) (VectorIDRange, error) {
	tx, err := d.sql.BeginTx(ctx, nil)
	if err != nil {
		return VectorIDRange{}, fmt.Errorf("delete image %d: begin tx: %w", id, err)
	}
	defer tx.Rollback()

	var rng VectorIDRange
	var count, total int64
	err = tx.QueryRowContext(ctx,
		`SELECT vector_count, total_vector_count FROM vector_stats WHERE id = ?`, id).
		Scan(&count, &total)
	if err == nil {
		rng = VectorIDRange{Lo: total - count + 1, Hi: total}
	} else if !errors.Is(err, sql.ErrNoRows) {
		return VectorIDRange{}, fmt.Errorf("delete image %d: read stats: %w", id, err)
	}

	if _, err := tx.ExecContext(ctx, `DELETE FROM vector WHERE id = ?`, id); err != nil {
		return VectorIDRange{}, fmt.Errorf("delete image %d: delete vector: %w", id, err)
	}
	if _, err := tx.ExecContext(ctx, `DELETE FROM vector_stats WHERE id = ?`, id); err != nil {
		return VectorIDRange{}, fmt.Errorf("delete image %d: delete stats: %w", id, err)
	}
	res, err := tx.ExecContext(ctx, `DELETE FROM image WHERE id = ?`, id)
	if err != nil {
		return VectorIDRange{}, fmt.Errorf("delete image %d: delete image: %w", id, err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return VectorIDRange{}, fmt.Errorf("delete image %d: rows affected: %w", id, err)
	}
	if n == 0 {
		return VectorIDRange{}, sql.ErrNoRows
	}
	if err := tx.Commit(); err != nil {
		return VectorIDRange{}, err
	}
	return rng, nil
}

type ImageInfo struct {
	ID          int64  `json:"id"`
	Path        string `json:"path"`
	Hash        string `json:"hash"`
	VectorCount int64  `json:"vector_count"`
	Indexed     bool   `json:"indexed"`
}

func (d *DB) ListImages(ctx context.Context, limit, offset int64) ([]ImageInfo, error) {
	if limit <= 0 || limit > 100 {
		limit = 20
	}
	rows, err := d.sql.QueryContext(ctx,
		`SELECT i.id, i.path, hex(i.hash), COALESCE(s.vector_count, 0), COALESCE(s.indexed, 0)
		 FROM image i LEFT JOIN vector_stats s ON i.id = s.id
		 ORDER BY i.id ASC LIMIT ? OFFSET ?`, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []ImageInfo
	for rows.Next() {
		var img ImageInfo
		var indexed int
		if err := rows.Scan(&img.ID, &img.Path, &img.Hash, &img.VectorCount, &indexed); err != nil {
			return nil, err
		}
		img.Indexed = indexed == 1
		out = append(out, img)
	}
	return out, rows.Err()
}

func (d *DB) GetImage(ctx context.Context, id int64) (*ImageInfo, error) {
	var img ImageInfo
	var indexed int
	err := d.sql.QueryRowContext(ctx,
		`SELECT i.id, i.path, hex(i.hash), COALESCE(s.vector_count, 0), COALESCE(s.indexed, 0)
		 FROM image i LEFT JOIN vector_stats s ON i.id = s.id
		 WHERE i.id = ?`, id).Scan(&img.ID, &img.Path, &img.Hash, &img.VectorCount, &indexed)
	if err != nil {
		return nil, err
	}
	img.Indexed = indexed == 1
	return &img, nil
}
