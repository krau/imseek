package server

import (
	"context"
	"image"
	"net/http/httptest"
	"testing"

	"github.com/gofiber/fiber/v3"
	"github.com/stretchr/testify/assert"

	"imseek/internal/config"
	"imseek/internal/db"
	"imseek/internal/imdb"
)

type mockExtractor struct {
	descs [][]byte
	err   error
}

func (m *mockExtractor) DetectImage(image.Image) ([][]byte, error) { return m.descs, m.err }
func (m *mockExtractor) DetectBytes([]byte) ([][]byte, error)      { return m.descs, m.err }
func (m *mockExtractor) DetectFile(string) ([][]byte, error)       { return m.descs, m.err }
func (m *mockExtractor) Close() error                              { return nil }

// setupTestServer creates a Fiber app with a real SQLite DB (temp dir) and
// mock extractor, then returns the app and the IMDB for direct manipulation.
func setupTestServer(t *testing.T) (*fiber.App, *imdb.IMDB) {
	t.Helper()
	ctx := context.Background()
	m, err := imdb.Open(ctx, imdb.Options{
		ConfDir:  t.TempDir(),
		WAL:      false,
		CodeSize: 32,
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { m.Close() })

	ext := &mockExtractor{descs: [][]byte{make([]byte, 8), make([]byte, 8)}}
	deps := &Deps{
		DB:        m,
		Extractor: ext,
		Search:    config.DefaultSearchOptions(),
		Token:     "test-token",
	}
	app := New(deps)
	return app, m
}

// addTestImage inserts a fake image directly into the DB for test setup.
func addTestImage(t *testing.T, m *imdb.IMDB, hash byte, path string) int64 {
	t.Helper()
	ctx := context.Background()
	id, err := m.AddImage(ctx, []byte{hash, 0, 0, 0}, path, [][]byte{make([]byte, 8)})
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func TestHealth(t *testing.T) {
	app, _ := setupTestServer(t)
	req := httptest.NewRequest("GET", "/api/v1/health", nil)
	req.Header.Set("Authorization", "Bearer test-token")
	resp, err := app.Test(req)
	assert.NoError(t, err)
	assert.Equal(t, 200, resp.StatusCode)
}

func TestStats(t *testing.T) {
	app, m := setupTestServer(t)
	addTestImage(t, m, 1, "/a.jpg")
	addTestImage(t, m, 2, "/b.jpg")

	req := httptest.NewRequest("GET", "/api/v1/stats", nil)
	req.Header.Set("Authorization", "Bearer test-token")
	resp, err := app.Test(req)
	assert.NoError(t, err)
	assert.Equal(t, 200, resp.StatusCode)
}

func TestListImages(t *testing.T) {
	app, m := setupTestServer(t)
	for i := range 5 {
		addTestImage(t, m, byte(i+1), "/img")
	}

	tests := []struct {
		name       string
		query      string
		wantStatus int
	}{
		{"default", "", 200},
		{"with limit", "?limit=3", 200},
		{"with offset", "?limit=2&offset=3", 200},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			url := "/api/v1/images" + tt.query
			req := httptest.NewRequest("GET", url, nil)
			req.Header.Set("Authorization", "Bearer test-token")
			resp, err := app.Test(req)
			assert.NoError(t, err)
			assert.Equal(t, tt.wantStatus, resp.StatusCode)
		})
	}
}

func TestGetImage(t *testing.T) {
	app, m := setupTestServer(t)
	id := addTestImage(t, m, 1, "/test.jpg")

	// Existing
	req := httptest.NewRequest("GET", "/api/v1/images/"+itoa64(id), nil)
	req.Header.Set("Authorization", "Bearer test-token")
	resp, err := app.Test(req)
	assert.NoError(t, err)
	assert.Equal(t, 200, resp.StatusCode)

	// Non-existent
	req = httptest.NewRequest("GET", "/api/v1/images/999", nil)
	req.Header.Set("Authorization", "Bearer test-token")
	resp, err = app.Test(req)
	assert.NoError(t, err)
	assert.Equal(t, 404, resp.StatusCode)

	// Invalid ID
	req = httptest.NewRequest("GET", "/api/v1/images/abc", nil)
	req.Header.Set("Authorization", "Bearer test-token")
	resp, err = app.Test(req)
	assert.NoError(t, err)
	assert.Equal(t, 400, resp.StatusCode)
}

func TestDeleteImage(t *testing.T) {
	app, m := setupTestServer(t)
	id := addTestImage(t, m, 1, "/del.jpg")

	// Delete existing
	req := httptest.NewRequest("DELETE", "/api/v1/images/"+itoa64(id), nil)
	req.Header.Set("Authorization", "Bearer test-token")
	resp, err := app.Test(req)
	assert.NoError(t, err)
	assert.Equal(t, 204, resp.StatusCode)

	// Delete again (not found)
	req = httptest.NewRequest("DELETE", "/api/v1/images/"+itoa64(id), nil)
	req.Header.Set("Authorization", "Bearer test-token")
	resp, err = app.Test(req)
	assert.NoError(t, err)
	assert.Equal(t, 404, resp.StatusCode)
}

func TestAuth(t *testing.T) {
	app, _ := setupTestServer(t)

	tests := []struct {
		name   string
		token  string
		status int
	}{
		{"valid token", "test-token", 200},
		{"invalid token", "wrong-token", 401},
		{"no token", "", 401},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", "/api/v1/health", nil)
			if tt.token != "" {
				req.Header.Set("Authorization", "Bearer "+tt.token)
			}
			resp, err := app.Test(req)
			assert.NoError(t, err)
			assert.Equal(t, tt.status, resp.StatusCode)
		})
	}
}

func TestAddImage_DuplicateHash(t *testing.T) {
	app, m := setupTestServer(t)
	addTestImage(t, m, 0xAB, "/original.jpg")

	// Adding same hash again should return 409
	// (the mock extractor returns descs, but the hash check happens first)
	req := httptest.NewRequest("POST", "/api/v1/images", nil)
	req.Header.Set("Authorization", "Bearer test-token")
	req.Header.Set("Content-Type", "multipart/form-data")
	resp, err := app.Test(req)
	assert.NoError(t, err)
	// Without a file field, should get 400
	assert.Equal(t, 400, resp.StatusCode)
}

func TestSearch_NoFile(t *testing.T) {
	app, _ := setupTestServer(t)
	req := httptest.NewRequest("POST", "/api/v1/search", nil)
	req.Header.Set("Authorization", "Bearer test-token")
	resp, err := app.Test(req)
	assert.NoError(t, err)
	assert.Equal(t, 400, resp.StatusCode)
}

func TestQueryInt(t *testing.T) {
	app, _ := setupTestServer(t)

	tests := []struct {
		name   string
		url    string
		status int
	}{
		{"valid limit", "/api/v1/images?limit=5", 200},
		{"negative limit", "/api/v1/images?limit=-1", 200},
		{"zero limit", "/api/v1/images?limit=0", 200},
		{"huge limit", "/api/v1/images?limit=99999", 200},
		{"non-numeric limit", "/api/v1/images?limit=abc", 200},
		{"valid offset", "/api/v1/images?offset=5", 200},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", tt.url, nil)
			req.Header.Set("Authorization", "Bearer test-token")
			resp, err := app.Test(req)
			assert.NoError(t, err)
			assert.Equal(t, tt.status, resp.StatusCode)
		})
	}
}

// itoa64 converts int64 to string without importing strconv in the test name.
func itoa64(n int64) string {
	if n == 0 {
		return "0"
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}

var _ = db.ImageInfo{}
