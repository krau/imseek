package server

import (
	"context"
	"database/sql"
	"errors"
	"image"
	"log"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/cors"

	_ "imseek/docs"
	"imseek/internal/config"
	"imseek/internal/imdb"
	"imseek/internal/index"
)

type Extractor interface {
	DetectImage(image.Image) ([][]byte, error)
	DetectBytes([]byte) ([][]byte, error)
	DetectFile(string) ([][]byte, error)
	Close() error
}

type Deps struct {
	DB        *imdb.IMDB
	Index     index.Searcher
	Extractor Extractor
	Search    config.SearchOptions
	Token     string
	Threads   int
}

type APIError struct {
	Error string `json:"error" example:"unauthorized"`
}

type StatsResponse struct {
	Images    int64 `json:"images" example:"499"`
	Vectors   int64 `json:"vectors" example:"253079"`
	Unindexed int64 `json:"unindexed" example:"0"`
}

type ListImagesResponse struct {
	Images []ImageInfo `json:"images"`
	Limit  int         `json:"limit" example:"20"`
	Offset int         `json:"offset" example:"0"`
}

type ImageInfo struct {
	ID          int64  `json:"id" example:"1"`
	Path        string `json:"path" example:"/data/img001.webp"`
	Hash        string `json:"hash" example:"ABCD1234EFGH5678"`
	VectorCount int64  `json:"vector_count" example:"508"`
	Indexed     bool   `json:"indexed"`
}

type AddImageResponse struct {
	ID          int64  `json:"id" example:"1"`
	Path        string `json:"path" example:"img001.webp"`
	Descriptors int    `json:"descriptors" example:"508"`
}

type SearchResult struct {
	Path  string  `json:"path" example:"img001.webp"`
	Score float64 `json:"score" example:"94.83"`
}

type SearchResponse struct {
	Time   int64          `json:"time" example:"45"`
	Result []SearchResult `json:"result"`
}

type BuildIndexResponse struct {
	Status string `json:"status" example:"building"`
}

func New(deps *Deps) *fiber.App {
	app := fiber.New(fiber.Config{
		BodyLimit: 50 * 1024 * 1024,
	})

	h := newHandler(deps)

	app.Hooks().OnPreShutdown(func() error {
		h.Shutdown()
		return nil
	})

	app.Use(recoverMiddleware())
	app.Use(cors.New(cors.Config{
		AllowOrigins:     []string{"*"},
		AllowMethods:     []string{"GET", "POST", "HEAD", "PUT", "DELETE", "PATCH"},
		AllowHeaders:     []string{"Origin", "Content-Type", "Accept", "Authorization"},
		ExposeHeaders:    []string{"Content-Length"},
		AllowCredentials: false,
		MaxAge:           300,
	}))
	app.Use(requestLogger())

	app.Get("/swagger", func(c fiber.Ctx) error {
		return c.Redirect().Status(301).To("/swagger/index.html")
	})
	app.Get("/swagger/*", swaggerHandler)

	app.Use(authMiddleware(deps.Token))

	api := app.Group("/api/v1")
	api.Get("/health", h.health)
	api.Get("/stats", h.stats)
	api.Get("/images", h.listImages)
	api.Get("/images/:id", h.getImage)
	api.Delete("/images/:id", h.deleteImage)
	api.Post("/images", h.addImage)
	api.Post("/search", h.search)
	api.Post("/index/build", h.buildIndex)

	return app
}

type handler struct {
	db         *imdb.IMDB
	idx        atomic.Pointer[index.Searcher]
	ext        Extractor
	searchOpts config.SearchOptions
	threads    int

	buildMu        sync.Mutex
	building       bool
	pendingRebuild bool
	buildTimer     *time.Timer
}

func newHandler(deps *Deps) *handler {
	h := &handler{
		db:         deps.DB,
		ext:        deps.Extractor,
		searchOpts: deps.Search,
		threads:    deps.Threads,
	}
	h.idx.Store(&deps.Index)
	return h
}

func (h *handler) searcher() index.Searcher {
	return *h.idx.Load()
}

func (h *handler) Shutdown() {
	h.buildMu.Lock()
	if h.buildTimer != nil {
		h.buildTimer.Stop()
		h.buildTimer = nil
	}
	h.buildMu.Unlock()
}

func (h *handler) reloadSearcher(ctx context.Context) error {
	if err := h.db.ReloadCache(ctx); err != nil {
		log.Printf("cache reload failed: %v", err)
	}
	newSearcher, _, err := h.db.OpenIndex(ctx, h.threads)
	if err != nil {
		return err
	}
	old := h.idx.Swap(&newSearcher)
	if old != nil {
		go (*old).Close()
	}
	return nil
}

func (h *handler) scheduleBuild(_ context.Context) {
	h.buildMu.Lock()
	defer h.buildMu.Unlock()

	if h.building {
		h.pendingRebuild = true
		return
	}

	if h.buildTimer != nil {
		h.buildTimer.Reset(2 * time.Second)
		return
	}
	// Use background context — the build outlives the HTTP request.
	bgCtx := context.Background()
	h.buildTimer = time.AfterFunc(2*time.Second, func() {
		h.buildMu.Lock()
		h.buildTimer = nil
		h.buildMu.Unlock()
		h.startBuild(bgCtx)
	})
}

func (h *handler) startBuild(ctx context.Context) {
	h.buildMu.Lock()
	if h.building {
		h.buildMu.Unlock()
		return
	}
	h.building = true
	h.pendingRebuild = false
	h.buildMu.Unlock()

	go func() {
		defer func() {
			h.buildMu.Lock()
			h.building = false
			needRebuild := h.pendingRebuild
			h.pendingRebuild = false
			h.buildMu.Unlock()

			if needRebuild {
				h.startBuild(ctx)
			}
		}()

		buildCtx := context.Background()
		t0 := time.Now()
		if err := h.db.BuildIndex(buildCtx, imdb.BuildOptions{BatchSize: 100000}); err != nil {
			log.Printf("auto build failed: %v", err)
			return
		}
		log.Printf("build completed in %v, reloading searcher", time.Since(t0))

		t1 := time.Now()
		if err := h.reloadSearcher(buildCtx); err != nil {
			log.Printf("searcher reload failed: %v", err)
		}
		log.Printf("searcher reloaded in %v", time.Since(t1))

		// Compact shards if they've accumulated past the threshold.
		if compacter, ok := h.db.Backend().(interface{ Compact() error }); ok {
			if shardCount, ok := h.db.Backend().(interface{ CountShards() int }); ok {
				if shardCount.CountShards() >= 8 {
					t2 := time.Now()
					if err := compacter.Compact(); err != nil {
						log.Printf("compact failed: %v", err)
					} else {
						log.Printf("compacted shards in %v", time.Since(t2))
						// Reload again with the compacted index.
						if err := h.reloadSearcher(buildCtx); err != nil {
							log.Printf("post-compact reload failed: %v", err)
						}
					}
				}
			}
		}
	}()
}

// health godoc
// @Summary      Health check
// @Description  Returns service health status.
// @Tags         system
// @Produce      json
// @Success      200 {object} map[string]string
// @Router       /health [get]
// @Security     Bearer
func (h *handler) health(c fiber.Ctx) error {
	return c.JSON(fiber.Map{"status": "ok"})
}

// stats godoc
// @Summary      Database statistics
// @Description  Returns image count, vector count, and unindexed count.
// @Tags         system
// @Produce      json
// @Success      200 {object} StatsResponse
// @Failure      500 {object} APIError
// @Router       /stats [get]
// @Security     Bearer
func (h *handler) stats(c fiber.Ctx) error {
	ctx := c.Context()
	imgCount, vecCount, err := h.db.Count(ctx)
	if err != nil {
		return apiError(c, fiber.StatusInternalServerError, "failed to get stats", err)
	}
	unindexed, err := h.db.CountUnindexed(ctx)
	if err != nil {
		return apiError(c, fiber.StatusInternalServerError, "failed to get unindexed count", err)
	}
	return c.JSON(fiber.Map{
		"images":    imgCount,
		"vectors":   vecCount,
		"unindexed": unindexed,
		"online":    h.db.Online(),
	})
}

// listImages godoc
// @Summary      List images
// @Description  Returns a paginated list of image metadata.
// @Tags         images
// @Produce      json
// @Param        limit  query int false "Items per page (1-100)" default(20) minimum(1) maximum(100)
// @Param        offset query int false "Number of items to skip" default(0) minimum(0)
// @Success      200 {object} ListImagesResponse
// @Failure      500 {object} APIError
// @Router       /images [get]
// @Security     Bearer
func (h *handler) listImages(c fiber.Ctx) error {
	ctx := c.Context()
	limit := queryInt(c, "limit", 20, 1, 100)
	offset := queryInt(c, "offset", 0, 0, 1<<30)

	images, err := h.db.ListImages(ctx, int64(limit), int64(offset))
	if err != nil {
		return apiError(c, fiber.StatusInternalServerError, "failed to list images", err)
	}
	return c.JSON(fiber.Map{
		"images": images,
		"limit":  limit,
		"offset": offset,
	})
}

// getImage godoc
// @Summary      Get image by ID
// @Description  Returns metadata for a single image.
// @Tags         images
// @Produce      json
// @Param        id path int true "Image ID"
// @Success      200 {object} ImageInfo
// @Failure      400 {object} APIError "Invalid image id"
// @Failure      404 {object} APIError "Image not found"
// @Router       /images/{id} [get]
// @Security     Bearer
func (h *handler) getImage(c fiber.Ctx) error {
	ctx := c.Context()
	id, err := parseID(c)
	if err != nil {
		return apiError(c, fiber.StatusBadRequest, "invalid image id", err)
	}
	img, err := h.db.GetImage(ctx, id)
	if err != nil {
		if errors.Is(err, sqlErrNoRows) {
			return apiError(c, fiber.StatusNotFound, "image not found", nil)
		}
		return apiError(c, fiber.StatusInternalServerError, "failed to get image", err)
	}
	return c.JSON(img)
}

// deleteImage godoc
// @Summary      Delete image
// @Description  Removes an image and its associated vectors and stats.
// @Tags         images
// @Param        id path int true "Image ID"
// @Success      204 "No content"
// @Failure      400 {object} APIError "Invalid image id"
// @Failure      404 {object} APIError "Image not found"
// @Router       /images/{id} [delete]
// @Security     Bearer
func (h *handler) deleteImage(c fiber.Ctx) error {
	ctx := c.Context()
	id, err := parseID(c)
	if err != nil {
		return apiError(c, fiber.StatusBadRequest, "invalid image id", err)
	}
	if err := h.db.DeleteImage(ctx, id); err != nil {
		if errors.Is(err, sqlErrNoRows) {
			return apiError(c, fiber.StatusNotFound, "image not found", nil)
		}
		return apiError(c, fiber.StatusInternalServerError, "failed to delete image", err)
	}
	// Local IVF rewrites invlists on disk; reopen searcher so readers see the purge.
	// Online (pgvector) deletes are visible immediately; no searcher reload needed.
	if !h.db.Online() {
		if err := h.reloadSearcher(ctx); err != nil {
			log.Printf("post-delete searcher reload failed: %v", err)
		}
	}
	return c.Status(fiber.StatusNoContent).Send(nil)
}

// addImage godoc
// @Summary      Upload image
// @Description  Uploads an image, extracts ORB descriptors, and adds it. On pgvector the image is immediately searchable; on local a background index build is scheduled. Rejects duplicates by content hash.
// @Tags         images
// @Accept       multipart/form-data
// @Produce      json
// @Param        file formData file true "Image file (jpg/png/webp, max 50MB)"
// @Param        path formData string false "Custom path label for the image"
// @Success      201 {object} AddImageResponse
// @Failure      400 {object} APIError "Missing file field"
// @Failure      409 {object} APIError "Image already exists"
// @Failure      422 {object} APIError "Feature extraction failed"
// @Failure      500 {object} APIError "Internal error"
// @Router       /images [post]
// @Security     Bearer
func (h *handler) addImage(c fiber.Ctx) error {
	ctx := c.Context()

	fileHeader, err := c.FormFile("file")
	if err != nil {
		return apiError(c, fiber.StatusBadRequest, "missing file field", nil)
	}
	path := c.FormValue("path")
	if path == "" {
		path = fileHeader.Filename
	}

	f, err := fileHeader.Open()
	if err != nil {
		return apiError(c, fiber.StatusBadRequest, "failed to open uploaded file", err)
	}
	defer f.Close()
	data := make([]byte, fileHeader.Size)
	if _, err := readFull(f, data); err != nil {
		return apiError(c, fiber.StatusBadRequest, "failed to read uploaded file", err)
	}

	desc, err := h.ext.DetectBytes(data)
	if err != nil {
		return apiError(c, fiber.StatusUnprocessableEntity, "feature extraction failed", err)
	}

	hash := blake3Hash(data)
	if _, exists, err := h.db.CheckHash(ctx, hash[:]); err != nil {
		return apiError(c, fiber.StatusInternalServerError, "hash check failed", err)
	} else if exists {
		return apiError(c, fiber.StatusConflict, "image already exists", nil)
	}

	id, err := h.db.AddImage(ctx, hash[:], path, desc)
	if err != nil {
		return apiError(c, fiber.StatusInternalServerError, "failed to add image", err)
	}

	// Local IVF needs a background build; pgvector is immediately searchable.
	if !h.db.Online() {
		h.scheduleBuild(ctx)
	}

	return c.Status(fiber.StatusCreated).JSON(fiber.Map{
		"id":          id,
		"path":        path,
		"descriptors": len(desc),
		"searchable":  h.db.Online(),
	})
}

// search godoc
// @Summary      Search similar images
// @Description  Uploads an image and returns the most similar images from the index.
// @Tags         search
// @Accept       multipart/form-data
// @Produce      json
// @Param        file     formData file   true  "Query image file"
// @Param        count    formData int    false "Number of results to return" default(10)
// @Param        distance formData int    false "Max Hamming distance threshold" default(64)
// @Param        nprobe   formData int    false "Number of inverted lists to probe" default(3)
// @Success      200 {object} SearchResponse
// @Failure      400 {object} APIError "Missing file field"
// @Failure      422 {object} APIError "Feature extraction failed"
// @Failure      500 {object} APIError "Search failed"
// @Router       /search [post]
// @Security     Bearer
func (h *handler) search(c fiber.Ctx) error {
	ctx := c.Context()
	start := time.Now()

	fileHeader, err := c.FormFile("file")
	if err != nil {
		return apiError(c, fiber.StatusBadRequest, "missing file field", nil)
	}
	f, err := fileHeader.Open()
	if err != nil {
		return apiError(c, fiber.StatusBadRequest, "failed to open uploaded file", err)
	}
	defer f.Close()
	data := make([]byte, fileHeader.Size)
	if _, err := readFull(f, data); err != nil {
		return apiError(c, fiber.StatusBadRequest, "failed to read uploaded file", err)
	}

	opts := h.searchOpts
	if v := c.FormValue("count"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			opts.Count = n
		}
	}
	if v := c.FormValue("distance"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 0 {
			opts.Distance = uint32(n)
		}
	}
	if v := c.FormValue("nprobe"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			opts.NProbe = n
		}
	}

	desc, err := h.ext.DetectBytes(data)
	if err != nil {
		return apiError(c, fiber.StatusUnprocessableEntity, "feature extraction failed", err)
	}

	results, err := h.db.Search(ctx, h.searcher(), desc, opts)
	if err != nil {
		return apiError(c, fiber.StatusInternalServerError, "search failed", err)
	}

	return c.JSON(fiber.Map{
		"time":   time.Since(start).Milliseconds(),
		"result": results,
	})
}

// buildIndex godoc
// @Summary      Build index
// @Description  Triggers an asynchronous index build for unindexed images (local backend only; no-op when online/pgvector).
// @Tags         index
// @Produce      json
// @Param        batch_size query int false "Vectors per flush batch" default(100000)
// @Success      202 {object} BuildIndexResponse
// @Router       /index/build [post]
// @Security     Bearer
func (h *handler) buildIndex(c fiber.Ctx) error {
	if h.db.Online() {
		return c.Status(fiber.StatusAccepted).JSON(fiber.Map{
			"status":  "noop",
			"message": "online backend: images are searchable on add, no build needed",
		})
	}

	h.buildMu.Lock()
	if h.buildTimer != nil {
		h.buildTimer.Stop()
		h.buildTimer = nil
	}
	h.buildMu.Unlock()

	h.startBuild(context.Background())

	return c.Status(fiber.StatusAccepted).JSON(fiber.Map{
		"status": "building",
	})
}

func parseID(c fiber.Ctx) (int64, error) {
	return strconv.ParseInt(c.Params("id"), 10, 64)
}

func queryInt(c fiber.Ctx, key string, def, min, max int) int {
	v := c.Query(key)
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < min {
		return def
	}
	if n > max {
		return max
	}
	return n
}

func apiError(c fiber.Ctx, status int, msg string, cause error) error {
	if cause != nil {
		log.Printf("apiError %d %s: %v", status, msg, cause)
		if os.Getenv("IMSEEK_DEBUG") != "" {
			return c.Status(status).JSON(APIError{Error: msg + ": " + cause.Error()})
		}
	}
	return c.Status(status).JSON(APIError{Error: msg})
}

var sqlErrNoRows = sql.ErrNoRows
