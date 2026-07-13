package server

import (
	"crypto/subtle"
	_ "embed"
	"io"
	"log"

	"github.com/gofiber/fiber/v3"
	"github.com/gofiber/fiber/v3/middleware/logger"
	"github.com/zeebo/blake3"

	docs "imseek/docs"
)

func readFull(r io.Reader, buf []byte) (int, error) {
	n, err := io.ReadFull(r, buf)
	if err == io.ErrUnexpectedEOF || err == io.EOF {
		return n, nil
	}
	return n, err
}

func blake3Hash(data []byte) [32]byte {
	return blake3.Sum256(data)
}

func authMiddleware(token string) fiber.Handler {
	return func(c fiber.Ctx) error {
		auth := c.Get("Authorization")
		cred := auth
		const prefix = "Bearer "
		if len(auth) > len(prefix) && auth[:len(prefix)] == prefix {
			cred = auth[len(prefix):]
		}
		if subtle.ConstantTimeCompare([]byte(cred), []byte(token)) != 1 {
			return c.Status(fiber.StatusUnauthorized).JSON(APIError{Error: "unauthorized"})
		}
		return c.Next()
	}
}

func recoverMiddleware() fiber.Handler {
	return func(c fiber.Ctx) error {
		defer func() {
			if r := recover(); r != nil {
				log.Printf("panic recovered: %v", r)
				_ = c.Status(fiber.StatusInternalServerError).JSON(APIError{Error: "internal server error"})
			}
		}()
		return c.Next()
	}
}

func requestLogger() fiber.Handler {
	return logger.New(logger.Config{
		Format:     "${time} ${ip} ${status} - ${latency} ${method} ${path} ${error}\n",
		TimeFormat: "2006-01-02 15:04:05",
		TimeZone:   "Local",
	})
}

//go:embed swagger.html
var swaggerHTML []byte

func swaggerHandler(c fiber.Ctx) error {
	path := c.Params("*")
	switch path {
	case "", "index.html":
		c.Set("Content-Type", "text/html; charset=utf-8")
		return c.Send(swaggerHTML)
	case "doc.json":
		c.Set("Content-Type", "application/json")
		return c.SendString(docs.SwaggerInfo.ReadDoc())
	case "swagger.yaml":
		c.Set("Content-Type", "application/yaml")
		return c.SendString(docs.SwaggerInfo.ReadDoc())
	default:
		return c.Status(404).SendString("not found")
	}
}
