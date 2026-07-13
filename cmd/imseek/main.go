// Command imseek is the CLI entry point.
//
// @title           ImSeek API
// @version         1.0
// @description     Image similarity search service.
// @BasePath        /api/v1
// @schemes         http https
//
// @securityDefinitions.apikey Bearer
// @in header
// @name Authorization
// @description Type "Bearer" followed by a space and the token.
package main

import "imseek/internal/cli"

func main() {
	cli.Execute()
}
