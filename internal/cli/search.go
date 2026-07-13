package cli

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/spf13/cobra"

	"imseek/internal/imdb"
)

type searchOpts struct {
	xf           extractorFlags
	scoreType    string
	outputFormat string
}

func newSearchCmd() *cobra.Command {
	o := &searchOpts{}
	cmd := &cobra.Command{
		Use:   "search IMAGE",
		Short: "Search for images similar to the given image",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			return runSearch(cmd.Context(), o, args[0])
		},
	}
	o.xf.register(cmd)
	cmd.Flags().StringVar(&o.scoreType, "score-type", "wilson", "scoring: wilson|count")
	cmd.Flags().StringVar(&o.outputFormat, "output-format", "table", "output format: table|json")
	return cmd
}

func runSearch(ctx context.Context, o *searchOpts, imagePath string) error {
	opts := searchOptions(o.scoreType)
	m, err := imdb.Open(ctx, dbOptions(codeSize(), false, opts.ScoreType).WithWAL(true))
	if err != nil {
		return err
	}
	defer m.Close()

	ext, err := o.xf.newExtractor()
	if err != nil {
		return err
	}
	defer ext.Close()

	desc, err := ext.DetectFile(imagePath)
	if err != nil {
		return err
	}

	searcher, closeIdx, err := m.OpenIndex(ctx, opts.Threads)
	if err != nil {
		return err
	}
	defer closeIdx()

	results, err := m.Search(ctx, searcher, desc, opts)
	if err != nil {
		return err
	}

	if o.outputFormat == "json" {
		b, err := json.MarshalIndent(results, "", "  ")
		if err != nil {
			return err
		}
		fmt.Println(string(b))
		return nil
	}
	for _, r := range results {
		fmt.Printf("%.2f\t%s\n", r.Score, r.Path)
	}
	return nil
}
