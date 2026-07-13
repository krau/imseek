package cli

import (
	"image"

	"github.com/spf13/cobra"

	"imseek/internal/orb"
)

type Extractor interface {
	DetectImage(image.Image) ([][]byte, error)
	DetectBytes([]byte) ([][]byte, error)
	DetectFile(string) ([][]byte, error)
	Close() error
}

type extractorFlags struct {
	maxHeight   int
	maxWidth    int
	nfeatures   int
	scaleFactor float64
	nlevels     int
}

func (f *extractorFlags) register(cmd *cobra.Command) {
	cmd.Flags().IntVar(&f.maxHeight, "max-height", orb.DefaultMaxSize.Height, "max image height")
	cmd.Flags().IntVar(&f.maxWidth, "max-width", orb.DefaultMaxSize.Width, "max image width")
	cmd.Flags().IntVar(&f.nfeatures, "orb-nfeatures", 500, "[orb] target keypoints")
	cmd.Flags().Float64Var(&f.scaleFactor, "orb-scale", 1.2, "[orb] pyramid scale factor")
	cmd.Flags().IntVar(&f.nlevels, "orb-levels", 8, "[orb] pyramid levels")
}

func (f *extractorFlags) newExtractor() (Extractor, error) {
	return orb.New(orb.Options{
		NFeatures:   f.nfeatures,
		ScaleFactor: f.scaleFactor,
		NLevels:     f.nlevels,
		MaxSize:     orb.MaxSize{Height: f.maxHeight, Width: f.maxWidth},
	}), nil
}
