package cli

import (
	"context"
	"math"

	"github.com/charmbracelet/log"
	"github.com/spf13/cobra"

	"imseek/internal/imdb"
	"imseek/internal/index"
)

type trainOpts struct {
	centers int
	images  int
	maxIter int
	init    string
	no2     bool
}

func newTrainCmd() *cobra.Command {
	o := &trainOpts{}
	cmd := &cobra.Command{
		Use:   "train",
		Short: "Train the IVF quantizer (K-Modes clustering)",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runTrain(cmd.Context(), o)
		},
	}
	cmd.Flags().IntVarP(&o.centers, "centers", "n", 0, "number of cluster centers (nlist; auto if 0)")
	cmd.Flags().IntVarP(&o.images, "images", "i", 0, "number of sample descriptors for training (auto if 0)")
	cmd.Flags().IntVarP(&o.maxIter, "max-iter", "m", 20, "max iterations")
	cmd.Flags().StringVar(&o.init, "init", "kmeans-plus-plus", "initialization: random|kmeans-plus-plus")
	cmd.Flags().BoolVar(&o.no2, "no-2level", false, "disable two-level clustering")
	return cmd
}

// autoTrainParams picks nlist ≈ √(totalVectors) clamped to [64, 16384],
// and samples ≈ min(30 × nlist, totalVectors).
func autoTrainParams(totalVecs int64) (nlist, samples int) {
	if totalVecs <= 0 {
		return 64, 1920
	}
	nlist = min(max(int(math.Sqrt(float64(totalVecs))), 64), 16384)
	samples = 30 * nlist
	if int64(samples) > totalVecs {
		samples = int(totalVecs)
	}
	return
}

func runTrain(ctx context.Context, o *trainOpts) error {
	m, err := imdb.Open(ctx, dbOptions(codeSize(), false, 0))
	if err != nil {
		return err
	}
	defer m.Close()

	_, totalVecs, err := m.Count(ctx)
	if err != nil {
		return err
	}

	nlist := o.centers
	sampleCount := o.images
	if nlist == 0 || sampleCount == 0 {
		autoNlist, _ := autoTrainParams(totalVecs)
		if nlist == 0 {
			nlist = autoNlist
		}
		if sampleCount == 0 {
			sampleCount = 30 * nlist
			if int64(sampleCount) > totalVecs {
				sampleCount = int(totalVecs)
			}
		}
		log.Info("auto train params", "total_vectors", totalVecs, "nlist", nlist, "samples", sampleCount)
	}

	log.Info("exporting training vectors", "count", sampleCount)
	vecs, err := m.ExportVectors(ctx, sampleCount)
	if err != nil {
		return err
	}
	log.Info("clustering", "vectors", len(vecs), "centers", nlist)

	res, err := m.TrainIndex(ctx, vecs, index.TrainOptions{
		NList:    nlist,
		MaxIter:  o.maxIter,
		Init:     o.init,
		TwoLevel: !o.no2,
	})
	if err != nil {
		return err
	}
	log.Info("clustering done", "centroids", res.Centroids, "imbalance", res.Imbalance)
	log.Info("quantizer saved", "path", m.QuantizerPath())
	return nil
}
