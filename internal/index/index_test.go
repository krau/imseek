package index

import (
	"context"
	"errors"
	"testing"
)

// mockBackend is a minimal Backend implementation for testing the interface
// contract and imdb layer interactions.
type mockBackend struct {
	trained     bool
	trainResult TrainResult
	trainErr    error
	builder     Builder
	searcher    Searcher
	searcherErr error
	closeCalled bool
}

func (m *mockBackend) Train(_ context.Context, _ [][]byte, _ TrainOptions) (TrainResult, error) {
	return m.trainResult, m.trainErr
}
func (m *mockBackend) Trained(_ context.Context) bool { return m.trained }
func (m *mockBackend) NewBuilder(_ context.Context) (Builder, error) {
	if m.builder == nil {
		return nil, errors.New("no builder configured")
	}
	return m.builder, nil
}
func (m *mockBackend) NewSearcher(_ context.Context, _ int) (Searcher, error) {
	return m.searcher, m.searcherErr
}
func (m *mockBackend) DeleteVectors(_ context.Context, _, _ uint64) error { return nil }
func (m *mockBackend) Close() error                                       { m.closeCalled = true; return nil }

type mockBuilder struct {
	addCalls    int
	flushCalls  int
	commitCalls int
	closeCalls  int
	addErr      error
}

func (m *mockBuilder) Add(_ context.Context, codes [][]byte, _ []uint64) error {
	if len(codes) == 0 {
		return nil
	}
	m.addCalls++
	return m.addErr
}
func (m *mockBuilder) Flush(_ context.Context) error  { m.flushCalls++; return nil }
func (m *mockBuilder) Commit(_ context.Context) error { m.commitCalls++; return nil }
func (m *mockBuilder) Close() error                   { m.closeCalls++; return nil }

type mockSearcher struct {
	searchCalls int
	results     []Neighbor
	closeErr    error
}

func (m *mockSearcher) Search(_ context.Context, _ [][]byte, _, _ int) []Neighbor {
	m.searchCalls++
	return m.results
}
func (m *mockSearcher) Close() error { return m.closeErr }

func TestMockBackendInterface(t *testing.T) {
	var _ Backend = (*mockBackend)(nil)
	var _ Builder = (*mockBuilder)(nil)
	var _ Searcher = (*mockSearcher)(nil)
}

func TestMockBackendTrain(t *testing.T) {
	ctx := context.Background()
	b := &mockBackend{
		trained:     true,
		trainResult: TrainResult{Centroids: 64, Imbalance: 1.2},
	}
	res, err := b.Train(ctx, nil, TrainOptions{NList: 64})
	if err != nil {
		t.Fatal(err)
	}
	if res.Centroids != 64 {
		t.Errorf("Centroids = %d want 64", res.Centroids)
	}
	if !b.Trained(ctx) {
		t.Error("Trained() = false want true")
	}
}

func TestMockBackendUntrained(t *testing.T) {
	ctx := context.Background()
	b := &mockBackend{trained: false}
	if b.Trained(ctx) {
		t.Error("Trained() = true want false")
	}
	_, err := b.NewBuilder(ctx)
	if err == nil {
		t.Error("NewBuilder on untrained backend should error")
	}
}

func TestMockBuilderAdd(t *testing.T) {
	ctx := context.Background()
	bl := &mockBuilder{}
	if err := bl.Add(ctx, [][]byte{{1, 2}}, []uint64{1}); err != nil {
		t.Fatal(err)
	}
	if bl.addCalls != 1 {
		t.Errorf("addCalls = %d want 1", bl.addCalls)
	}
	// Empty add should not count
	bl.Add(ctx, nil, nil)
	if bl.addCalls != 1 {
		t.Errorf("addCalls after empty = %d want 1", bl.addCalls)
	}
}

func TestMockSearcherSearch(t *testing.T) {
	ctx := context.Background()
	s := &mockSearcher{
		results: []Neighbor{{ID: 1, Distance: 10}, {ID: 2, Distance: 20}},
	}
	got := s.Search(ctx, [][]byte{{1}}, 5, 3)
	if len(got) != 2 {
		t.Fatalf("len = %d want 2", len(got))
	}
	if got[0].ID != 1 || got[0].Distance != 10 {
		t.Errorf("got[0] = %+v", got[0])
	}
}

func TestMockBackendClose(t *testing.T) {
	b := &mockBackend{}
	b.Close()
	if !b.closeCalled {
		t.Error("Close() did not set closeCalled")
	}
}
