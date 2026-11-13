package web

import (
	"os"
	"testing"
)

func readFixture(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile("testdata/" + name)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestDuckDuckGoFixtureUnwrapsRedirects(t *testing.T) {
	results, err := parseDDGResults(readFixture(t, "ddg.html"), 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %+v", results)
	}
	if results[0].URL != "https://go.dev/doc/" {
		t.Fatalf("redirect not unwrapped: %q", results[0].URL)
	}
	if results[0].Domain != "go.dev" || results[0].Snippet == "" {
		t.Fatalf("unexpected first result: %+v", results[0])
	}
	if results[1].URL != "https://example.org/guide?q=1" {
		t.Fatalf("unexpected second result: %+v", results[1])
	}
}

func TestMojeekFixtureIgnoresNavigation(t *testing.T) {
	results, err := parseMojeekResults(readFixture(t, "mojeek.html"), 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %+v", results)
	}
	if results[0].URL != "https://go.dev/doc/" || results[0].Title != "Go Documentation" {
		t.Fatalf("unexpected first result: %+v", results[0])
	}
}
