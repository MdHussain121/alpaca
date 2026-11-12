package web

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"mime"
	"net/http"
	"strings"
	"unicode/utf8"
)

const defaultFetchMaxChars = 20000

type FetchOptions struct {
	URL        string
	MaxChars   int
	MaxBytes   int
	ClientOpts ClientOptions
}

type FetchResult struct {
	URL         string
	FinalURL    string
	StatusCode  int
	ContentType string
	Title       string
	Body        string // clean text/markdown before untrusted wrapping
	Wrapped     string // untrusted delimiter-wrapped payload for LLM consumption
	ByteLen     int
	ByteCapped  bool
	CharTrunc   bool
}

// Fetch retrieves a web resource safely via NewSafeHTTPClient, enforces a 2 MB byte cap,
// rejects binaries, converts HTML to readable Markdown, and wraps output as untrusted content.
func Fetch(ctx context.Context, opts FetchOptions) (FetchResult, error) {
	u, _, _, err := ValidateAndResolveURL(ctx, opts.URL, opts.ClientOpts)
	if err != nil {
		return FetchResult{}, err
	}
	maxBytes := opts.MaxBytes
	if maxBytes <= 0 || maxBytes > DefaultMaxBytes {
		maxBytes = DefaultMaxBytes
	}
	maxChars := opts.MaxChars
	if maxChars <= 0 {
		maxChars = defaultFetchMaxChars
	}

	client := NewSafeHTTPClient(opts.ClientOpts)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return FetchResult{}, err
	}
	req.Header.Set("User-Agent", "github.com/MdHussain121/alpaca/1.2 (+local-agent)")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.5")

	resp, err := client.Do(req)
	if err != nil {
		return FetchResult{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return FetchResult{}, fmt.Errorf("HTTP %d fetching %s", resp.StatusCode, resp.Request.URL.String())
	}

	ct := strings.TrimSpace(resp.Header.Get("Content-Type"))
	mediaType, _, _ := mime.ParseMediaType(ct)
	mediaType = strings.ToLower(mediaType)
	if mediaType == "" {
		mediaType = strings.ToLower(strings.Split(http.DetectContentType(nil), ";")[0])
	}
	if isRejectedBinaryMIME(mediaType) {
		return FetchResult{}, fmt.Errorf("refusing binary content-type %q", mediaType)
	}

	raw, err := io.ReadAll(io.LimitReader(resp.Body, int64(maxBytes)+1))
	if err != nil {
		return FetchResult{}, fmt.Errorf("reading response body: %w", err)
	}
	byteCapped := false
	if len(raw) > maxBytes {
		raw = raw[:maxBytes]
		byteCapped = true
	}
	if isBinaryPayload(mediaType, raw) {
		return FetchResult{}, fmt.Errorf("refusing binary payload (%d bytes)", len(raw))
	}

	finalURL := resp.Request.URL.String()
	var title, body string
	switch {
	case mediaType == "text/html" || mediaType == "application/xhtml+xml" || looksLikeHTML(raw):
		title, body = ExtractReadableMarkdown(string(raw))
	default:
		body = strings.TrimSpace(string(raw))
	}

	charTrunc := false
	runes := []rune(body)
	if len(runes) > maxChars {
		body = strings.TrimSpace(string(runes[:maxChars]))
		charTrunc = true
	}

	var notes []string
	if byteCapped {
		notes = append(notes, fmt.Sprintf("[response truncated at %d byte limit]", maxBytes))
	}
	if charTrunc {
		notes = append(notes, fmt.Sprintf("[text truncated to %d chars]", maxChars))
	}
	decorated := body
	if len(notes) > 0 {
		decorated += "\n\n" + strings.Join(notes, " ")
	}

	return FetchResult{
		URL:         u.String(),
		FinalURL:    finalURL,
		StatusCode:  resp.StatusCode,
		ContentType: mediaType,
		Title:       title,
		Body:        decorated,
		Wrapped:     WrapUntrustedContent(finalURL, decorated),
		ByteLen:     len(raw),
		ByteCapped:  byteCapped,
		CharTrunc:   charTrunc,
	}, nil
}

func isRejectedBinaryMIME(mt string) bool {
	if mt == "" {
		return false
	}
	if strings.HasPrefix(mt, "text/") {
		return false
	}
	switch mt {
	case "application/json", "application/ld+json", "application/xml", "application/xhtml+xml", "application/javascript":
		return false
	}
	if strings.HasSuffix(mt, "+json") || strings.HasSuffix(mt, "+xml") {
		return false
	}
	if strings.HasPrefix(mt, "image/") || strings.HasPrefix(mt, "audio/") || strings.HasPrefix(mt, "video/") || strings.HasPrefix(mt, "font/") {
		return true
	}
	switch mt {
	case "application/octet-stream", "application/pdf", "application/zip", "application/gzip", "application/x-tar", "application/wasm":
		return true
	}
	return false
}

func isBinaryPayload(mt string, raw []byte) bool {
	head := raw
	if len(head) > 4096 {
		head = head[:4096]
	}
	if bytes.IndexByte(head, 0) >= 0 {
		return true
	}
	if !strings.HasPrefix(mt, "text/") && mt != "application/json" && mt != "application/xhtml+xml" {
		if !utf8.Valid(raw) {
			return true
		}
	}
	return false
}

func looksLikeHTML(raw []byte) bool {
	s := strings.ToLower(strings.TrimSpace(string(raw[:min(len(raw), 512)])))
	return strings.HasPrefix(s, "<!doctype html") || strings.HasPrefix(s, "<html") || strings.Contains(s, "<body")
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
