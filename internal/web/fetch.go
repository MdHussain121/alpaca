package web

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"mime"
	"net/http"
	"net/url"
	"strings"
	"unicode/utf8"

	xhtml "golang.org/x/net/html"
	"golang.org/x/net/html/charset"
)

const (
	defaultFetchMaxTokens = 1500
	browserUserAgent      = "Mozilla/5.0 (compatible; alpaca/1.2; +https://github.com/MdHussain121/alpaca)"
)

type FetchOptions struct {
	URL          string
	Query        string
	MaxTokens    int
	OffsetTokens int
	MaxChars     int // legacy character cap; converted to a token budget
	MaxBytes     int
	ClientOpts   ClientOptions
	Cache        *Cache
	NoCache      bool
}

type FetchResult struct {
	URL           string
	FinalURL      string
	StatusCode    int
	ContentType   string
	Title         string
	Body          string
	Wrapped       string
	Links         []string
	ByteLen       int
	ByteCapped    bool
	CharTrunc     bool
	TokenEstimate int
	FromCache     bool
}

type cachedPage struct {
	FinalURL    string   `json:"final_url"`
	StatusCode  int      `json:"status"`
	ContentType string   `json:"content_type"`
	Title       string   `json:"title"`
	Body        string   `json:"body"`
	RawHTML     string   `json:"-"`
	Links       []string `json:"links,omitempty"`
	ByteLen     int      `json:"byte_len,omitempty"`
	ByteCapped  bool     `json:"byte_capped,omitempty"`
	HasScript   bool     `json:"has_script,omitempty"`
}

// Fetch retrieves a web resource safely via NewSafeHTTPClient, converts HTML to
// readable Markdown, selects query-relevant passages within a token budget, and
// wraps the result as untrusted data.
func Fetch(ctx context.Context, opts FetchOptions) (FetchResult, error) {
	return fetch(ctx, opts, false)
}

func fetch(ctx context.Context, opts FetchOptions, ampRetry bool) (FetchResult, error) {
	u, _, _, err := ValidateAndResolveURL(ctx, opts.URL, opts.ClientOpts)
	if err != nil {
		return FetchResult{}, err
	}
	maxBytes := opts.MaxBytes
	if maxBytes <= 0 || maxBytes > DefaultMaxBytes {
		maxBytes = DefaultMaxBytes
	}
	maxTokens := opts.MaxTokens
	if maxTokens <= 0 && opts.MaxChars > 0 {
		maxTokens = opts.MaxChars / 4
	}
	if maxTokens <= 0 {
		maxTokens = defaultFetchMaxTokens
	}

	key := CacheKey("page", u.String())
	var page cachedPage
	cacheHit := false
	if opts.Cache != nil && !opts.NoCache {
		cacheHit = opts.Cache.GetJSON(key, &page)
	}
	if !cacheHit {
		page, err = fetchPage(ctx, u.String(), maxBytes, opts.ClientOpts)
		if err != nil {
			return FetchResult{}, err
		}
		if opts.Cache != nil && !opts.NoCache {
			opts.Cache.PutJSON(key, page)
		}
	}

	body := page.Body
	if opts.OffsetTokens > 0 {
		body = dropLeadingTokens(body, opts.OffsetTokens)
	}
	if strings.TrimSpace(opts.Query) != "" && EstimateTokens(body) > maxTokens {
		body = SelectExcerpt(body, opts.Query, maxTokens)
	} else {
		body = TruncateToTokens(body, maxTokens)
	}
	truncated := EstimateTokens(body) > maxTokens
	body = TruncateToTokens(body, maxTokens)

	// A JS-only shell often parses to almost nothing. Retry the AMP page once.
	if !ampRetry && EstimateTokens(body) < 12 && page.HasScript {
		if amp := findAMPLink(page.RawHTML); amp != "" {
			ampOpts := opts
			ampOpts.URL = amp
			if res, ampErr := fetch(ctx, ampOpts, true); ampErr == nil {
				res.Links = append(res.Links, page.Links...)
				return res, nil
			}
		}
		return FetchResult{}, fmt.Errorf("page appears JavaScript-only (little readable text). Try the next result or the site's reader/AMP version.")
	}
	if EstimateTokens(body) < 4 && strings.Contains(strings.ToLower(page.ContentType), "html") {
		return FetchResult{}, fmt.Errorf("page has no readable article text (paywall, login wall, or JavaScript-only). Try another result.")
	}

	var notes []string
	if page.ByteLen > 0 {
		// ByteLen is only available on fresh fetches; cache keeps the body.
	}
	if truncated {
		notes = append(notes, fmt.Sprintf("[text limited to about %d tokens]", maxTokens))
	}
	decorated := body
	if len(notes) > 0 {
		decorated += "\n\n" + strings.Join(notes, " ")
	}
	return FetchResult{
		URL:           u.String(),
		FinalURL:      page.FinalURL,
		StatusCode:    page.StatusCode,
		ContentType:   page.ContentType,
		Title:         page.Title,
		Body:          decorated,
		Wrapped:       WrapUntrustedContent(page.FinalURL, decorated),
		Links:         page.Links,
		ByteLen:       page.ByteLen,
		ByteCapped:    page.ByteCapped,
		CharTrunc:     truncated,
		TokenEstimate: EstimateTokens(decorated),
		FromCache:     cacheHit,
	}, nil
}

func fetchPage(ctx context.Context, rawURL string, maxBytes int, opts ClientOptions) (cachedPage, error) {
	client := NewSafeHTTPClient(opts)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return cachedPage{}, err
	}
	req.Header.Set("User-Agent", browserUserAgent)
	req.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.5")
	req.Header.Set("Accept-Language", "en-US,en;q=0.8")

	resp, err := client.Do(req)
	if err != nil {
		return cachedPage{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return cachedPage{}, fetchStatusError(resp.StatusCode, resp.Request.URL.String())
	}

	ct := strings.TrimSpace(resp.Header.Get("Content-Type"))
	mediaType, params, _ := mime.ParseMediaType(ct)
	mediaType = strings.ToLower(mediaType)
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, int64(maxBytes)+1))
	if err != nil {
		return cachedPage{}, fmt.Errorf("reading response body: %w", err)
	}
	byteCapped := false
	if len(raw) > maxBytes {
		raw = raw[:maxBytes]
		byteCapped = true
	}

	if mediaType == "application/pdf" || bytes.HasPrefix(bytes.TrimSpace(raw), []byte("%PDF")) {
		return cachedPage{}, fmt.Errorf("PDF text extraction is not available in this build. Try an HTML version or another result.")
	}
	if isRejectedBinaryMIME(mediaType) || isBinaryPayload(mediaType, raw) {
		return cachedPage{}, fmt.Errorf("refusing binary content-type %q. Try another result.", mediaType)
	}

	finalURL := resp.Request.URL.String()
	page := cachedPage{FinalURL: finalURL, StatusCode: resp.StatusCode, ContentType: mediaType, ByteLen: len(raw), ByteCapped: byteCapped}

	switch {
	case mediaType == "text/html" || mediaType == "application/xhtml+xml" || looksLikeHTML(raw):
		decoded := decodeBody(raw, ct, params)
		title, body, links := ExtractReadable(decoded)
		page.Title = title
		page.Body = body
		page.RawHTML = decoded
		page.HasScript = strings.Contains(strings.ToLower(decoded), "<script")
		for _, href := range links {
			if abs := resolveLink(finalURL, href); abs != "" {
				page.Links = append(page.Links, abs)
			}
		}
	default:
		page.Body = strings.TrimSpace(string(raw))
	}
	if page.Body == "" {
		return cachedPage{}, fmt.Errorf("empty response body. Try another result.")
	}
	return page, nil
}

func decodeBody(raw []byte, contentType string, params map[string]string) string {
	if utf8.Valid(raw) {
		return string(raw)
	}
	r, err := charset.NewReader(bytes.NewReader(raw), contentType)
	if err != nil {
		return string(raw)
	}
	decoded, err := io.ReadAll(r)
	if err != nil {
		return string(raw)
	}
	return string(decoded)
}

func fetchStatusError(code int, u string) error {
	switch code {
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("HTTP %d (forbidden or paywall). Try another result.", code)
	case http.StatusNotFound:
		return fmt.Errorf("HTTP 404 (page not found). The URL may be invented; use a search result number instead.")
	case http.StatusTooManyRequests:
		return fmt.Errorf("HTTP 429 (rate limited). Wait briefly, then try another result.")
	case http.StatusUnsupportedMediaType:
		return fmt.Errorf("HTTP 415 (unsupported content). Try another result.")
	}
	if code >= 500 {
		return fmt.Errorf("HTTP %d (server error at %s). Try another result.", code, DomainOf(u))
	}
	return fmt.Errorf("HTTP %d fetching %s. Try another result.", code, DomainOf(u))
}

func findAMPLink(rawHTML string) string {
	doc, err := xhtml.Parse(strings.NewReader(rawHTML))
	if err != nil {
		return ""
	}
	var found string
	walkNodes(doc, func(n *xhtml.Node) {
		if found != "" || n.Type != xhtml.ElementNode || !strings.EqualFold(n.Data, "link") {
			return
		}
		if !strings.EqualFold(nodeAttr(n, "rel"), "amphtml") {
			return
		}
		found = strings.TrimSpace(nodeAttr(n, "href"))
	})
	return found
}

func resolveLink(base, href string) string {
	href = strings.TrimSpace(href)
	if href == "" || strings.HasPrefix(href, "#") || strings.HasPrefix(strings.ToLower(href), "javascript:") {
		return ""
	}
	b, err := url.Parse(base)
	if err != nil {
		return ""
	}
	ref, err := url.Parse(href)
	if err != nil {
		return ""
	}
	abs := b.ResolveReference(ref)
	if abs.Scheme != "http" && abs.Scheme != "https" {
		return ""
	}
	abs.Fragment = ""
	return abs.String()
}

func dropLeadingTokens(s string, tokens int) string {
	if tokens <= 0 {
		return strings.TrimSpace(s)
	}
	cut := tokens * 4
	if cut >= len(s) {
		return ""
	}
	if i := strings.IndexByte(s[cut:], '\n'); i >= 0 {
		cut += i + 1
	}
	return strings.TrimSpace(s[cut:])
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
