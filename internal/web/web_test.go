package web

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"

	"github.com/MdHussain121/alpaca/internal/config"
)

type fakeResolver map[string][]string

func (f fakeResolver) LookupIPAddr(ctx context.Context, host string) ([]net.IPAddr, error) {
	ips, ok := f[strings.ToLower(host)]
	if !ok {
		return nil, fmt.Errorf("no such host %q", host)
	}
	out := make([]net.IPAddr, 0, len(ips))
	for _, s := range ips {
		ip := net.ParseIP(s)
		if ip == nil {
			continue
		}
		out = append(out, net.IPAddr{IP: ip})
	}
	return out, nil
}

func TestSSRFBlocksHostileURLsDNSAndRedirects(t *testing.T) {
	// Live internal target that must never be reached by production ValidateAndResolveURL / Fetch.
	internalSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("TOP SECRET INTERNAL DATA"))
	}))
	defer internalSrv.Close()

	res := fakeResolver{
		"public.example":       {"93.184.216.34"},
		"rebind.example":       {"93.184.216.34", "10.0.0.5"}, // mixed public + private -> MUST reject
		"private172.example":   {"172.16.0.10"},
		"private192.example":   {"192.168.1.50"},
		"metadata.example":     {"169.254.169.254"},
		"ecs.example":          {"169.254.170.2"},
		"cgnat.example":        {"100.64.0.1"},
		"v6loop.example":       {"::1"},
		"v6ula.example":        {"fd12:3456:789a::1"},
		"v6linklocal.example":  {"fe80::1"},
		"v4mapped.example":     {"::ffff:127.0.0.1"},
		"v4mappedmeta.example": {"::ffff:169.254.169.254"},
	}

	blocked := []string{
		internalSrv.URL,
		"http://127.0.0.1:8080/v1/models",
		"http://localhost:8080/props",
		"http://[::1]:8080/",
		"http://0.0.0.0:8080/",
		"http://10.0.0.1/admin",
		"http://172.16.5.4/",
		"http://192.168.0.1/",
		"http://169.254.169.254/latest/meta-data/",
		"http://169.254.170.2/v2/credentials",
		"http://100.100.100.200/latest/meta-data/",
		"http://[::ffff:127.0.0.1]:8080/",
		"http://[::ffff:169.254.169.254]/",
		"http://metadata.google.internal/computeMetadata/v1/",
		"http://printer.local/",
		"file:///C:/Windows/System32/drivers/etc/hosts",
		"ftp://example.com/file",
		"gopher://example.com:70/1",
		"javascript:alert(1)",
		"http://user:pass@public.example/",
		"http://rebind.example/",
		"http://private172.example/",
		"http://private192.example/",
		"http://metadata.example/",
		"http://ecs.example/",
		"http://cgnat.example/",
		"http://v6loop.example/",
		"http://v6ula.example/",
		"http://v6linklocal.example/",
		"http://v4mapped.example/",
		"http://v4mappedmeta.example/",
	}

	for _, u := range blocked {
		t.Run(u, func(t *testing.T) {
			_, _, _, err := ValidateAndResolveURL(context.Background(), u, ClientOptions{Resolver: res})
			if err == nil {
				t.Fatalf("expected SSRF error for %q, got nil", u)
			}
			if _, ferr := Fetch(context.Background(), FetchOptions{URL: u, ClientOpts: ClientOptions{Resolver: res}}); ferr == nil {
				t.Fatalf("Fetch unexpectedly succeeded for blocked URL %q", u)
			}
		})
	}

	// Verify public DNS IP vets cleanly.
	_, ip, port, err := ValidateAndResolveURL(context.Background(), "https://public.example/docs", ClientOptions{Resolver: res})
	if err != nil {
		t.Fatalf("expected public.example to pass validation: %v", err)
	}
	if ip.String() != "93.184.216.34" || port != "443" {
		t.Fatalf("unexpected resolved endpoint: %s:%s", ip, port)
	}

	// Verify redirect protection: a server on loopback (test-opted-in) redirects to a private DNS name OR 169.254.169.254 -> MUST FAIL on CheckRedirect!
	redirectCases := []struct {
		name   string
		target string
	}{
		{"redirect_to_metadata", "http://169.254.169.254/latest/meta-data/"},
		{"redirect_to_private_dns", "http://private192.example/secret"},
		{"redirect_to_file_scheme", "file:///C:/Windows/win.ini"},
	}
	for _, rc := range redirectCases {
		t.Run(rc.name, func(t *testing.T) {
			redirSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.Redirect(w, r, rc.target, http.StatusFound)
			}))
			defer redirSrv.Close()

			_, err := Fetch(context.Background(), FetchOptions{
				URL: redirSrv.URL,
				ClientOpts: ClientOptions{
					Resolver:             res,
					AllowLoopbackForTest: true,
				},
			})
			if err == nil {
				t.Fatalf("expected redirect target %q to be blocked", rc.target)
			}
		})
	}

	// Also verify redirect to an internal loopback target is blocked when AllowLoopbackForTest=false on CheckRedirect hop (simulated via public->loopback):
	hopSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, internalSrv.URL, http.StatusFound)
	}))
	defer hopSrv.Close()
	hopURL, _ := url.Parse(hopSrv.URL)
	// Even if initial host were vetted public IP, CheckRedirect calls ValidateAndResolveURL with opts (when AllowLoopbackForTest=false):
	client := NewSafeHTTPClient(ClientOptions{Resolver: res})
	req, _ := http.NewRequest(http.MethodGet, "http://10.0.0.1/", nil)
	if err := client.CheckRedirect(req, []*http.Request{{URL: hopURL}}); err == nil {
		t.Fatal("CheckRedirect must reject 10.0.0.1")
	}
}

func TestFetchHTMLReadabilityBinaryRefusal2MBCapAndUntrustedDelimiters(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/article":
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			_, _ = w.Write([]byte(`<!doctype html><html><head><title>Go Agent Guide</title></head><body>
				<nav><a href="/home">Home</a> | <a href="/login">Login</a></nav>
				<script>alert("xss")</script>
				<article>
					<h1>Writing Safe Agents</h1>
					<p>Always sandbox <strong>workspace</strong> files carefully &amp; validate paths.</p>
					<ul><li>Check traversal</li><li>Reject symlinks</li></ul>
					<pre><code>fmt.Println("ok")</code></pre>
					<p>Extra padding text to exceed article length heuristic cleanly and reliably.</p>
				</article>
				<footer>Copyright 2026</footer>
			</body></html>`))
		case "/binary":
			w.Header().Set("Content-Type", "application/octet-stream")
			_, _ = w.Write([]byte{0x89, 0x50, 0x4e, 0x47, 0x00, 0x01})
		case "/huge":
			w.Header().Set("Content-Type", "text/plain")
			_, _ = w.Write([]byte(strings.Repeat("A", DefaultMaxBytes+50000)))
		}
	}))
	defer srv.Close()

	// 1. HTML -> Readable Markdown + Untrusted Delimiters
	res, err := Fetch(context.Background(), FetchOptions{
		URL:        srv.URL + "/article",
		ClientOpts: ClientOptions{AllowLoopbackForTest: true},
	})
	if err != nil {
		t.Fatalf("Fetch article failed: %v", err)
	}
	if !strings.Contains(res.Body, "# Writing Safe Agents") || !strings.Contains(res.Body, "**workspace**") {
		t.Fatalf("unexpected markdown readability output:\n%s", res.Body)
	}
	if strings.Contains(res.Body, "Copyright 2026") || strings.Contains(res.Body, `alert("xss")`) || strings.Contains(res.Body, "Login") {
		t.Fatalf("boilerplate nav/script/footer leaked into readable output:\n%s", res.Body)
	}
	if !strings.HasPrefix(res.Wrapped, "<<<UNTRUSTED_WEB_CONTENT") || !strings.HasSuffix(res.Wrapped, "<<<END_UNTRUSTED_WEB_CONTENT>>>") {
		t.Fatalf("missing untrusted wrapper:\n%s", res.Wrapped)
	}

	// 2. Binary refusal
	if _, err := Fetch(context.Background(), FetchOptions{
		URL:        srv.URL + "/binary",
		ClientOpts: ClientOptions{AllowLoopbackForTest: true},
	}); err == nil {
		t.Fatal("expected error fetching binary content")
	}

	// 3. 2 MB cap
	huge, err := Fetch(context.Background(), FetchOptions{
		URL:        srv.URL + "/huge",
		MaxChars:   3000000,
		ClientOpts: ClientOptions{AllowLoopbackForTest: true},
	})
	if err != nil {
		t.Fatalf("Fetch huge failed: %v", err)
	}
	if !huge.ByteCapped || huge.ByteLen != DefaultMaxBytes {
		t.Fatalf("expected ByteCapped=true ByteLen=%d, got ByteCapped=%v ByteLen=%d", DefaultMaxBytes, huge.ByteCapped, huge.ByteLen)
	}
}

func TestSearchProvidersConfigAndSecretRedaction(t *testing.T) {
	// 1. Unconfigured returns helpful reason and nil provider
	t.Setenv("BRAVE_API_KEY", "")
	t.Setenv("TAVILY_API_KEY", "")
	t.Setenv("ALPACA_WEB_SEARCH_KEY", "")
	t.Setenv("LLM_WEB_SEARCH_KEY", "")
	t.Setenv("ALPACA_WEB_SEARCH_PROVIDER", "")
	t.Setenv("LLM_WEB_SEARCH_PROVIDER", "")
	t.Setenv("SEARXNG_URL", "")
	t.Setenv("ALPACA_SEARXNG_URL", "")
	t.Setenv("LLM_SEARXNG_URL", "")

	prov, reason := ResolveSearchProvider(SearchConfigOptions{})
	if prov != nil || reason == "" {
		t.Fatalf("expected nil provider with reason when unconfigured, got prov=%v reason=%q", prov, reason)
	}

	// 2. SearXNG against local test server
	searxSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/search" || r.URL.Query().Get("format") != "json" {
			http.Error(w, "bad req", 400)
			return
		}
		_, _ = w.Write([]byte(`{"results":[{"title":"Go Blog","url":"https://go.dev/blog","content":"Go releases"},{"title":"Two","url":"https://example.org","content":"Two desc"}]}`))
	}))
	defer searxSrv.Close()

	sProv, errReason := ResolveSearchProvider(SearchConfigOptions{
		Web: config.WebConfig{SearXNGURL: searxSrv.URL},
	})
	if sProv == nil {
		t.Fatalf("expected searxng provider, got reason=%q", errReason)
	}
	results, err := sProv.Search(context.Background(), "golang", 1)
	if err != nil || len(results) != 1 || results[0].Title != "Go Blog" {
		t.Fatalf("searxng results mismatch: results=%+v err=%v", results, err)
	}

	// 3. Brave provider secret redaction on HTTP error
	secretKey := "BSA-SUPER-SECRET-TOKEN-123456"
	braveSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Subscription-Token") != secretKey {
			http.Error(w, "unauthorized", 401)
			return
		}
		http.Error(w, "rate limited for key "+secretKey, 429)
	}))
	defer braveSrv.Close()

	bp := &braveProvider{apiKey: secretKey, baseURL: braveSrv.URL, httpClient: braveSrv.Client()}
	_, bErr := bp.Search(context.Background(), "test", 5)
	if bErr == nil {
		t.Fatal("expected error from brave mock")
	}
	if strings.Contains(bErr.Error(), secretKey) || !strings.Contains(bErr.Error(), "[REDACTED]") {
		t.Fatalf("secret leaked in error output: %v", bErr)
	}
}
