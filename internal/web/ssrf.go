// Package web implements SSRF-safe web_fetch, configurable web_search providers,
// secret redaction, and untrusted web content wrappers.
package web

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/netip"
	"net/url"
	"strings"
	"time"
)

const (
	DefaultTimeout      = 15 * time.Second
	DefaultMaxRedirects = 5
	DefaultMaxBytes     = 2 * 1024 * 1024 // 2 MB strict body cap
)

// Resolver abstracts DNS lookup so SSRF protections can be unit-tested deterministically.
type Resolver interface {
	LookupIPAddr(ctx context.Context, host string) ([]net.IPAddr, error)
}

var blockedPrefixes = mustParsePrefixes([]string{
	"0.0.0.0/8",          // "this network"
	"10.0.0.0/8",         // RFC1918
	"100.64.0.0/10",      // CGNAT / carrier NAT (incl. 100.100.100.200 metadata)
	"127.0.0.0/8",        // IPv4 loopback
	"169.254.0.0/16",     // Link-local & cloud metadata (169.254.169.254, 169.254.170.2)
	"172.16.0.0/12",      // RFC1918
	"192.0.0.0/24",       // IETF protocol assignments
	"192.0.2.0/24",       // TEST-NET-1
	"192.168.0.0/16",     // RFC1918
	"198.18.0.0/15",      // Benchmarking
	"198.51.100.0/24",    // TEST-NET-2
	"203.0.113.0/24",     // TEST-NET-3
	"224.0.0.0/4",        // Multicast
	"240.0.0.0/4",        // Reserved / broadcast
	"255.255.255.255/32", // Limited broadcast
	"::/128",             // IPv6 unspecified
	"::1/128",            // IPv6 loopback
	"100::/64",           // Discard-only
	"2001:db8::/32",      // Documentation
	"fc00::/7",           // IPv6 ULA private
	"fe80::/10",          // IPv6 link-local
	"ff00::/8",           // IPv6 multicast
})

func mustParsePrefixes(cidrs []string) []netip.Prefix {
	out := make([]netip.Prefix, 0, len(cidrs))
	for _, c := range cidrs {
		out = append(out, netip.MustParsePrefix(c))
	}
	return out
}

// IsBlockedAddr checks whether a netip.Addr is non-public (loopback, private, link-local,
// cloud metadata, unspecified, multicast, CGNAT, or IPv4-mapped non-public IPv6).
func IsBlockedAddr(addr netip.Addr) (bool, string) {
	if !addr.IsValid() {
		return true, "invalid IP address"
	}
	// Strip zone and unmap ::ffff:a.b.c.d -> a.b.c.d so mapped addresses cannot slip past IPv4 rules.
	clean := addr.WithZone("").Unmap()
	if clean.IsLoopback() {
		return true, "loopback address"
	}
	if clean.IsPrivate() {
		return true, "private address"
	}
	if clean.IsLinkLocalUnicast() || clean.IsLinkLocalMulticast() {
		return true, "link-local / metadata address"
	}
	if clean.IsMulticast() || clean.IsInterfaceLocalMulticast() {
		return true, "multicast address"
	}
	if clean.IsUnspecified() {
		return true, "unspecified address"
	}
	for _, p := range blockedPrefixes {
		if p.Contains(clean) {
			return true, fmt.Sprintf("restricted range (%s)", p)
		}
	}
	return false, ""
}

// IsBlockedIP checks a net.IP against SSRF block rules.
func IsBlockedIP(ip net.IP) (bool, string) {
	if ip == nil {
		return true, "nil IP"
	}
	addr, ok := netip.AddrFromSlice(ip)
	if !ok {
		return true, "invalid IP slice"
	}
	return IsBlockedAddr(addr)
}

type ClientOptions struct {
	Resolver             Resolver
	Timeout              time.Duration
	MaxRedirects         int
	AllowLoopbackForTest bool // test-only hook; production callers MUST leave false
}

// ValidateAndResolveURL enforces http/https schemes, resolves DNS explicitly,
// rejects any loopback/private/link-local/metadata IP, and returns a single pinned IP + port.
func ValidateAndResolveURL(ctx context.Context, rawURL string, opts ClientOptions) (*url.URL, netip.Addr, string, error) {
	u, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil {
		return nil, netip.Addr{}, "", fmt.Errorf("invalid URL: %w", err)
	}
	scheme := strings.ToLower(u.Scheme)
	if scheme != "http" && scheme != "https" {
		return nil, netip.Addr{}, "", fmt.Errorf("blocked URL scheme %q (only http and https allowed)", u.Scheme)
	}
	if u.User != nil {
		return nil, netip.Addr{}, "", errors.New("URL userinfo credentials are not permitted")
	}
	host := u.Hostname()
	if host == "" {
		return nil, netip.Addr{}, "", errors.New("URL missing host")
	}
	port := u.Port()
	if port == "" {
		if scheme == "https" {
			port = "443"
		} else {
			port = "80"
		}
	}

	// Reject explicit localhost / .local aliases early unless loopback explicitly opted in for tests.
	lowHost := strings.ToLower(strings.TrimSuffix(host, "."))
	if !opts.AllowLoopbackForTest {
		if lowHost == "localhost" || strings.HasSuffix(lowHost, ".localhost") || strings.HasSuffix(lowHost, ".local") || lowHost == "metadata.google.internal" {
			return nil, netip.Addr{}, "", fmt.Errorf("SSRF blocked: host %q is internal/loopback", host)
		}
	}

	if ipAddr, err := netip.ParseAddr(host); err == nil {
		clean := ipAddr.WithZone("").Unmap()
		if opts.AllowLoopbackForTest && clean.IsLoopback() {
			return u, clean, port, nil
		}
		if blocked, reason := IsBlockedAddr(clean); blocked {
			return nil, netip.Addr{}, "", fmt.Errorf("SSRF blocked %s: %s", clean, reason)
		}
		return u, clean, port, nil
	}

	res := opts.Resolver
	if res == nil {
		res = net.DefaultResolver
	}
	addrs, err := res.LookupIPAddr(ctx, host)
	if err != nil {
		return nil, netip.Addr{}, "", fmt.Errorf("DNS resolution failed for %s: %w", host, err)
	}
	if len(addrs) == 0 {
		return nil, netip.Addr{}, "", fmt.Errorf("DNS returned no IP addresses for %s", host)
	}

	var vetted []netip.Addr
	for _, a := range addrs {
		parsed, ok := netip.AddrFromSlice(a.IP)
		if !ok {
			return nil, netip.Addr{}, "", fmt.Errorf("SSRF blocked: invalid resolved address for %s", host)
		}
		clean := parsed.WithZone("").Unmap()
		if opts.AllowLoopbackForTest && clean.IsLoopback() {
			vetted = append(vetted, clean)
			continue
		}
		if blocked, reason := IsBlockedAddr(clean); blocked {
			return nil, netip.Addr{}, "", fmt.Errorf("SSRF blocked %s (%s resolved to %s)", reason, host, clean)
		}
		vetted = append(vetted, clean)
	}
	if len(vetted) == 0 {
		return nil, netip.Addr{}, "", fmt.Errorf("no valid public IPs for %s", host)
	}
	return u, vetted[0], port, nil
}

// NewSafeHTTPClient builds an http.Client that pins connections to the SSRF-vetted IP
// and re-validates every HTTP redirect hop so redirects or DNS rebinding cannot reach
// private, loopback, or cloud-metadata targets.
func NewSafeHTTPClient(opts ClientOptions) *http.Client {
	if opts.Timeout <= 0 {
		opts.Timeout = DefaultTimeout
	}
	maxRedir := opts.MaxRedirects
	if maxRedir <= 0 {
		maxRedir = DefaultMaxRedirects
	}

	dialer := &net.Dialer{Timeout: opts.Timeout}
	tr := &http.Transport{
		DisableKeepAlives: true,
		DialContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(addr)
			if err != nil {
				return nil, fmt.Errorf("invalid dial addr %q: %w", addr, err)
			}
			scheme := "http"
			if port == "443" {
				scheme = "https"
			}
			_, ip, _, err := ValidateAndResolveURL(ctx, scheme+"://"+net.JoinHostPort(host, port), opts)
			if err != nil {
				return nil, err
			}
			return dialer.DialContext(ctx, network, net.JoinHostPort(ip.String(), port))
		},
		DialTLSContext: func(ctx context.Context, network, addr string) (net.Conn, error) {
			host, port, err := net.SplitHostPort(addr)
			if err != nil {
				return nil, fmt.Errorf("invalid TLS dial addr %q: %w", addr, err)
			}
			_, ip, _, err := ValidateAndResolveURL(ctx, "https://"+net.JoinHostPort(host, port), opts)
			if err != nil {
				return nil, err
			}
			rawConn, err := dialer.DialContext(ctx, network, net.JoinHostPort(ip.String(), port))
			if err != nil {
				return nil, err
			}
			tlsConn := tls.Client(rawConn, &tls.Config{
				ServerName: host,
				MinVersion: tls.VersionTLS12,
			})
			if err := tlsConn.HandshakeContext(ctx); err != nil {
				_ = rawConn.Close()
				return nil, err
			}
			return tlsConn, nil
		},
	}

	return &http.Client{
		Timeout:   opts.Timeout,
		Transport: tr,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= maxRedir {
				return fmt.Errorf("stopped after %d redirects", maxRedir)
			}
			if _, _, _, err := ValidateAndResolveURL(req.Context(), req.URL.String(), opts); err != nil {
				return fmt.Errorf("redirect target blocked: %w", err)
			}
			return nil
		},
	}
}
