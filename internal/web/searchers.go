package web

import (
	"context"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	xhtml "golang.org/x/net/html"
)

const searchUserAgent = "Mozilla/5.0 (compatible; alpaca/1.2; +https://github.com/MdHussain121/alpaca)"

var errRateLimited = errors.New("rate limited (HTTP 429)")

type Searcher interface {
	Name() string
	Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error)
}

func searchGet(ctx context.Context, rawURL, accept string, opts ClientOptions) ([]byte, error) {
	client := NewSafeHTTPClient(opts)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", searchUserAgent)
	req.Header.Set("Accept", accept)
	req.Header.Set("Accept-Language", "en-US,en;q=0.8")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, errRateLimited
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("%s HTTP %d: %s", DomainOf(rawURL), resp.StatusCode, ShortError(string(body), 120))
	}
	return body, nil
}

func parseDoc(raw []byte) (*xhtml.Node, error) {
	return xhtml.Parse(strings.NewReader(string(raw)))
}

func nodeText(n *xhtml.Node) string {
	if n == nil {
		return ""
	}
	if n.Type == xhtml.TextNode {
		return html.UnescapeString(n.Data)
	}
	var b strings.Builder
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		b.WriteString(nodeText(c))
	}
	return b.String()
}

func nodeAttr(n *xhtml.Node, key string) string {
	for _, a := range n.Attr {
		if strings.EqualFold(a.Key, key) {
			return a.Val
		}
	}
	return ""
}

func hasClass(n *xhtml.Node, part string) bool {
	for _, c := range strings.Fields(nodeAttr(n, "class")) {
		if c == part {
			return true
		}
	}
	return false
}

func walkNodes(n *xhtml.Node, fn func(*xhtml.Node)) {
	if n == nil {
		return
	}
	fn(n)
	for c := n.FirstChild; c != nil; c = c.NextSibling {
		walkNodes(c, fn)
	}
}

func stripTags(s string) string {
	s = regexp.MustCompile(`(?s)<[^>]*>`).ReplaceAllString(s, " ")
	return strings.Join(strings.Fields(html.UnescapeString(s)), " ")
}

func unwrapDDG(href string) string {
	href = strings.TrimSpace(href)
	if strings.HasPrefix(href, "//") {
		href = "https:" + href
	}
	u, err := url.Parse(href)
	if err != nil {
		return ""
	}
	if strings.Contains(strings.ToLower(u.Host), "duckduckgo.com") && u.Path == "/l/" {
		if target := u.Query().Get("uddg"); target != "" {
			return target
		}
	}
	return href
}

type ddgSearcher struct{ opts ClientOptions }

func (d *ddgSearcher) Name() string { return "duckduckgo" }

func (d *ddgSearcher) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	endpoint := "https://html.duckduckgo.com/html/?q=" + url.QueryEscape(query)
	raw, err := searchGet(ctx, endpoint, "text/html,application/xhtml+xml", d.opts)
	if err != nil {
		return nil, err
	}
	return parseDDGResults(raw, maxResults)
}

func parseDDGResults(raw []byte, maxResults int) ([]SearchResult, error) {
	doc, err := parseDoc(raw)
	if err != nil {
		return nil, err
	}
	var links []*xhtml.Node
	var snippets []string
	walkNodes(doc, func(n *xhtml.Node) {
		if n.Type != xhtml.ElementNode {
			return
		}
		if n.Data == "a" && hasClass(n, "result__a") {
			links = append(links, n)
		}
		if hasClass(n, "result__snippet") {
			snippets = append(snippets, strings.Join(strings.Fields(nodeText(n)), " "))
		}
	})
	if len(links) == 0 {
		low := strings.ToLower(string(raw))
		if strings.Contains(low, "anomaly") || strings.Contains(low, "bots use duckduckgo") {
			return nil, errors.New("duckduckgo returned a bot challenge")
		}
		return nil, errors.New("duckduckgo returned no parseable results")
	}
	out := make([]SearchResult, 0, maxResults)
	for i, a := range links {
		if len(out) >= maxResults {
			break
		}
		target := unwrapDDG(nodeAttr(a, "href"))
		if target == "" {
			continue
		}
		snip := ""
		if i < len(snippets) {
			snip = snippets[i]
		}
		out = append(out, SearchResult{
			Title:   strings.Join(strings.Fields(nodeText(a)), " "),
			URL:     target,
			Snippet: snip,
			Domain:  DomainOf(target),
		})
	}
	if len(out) == 0 {
		return nil, errors.New("duckduckgo results had no usable links")
	}
	return out, nil
}

type mojeekSearcher struct{ opts ClientOptions }

func (m *mojeekSearcher) Name() string { return "mojeek" }

func (m *mojeekSearcher) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	endpoint := "https://www.mojeek.com/search?q=" + url.QueryEscape(query)
	raw, err := searchGet(ctx, endpoint, "text/html,application/xhtml+xml", m.opts)
	if err != nil {
		return nil, err
	}
	return parseMojeekResults(raw, maxResults)
}

func parseMojeekResults(raw []byte, maxResults int) ([]SearchResult, error) {
	doc, err := parseDoc(raw)
	if err != nil {
		return nil, err
	}
	var out []SearchResult
	seen := map[string]bool{}
	walkNodes(doc, func(n *xhtml.Node) {
		if len(out) >= maxResults || n.Type != xhtml.ElementNode || n.Data != "a" {
			return
		}
		href := strings.TrimSpace(nodeAttr(n, "href"))
		if !strings.HasPrefix(href, "http") {
			return
		}
		title := strings.Join(strings.Fields(nodeText(n)), " ")
		if len(title) < 4 || seen[href] {
			return
		}
		// Mojeek result links live inside li.result; skip navigation and
		// footer links by requiring a result-ish ancestor class.
		parent := n.Parent
		ok := false
		for depth := 0; parent != nil && depth < 3; depth++ {
			if strings.Contains(strings.ToLower(nodeAttr(parent, "class")), "result") {
				ok = true
				break
			}
			parent = parent.Parent
		}
		if !ok {
			return
		}
		seen[href] = true
		out = append(out, SearchResult{Title: title, URL: href, Domain: DomainOf(href)})
	})
	if len(out) == 0 {
		return nil, errors.New("mojeek returned no parseable results")
	}
	return out, nil
}

type wikipediaSearcher struct{ opts ClientOptions }

func (w *wikipediaSearcher) Name() string { return "wikipedia" }

func (w *wikipediaSearcher) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	endpoint := "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&utf8=1&srlimit=" +
		fmt.Sprint(maxResults) + "&srsearch=" + url.QueryEscape(query)
	raw, err := searchGet(ctx, endpoint, "application/json", w.opts)
	if err != nil {
		return nil, err
	}
	var payload struct {
		Query struct {
			Search []struct {
				Title   string `json:"title"`
				Snippet string `json:"snippet"`
			} `json:"search"`
		} `json:"query"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(payload.Query.Search))
	for _, r := range payload.Query.Search {
		u := "https://en.wikipedia.org/wiki/" + url.PathEscape(strings.ReplaceAll(r.Title, " ", "_"))
		out = append(out, SearchResult{Title: r.Title, URL: u, Snippet: stripTags(r.Snippet), Domain: "en.wikipedia.org"})
	}
	if len(out) == 0 {
		return nil, errors.New("wikipedia returned no results")
	}
	return out, nil
}

type wikidataSearcher struct{ opts ClientOptions }

func (w *wikidataSearcher) Name() string { return "wikidata" }

func (w *wikidataSearcher) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	endpoint := "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=en&limit=" +
		fmt.Sprint(maxResults) + "&search=" + url.QueryEscape(query)
	raw, err := searchGet(ctx, endpoint, "application/json", w.opts)
	if err != nil {
		return nil, err
	}
	var payload struct {
		Search []struct {
			ID          string `json:"id"`
			Label       string `json:"label"`
			Description string `json:"description"`
		} `json:"search"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(payload.Search))
	for _, r := range payload.Search {
		out = append(out, SearchResult{
			Title:   r.Label + " (" + r.ID + ")",
			URL:     "https://www.wikidata.org/wiki/" + r.ID,
			Snippet: r.Description,
			Domain:  "www.wikidata.org",
		})
	}
	if len(out) == 0 {
		return nil, errors.New("wikidata returned no results")
	}
	return out, nil
}

type hnSearcher struct{ opts ClientOptions }

func (h *hnSearcher) Name() string { return "hackernews" }

func (h *hnSearcher) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	endpoint := "https://hn.algolia.com/api/v1/search?tags=story&hitsPerPage=" + fmt.Sprint(maxResults) + "&query=" + url.QueryEscape(query)
	raw, err := searchGet(ctx, endpoint, "application/json", h.opts)
	if err != nil {
		return nil, err
	}
	var payload struct {
		Hits []struct {
			Title     string `json:"title"`
			URL       string `json:"url"`
			ObjectID  string `json:"objectID"`
			StoryText string `json:"story_text"`
		} `json:"hits"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(payload.Hits))
	for _, hit := range payload.Hits {
		u := hit.URL
		if u == "" {
			u = "https://news.ycombinator.com/item?id=" + hit.ObjectID
		}
		out = append(out, SearchResult{Title: hit.Title, URL: u, Snippet: stripTags(hit.StoryText), Domain: DomainOf(u)})
	}
	if len(out) == 0 {
		return nil, errors.New("hacker news returned no results")
	}
	return out, nil
}

type githubSearcher struct{ opts ClientOptions }

func (g *githubSearcher) Name() string { return "github" }

func (g *githubSearcher) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	endpoint := "https://api.github.com/search/repositories?sort=stars&order=desc&per_page=" + fmt.Sprint(maxResults) + "&q=" + url.QueryEscape(query)
	client := NewSafeHTTPClient(g.opts)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("User-Agent", searchUserAgent)
	req.Header.Set("Accept", "application/vnd.github+json")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, errRateLimited
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("github HTTP %d", resp.StatusCode)
	}
	var payload struct {
		Items []struct {
			FullName    string `json:"full_name"`
			HTMLURL     string `json:"html_url"`
			Description string `json:"description"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(payload.Items))
	for _, r := range payload.Items {
		out = append(out, SearchResult{Title: r.FullName, URL: r.HTMLURL, Snippet: r.Description, Domain: "github.com"})
	}
	if len(out) == 0 {
		return nil, errors.New("github returned no results")
	}
	return out, nil
}

type stackSearcher struct{ opts ClientOptions }

func (s *stackSearcher) Name() string { return "stackexchange" }

func (s *stackSearcher) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	endpoint := "https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance&site=stackoverflow&pagesize=" +
		fmt.Sprint(maxResults) + "&q=" + url.QueryEscape(query)
	raw, err := searchGet(ctx, endpoint, "application/json", s.opts)
	if err != nil {
		return nil, err
	}
	var payload struct {
		Items []struct {
			Title string `json:"title"`
			Link  string `json:"link"`
		} `json:"items"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(payload.Items))
	for _, r := range payload.Items {
		out = append(out, SearchResult{Title: stripTags(r.Title), URL: r.Link, Domain: "stackoverflow.com"})
	}
	if len(out) == 0 {
		return nil, errors.New("stackexchange returned no results")
	}
	return out, nil
}

type arxivSearcher struct{ opts ClientOptions }

func (a *arxivSearcher) Name() string { return "arxiv" }

func (a *arxivSearcher) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	endpoint := "https://export.arxiv.org/api/query?search_query=all:" + url.QueryEscape(query) + "&start=0&max_results=" + fmt.Sprint(maxResults)
	raw, err := searchGet(ctx, endpoint, "application/atom+xml", a.opts)
	if err != nil {
		return nil, err
	}
	var feed struct {
		Entries []struct {
			Title   string `xml:"title"`
			Summary string `xml:"summary"`
			ID      string `xml:"id"`
		} `xml:"entry"`
	}
	if err := xml.Unmarshal(raw, &feed); err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(feed.Entries))
	for _, e := range feed.Entries {
		out = append(out, SearchResult{
			Title:   strings.Join(strings.Fields(e.Title), " "),
			URL:     strings.TrimSpace(e.ID),
			Snippet: ShortError(strings.Join(strings.Fields(e.Summary), " "), 240),
			Domain:  "arxiv.org",
		})
	}
	if len(out) == 0 {
		return nil, errors.New("arxiv returned no results")
	}
	return out, nil
}

type npmSearcher struct{ opts ClientOptions }

func (n *npmSearcher) Name() string { return "npm" }

func (n *npmSearcher) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	endpoint := "https://registry.npmjs.org/-/v1/search?size=" + fmt.Sprint(maxResults) + "&text=" + url.QueryEscape(query)
	raw, err := searchGet(ctx, endpoint, "application/json", n.opts)
	if err != nil {
		return nil, err
	}
	var payload struct {
		Objects []struct {
			Package struct {
				Name        string `json:"name"`
				Description string `json:"description"`
				Links       struct {
					NPM string `json:"npm"`
				} `json:"links"`
			} `json:"package"`
		} `json:"objects"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(payload.Objects))
	for _, o := range payload.Objects {
		u := o.Package.Links.NPM
		if u == "" {
			u = "https://www.npmjs.com/package/" + o.Package.Name
		}
		out = append(out, SearchResult{Title: o.Package.Name, URL: u, Snippet: o.Package.Description, Domain: "www.npmjs.com"})
	}
	if len(out) == 0 {
		return nil, errors.New("npm returned no results")
	}
	return out, nil
}

type searxSearcher struct {
	base string
	opts ClientOptions
}

func (s *searxSearcher) Name() string { return "searxng" }

func (s *searxSearcher) Search(ctx context.Context, query string, maxResults int) ([]SearchResult, error) {
	endpoint := strings.TrimRight(s.base, "/") + "/search?format=json&q=" + url.QueryEscape(query)
	raw, err := searchGet(ctx, endpoint, "application/json", s.opts)
	if err != nil {
		return nil, err
	}
	var payload struct {
		Results []struct {
			Title   string `json:"title"`
			URL     string `json:"url"`
			Content string `json:"content"`
		} `json:"results"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	out := make([]SearchResult, 0, len(payload.Results))
	for _, r := range payload.Results {
		if len(out) >= maxResults {
			break
		}
		out = append(out, SearchResult{Title: r.Title, URL: r.URL, Snippet: r.Content, Domain: DomainOf(r.URL)})
	}
	if len(out) == 0 {
		return nil, errors.New("searxng returned no results")
	}
	return out, nil
}

var _ = time.Second
