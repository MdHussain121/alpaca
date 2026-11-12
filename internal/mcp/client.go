package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/MdHussain121/alpaca/internal/runner"
)

const mcpProtocolVersion = "2024-11-05"

// ToolSchema describes one tool exposed by an MCP server from tools/list.
type ToolSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"inputSchema"`
}

// ClientOptions configures spawning a single stdio MCP server process.
type ClientOptions struct {
	Name    string
	RootDir string
	WorkDir string
	Config  ServerConfig
	OnExit  func(err error)
}

type rpcRequest struct {
	JSONRPC string `json:"jsonrpc"`
	ID      *int64 `json:"id,omitempty"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

type ringBuf struct {
	mu   sync.Mutex
	max  int
	data []byte
}

func (b *ringBuf) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.data = append(b.data, p...)
	if b.max > 0 && len(b.data) > b.max {
		b.data = append([]byte(nil), b.data[len(b.data)-b.max:]...)
	}
	return len(p), nil
}

func (b *ringBuf) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return strings.TrimSpace(string(b.data))
}

// Client owns one stdio MCP server child process, Job Object, and JSON-RPC 2.0 session.
type Client struct {
	opts    ClientOptions
	cmd     *exec.Cmd
	job     *runner.JobHandle
	stdin   io.WriteCloser
	stderr  *ringBuf
	writeMu sync.Mutex
	pendMu  sync.Mutex
	pending map[int64]chan rpcResponse
	nextID  atomic.Int64
	tools   []ToolSchema
	closed  atomic.Bool
	doneCh  chan struct{}
	waitErr error
	closeMu sync.Mutex
}

// StartClient launches the MCP server process, binds it to the Windows Job Object,
// completes the initialize + initialized handshake, and lists all tools.
func StartClient(ctx context.Context, opts ClientOptions) (*Client, error) {
	if strings.TrimSpace(opts.Config.Command) == "" {
		return nil, errors.New("mcp server command cannot be empty")
	}
	resolved := ResolveCommand(opts.RootDir, opts.Config.Command)
	cmd := exec.Command(resolved, opts.Config.Args...)
	if opts.WorkDir != "" {
		cmd.Dir = opts.WorkDir
	} else if opts.RootDir != "" {
		cmd.Dir = opts.RootDir
	}
	cmd.Env = buildMergedEnv(opts.Config.Env)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("stdout pipe: %w", err)
	}
	stderrBuf := &ringBuf{max: 4096}
	cmd.Stderr = stderrBuf

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("start %s: %w", opts.Name, err)
	}

	var job *runner.JobHandle
	if cmd.Process != nil {
		if j, jErr := runner.AttachJob(uint32(cmd.Process.Pid)); jErr == nil {
			job = j
		}
	}

	c := &Client{
		opts:    opts,
		cmd:     cmd,
		job:     job,
		stdin:   stdin,
		stderr:  stderrBuf,
		pending: make(map[int64]chan rpcResponse),
		doneCh:  make(chan struct{}),
	}

	go c.readLoop(stdout)
	go c.waitLoop()

	hsCtx, cancel := context.WithTimeout(ctx, minDuration(opts.Config.Timeout(), 15*time.Second))
	defer cancel()
	if err := c.handshakeAndList(hsCtx); err != nil {
		stderrMsg := RedactSecrets(opts.Config.Env, stderrBuf.String())
		_ = c.Close()
		if stderrMsg != "" {
			return nil, fmt.Errorf("%w (stderr: %s)", err, stderrMsg)
		}
		return nil, err
	}
	return c, nil
}

func minDuration(a, b time.Duration) time.Duration {
	if a <= 0 || (b > 0 && b < a) {
		return b
	}
	return a
}

func buildMergedEnv(overrides map[string]string) []string {
	base := os.Environ()
	if len(overrides) == 0 {
		return base
	}
	idxByUpper := make(map[string]int, len(base))
	out := make([]string, 0, len(base)+len(overrides))
	for _, entry := range base {
		eq := strings.IndexByte(entry, '=')
		if eq <= 0 {
			continue
		}
		k := strings.ToUpper(entry[:eq])
		idxByUpper[k] = len(out)
		out = append(out, entry)
	}
	for k, v := range overrides {
		if strings.TrimSpace(k) == "" {
			continue
		}
		kv := k + "=" + v
		if idx, ok := idxByUpper[strings.ToUpper(k)]; ok {
			out[idx] = kv
		} else {
			idxByUpper[strings.ToUpper(k)] = len(out)
			out = append(out, kv)
		}
	}
	return out
}

func (c *Client) Tools() []ToolSchema {
	out := make([]ToolSchema, len(c.tools))
	copy(out, c.tools)
	return out
}

func (c *Client) Done() <-chan struct{} {
	return c.doneCh
}

func (c *Client) handshakeAndList(ctx context.Context) error {
	initParams := map[string]any{
		"protocolVersion": mcpProtocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo": map[string]string{
			"name":    "alpaca",
			"version": "1.0.0",
		},
	}
	if _, err := c.call(ctx, "initialize", initParams); err != nil {
		return fmt.Errorf("mcp initialize failed: %w", err)
	}
	if err := c.notify("notifications/initialized", map[string]any{}); err != nil {
		return fmt.Errorf("mcp notifications/initialized failed: %w", err)
	}
	tools, err := c.fetchAllTools(ctx)
	if err != nil {
		return fmt.Errorf("mcp tools/list failed: %w", err)
	}
	c.tools = tools
	return nil
}

func (c *Client) fetchAllTools(ctx context.Context) ([]ToolSchema, error) {
	var all []ToolSchema
	cursor := ""
	for page := 0; page < 32; page++ {
		var params any = map[string]any{}
		if cursor != "" {
			params = map[string]any{"cursor": cursor}
		}
		raw, err := c.call(ctx, "tools/list", params)
		if err != nil {
			return nil, err
		}
		var parsed struct {
			Tools      []ToolSchema `json:"tools"`
			NextCursor string       `json:"nextCursor"`
		}
		if err := json.Unmarshal(raw, &parsed); err != nil {
			return nil, fmt.Errorf("decode tools/list: %w", err)
		}
		for _, t := range parsed.Tools {
			if strings.TrimSpace(t.Name) == "" {
				continue
			}
			if t.InputSchema == nil {
				t.InputSchema = map[string]any{"type": "object", "properties": map[string]any{}}
			}
			all = append(all, t)
		}
		if strings.TrimSpace(parsed.NextCursor) == "" {
			break
		}
		cursor = parsed.NextCursor
	}
	return all, nil
}

// CallTool invokes a named tool with per-call timeout enforcement.
func (c *Client) CallTool(ctx context.Context, name string, args map[string]any) (string, bool, error) {
	if c.closed.Load() {
		return "", true, errors.New("mcp client is closed")
	}
	if args == nil {
		args = map[string]any{}
	}
	timeout := c.opts.Config.Timeout()
	callCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	raw, err := c.call(callCtx, "tools/call", map[string]any{
		"name":      name,
		"arguments": args,
	})
	if err != nil {
		return "", true, errors.New(RedactSecrets(c.opts.Config.Env, err.Error()))
	}
	var res struct {
		Content []struct {
			Type     string `json:"type"`
			Text     string `json:"text"`
			Data     string `json:"data"`
			MimeType string `json:"mimeType"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(raw, &res); err != nil {
		txt := RedactSecrets(c.opts.Config.Env, string(raw))
		return txt, false, nil
	}
	var parts []string
	for _, item := range res.Content {
		switch {
		case item.Text != "":
			parts = append(parts, item.Text)
		case item.Type == "image":
			parts = append(parts, fmt.Sprintf("[image %s (%d bytes)]", item.MimeType, len(item.Data)))
		case item.Type != "":
			parts = append(parts, fmt.Sprintf("[%s]", item.Type))
		}
	}
	out := RedactSecrets(c.opts.Config.Env, strings.Join(parts, "\n"))
	if res.IsError {
		if out == "" {
			out = "mcp tool returned error"
		}
		return out, true, errors.New(out)
	}
	if out == "" {
		out = "(ok)"
	}
	return out, false, nil
}

func (c *Client) notify(method string, params any) error {
	if c.closed.Load() {
		return errors.New("mcp client closed")
	}
	req := rpcRequest{
		JSONRPC: "2.0",
		Method:  method,
		Params:  params,
	}
	b, err := json.Marshal(req)
	if err != nil {
		return err
	}
	b = append(b, '\n')
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	_, err = c.stdin.Write(b)
	return err
}

func (c *Client) call(ctx context.Context, method string, params any) (json.RawMessage, error) {
	if c.closed.Load() {
		return nil, errors.New("mcp client closed")
	}
	id := c.nextID.Add(1)
	ch := make(chan rpcResponse, 1)

	c.pendMu.Lock()
	if c.closed.Load() {
		c.pendMu.Unlock()
		return nil, errors.New("mcp client closed")
	}
	c.pending[id] = ch
	c.pendMu.Unlock()

	defer func() {
		c.pendMu.Lock()
		delete(c.pending, id)
		c.pendMu.Unlock()
	}()

	req := rpcRequest{
		JSONRPC: "2.0",
		ID:      &id,
		Method:  method,
		Params:  params,
	}
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	b = append(b, '\n')

	c.writeMu.Lock()
	_, wErr := c.stdin.Write(b)
	c.writeMu.Unlock()
	if wErr != nil {
		return nil, fmt.Errorf("write %s: %w", method, wErr)
	}

	select {
	case resp, ok := <-ch:
		if !ok {
			return nil, fmt.Errorf("mcp server closed while waiting for %s", method)
		}
		if resp.Error != nil {
			return nil, fmt.Errorf("rpc error %d: %s", resp.Error.Code, resp.Error.Message)
		}
		return resp.Result, nil
	case <-c.doneCh:
		return nil, fmt.Errorf("mcp server exited during %s", method)
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (c *Client) readLoop(stdout io.Reader) {
	r := bufio.NewReaderSize(stdout, 64*1024)
	for {
		payload, err := readMCPMessage(r)
		if err != nil {
			return
		}
		if len(payload) == 0 {
			continue
		}
		var msg rpcResponse
		if err := json.Unmarshal(payload, &msg); err != nil {
			continue
		}
		if len(msg.ID) == 0 || string(msg.ID) == "null" {
			// Server notification or log message; ignore cleanly.
			continue
		}
		var id int64
		if err := json.Unmarshal(msg.ID, &id); err != nil {
			continue
		}
		c.pendMu.Lock()
		ch := c.pending[id]
		if ch != nil {
			delete(c.pending, id)
		}
		c.pendMu.Unlock()
		if ch != nil {
			ch <- msg
		}
	}
}

func readMCPMessage(r *bufio.Reader) ([]byte, error) {
	for {
		line, err := r.ReadString('\n')
		if err != nil && len(line) == 0 {
			return nil, err
		}
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			if err != nil {
				return nil, err
			}
			continue
		}
		if strings.HasPrefix(strings.ToLower(trimmed), "content-length:") {
			valStr := strings.TrimSpace(trimmed[len("content-length:"):])
			n, parseErr := strconv.Atoi(valStr)
			if parseErr != nil || n <= 0 || n > 16*1024*1024 {
				continue
			}
			// Drain header terminator blank line.
			for {
				hdr, hErr := r.ReadString('\n')
				if hErr != nil && len(hdr) == 0 {
					return nil, hErr
				}
				if strings.TrimSpace(hdr) == "" {
					break
				}
			}
			buf := make([]byte, n)
			if _, rErr := io.ReadFull(r, buf); rErr != nil {
				return nil, rErr
			}
			return buf, nil
		}
		return []byte(trimmed), err
	}
}

func (c *Client) waitLoop() {
	err := c.cmd.Wait()
	c.waitErr = err
	c.failPending()
	close(c.doneCh)
	if !c.closed.Load() && c.opts.OnExit != nil {
		c.opts.OnExit(err)
	}
}

func (c *Client) failPending() {
	c.pendMu.Lock()
	for id, ch := range c.pending {
		close(ch)
		delete(c.pending, id)
	}
	c.pendMu.Unlock()
}

// Close shuts down the child MCP server and terminates its Job Object cleanly.
func (c *Client) Close() error {
	c.closeMu.Lock()
	defer c.closeMu.Unlock()

	if c.closed.Swap(true) {
		<-c.doneCh
		return nil
	}
	c.failPending()
	if c.stdin != nil {
		_ = c.stdin.Close()
	}
	select {
	case <-c.doneCh:
	case <-time.After(250 * time.Millisecond):
		if c.job != nil {
			_ = c.job.Terminate()
		} else if c.cmd.Process != nil {
			_ = c.cmd.Process.Kill()
		}
		select {
		case <-c.doneCh:
		case <-time.After(1 * time.Second):
		}
	}
	if c.job != nil {
		_ = c.job.Terminate()
		c.job = nil
	}
	return nil
}
