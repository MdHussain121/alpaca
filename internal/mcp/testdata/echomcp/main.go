// echomcp is a standalone stdio MCP server binary used for integration tests.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

type rpcMsg struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
}

func main() {
	manyTools := flag.Int("many", 0, "number of extra tools to expose")
	crashOnceFile := flag.String("crash-once-file", "", "if set and missing, create file and exit non-zero once")
	crashAlways := flag.Bool("crash-always", false, "exit non-zero immediately")
	crashAfterMs := flag.Int("crash-after-ms", 0, "exit non-zero N ms after startup")
	spawnChild := flag.Bool("spawn-child", false, "spawn a long-running child process to verify Job Object tree termination")
	childPidFile := flag.String("child-pid-file", "", "write spawned child PID to this path")
	flag.Parse()

	if *crashAlways {
		os.Stderr.WriteString("echomcp fatal crash-always " + os.Getenv("MCP_SECRET") + "\n")
		os.Exit(2)
	}
	if *crashAfterMs > 0 {
		go func() {
			time.Sleep(time.Duration(*crashAfterMs) * time.Millisecond)
			os.Stderr.WriteString("echomcp delayed crash " + os.Getenv("MCP_SECRET") + "\n")
			os.Exit(3)
		}()
	}
	if *crashOnceFile != "" {
		if _, err := os.Stat(*crashOnceFile); os.IsNotExist(err) {
			_ = os.WriteFile(*crashOnceFile, []byte("crashed"), 0o644)
			os.Stderr.WriteString("echomcp simulated crash once " + os.Getenv("MCP_SECRET") + "\n")
			os.Exit(1)
		}
	}

	var childCmd *exec.Cmd
	if *spawnChild {
		// Spawn a quiescent child process (ping localhost repeatedly on Windows, sleep on Unix)
		childCmd = exec.Command("cmd.exe", "/c", "ping -n 60 127.0.0.1 >NUL")
		if err := childCmd.Start(); err == nil && childCmd.Process != nil {
			if *childPidFile != "" {
				_ = os.WriteFile(*childPidFile, []byte(strconv.Itoa(childCmd.Process.Pid)), 0o644)
			}
		}
	}
	defer func() {
		if childCmd != nil && childCmd.Process != nil {
			_ = childCmd.Process.Kill()
		}
	}()

	r := bufio.NewReaderSize(os.Stdin, 256*1024)
	enc := json.NewEncoder(os.Stdout)

	for {
		line, err := r.ReadString('\n')
		if len(strings.TrimSpace(line)) == 0 && err != nil {
			return
		}
		line = strings.TrimSpace(line)
		if line == "" {
			if err != nil {
				return
			}
			continue
		}
		var req rpcMsg
		if json.Unmarshal([]byte(line), &req) != nil {
			continue
		}
		if len(req.ID) == 0 || string(req.ID) == "null" {
			// Notification (e.g. notifications/initialized)
			continue
		}

		switch req.Method {
		case "initialize":
			_ = enc.Encode(map[string]any{
				"jsonrpc": "2.0",
				"id":      req.ID,
				"result": map[string]any{
					"protocolVersion": "2024-11-05",
					"capabilities":    map[string]any{"tools": map[string]any{}},
					"serverInfo":      map[string]string{"name": "echomcp", "version": "1.0.0"},
				},
			})

		case "tools/list":
			tools := []map[string]any{
				{
					"name":        "echo",
					"description": "Echo back input text verbatim.",
					"inputSchema": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"text": map[string]any{"type": "string", "description": "Text to echo back"},
						},
						"required": []string{"text"},
					},
				},
				{
					"name":        "add",
					"description": "Add two integers a and b.",
					"inputSchema": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"a": map[string]any{"type": "number"},
							"b": map[string]any{"type": "number"},
						},
						"required": []string{"a", "b"},
					},
				},
				{
					"name":        "sleep",
					"description": "Sleep ms milliseconds (for timeout testing).",
					"inputSchema": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"ms": map[string]any{"type": "number"},
						},
						"required": []string{"ms"},
					},
				},
				{
					"name":        "env_probe",
					"description": "Return configured MCP_SECRET value (for redaction test).",
					"inputSchema": map[string]any{
						"type":       "object",
						"properties": map[string]any{},
					},
				},
			}
			for i := 1; i <= *manyTools; i++ {
				tools = append(tools, map[string]any{
					"name":        fmt.Sprintf("extra_%d", i),
					"description": fmt.Sprintf("Extra synthetic tool #%d for schema token accounting.", i),
					"inputSchema": map[string]any{
						"type": "object",
						"properties": map[string]any{
							"input": map[string]any{"type": "string"},
						},
					},
				})
			}
			_ = enc.Encode(map[string]any{
				"jsonrpc": "2.0",
				"id":      req.ID,
				"result":  map[string]any{"tools": tools},
			})

		case "tools/call":
			var p struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			_ = json.Unmarshal(req.Params, &p)
			var text string
			var isErr bool
			switch p.Name {
			case "echo":
				v, _ := p.Arguments["text"].(string)
				text = "echomcp: " + v
			case "add":
				a, _ := p.Arguments["a"].(float64)
				b, _ := p.Arguments["b"].(float64)
				text = fmt.Sprintf("%.0f", a+b)
			case "sleep":
				ms, _ := p.Arguments["ms"].(float64)
				if ms > 0 {
					time.Sleep(time.Duration(ms) * time.Millisecond)
				}
				text = fmt.Sprintf("slept %.0f ms", ms)
			case "env_probe":
				text = "secret=" + os.Getenv("MCP_SECRET")
			default:
				if strings.HasPrefix(p.Name, "extra_") {
					v, _ := p.Arguments["input"].(string)
					text = p.Name + ": " + v
				} else {
					isErr = true
					text = "unknown tool: " + p.Name
				}
			}
			_ = enc.Encode(map[string]any{
				"jsonrpc": "2.0",
				"id":      req.ID,
				"result": map[string]any{
					"content": []map[string]any{{"type": "text", "text": text}},
					"isError": isErr,
				},
			})

		default:
			_ = enc.Encode(map[string]any{
				"jsonrpc": "2.0",
				"id":      req.ID,
				"error":   map[string]any{"code": -32601, "message": "method not found"},
			})
		}

		if err != nil {
			return
		}
	}
}
