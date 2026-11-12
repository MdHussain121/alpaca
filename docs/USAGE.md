# Using alpaca

`alpaca` is a portable terminal launcher, VRAM/RAM estimator, and coding-agent chat client for local GGUF models powered by `llama.cpp` (`llama-server.exe`).

## Quick Start

1. Place one or more `.gguf` files next to `alpaca.exe` (or configure `models_dir` / `extra_models` in `config.toml`, or press `o` in the launcher).
2. Run `alpaca.exe` from Windows Terminal, PowerShell, or Command Prompt.
3. Inspect the live VRAM/RAM fit gauge for the selected model, adjust sliders with `←` / `→`, and press `Enter` to start `llama-server.exe`.
4. Switch between **Chat**, **Metrics**, and **Logs** (`Tab`) or type `/help` in chat.

## CLI Commands & Subcommands

Alpaca can be launched interactively or invoked for benchmarks and model evaluations:

```powershell
# Interactive launcher and coding assistant (Bubbletea TUI)
.\alpaca.exe

# Run model speed, throughput, and latency benchmarks
.\alpaca.exe bench --model models\my-model.gguf

# Run 10-task tool calling accuracy evaluation & update launcher badges
.\alpaca.exe eval

# CLI options
.\alpaca.exe --version
.\alpaca.exe --help
```

For complete benchmarking instructions, see [BENCHMARKS.md](BENCHMARKS.md).

## CLI Flags

- `--version`: Print version, commit, and bundled `llama.cpp` build tag.
- `--ascii`: Force ASCII borders and progress bars instead of UTF-8 box drawing (also controlled by `ALPACA_ASCII=1` or `ascii = true` in `config.toml`).
- `--models-dir <dir>`: Scan an explicit models folder on launch (`ALPACA_MODELS_DIR` override also supported).

## Launcher Keyboard Shortcuts

| Key | Action |
| --- | --- |
| `↑` / `↓`, `k` / `j` | Move model selection or setting field |
| `←` / `→`, `h` / `l` | Adjust numeric/enum setting (GPU offload, context, KV quant, reasoning) |
| `Tab` | Switch focus between Models list and Settings panel |
| `Enter` | Launch selected model or edit/toggle focused setting |
| `/` | Filter model list (Unicode-safe backspace) |
| `o` | Open an extra `.gguf` path or change models folder |
| `p` / `s` | Open presets modal (`r` rename, `d` delete) / save preset |
| `c` / `C` | Copy `llama-server.exe` CLI command / copy model to local drive (checks FAT32 4 GiB limit & disk space) |
| `t` | Cycle TUI color theme |
| `?` / `q` | Toggle help modal / quit |

## Chat Slash Commands

- `/help`: Show available chat commands.
- `/system [show|off|on|reset|clear|<text>]`: Inspect the rendered system prompt, toggle the built-in tool/skills/MCP instruction template (`off`/`on`), add a custom instruction suffix, or `reset` back to defaults.
- `/context`, `/context compact`, `/context budget`: Inspect live per-category token breakdown (`System`, `AGENTS.md`, `Tool defs`, `Skills`, `History`, KV cache savings) or force conversation compaction.
- `/mode [ask|read-auto|workspace-auto]`: Switch tool execution permission policy.
- `/workspace [<dir>]`: Show or change the active workspace directory (also loads workspace `AGENTS.md`).
- `/skills`, `/skill <name>`: Manage, reload, inspect, enable/disable, or delete Agent Skills.
- `/mcp`, `/tools`: Inspect configured MCP servers, restart crashed servers, or toggle individual tools.
- `/session [list|load <id>|branch <id>|export]`: Save, resume, branch (`/edit`, `/regen`), or export conversations as Markdown.
- `/web [on|off]`: Inspect or configure web search and fetch status.
- `/compact`: Toggle compact response density or trigger compaction.
- `/effort`, `/thinking`: Toggle reasoning effort / visible thinking panel.
- `/profile [<name>]`: Switch sampling profile (`coding`, `precise`, `creative`, or custom profiles from `config.toml`).
- `/bell`, `/numbers`: Toggle long-generation terminal bell or code-block line numbers.
- `/restart`, `/log`, `/clear`: Restart `llama-server.exe`, view server logs, or clear chat history.

## Built-In System Prompt

By default, every agent conversation includes a runtime-interpolated instruction prompt tailored to your active workspace, permission mode, enabled tools, loaded skills, MCP servers, `AGENTS.md`, and context size:

- **Full mode** (`>= 8192` context): Detailed tool-usage rules (read before edit, safe shell commands, web untrusted-content boundary, `use_skill`/`save_skill`, MCP usage).
- **Compact mode** (`< 8192` context, ~150 tokens): Token-frugal variant for smaller context windows.
- **On-disk customization**: Create `prompts/system.md` (or `prompts/system-compact.md`) alongside `alpaca.exe` to customize the template, or run `/system off` / set `disable_system_prompt = true` in `config.toml`.

## Configuration (`config.toml`)

Copy [`config.example.toml`](../config.example.toml) to `config.toml` next to `alpaca.exe`. Legacy `.llm` / `pllama.toml` settings migrate automatically on first launch if present.
