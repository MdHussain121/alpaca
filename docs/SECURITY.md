# Security Model

`alpaca` runs local models and local coding-agent tools on your machine. It enforces workspace isolation, explicit permission gates, secret redaction, and safe subprocess execution.

## 1. Workspace & Path Boundaries

- All file tools (`list_dir`, `glob`, `grep`, `read_file`, `edit_file`, `write_file`, `make_dir`) canonicalize paths (including symlinks and Windows junctions via `EvalSymlinks`) and refuse reads or writes outside the active workspace directory.
- Deleting files outside Trash or workspace boundaries is blocked, and reversible file checkpoints are stored under `.alpaca/checkpoints/` before modifications.
- Copying `.gguf` models onto FAT32 drives (`> 4 GiB` limit) or read-only destinations is rejected upfront before transferring bytes.

## 2. Permission Modes (`/mode`)

| Mode | Read Tools | Edit/Write/mkdir | Shell (`run_command`) | Web (`web_search`/`web_fetch`) | MCP Tools |
| --- | --- | --- | --- | --- | --- |
| `ask` | Ask | Ask | Ask | Ask | Always Ask |
| `read-auto` *(default)* | Auto | Ask | Ask | Ask | Always Ask |
| `workspace-auto` | Auto | Auto | Ask | Ask | Always Ask |

- **`run_command` and MCP tools (`server.tool`) always prompt for confirmation** unless explicitly allowlisted for the session by the user.
- Prompt approvals never silently elevate permissions from tool output.

## 3. Web Content & Prompt Injection Defense

- `web_fetch` validates outbound URLs against SSRF targets: private, loopback, link-local, metadata (`169.254.169.254`), and non-HTTP/HTTPS schemes are blocked, including DNS-rebind checks and redirect re-validation.
- Fetched web bodies are wrapped in untrusted data delimiters and marked tainted; built-in system instructions explicitly instruct the model to treat web content as data rather than orders.

## 4. Secrets & Redaction

- API keys and configured environment tokens (e.g., `mcp.toml` server `env`, web search API keys) are redacted in chat logs, transcripts, session exports, and `/mcp` diagnostics.
- MCP subprocesses inherit a minimal filtered environment instead of dumping arbitrary user environment variables.
- Built-in system rules instruct the model never to embed secrets into files or commands.

## 5. Binary Verification

- Always verify downloaded release archives against `SHA256SUMS.txt`:

```powershell
Get-FileHash .\alpaca-v1.0.0-win-x64.zip -Algorithm SHA256
```

- Because binaries are unsigned open-source builds, Windows SmartScreen may display a prompt on first launch; verify the SHA-256 hash matches `SHA256SUMS.txt` before allowing execution.
