# Benchmarks & Tool Recommendation Evaluation

Alpaca provides built-in tools for evaluating local LLM performance and tool-calling capabilities. Users can run these benchmarks directly from the command line using the single portable `alpaca.exe` binary.

---

## 1. Tool Recommendation Evaluation (`alpaca eval`)

The evaluation harness tests local GGUF models against a 10-task suite designed to measure real-world tool execution, file editing, shell command reactivity, skill creation, and workspace permission boundaries.

### How UI Badges Work

The launcher TUI automatically displays data-driven badges next to models in the model picker:
- **`tools:recommended (80%–100%)`** (green): Model reliably understands tool calling syntax, argument formatting, and error recovery.
- **`tools:works with limits (50%–79%)`** (yellow): Model executes simple tools but may need manual guidance on multi-step workflows.
- **`tools:not-recommended (<50%)`** (red): Model struggles with tool calling schemas and should be used primarily for plain text and coding assistance.

### Running the Evaluation

To evaluate all models in your `models/` folder:
```powershell
.\alpaca.exe eval
```

#### Fast Single-Pass Test
By default, each task is attempted 3 times per model to verify consistency. For a quick single-pass check:
```powershell
.\alpaca.exe eval --attempts 1
```

#### Evaluate a Specific Model
Filter by substring in the model filename:
```powershell
.\alpaca.exe eval --filter Qwen3.5
```

#### Custom Models or Server
```powershell
.\alpaca.exe eval --models-dir custom_models --server bin\cuda\llama-server.exe
```

When evaluation finishes, Alpaca writes `eval_ratings.json`. The next time you open the launcher UI (`alpaca.exe`), your evaluated models will display their updated tool recommendation badges automatically.

---

## 2. Performance & Speed Benchmarks (`alpaca bench`)

The benchmark harness tests inference speed, time-to-first-token (TTFT), memory usage, and consistency across prompts.

### What is Measured
- **Ready Time**: Total milliseconds from process launch until HTTP `/health` is ready.
- **TTFT (Time to First Token)**: Latency from HTTP request to the first generated token.
- **Prefill Speed**: Prompt processing throughput in tokens per second.
- **Decode Speed**: Autoregressive generation throughput in tokens per second.
- **Memory Footprint**: Process working set (RSS), private commit, and GPU VRAM usage.
- **Consistency**: Output bundle hash across repeated runs (verifies deterministic sampling).

### Running a Benchmark

Benchmark a model with default settings:
```powershell
.\alpaca.exe bench --model models\Spark-X2.6-4B-abliterated-Q4_K_M.gguf
```

#### Repeated Runs with CSV Export
Run 3 repeats and append summary rows to a CSV file:
```powershell
.\alpaca.exe bench --model models\model.gguf --repeats 3 --out results.csv
```

#### Prompt Filter (Short, Medium, Long)
Test only short prompts (latency focus) or long prompts (prefill throughput focus):
```powershell
.\alpaca.exe bench --model models\model.gguf --filter short
.\alpaca.exe bench --model models\model.gguf --filter long
```

#### Needle-in-a-Haystack Retrieval Test
Test long-context retrieval accuracy at 16K tokens:
```powershell
.\alpaca.exe bench --model models\model.gguf --needle-ctx 16384
```

#### KV Cache Prefix Hit Test (2-Turn Conversation)
```powershell
.\alpaca.exe bench --model models\model.gguf --warm-repeat
```

---

## 3. Standalone Utilities

If you prefer separate executables, standalone tools are also provided:
- `alpaca-bench.exe`: Standalone benchmark runner.
- `alpaca-eval.exe`: Standalone tool evaluation runner.

Both support identical command-line flags and options.
