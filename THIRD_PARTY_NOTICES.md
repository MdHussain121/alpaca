# Third-Party Notices

`alpaca` (`github.com/MdHussain121/alpaca`) is licensed under the MIT License (see [`LICENSE`](LICENSE)).
Official binary distributions include or link the third-party components listed below.

---

## 1. llama.cpp / ggml (`llama-server.exe`, `llama-bench.exe`)

- **Upstream:** https://github.com/ggml-org/llama.cpp (`b12284`, commit `9e9619d030bbe6b584d826a0a6cf06e38251f52b`)
- **License:** MIT License

```text
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 2. NVIDIA CUDA Toolkit Redistributables (`alpaca-*-win-x64-cuda.zip`)

The CUDA release archive bundles **`cublas64_13.dll`** and **`cublasLt64_13.dll`** from NVIDIA CUDA Toolkit `13.2`, which are listed as redistributable files under **Attachment A** of the NVIDIA CUDA Toolkit End User License Agreement (`EULA.txt`). Redistribution complies with the NVIDIA CUDA Toolkit EULA and includes the following third-party notices required by Sections 6 and 7 of the NVIDIA CUDA Toolkit license notice:

### Vasily Volkov cuBLAS Routines (Modified BSD)

```text
Copyright (c) 2007-2009, Regents of the University of California

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of the University of California, Berkeley nor
      the names of its contributors may be used to endorse or promote
      products derived from this software without specific prior
      written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

### Davide Barbieri cuBLAS Routines (Modified BSD)

```text
Copyright (c) 2008-2009 Davide Barbieri @ University of Rome Tor Vergata.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.
    * Neither the name of the University of Rome Tor Vergata nor the
      names of its contributors may be used to endorse or promote
      products derived from this software without specific prior
      written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
```

---

## 3. Go Module Dependencies (`go.mod`)

All Go dependencies linked into `alpaca.exe`:

| Module | Version | License |
| --- | --- | --- |
| `github.com/BurntSushi/toml` | `v1.6.0` | MIT License |
| `github.com/alecthomas/chroma/v2` | `v2.27.0` | MIT License |
| `github.com/atotto/clipboard` | `v0.1.4` | BSD-3-Clause |
| `github.com/aymanbagabas/go-osc52/v2` | `v2.0.1` | MIT License |
| `github.com/aymanbagabas/go-udiff` | `v0.3.1` | BSD-3-Clause (Go Authors) |
| `github.com/charmbracelet/bubbles` | `v1.0.0` | MIT License |
| `github.com/charmbracelet/bubbletea` | `v1.3.10` | MIT License |
| `github.com/charmbracelet/colorprofile` | `v0.4.1` | MIT License |
| `github.com/charmbracelet/harmonica` | `v0.2.0` | MIT License |
| `github.com/charmbracelet/lipgloss` | `v1.1.0` | MIT License |
| `github.com/charmbracelet/x/ansi` | `v0.11.6` | MIT License |
| `github.com/charmbracelet/x/cellbuf` | `v0.0.15` | MIT License |
| `github.com/charmbracelet/x/exp/golden` | `v0.0.0-20241011142426-46044092ad91` | MIT License |
| `github.com/charmbracelet/x/exp/teatest` | `v0.0.0-20261101151554-27ecf3e4d454` | MIT License |
| `github.com/charmbracelet/x/term` | `v0.2.2` | MIT License |
| `github.com/clipperhouse/displaywidth` | `v0.9.0` | MIT License |
| `github.com/clipperhouse/stringish` | `v0.1.1` | MIT License |
| `github.com/clipperhouse/uax29/v2` | `v2.5.0` | MIT License |
| `github.com/dlclark/regexp2/v2` | `v2.2.1` | MIT License |
| `github.com/erikgeiser/coninput` | `v0.0.0-20211004153227-1c3628e74d0f` | MIT License |
| `github.com/lucasb-eyer/go-colorful` | `v1.3.0` | MIT License |
| `github.com/mattn/go-isatty` | `v0.0.20` | MIT License |
| `github.com/mattn/go-localereader` | `v0.0.1` | MIT License |
| `github.com/mattn/go-runewidth` | `v0.0.19` | MIT License |
| `github.com/muesli/ansi` | `v0.0.0-20230316100256-276c6243b2f6` | MIT License |
| `github.com/muesli/cancelreader` | `v0.2.2` | MIT License |
| `github.com/muesli/termenv` | `v0.16.0` | MIT License |
| `github.com/rivo/uniseg` | `v0.4.7` | MIT License |
| `github.com/xo/terminfo` | `v0.0.0-20220910002029-abceb7e1c41e` | MIT License |
| `golang.org/x/net` | `v0.62.0` | BSD-3-Clause (Go Authors) |
| `golang.org/x/sys` | `v0.50.0` | BSD-3-Clause (Go Authors) |
| `golang.org/x/text` | `v0.43.0` | BSD-3-Clause (Go Authors) |
