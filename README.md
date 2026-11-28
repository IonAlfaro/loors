<div align="center">
  <img src="docs/logo.png" alt="Loors Logo" width="88" height="88" />
  <h1>Loors</h1>
  <p><strong>Local-first AI Agent, Visual Cyclic Harness Studio &amp; Headless Server</strong></p>
  <p>
    Run autonomous coding &amp; research agents, design cyclic multi-step AI workflows, browse interactively alongside your agent, and deploy headlessly on any server — powered by 100% local GGUF models (embedded <code>llama.cpp</code>) or your favorite cloud APIs.
  </p>

  <p>
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-FSL--1.1--MIT-blue.svg" alt="FSL-1.1-MIT License" /></a>
    <img src="https://img.shields.io/badge/Tauri-v2-24C8DB?logo=tauri&logoColor=white" alt="Tauri v2" />
    <img src="https://img.shields.io/badge/Rust-Backend-orange?logo=rust&logoColor=white" alt="Rust" />
    <img src="https://img.shields.io/badge/React-TypeScript-61DAFB?logo=react&logoColor=black" alt="React + TypeScript" />
    <img src="https://img.shields.io/badge/Platforms-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey" alt="Platforms" />
    <a href="https://github.com/sponsors/IonAlfaro"><img src="https://img.shields.io/badge/Sponsor-Loors-ea4aaa?logo=githubsponsors&amp;logoColor=white" alt="Sponsor Loors" /></a>
  </p>

  <p>
    <a href="https://github.com/IonAlfaro/loors/releases/latest">Download</a> ·
    <a href="#features">Features</a> ·
    <a href="#screenshots">Screenshots</a> ·
    <a href="#harness-templates">Templates</a> ·
    <a href="#headless-cli--web-server">Headless CLI &amp; Server</a> ·
    <a href="#installation">Installation</a> ·
    <a href="#architecture">Architecture</a> ·
    <a href="#building-from-source">Build from Source</a> ·
    <a href="CONTRIBUTING.md">Contributing</a> ·
    <a href="#support-loors">Support Loors</a>
  </p>
</div>

---

<p align="center">
  <img src="docs/screenshots/harness-studio.png" alt="Loors Visual Harness Studio" width="100%" />
</p>

## Get started

1. Download the latest desktop installer from [GitHub Releases](https://github.com/IonAlfaro/loors/releases/latest).
2. Choose a local GGUF model or connect an existing local or cloud API in Settings.
3. Open **Agent Mode** to work with a project, or **Harness Studio** to build a repeatable workflow from a template or an empty canvas.

Loors is designed for local-first operation. Cloud providers are optional; using external APIs or web tools may transmit data to those services. The project is source-available under FSL-1.1-MIT (see [License](#license)).

## Why Loors?

Most AI tools force you to choose between a simple chat window and brittle CLI scripts — or lock you into a single cloud provider. **Loors** brings desktop polish, visual cyclic harnesses, an interactive shared browser, and a headless server binary together in a native Rust and Tauri stack:

- **Agent Mode** — A clean, Codex/Cursor-inspired workspace with file tree, live Git diff, **interactive embedded browser**, subagents, live checklists, persistent artifacts, global MCP servers, 4-tier safety permissions, live model-context usage, structured task verification and bounded recovery.
- **Visual Harness Studio** — Build repeatable, **cyclic** agent loops (e.g., *Coder → Tests → LLM Judge → Loop Guard → Human Gate → Done*) with `For Each` iterators, `JSON Validator`, JSON transformation, document generation, swarm coordination, global run budgets, *Run from here* resume, webhooks, cron, and Telegram triggers.
- **AI Architect** — Built-in assistant that inspects your workspace and designs or refactors `.loors.json` harnesses for you, with safeguards around applying and saving workflow changes.
- **Headless Agent, CLI & Self-Hosted Web Panel (`loors-cli`)** — Run an agent from a terminal, optionally in an isolated Git worktree, select a local inference backend and context size, then review the saved session in the desktop app. Also execute harnesses in CI/CD, cron, or SSH, or launch the web panel with Bearer token auth and SSE streaming.
- **Local AI or Cloud APIs — Your Choice** — Run 100% private local GGUF models with 1-click `llama.cpp` GPU installation (CUDA, Vulkan, Metal, CPU), dynamic Hugging Face discovery, or connect OpenAI, Anthropic (Claude), Google Gemini, GitHub Models/Copilot, OpenRouter, LM Studio, Ollama, or any custom endpoint.

---

## Features

| Area | Capabilities |
| :--- | :--- |
| **Local Inference (`llama.cpp`)** | Automatic GPU detection (CUDA / Vulkan / Metal / CPU), 1-click runtime download, **dynamic Hugging Face GGUF explorer** (Trending, Most Downloaded, Most Liked, Most Recent), automatic LM Studio local model detection, custom `llama-server` binary support, KV-cache & context tuning, and live VRAM load/unload progress. |
| **Cloud & Local API Providers** | Native streaming SSE for **OpenAI / ChatGPT**, **Anthropic / Claude** (including extended thinking & tool use), **Google Gemini**, **GitHub Models / Copilot**, **OpenRouter / OpenCode**, dedicated **LM Studio** (`:1234`) & **Ollama** (`:11434`) with selectable OpenAI/Anthropic compatibility & optional server secret, and custom endpoints. Automatic `/models` discovery + `${ENV_VAR}` secret expansion. |
| **Autonomous Agent Mode** | 4-tier safety permissions (**Read-only**, **Ask**, **Auto** with AI safety reviewer, **Turbo**), isolated & parallel subagents, live todo checklist, background process manager, scheduled tasks, interactive user questions, persistent Markdown/HTML artifacts, global MCP servers, Thinking/Effort selector, live context counter for local-model context tracking, task reviews, persisted execution outcomes and configurable agent limits. |
| **Interactive Embedded Browser** | Pure-Rust Chrome DevTools Protocol (CDP) browser with Chrome-style tabs, editable omnibox, direct click/type/scroll forwarding, and a persistent profile shared live between you and the AI Agent (log in manually and let the agent continue on the same authenticated page). Zero Node.js or Playwright dependencies. |
| **Cyclic Visual Harnesses** | Drag-and-drop React Flow canvas with `Agent`, `Shell / Test`, `For Each / Iterator`, `JSON Validator`, `Data Transform`, document creation and reading (Excel, CSV, Word, PDF), `Swarm`, `LLM Judge`, `Loop Guard`, `Human Gate`, `Git Checkpoint`, `Context Compactor`, `Knowledge / RAG`, `Memory / State`, `HTTP Request`, `Notify`, `Trigger` (Webhook / Cron / Telegram), `MCP Server`, `Skills`, `Parallel Fan-Out`, `Subgraph`, and `Custom Nodes` — plus global run budgets, per-node latency badges, 1-click resource wiring, undo/redo, template catalog, and *Run from here* resume. |
| **Headless Agent, CLI & Web Server** | Dedicated `loors-cli` binary sharing 100% of the Rust engine: run an agent with a GGUF or cloud model, choose `--backend` and per-slot `--context`, optionally isolate changes in a Git worktree, and review the saved conversation in desktop Agent Mode. It also runs harnesses one-shot (`run --jsonl` / `--quiet`), hosts the React Web Panel over HTTP + SSE, and exposes run-control REST APIs. |

---

## Agent execution and verification

For multi-step software engineering tasks, Loors supports structured planning, task tracking, tool execution and review. The agent can work through ready tasks, run relevant checks and record their results before marking work complete. Independent task reviews, stagnation detection and bounded recovery attempts help it revisit unsuccessful approaches rather than repeat them indefinitely.

- **Verification:** run checks after changes and distinguish observed test results from manual inspection. Permission denials and unsuccessful checks are treated as blockers rather than successful verification.
- **Execution state:** session-scoped progress, active task, recovery phase and persisted execution outcomes help users understand where a run stopped.
- **Configurable limits:** per-session limits for agent iterations, total model calls (including subagents and automated reviews), recovery attempts, identical results, reviews per task and verifier steps.
- **Permissions:** select the appropriate permission mode for the workspace. Review tool requests and changes before approving operations that affect important files or systems.

These mechanisms improve controllability; they do not guarantee that generated changes are correct. Review diffs and verification results before using them in production.

## Screenshots

### Agent Mode
Autonomous coding, research, and interactive browser workflows with 4-tier permissions and 1-click local or cloud setup.
<p align="center">
  <img src="docs/screenshots/agent-mode.png" alt="Loors Agent Mode" width="100%" />
</p>

### Visual Harness Studio
Design cyclic, self-healing multi-agent pipelines with live inspection, variable chips, and an integrated AI Architect.
<p align="center">
  <img src="docs/screenshots/harness-studio.png" alt="Loors Visual Harness Studio" width="100%" />
</p>

### Unified Settings & API Discovery
Manage local GGUF models, dynamic Hugging Face downloads, MCP servers, `llama.cpp` runtimes, and cloud/local APIs in one place.
<p align="center">
  <img src="docs/screenshots/settings-apis.png" alt="Loors Settings" width="100%" />
</p>

---

## Harness templates

The built-in bilingual template catalog describes the required inputs and expected outputs of each workflow. Templates open in a separate editor tab, and templates with demo variants can be tried using sample inputs. The visual editor supports undo/redo, node navigation and inspection of execution results.

| Template | Input | Output |
| :--- | :--- | :--- |
| **Invoices to Excel and PDF** | Workspace, tool-capable model and PDF paths (selectable text required) | Excel spreadsheet and PDF summary with totals by currency and exceptions |
| **Fix a failing check** | Git workspace, bug description and verification command | Check results and Git diff for approval; up to three repair attempts |
| **Inspect a binary** | Local PE, ELF or Mach-O file and tool-capable model | PDF inventory of format, architecture, sections, available symbols and relevant strings |
| **Review a repository with a swarm** | Repository and optional area of focus | Prioritized PDF report with file references and swarm-board transcript |

### Data and document workflows

Harnesses can also transform JSON using JSON Pointer paths, select or rename fields, generate Excel (.xlsx), CSV, Word (.docx) and PDF files, and read text from supported document formats. Document nodes work with workspace-relative paths and can pass generated file paths to downstream nodes. Use `{{run_id}}` in output paths to distinguish files created by different runs.

## Headless CLI & Web Server

Loors ships with a dedicated headless binary (`loors-cli`) for servers, SSH, cron jobs, and CI/CD pipelines:

```bash
# Start an agent from committed HEAD in an isolated worktree, using llama.cpp Vulkan
loors-cli agent --worktree --model /path/to/model.gguf --backend vulkan --context 16384 "Fix the failing test"

# Run a harness once (with optional overrides & structured JSONL output)
loors-cli run ./my-harness.loors.json --workspace ./repo --prompt "Fix failing tests" --jsonl

# Print only the final output (great for Unix pipes)
loors-cli run ./my-harness.loors.json --quiet

# Start the headless HTTP/SSE server + full Web Panel (localhost by default)
loors-cli serve --port 7700

# Expose on a server / LAN (automatically enforces Bearer token authentication)
loors-cli serve --host 0.0.0.0 --port 7700 --token "$LOORS_TOKEN"

# Inspect server state
loors-cli status
loors-cli workflows
loors-cli models
```

The agent asks before running commands or editing files. It prints the session ID and workspace path, and saves the conversation in Agent Mode for desktop review. With `--worktree`, Loors creates an isolated worktree from committed `HEAD` and keeps it after the CLI exits so you can inspect the diff. `--backend` selects `auto`, `cuda`, `vulkan`, `metal`, or `cpu`; `--context` sets the requested token window for each local agent slot. Use `cloud::provider/model` with `--model` for cloud APIs. You can set `LOORS_MODEL` to reuse a model across runs.

| Exit Code | Meaning |
| :--- | :--- |
| `0` | Harness completed successfully |
| `1` | Harness failed or configuration error |
| `2` | Run budget (`--timeout` / `--max-llm-calls`) or Loop Guard exhausted |
| `130` | Interrupted (`Ctrl+C`) with clean `llama-server` and browser shutdown |

---

## Installation

### Prebuilt Installers
Download the latest installer for your operating system from [**GitHub Releases**](../../releases):
- **Windows**: `Loors_*_x64-setup.exe` (branded NSIS installer) or `.msi`
- **macOS**: `Loors_*_aarch64.dmg`
- **Linux**: `Loors_*.AppImage` or `.deb`

> **Windows signing notice:** The release pipeline supports SignPath signing when configured. Check the digital signature of the installer you download. Unsigned builds may trigger Microsoft Defender SmartScreen warnings or be blocked by Smart App Control on Windows 11.

On first launch, choose either:
1. **100% Local AI** — Click **"Install llama.cpp automatically (1 click)"** to detect your GPU and download the inference engine, then download or reuse any GGUF model.
2. **External / Local API** — Open **Settings → APIs / Cloud**, expand a provider (OpenAI, Anthropic, Gemini, GitHub, OpenRouter, LM Studio, Ollama, or Custom), enter your key/URL, and click **"Connect & load models"**.

---

## Building from Source

### Prerequisites
- **Node.js** 20+
- **Rust** stable toolchain (`cargo`, `rustc`)
- [Tauri v2 platform prerequisites](https://v2.tauri.app/start/prerequisites/)

### Development
```bash
npm ci
npm run tauri dev
```

### Production Build, CLI & Installers
```bash
# Type-check & build frontend (dist/)
npm run build

# Build headless CLI & server binary
cargo build --release --bin loors-cli --manifest-path src-tauri/Cargo.toml

# Run Rust test suite
cargo test --manifest-path src-tauri/Cargo.toml

# Build native desktop installer for your OS
npm run tauri build
```

---

## Architecture

```text
loors/
├── src/                                  # React 19 + TypeScript + React Flow + Zustand UI
│   ├── lib/bridge.ts                     # Unified Tauri IPC / Web HTTP+SSE transport adapter
│   ├── components/
│   │   ├── AgentWorkspaceView.tsx        # Autonomous Agent Mode UI
│   │   ├── InteractiveBrowserPane.tsx    # Chrome-style interactive embedded CDP browser
│   │   ├── WebAuthGate.tsx               # Bearer token login modal for headless web mode
│   │   ├── SettingsModal.tsx             # Unified General, Models, APIs, MCP, Runtime & Updates
│   │   ├── ModelsModal.tsx               # Local GGUF, dynamic HF explorer & API provider drawer
│   │   ├── ArchitectChat.tsx             # AI Architect visual workflow assistant
│   │   └── Inspector.tsx                 # Node inspector, tool presets & 1-click wiring
│   ├── store/useGraphStore.ts            # Reactive canvas, layout, model & i18n state
│   └── i18n.ts                           # English (default) & Spanish translations
├── src-tauri/                            # Rust + Tauri v2 native backend
│   └── src/
│       ├── bin/loors-cli.rs              # Headless CLI & self-hosted web server entrypoint
│       ├── headless.rs                   # Unified RPC dispatcher, HTTP/SSE server & CLI runner
│       ├── agent_chat/                   # Autonomous agent loop, subagents, permissions & artifacts
│       ├── architect/                    # AI Architect graph designer + read-only workspace tools
│       ├── engine/                       # Cyclic harness executor, For Each, Validator, Judge & RAG
│       ├── inference/                    # Embedded llama.cpp manager + Cloud/Anthropic/OpenAI SSE
│       ├── mcp/                          # Global & per-node Model Context Protocol (stdio) client
│       ├── runtime/                      # GPU detection, llama.cpp auto-installer & custom binary
│       ├── tools/                        # File ops, shell, pure-Rust interactive CDP browser, bg tasks
│       ├── triggers/                     # Webhook HTTP, Cron interval & Telegram Bot triggers
│       └── updater.rs                    # Native GitHub Releases auto-updater
├── .github/workflows/
│   ├── ci.yml                            # PR & push CI (frontend build + Rust tests + loors-cli)
│   └── release.yml                       # Multi-OS release builds + SignPath Windows code signing
└── docs/                                 # Logo & showcase screenshots
```

---

## Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl + 1` | Switch to **Agent** mode |
| `Ctrl + 2` | Switch to **Harnesses** studio |
| `Ctrl + B` | Toggle left sidebar |
| `Ctrl + Shift + B` | Toggle right panel |
| `Ctrl + J` | Toggle bottom console (Harnesses) |

---

## Contributing, Security & Code Signing

- Read [CONTRIBUTING.md](CONTRIBUTING.md) to get started with pull requests.
- Please follow our [Code of Conduct](CODE_OF_CONDUCT.md).
- To report a security issue or review our privacy & signing policy, see [SECURITY.md](SECURITY.md).

## Support Loors

Loors is an independent, local-first AI project. If it helps you build software, automate workflows, or run AI on your own hardware, consider supporting its development through GitHub Sponsors.

Your sponsorship helps fund ongoing development, local model integrations, agent reliability, cross-platform testing, documentation, and maintenance.

[**Become a sponsor on GitHub**](https://github.com/sponsors/IonAlfaro)

You can also support Loors for free by starring the repository, reporting bugs, contributing improvements, or sharing the project with others. Thank you!

## License

Loors is licensed under the [Functional Source License, Version 1.1, MIT Future License (FSL-1.1-MIT)](LICENSE).

Loors is **source-available**, not currently distributed under an unrestricted open-source license. The FSL permits use, modification, self-hosting and redistribution for non-competing purposes, subject to its terms, including internal business use and specified educational, research and professional-service uses. Competing commercial products or services are restricted under the FSL. Each version is additionally available under the MIT License starting two years after that version is made available. See the full [LICENSE](LICENSE) for the governing terms.
