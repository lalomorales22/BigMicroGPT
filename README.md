# BigMicroGPT

a fully local AI workstation with an IDE-style interface. persistent vector memory with temporal decay RAG, web search, image generation, and streaming chat — all running through Ollama. single-file Python app, zero cloud, zero API costs.

built by [@lalomorales22](https://github.com/lalomorales22)

---

## what it does

- **IDE-style interface** — neon-bordered window panels, session explorer sidebar, chat terminal, image lab, system status, memory bank, RAG context viewer, and a live console log. feels like a desktop app, not a web page
- **persistent vector memory** — every message is embedded into a vector and stored in SQLite. the more you talk to it, the smarter it gets
- **temporal decay RAG** — memory retrieval combines cosine similarity with exponential time decay (0.96/day). recent conversations get boosted, old ones fade naturally but still surface when highly relevant
- **image generation** — built-in image lab powered by `x/flux2-klein:latest` through Ollama. type a prompt, get an image, gallery tracks your history
- **streaming chat** — tokens stream in real time with a live typing cursor
- **web search** — toggle DuckDuckGo search on/off. results inject as context automatically
- **model switching** — dropdown in the title bar lets you swap Ollama models instantly
- **multi-session** — full session management with auto-naming, delete, sidebar navigation
- **RAG context panel** — see exactly which memories were retrieved, their similarity scores, and temporal weights in real time
- **console log** — boot sequence, event logging, status tracking — all visible in the bottom console bar
- **window management** — minimize/expand any panel with the yellow dot
- **single file** — everything in one `app.py`. no build step, no node_modules, no framework. just Python

---

## screenshot

the interface is modeled after an IDE/workstation layout:

```
┌──────────────────────────────────────────────────────────────────────┐
│  ● ● ●  MICRO GPT 2.0 — LOCAL AI WORKSTATION   [model ▾] [SEARCH]    │
├──────────┬───────────────────────────────┬───────────────────────────┤
│          │  CHAT_TERMINAL.gpt        ● ● │  SYSTEM_STATUS        ● ● │
│ SESSION  │                               │  ● Ollama: Connected      │
│ EXPLORER │  streaming chat with          │  ● Model: gpt-oss:20b     │
│          │  markdown rendering           │  ● Search: ON             │
│ .db      │                               ├───────────────────────────┤
│          │                               │  MEMORY_BANK.vec      ● ● │
│ ▸ Chat 1 │                               │  Messages: 128            │
│ ▸ Chat 2 │                               │  Vectorized: 98           │
│ ▸ Chat 3 ├───────────────────────────────┤  ██████████░░ 76%         │
│          │  IMAGE_LAB.flux           ● ● ├───────────────────────────┤
│ msgs: 42 │                               │  RAG_CONTEXT          ● ● │
│ vecs: 38 │  [generated image]            │  [0.89] what about...     │
│ sess: 5  │  [prompt] [GENERATE ⚡]        │  [0.72] remember when...  │
├──────────┴───────────────────────────────┴───────────────────────────┤
│  CONSOLE  [14:32:01] System ready ● [14:32:15] RAG: 3 memories       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## requirements

- Python 3.9+
- [Ollama](https://ollama.com) installed and running

---

## quick start

```bash
# clone
git clone https://github.com/lalomorales22/BigMicroGPT.git
cd BigMicroGPT

# install deps
pip install -r requirements.txt

# start ollama
ollama serve

# pull models
ollama pull gpt-oss:20b           # chat model
ollama pull nomic-embed-text      # embedding model
ollama pull x/flux2-klein:latest  # image generation

# run
python app.py

# open http://localhost:5000
```

---

## how the memory works

```
you send a message
        ↓
nomic-embed-text turns it into a 768-dim vector
        ↓
cosine similarity + temporal decay search against ALL past messages
        ↓
top-k memories (score > 0.3) inject into system prompt
        ↓
[optional] DuckDuckGo search results also inject as context
        ↓
Ollama generates a streaming response with full context
        ↓
response gets embedded + saved to SQLite
        ↓
repeat — memory grows with every conversation
```

### temporal decay

memory retrieval isn't just similarity anymore. each memory gets a combined score:

```
combined = similarity * 0.7 + similarity * temporal_weight * 0.3
temporal_weight = 0.96 ^ (hours_since_message / 24)
```

this means:
- a memory from 1 hour ago with 0.8 similarity → score: **0.79**
- a memory from 7 days ago with 0.8 similarity → score: **0.72**
- a memory from 30 days ago with 0.8 similarity → score: **0.65**

recent context gets naturally boosted. old but highly relevant memories still surface. you can see both scores (similarity + temporal) in the RAG_CONTEXT panel.

---

## models

**chat model** (default: `gpt-oss:20b`)

any model in Ollama works. switch instantly from the title bar dropdown or in settings.

```bash
ollama pull gpt-oss:20b         # solid default
ollama pull llama3.2            # fast, lightweight
ollama pull mistral             # great instruction following
ollama pull deepseek-r1         # reasoning focused
ollama pull phi4                # small + capable
```

**embedding model** (default: `nomic-embed-text`)

turns text into vectors. don't change this after building up memory — old vectors won't be comparable to new ones.

```bash
ollama pull nomic-embed-text    # recommended
ollama pull mxbai-embed-large   # higher quality, slower
```

**image model** (default: `x/flux2-klein:latest`)

generates images in the IMAGE_LAB panel.

```bash
ollama pull x/flux2-klein:latest
```

---

## interface panels

| panel | color | purpose |
|---|---|---|
| SESSION_EXPLORER.db | cyan | session list, new chat, stats |
| CHAT_TERMINAL.gpt | green | streaming chat with markdown |
| IMAGE_LAB.flux | pink | image generation + gallery |
| SYSTEM_STATUS | yellow | model info, search, ollama status |
| MEMORY_BANK.vec | purple | vector stats, coverage bar |
| RAG_CONTEXT | cyan | live memory retrieval display |
| CONSOLE | yellow | boot log, events, quick stats |

every panel has window controls — the yellow dot minimizes/expands.

---

## settings

open settings from the title bar to configure:

| setting | default | description |
|---|---|---|
| chat model | gpt-oss:20b | which Ollama model to chat with |
| embed model | nomic-embed-text | model for vectorizing messages |
| memory depth (k) | 5 | how many past memories to inject per message |
| system prompt | see app.py | base instructions for the model |

web search toggle is in the title bar.

---

## keyboard shortcuts

| shortcut | action |
|---|---|
| Enter | send message |
| Shift+Enter | newline in message |
| Ctrl+N | new session |
| Escape | close settings |

---

## file structure

```
BigMicroGPT/
├── app.py           # everything — flask, RAG, temporal memory, image gen, full UI
├── requirements.txt # flask + duckduckgo-search
├── memory.db        # auto-created, grows over time — this is your brain
└── README.md
```

`memory.db` is your entire knowledge base. back it up to preserve memory across machines.

---

## troubleshooting

**ollama connection error** — make sure ollama is running: `ollama serve`

**model not found** — pull it first: `ollama pull gpt-oss:20b`

**no memory badges** — pull the embed model: `ollama pull nomic-embed-text`

**image generation fails** — pull the image model: `ollama pull x/flux2-klein:latest`

**search not working** — install the dep: `pip install duckduckgo-search`. DuckDuckGo occasionally rate limits — degrades gracefully

**reset all memory** — delete `memory.db`, a fresh one is created on next run

---

## built with

- [Flask](https://flask.palletsprojects.com) — web server
- [Ollama](https://ollama.com) — local model inference, embeddings, image generation
- [duckduckgo-search](https://github.com/deedy5/duckduckgo_search) — free web search, no API key
- SQLite — vector + conversation storage
- pure Python cosine similarity — no numpy, no external vector db
- zero external JS/CSS frameworks — all hand-rolled

---

## license

MIT

---

built by [Lalo Morales](https://github.com/lalomorales22)
