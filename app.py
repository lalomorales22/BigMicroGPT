#!/usr/bin/env python3
"""
MICRO GPT 2.0 — LOCAL AI WORKSTATION
    · talks to any ollama model (text + image generation)
    · vectorizes every message and saves to sqlite
    · retrieves relevant memories via cosine similarity + temporal decay (RAG)
    · optional duckduckgo web search injected as context
    · streams responses token by token
    · image generation via flux2-klein or any ollama vision model
    · 100% local, zero cloud, zero api costs

deps: pip install flask ddgs
ollama: ollama pull llama3.2 && ollama pull nomic-embed-text
run:   python app.py
"""

import os
import json
import math
import sqlite3
import urllib.request
import urllib.error
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, Response, stream_with_context

app = Flask(__name__)
DB_PATH = os.path.join(os.path.dirname(__file__), "memory.db")
OLLAMA_BASE = "http://localhost:11434"

# ───────────────────────────────────────────────────────────────
# DATABASE
# ───────────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    DEFAULT 'New Chat',
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL,
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                embedding       TEXT,
                search_used     INTEGER DEFAULT 0,
                timestamp       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS settings (
                key   TEXT PRIMARY KEY,
                value TEXT
            );

            CREATE TABLE IF NOT EXISTS documents (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content     TEXT NOT NULL,
                embedding   TEXT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS images (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL,
                prompt      TEXT NOT NULL,
                image_data  TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            INSERT OR IGNORE INTO settings VALUES ('model',         'llama3.2');
            INSERT OR IGNORE INTO settings VALUES ('embed_model',   'nomic-embed-text');
            INSERT OR IGNORE INTO settings VALUES ('web_search',    'false');
            INSERT OR IGNORE INTO settings VALUES ('memory_k',      '5');
            INSERT OR IGNORE INTO settings VALUES ('system_prompt', 'You are a sharp, helpful assistant with persistent memory across conversations. When relevant memories are provided, reference them naturally. Be concise and direct.');
        """)

# ───────────────────────────────────────────────────────────────
# SETTINGS HELPERS
# ───────────────────────────────────────────────────────────────

def get_setting(key):
    with get_db() as conn:
        row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row['value'] if row else None

def set_setting(key, value):
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?,?)", (key, str(value)))

# ───────────────────────────────────────────────────────────────
# VECTOR MATH  (pure python, no numpy)
# ───────────────────────────────────────────────────────────────

def cosine_similarity(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

# ───────────────────────────────────────────────────────────────
# OLLAMA
# ───────────────────────────────────────────────────────────────

def ollama_request(path, payload, timeout=30):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{OLLAMA_BASE}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())

def ollama_embed(text, model):
    """Get embedding vector for text"""
    try:
        result = ollama_request("/api/embeddings", {"model": model, "prompt": text}, timeout=30)
        return result.get("embedding", [])
    except Exception as e:
        print(f"[embed error] {e}")
        return []

def ollama_models():
    """List available local models"""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return [m['name'] for m in data.get('models', [])]
    except:
        return []

def ollama_chat_stream(messages, model):
    """Stream chat response token by token"""
    payload = json.dumps({
        "model":    model,
        "messages": messages,
        "stream":   True
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw_line in resp:
                line = raw_line.decode('utf-8').strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    if 'message' in chunk and 'content' in chunk['message']:
                        yield chunk['message']['content']
                    if chunk.get('done', False):
                        break
                except json.JSONDecodeError:
                    continue
    except urllib.error.URLError as e:
        yield f"\n\n[❌ ollama connection error: {e}]\n[make sure `ollama serve` is running]"

# ───────────────────────────────────────────────────────────────
# DUCKDUCKGO SEARCH
# ───────────────────────────────────────────────────────────────

def web_search(query, max_results=4):
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except ImportError:
        return [{"title": "duckduckgo_search not installed", "body": "run: pip install duckduckgo-search", "href": ""}]
    except Exception as e:
        return [{"title": "search failed", "body": str(e), "href": ""}]

def format_search_context(results):
    if not results:
        return ""
    lines = ["[WEB SEARCH RESULTS — use these to answer if relevant]"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '')}")
        lines.append(f"   {r.get('body', '')}")
        if r.get('href'):
            lines.append(f"   source: {r['href']}")
    return "\n".join(lines)

# ───────────────────────────────────────────────────────────────
# DOCUMENT CHUNKING & FILE READING
# ───────────────────────────────────────────────────────────────

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks, paragraph-aware with overlap."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if current and len(current) + len(para) + 2 > chunk_size:
            chunks.append(current)
            # keep overlap from end of previous chunk
            if overlap > 0 and len(current) > overlap:
                current = current[-overlap:] + "\n\n" + para
            else:
                current = para
        else:
            current = (current + "\n\n" + para) if current else para

    if current:
        chunks.append(current)

    # if no paragraph breaks, fall back to hard splitting
    if len(chunks) == 1 and len(chunks[0]) > chunk_size * 2:
        text = chunks[0]
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])

    return chunks

def read_file_text(file_storage):
    """Read text from uploaded file (txt, md, pdf). Returns (text, error)."""
    filename = file_storage.filename.lower()

    if filename.endswith('.txt') or filename.endswith('.md'):
        try:
            return file_storage.read().decode('utf-8'), None
        except UnicodeDecodeError:
            return None, "File is not valid UTF-8 text"

    if filename.endswith('.pdf'):
        try:
            import pypdf
            reader = pypdf.PdfReader(file_storage)
            pages = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            if not pages:
                return None, "PDF has no extractable text"
            return "\n\n".join(pages), None
        except ImportError:
            return None, "pypdf not installed. Run: pip install pypdf"
        except Exception as e:
            return None, f"PDF read error: {e}"

    return None, f"Unsupported file type: {filename.rsplit('.', 1)[-1]}"

# ───────────────────────────────────────────────────────────────
# MEMORY / RAG
# ───────────────────────────────────────────────────────────────

def save_message(session_id, role, content, embedding=None, search_used=False):
    emb_str = json.dumps(embedding) if embedding else None
    with get_db() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, embedding, search_used) VALUES (?,?,?,?,?)",
            (session_id, role, content, emb_str, int(search_used))
        )

def retrieve_memories(query_embedding, current_session_id, k=5):
    """
    Find top-k most semantically similar past messages AND document chunks.
    Messages use cosine similarity weighted by temporal decay.
    Document chunks use cosine similarity with fixed temporal weight of 1.0.
    """
    if not query_embedding:
        return []

    scored = []
    now = datetime.now()
    decay = 0.96  # temporal decay factor per day

    with get_db() as conn:
        # search message memories
        rows = conn.execute(
            """SELECT m.role, m.content, m.embedding, m.timestamp, s.name as session_name
               FROM messages m
               JOIN sessions s ON m.session_id = s.id
               WHERE m.embedding IS NOT NULL
               AND m.role = 'user'""",
        ).fetchall()

        for row in rows:
            try:
                emb = json.loads(row['embedding'])
                sim = cosine_similarity(query_embedding, emb)

                time_factor = 0.5
                try:
                    ts = row['timestamp']
                    if ts:
                        msg_time = datetime.fromisoformat(ts) if 'T' in ts else datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                        hours_ago = max((now - msg_time).total_seconds() / 3600, 0)
                        time_factor = decay ** (hours_ago / 24)
                except Exception:
                    pass

                combined = sim * 0.7 + sim * time_factor * 0.3

                scored.append({
                    "score":        round(combined, 3),
                    "similarity":   round(sim, 3),
                    "temporal":     round(time_factor, 3),
                    "content":      row['content'],
                    "timestamp":    row['timestamp'],
                    "session_name": row['session_name'],
                    "source":       "memory"
                })
            except Exception:
                continue

        # search document chunks
        doc_rows = conn.execute(
            """SELECT filename, chunk_index, content, embedding, uploaded_at
               FROM documents
               WHERE embedding IS NOT NULL"""
        ).fetchall()

        for row in doc_rows:
            try:
                emb = json.loads(row['embedding'])
                sim = cosine_similarity(query_embedding, emb)
                # documents don't decay — they're reference material
                combined = sim * 0.7 + sim * 1.0 * 0.3

                scored.append({
                    "score":        round(combined, 3),
                    "similarity":   round(sim, 3),
                    "temporal":     1.0,
                    "content":      row['content'],
                    "timestamp":    row['uploaded_at'],
                    "session_name": row['filename'],
                    "source":       "doc"
                })
            except Exception:
                continue

    scored.sort(key=lambda x: x['score'], reverse=True)
    return [m for m in scored[:k] if m['score'] > 0.3]

def build_memory_context(memories):
    if not memories:
        return ""
    mem_items = [m for m in memories if m.get('source') == 'memory']
    doc_items = [m for m in memories if m.get('source') == 'doc']
    lines = []
    if mem_items:
        lines.append("\n[RELEVANT MEMORIES FROM PAST CONVERSATIONS — reference naturally if helpful]")
        for m in mem_items:
            ts = m['timestamp'][:10] if m['timestamp'] else ''
            lines.append(f"• [{m['score']:.2f} relevance | {ts}] {m['content']}")
    if doc_items:
        lines.append("\n[RELEVANT DOCUMENT EXCERPTS — reference if helpful]")
        for m in doc_items:
            lines.append(f"• [{m['score']:.2f} relevance | {m['session_name']}] {m['content']}")
    return "\n".join(lines)

def get_session_history(session_id, limit=20):
    with get_db() as conn:
        rows = conn.execute(
            """SELECT role, content, search_used, timestamp
               FROM messages WHERE session_id=?
               ORDER BY id DESC LIMIT ?""",
            (session_id, limit)
        ).fetchall()
    return list(reversed([dict(r) for r in rows]))

# ───────────────────────────────────────────────────────────────
# FLASK ROUTES
# ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, created_at FROM sessions ORDER BY id DESC LIMIT 60"
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/sessions', methods=['POST'])
def create_session():
    name = request.json.get('name', 'New Chat') if request.json else 'New Chat'
    with get_db() as conn:
        cur = conn.execute("INSERT INTO sessions (name) VALUES (?)", (name,))
        sid = cur.lastrowid
    return jsonify({"id": sid, "name": name})

@app.route('/api/sessions/<int:sid>', methods=['PATCH'])
def rename_session(sid):
    name = request.json.get('name', 'Chat') if request.json else 'Chat'
    with get_db() as conn:
        conn.execute("UPDATE sessions SET name=? WHERE id=?", (name, sid))
    return jsonify({"ok": True})

@app.route('/api/sessions/<int:sid>', methods=['DELETE'])
def delete_session(sid):
    with get_db() as conn:
        conn.execute("DELETE FROM images WHERE session_id=?", (sid,))
        conn.execute("DELETE FROM messages WHERE session_id=?", (sid,))
        conn.execute("DELETE FROM sessions WHERE id=?", (sid,))
    return jsonify({"ok": True})

@app.route('/api/history/<int:sid>')
def get_history(sid):
    return jsonify(get_session_history(sid, limit=200))

@app.route('/api/settings', methods=['GET'])
def api_get_settings():
    with get_db() as conn:
        rows = conn.execute("SELECT key, value FROM settings").fetchall()
    return jsonify({r['key']: r['value'] for r in rows})

@app.route('/api/settings', methods=['POST'])
def api_set_settings():
    for key, val in (request.json or {}).items():
        set_setting(key, val)
    return jsonify({"ok": True})

@app.route('/api/models')
def api_models():
    return jsonify(ollama_models())

@app.route('/api/memory/stats')
def memory_stats():
    with get_db() as conn:
        total    = conn.execute("SELECT COUNT(*) as c FROM messages").fetchone()['c']
        vectors  = conn.execute("SELECT COUNT(*) as c FROM messages WHERE embedding IS NOT NULL").fetchone()['c']
        sessions = conn.execute("SELECT COUNT(*) as c FROM sessions").fetchone()['c']
    return jsonify({"total": total, "vectorized": vectors, "sessions": sessions})

@app.route('/api/chat', methods=['POST'])
def chat():
    body       = request.json or {}
    session_id = body.get('session_id')
    user_msg   = (body.get('message') or '').strip()

    if not session_id or not user_msg:
        return jsonify({"error": "missing session_id or message"}), 400

    # load settings
    model        = get_setting('model')        or 'llama3.2'
    embed_model  = get_setting('embed_model')  or 'nomic-embed-text'
    search_on    = get_setting('web_search')   == 'true'
    memory_k     = int(get_setting('memory_k') or 5)
    system_base  = get_setting('system_prompt') or ''

    # 1. embed the incoming user message
    user_emb = ollama_embed(user_msg, embed_model)

    # 2. retrieve relevant memories from all past sessions (RAG)
    memories     = retrieve_memories(user_emb, session_id, k=memory_k)
    memory_ctx   = build_memory_context(memories)

    # 3. optional duckduckgo web search
    search_results = []
    search_ctx     = ""
    if search_on:
        search_results = web_search(user_msg)
        search_ctx     = format_search_context(search_results)

    # 4. build full system prompt
    system_prompt = system_base
    if memory_ctx:
        system_prompt += "\n\n" + memory_ctx
    if search_ctx:
        system_prompt += "\n\n" + search_ctx

    # 5. build message history for this session (recent context window)
    history  = get_session_history(session_id, limit=16)
    messages = [{"role": "system", "content": system_prompt}]
    for h in history:
        messages.append({"role": h['role'], "content": h['content']})
    messages.append({"role": "user", "content": user_msg})

    # 6. save user message immediately
    save_message(session_id, 'user', user_msg, user_emb, search_used=bool(search_results))

    # 7. auto-name session from first user message
    with get_db() as conn:
        msg_count = conn.execute(
            "SELECT COUNT(*) as c FROM messages WHERE session_id=?", (session_id,)
        ).fetchone()['c']
        if msg_count == 1:
            short = user_msg[:45] + ('…' if len(user_msg) > 45 else '')
            conn.execute("UPDATE sessions SET name=? WHERE id=?", (short, session_id))

    def generate():
        full_response = []

        # send metadata first so the UI can show badges + RAG panel
        meta = {
            "type":     "meta",
            "memories": len(memories),
            "searched": bool(search_results),
            "snippets": [{"score": m['score'], "similarity": m.get('similarity', m['score']), "temporal": m.get('temporal', 1.0), "text": m['content'][:120], "session": m.get('session_name',''), "ts": (m.get('timestamp') or '')[:16], "source": m.get('source', 'memory')} for m in memories[:5]]
        }
        yield f"data: {json.dumps(meta)}\n\n"

        # stream the model response
        for chunk in ollama_chat_stream(messages, model):
            full_response.append(chunk)
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"

        # save assistant response with its own embedding
        full_text    = ''.join(full_response)
        asst_emb     = ollama_embed(full_text, embed_model)
        save_message(session_id, 'assistant', full_text, asst_emb)

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )

# ───────────────────────────────────────────────────────────────
# DOCUMENT UPLOAD & MANAGEMENT
# ───────────────────────────────────────────────────────────────

@app.route('/api/upload-doc', methods=['POST'])
def upload_doc():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    allowed = ('.txt', '.md', '.pdf')
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        return jsonify({"error": f"Unsupported file type. Use: {', '.join(allowed)}"}), 400

    text, err = read_file_text(file)
    if err:
        return jsonify({"error": err}), 400

    chunks = chunk_text(text)
    if not chunks:
        return jsonify({"error": "No text content found in file"}), 400

    embed_model = get_setting('embed_model') or 'nomic-embed-text'
    filename = file.filename

    with get_db() as conn:
        for i, chunk in enumerate(chunks):
            emb = ollama_embed(chunk, embed_model)
            emb_str = json.dumps(emb) if emb else None
            conn.execute(
                "INSERT INTO documents (filename, chunk_index, content, embedding) VALUES (?,?,?,?)",
                (filename, i, chunk, emb_str)
            )

    return jsonify({
        "ok": True,
        "filename": filename,
        "chunks": len(chunks),
        "embedded": len(chunks)
    })

@app.route('/api/documents', methods=['GET'])
def list_documents():
    with get_db() as conn:
        rows = conn.execute(
            """SELECT filename, COUNT(*) as chunks,
                      MIN(uploaded_at) as uploaded_at
               FROM documents
               GROUP BY filename
               ORDER BY MIN(uploaded_at) DESC"""
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/documents/<path:filename>', methods=['DELETE'])
def delete_document(filename):
    with get_db() as conn:
        conn.execute("DELETE FROM documents WHERE filename=?", (filename,))
    return jsonify({"ok": True})

@app.route('/api/documents/stats', methods=['GET'])
def document_stats():
    with get_db() as conn:
        docs = conn.execute("SELECT COUNT(DISTINCT filename) as c FROM documents").fetchone()['c']
        chunks = conn.execute("SELECT COUNT(*) as c FROM documents").fetchone()['c']
        embedded = conn.execute("SELECT COUNT(*) as c FROM documents WHERE embedding IS NOT NULL").fetchone()['c']
    return jsonify({"docs": docs, "chunks": chunks, "embedded": embedded})

# ───────────────────────────────────────────────────────────────
# IMAGE GENERATION
# ───────────────────────────────────────────────────────────────

IMAGE_MODEL = "x/flux2-klein:latest"

@app.route('/api/generate-image', methods=['POST'])
def generate_image():
    body = request.json or {}
    prompt = body.get('prompt', '').strip()
    session_id = body.get('session_id')

    if not prompt:
        return jsonify({"error": "missing prompt"}), 400

    # always use the dedicated image model, ignore any model param
    model = IMAGE_MODEL

    # try non-streaming first, then fall back to streaming collection
    images = []
    response_text = ''

    try:
        # attempt 1: non-streaming (works on some ollama versions)
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read())

        # ollama returns "image" (singular) for diffusion models, "images" (plural) for others
        if result.get('image'):
            images = [result['image']]
        elif result.get('images'):
            images = result['images']
        response_text = result.get('response', '')

    except urllib.error.URLError as e:
        return jsonify({"error": f"Connection error: {e}. Is Ollama running?"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # save generated images to DB if session_id provided
    saved_ids = []
    if session_id and images:
        with get_db() as conn:
            for img_b64 in images:
                cur = conn.execute(
                    "INSERT INTO images (session_id, prompt, image_data) VALUES (?,?,?)",
                    (session_id, prompt, img_b64)
                )
                saved_ids.append(cur.lastrowid)

    return jsonify({
        "images": images,
        "image_ids": saved_ids,
        "response": response_text,
        "model": model,
        "done": True
    })

@app.route('/api/images/<int:sid>')
def get_session_images(sid):
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, prompt, image_data, created_at FROM images WHERE session_id=? ORDER BY created_at ASC",
            (sid,)
        ).fetchall()
    return jsonify([dict(r) for r in rows])

@app.route('/api/images/item/<int:image_id>', methods=['DELETE'])
def delete_image(image_id):
    with get_db() as conn:
        conn.execute("DELETE FROM images WHERE id=?", (image_id,))
    return jsonify({"ok": True})

# ───────────────────────────────────────────────────────────────
# HTML / CSS / JS  (single-file UI)
# ───────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MICRO GPT 2.0</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #080811;
  --s1:       #0c0c18;
  --s2:       #10101e;
  --s3:       #161628;
  --border:   #1a1a30;
  --border2:  #252540;
  --cyan:     #00e5ff;
  --pink:     #ff2d78;
  --yellow:   #ffd700;
  --green:    #39ff8a;
  --purple:   #b06cff;
  --red:      #ff4d6d;
  --orange:   #ff9f43;
  --text:     #dde0f0;
  --text2:    #a0a0c0;
  --muted:    #4a4a6a;
  --muted2:   #2a2a42;
}

* { margin:0; padding:0; box-sizing:border-box; }
html, body { height:100vh; overflow:hidden; background:var(--bg); }

body {
  font-family: 'JetBrains Mono', monospace;
  color: var(--text);
}

/* ── APP LAYOUT (flex, resizable) ──────── */
#app {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
  overflow: hidden;
  background: var(--bg);
  background-image:
    linear-gradient(rgba(0,229,255,0.015) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,0.015) 1px, transparent 1px);
  background-size: 50px 50px;
}

#main-area {
  flex: 1;
  display: flex;
  flex-direction: row;
  overflow: hidden;
  min-height: 0;
}

/* ── SPLITTERS ─────────────────────────── */
.splitter {
  flex-shrink: 0;
  background: var(--border);
  transition: background 0.15s;
  z-index: 10;
}
.splitter:hover, .splitter.dragging {
  background: var(--cyan);
  box-shadow: 0 0 6px rgba(0,229,255,0.3);
}
.splitter-v {
  width: 4px;
  cursor: col-resize;
}
.splitter-h {
  height: 4px;
  cursor: row-resize;
}

/* ── SCANLINE OVERLAY ──────────────────── */
#scanline {
  position: fixed; inset:0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(0,0,0,0.02) 2px, rgba(0,0,0,0.02) 4px
  );
  pointer-events: none;
  z-index: 9999;
}

/* ── TITLE BAR ─────────────────────────── */
#titlebar {
  flex-shrink: 0;
  height: 36px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 12px;
  background: var(--s1);
  border-bottom: 1px solid var(--border);
  user-select: none;
  -webkit-app-region: drag;
}

#tb-left {
  display: flex;
  align-items: center;
  gap: 6px;
}

.tb-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  -webkit-app-region: no-drag;
  cursor: pointer;
}
.tb-dot.r { background: #ff5f56; }
.tb-dot.y { background: #ffbd2e; }
.tb-dot.g { background: #27c93f; }

#tb-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  background: linear-gradient(135deg, var(--cyan), var(--purple));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

#tb-right {
  display: flex;
  align-items: center;
  gap: 8px;
  -webkit-app-region: no-drag;
}

#model-select {
  background: var(--s2);
  border: 1px solid var(--border2);
  border-radius: 4px;
  padding: 3px 6px;
  color: var(--cyan);
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  outline: none;
  cursor: pointer;
}
#model-select option { background: var(--s2); color: var(--text); }

.tb-btn {
  background: none;
  border: 1px solid var(--border2);
  border-radius: 4px;
  padding: 3px 8px;
  color: var(--muted);
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  cursor: pointer;
  transition: all .15s;
  letter-spacing: 0.3px;
}
.tb-btn:hover { border-color: var(--cyan); color: var(--cyan); }
.tb-btn.on { border-color: var(--green); color: var(--green); background: rgba(57,255,138,0.06); }

#tb-clock {
  font-size: 10px;
  color: var(--muted);
  min-width: 42px;
  text-align: right;
}

/* ── WINDOW COMPONENT ──────────────────── */
.window {
  display: flex;
  flex-direction: column;
  border: 1px solid var(--win-color, var(--border));
  border-radius: 3px;
  background: var(--s1);
  overflow: hidden;
  min-height: 0;
  transition: flex 0.3s ease;
}
.window.minimized { flex: 0 0 auto !important; }
.window.minimized .win-body,
.window.minimized .win-input,
.window.minimized .win-footer { display: none; }

.win-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 10px;
  background: rgba(255,255,255,0.02);
  border-bottom: 1px solid rgba(255,255,255,0.04);
  flex-shrink: 0;
  cursor: default;
  user-select: none;
  min-height: 26px;
}

.win-title {
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  color: var(--win-color, var(--text));
}

.win-right {
  display: flex;
  align-items: center;
  gap: 6px;
}

.win-dots {
  display: flex;
  gap: 5px;
}

.wd {
  width: 9px; height: 9px;
  border-radius: 50%;
  cursor: pointer;
  opacity: 0.7;
  transition: opacity .15s;
}
.wd:hover { opacity: 1; }
.wd-min { background: #ffbd2e; }
.wd-max { background: #27c93f; }
.wd-close { background: #ff5f56; }

.win-body {
  flex: 1;
  overflow-y: auto;
  overflow-x: hidden;
  min-height: 0;
}

.win-body::-webkit-scrollbar { width: 3px; }
.win-body::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

.win-input {
  flex-shrink: 0;
  border-top: 1px solid rgba(255,255,255,0.04);
}

.win-footer {
  flex-shrink: 0;
  border-top: 1px solid rgba(255,255,255,0.04);
}

/* Window accent variants */
.window.ac-cyan   { --win-color: var(--cyan);   border-color: rgba(0,229,255,0.35);   box-shadow: 0 0 12px rgba(0,229,255,0.06); }
.window.ac-pink   { --win-color: var(--pink);   border-color: rgba(255,45,120,0.35);   box-shadow: 0 0 12px rgba(255,45,120,0.06); }
.window.ac-yellow { --win-color: var(--yellow); border-color: rgba(255,215,0,0.35);    box-shadow: 0 0 12px rgba(255,215,0,0.06); }
.window.ac-green  { --win-color: var(--green);  border-color: rgba(57,255,138,0.35);   box-shadow: 0 0 12px rgba(57,255,138,0.06); }
.window.ac-purple { --win-color: var(--purple); border-color: rgba(176,108,255,0.35);  box-shadow: 0 0 12px rgba(176,108,255,0.06); }

/* ── LEFT SIDEBAR ──────────────────────── */
#sidebar {
  width: 220px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  padding: 3px;
  overflow: hidden;
  min-width: 140px;
  max-width: 400px;
}

#explorer-window { flex: 1; }

#new-btn {
  display: block;
  width: calc(100% - 12px);
  margin: 6px auto 4px;
  padding: 7px 10px;
  background: linear-gradient(135deg, var(--cyan), var(--purple));
  color: #fff;
  border: none;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 600;
  cursor: pointer;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  transition: opacity .15s;
}
#new-btn:hover { opacity: .85; }

#sessions { padding: 2px 4px; }

.sess {
  padding: 5px 8px;
  border-radius: 3px;
  font-size: 10px;
  color: var(--muted);
  cursor: pointer;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  transition: all .12s;
  margin-bottom: 1px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 4px;
}
.sess:hover { background: var(--s2); color: var(--text); }
.sess.active { background: var(--s2); color: var(--cyan); border-left: 2px solid var(--cyan); padding-left: 6px; }
.sess-name { flex:1; overflow:hidden; text-overflow:ellipsis; }
.sess-ico { color: var(--muted); font-size: 8px; flex-shrink:0; }
.sess-del {
  opacity: 0;
  font-size: 9px;
  color: var(--red);
  padding: 1px 3px;
  flex-shrink: 0;
  cursor: pointer;
}
.sess:hover .sess-del { opacity: 1; }

.sidebar-stats {
  padding: 8px 10px;
  font-size: 9px;
  color: var(--muted);
  line-height: 2;
}
.sidebar-stats .sr { display:flex; justify-content:space-between; }
.sidebar-stats .sv { color: var(--cyan); }

/* ── CENTER AREA ───────────────────────── */
#center {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 3px;
  overflow: hidden;
  min-width: 0;
}

#chat-window { flex: 6; min-height: 80px; }
#image-window { flex: 4; min-height: 80px; }

/* ── CHAT MESSAGES ─────────────────────── */
#msgs {
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.msg { animation: fadeUp .2s ease; }
@keyframes fadeUp {
  from { opacity:0; transform:translateY(4px); }
  to   { opacity:1; transform:translateY(0); }
}

.msg-head {
  font-size: 9px;
  color: var(--muted);
  margin-bottom: 3px;
  display: flex;
  align-items: center;
  gap: 6px;
}
.msg.user .msg-head { justify-content:flex-end; }

.role-tag {
  font-weight: 700;
  font-size: 8px;
  padding: 1px 5px;
  border-radius: 2px;
  text-transform: uppercase;
  letter-spacing: .6px;
}
.msg.user .role-tag { background: rgba(0,229,255,.12); color: var(--cyan); }
.msg.assistant .role-tag { background: rgba(57,255,138,.1); color: var(--green); }

.msg.user { align-self:flex-end; max-width:70%; }
.msg.assistant { align-self:flex-start; max-width:82%; }

.bubble {
  padding: 10px 14px;
  border-radius: 6px;
  font-size: 12px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
}

.msg.user .bubble {
  background: var(--s2);
  border: 1px solid var(--border2);
  border-top-right-radius: 1px;
}
.msg.assistant .bubble {
  background: var(--s1);
  border: 1px solid var(--border);
  border-top-left-radius: 1px;
}

.bubble pre {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 8px 10px;
  margin: 6px 0;
  overflow-x: auto;
  font-size: 11px;
  line-height: 1.5;
}
.bubble code.inline {
  background: var(--s3);
  padding: 1px 5px;
  border-radius: 3px;
  font-size: 11px;
  color: var(--orange);
}

.badges { display:flex; gap:4px; flex-wrap:wrap; margin-top:5px; }
.badge {
  padding: 2px 7px;
  border-radius: 3px;
  font-size: 8px;
  font-weight: 600;
  letter-spacing: .4px;
  text-transform: uppercase;
}
.badge.mem { background:rgba(176,108,255,.1); color:var(--purple); border:1px solid rgba(176,108,255,.25); }
.badge.web { background:rgba(0,229,255,.08); color:var(--cyan); border:1px solid rgba(0,229,255,.2); }
.badge.temporal { background:rgba(255,215,0,.08); color:var(--yellow); border:1px solid rgba(255,215,0,.2); }

/* typing cursor */
.tcursor {
  display: inline-block;
  width: 7px; height: 13px;
  background: var(--green);
  margin-left: 1px;
  vertical-align: text-bottom;
  animation: blink .6s infinite;
}
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0; } }

/* empty state */
#empty {
  flex:1; display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  gap:8px; text-align:center; padding:20px;
}
#empty-icon {
  font-size: 36px;
  opacity: 0.6;
  margin-bottom: 4px;
}
#empty-title {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 22px;
  font-weight: 700;
  background: linear-gradient(135deg, var(--cyan), var(--green));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.empty-sub { font-size: 10px; color: var(--muted); }
.hints { margin-top:12px; display:flex; flex-wrap:wrap; gap:5px; justify-content:center; }
.hint {
  padding: 5px 10px;
  background: var(--s2);
  border: 1px solid var(--border2);
  border-radius: 4px;
  font-size: 10px;
  cursor: pointer;
  transition: all .15s;
  color: var(--muted);
}
.hint:hover { border-color: var(--cyan); color: var(--cyan); }

/* chat input */
.chat-input-row {
  display: flex;
  gap: 6px;
  padding: 8px 10px;
  align-items: flex-end;
}

#user-input {
  flex: 1;
  background: var(--s2);
  border: 1px solid var(--border2);
  border-radius: 4px;
  padding: 8px 10px;
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  resize: none;
  min-height: 36px;
  max-height: 120px;
  outline: none;
  line-height: 1.5;
  transition: border-color .15s;
}
#user-input:focus { border-color: var(--cyan); box-shadow: 0 0 8px rgba(0,229,255,0.1); }
#user-input::placeholder { color: var(--muted2); }

#send-btn {
  height: 36px;
  padding: 0 14px;
  background: linear-gradient(135deg, var(--green), var(--cyan));
  color: var(--bg);
  border: none;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  cursor: pointer;
  transition: opacity .15s;
  white-space: nowrap;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
#send-btn:hover { opacity:.85; }
#send-btn:disabled { opacity:.3; cursor:not-allowed; }

/* ── IMAGE LAB ─────────────────────────── */
#image-display {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 10px;
  min-height: 60px;
}

#img-empty {
  color: var(--muted);
  font-size: 10px;
  text-align: center;
  padding: 16px;
}

#img-result {
  max-width: 100%;
  max-height: 100%;
  border-radius: 4px;
  border: 1px solid var(--border2);
  display: none;
}

.img-loading {
  text-align: center;
  padding: 20px;
}
.img-loading-text {
  color: var(--pink);
  font-size: 10px;
  letter-spacing: 1px;
  margin-bottom: 8px;
  animation: pulse 1.5s infinite;
}
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.4; } }
.img-loading-bar {
  width: 80%;
  max-width: 200px;
  height: 2px;
  background: var(--s3);
  margin: 0 auto;
  border-radius: 1px;
  overflow: hidden;
  position: relative;
}
.img-loading-bar::after {
  content: '';
  position: absolute;
  width: 40%;
  height: 100%;
  background: linear-gradient(90deg, transparent, var(--pink), transparent);
  animation: loadSlide 1.2s infinite;
}
@keyframes loadSlide { from { left: -40%; } to { left: 100%; } }

.img-input-row {
  display: flex;
  gap: 6px;
  padding: 8px 10px;
  align-items: center;
}

#image-prompt {
  flex: 1;
  background: var(--s2);
  border: 1px solid var(--border2);
  border-radius: 4px;
  padding: 7px 10px;
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  outline: none;
  transition: border-color .15s;
}
#image-prompt:focus { border-color: var(--pink); box-shadow: 0 0 8px rgba(255,45,120,0.1); }
#image-prompt::placeholder { color: var(--muted2); }

#gen-btn {
  height: 32px;
  padding: 0 12px;
  background: linear-gradient(135deg, var(--pink), var(--purple));
  color: #fff;
  border: none;
  border-radius: 4px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  font-weight: 700;
  cursor: pointer;
  transition: opacity .15s;
  white-space: nowrap;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
#gen-btn:hover { opacity:.85; }
#gen-btn:disabled { opacity:.3; cursor:not-allowed; }

#img-gallery {
  display: flex;
  gap: 4px;
  padding: 4px 10px 6px;
  overflow-x: auto;
}
#img-gallery::-webkit-scrollbar { height: 2px; }
#img-gallery::-webkit-scrollbar-thumb { background: var(--border2); }

.gallery-thumb {
  width: 40px; height: 40px;
  border-radius: 3px;
  border: 1px solid var(--border2);
  object-fit: cover;
  cursor: pointer;
  opacity: 0.6;
  transition: opacity .15s;
  flex-shrink: 0;
}
.gallery-thumb:hover { opacity: 1; border-color: var(--pink); }
.gallery-thumb.active { opacity: 1; border-color: var(--pink); }

/* ── RIGHT SIDEBAR ─────────────────────── */
#rightbar {
  width: 260px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 3px;
  padding: 3px 3px 3px 0;
  overflow: hidden;
  min-width: 180px;
  max-width: 420px;
}

#status-window { flex: 0 0 auto; }
#memory-window { flex: 0 0 auto; }
#rag-window { flex: 1; min-height: 0; }

/* status panel */
.status-grid {
  padding: 6px 10px;
}
.status-row {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 0;
  font-size: 10px;
}
.status-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}
.sd-green { background: var(--green); box-shadow: 0 0 4px var(--green); }
.sd-cyan { background: var(--cyan); box-shadow: 0 0 4px var(--cyan); }
.sd-yellow { background: var(--yellow); box-shadow: 0 0 4px var(--yellow); }
.sd-red { background: var(--red); box-shadow: 0 0 4px var(--red); }
.sd-off { background: var(--muted2); }

.status-label { color: var(--text2); flex:1; }
.status-val { color: var(--win-color, var(--text)); font-weight: 500; text-align:right; max-width:120px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

/* memory panel */
.mem-grid { padding: 6px 10px; }
.mem-row { display:flex; justify-content:space-between; align-items:center; padding:2px 0; font-size:10px; }
.mem-label { color: var(--text2); }
.mem-val { color: var(--purple); font-weight:600; }
.mem-bar-wrap { margin-top:4px; }
.mem-bar {
  width: 100%;
  height: 4px;
  background: var(--s3);
  border-radius: 2px;
  overflow: hidden;
}
.mem-bar-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--purple), var(--cyan));
  border-radius: 2px;
  transition: width 0.5s ease;
  width: 0%;
}
.mem-bar-label {
  font-size: 8px;
  color: var(--muted);
  margin-top: 2px;
  text-align: right;
}

/* rag panel */
.rag-content { padding: 6px 10px; }
#rag-empty { color: var(--muted); font-size: 9px; padding: 12px 0; text-align:center; }

.rag-item {
  padding: 5px 0;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  animation: fadeUp .2s ease;
}
.rag-item:last-child { border-bottom: none; }
.rag-head {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 2px;
}
.rag-score {
  font-size: 8px;
  font-weight: 700;
  padding: 1px 4px;
  border-radius: 2px;
  background: rgba(0,229,255,.1);
  color: var(--cyan);
}
.rag-ts {
  font-size: 8px;
  color: var(--muted);
}
.rag-text {
  font-size: 9px;
  color: var(--text2);
  line-height: 1.4;
  overflow: hidden;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}
.rag-temporal {
  font-size: 8px;
  color: var(--yellow);
  margin-top: 1px;
}
.rag-source {
  font-size: 7px;
  font-weight: 700;
  padding: 1px 4px;
  border-radius: 2px;
  letter-spacing: .4px;
  text-transform: uppercase;
}
.rag-source.src-mem { background: rgba(176,108,255,.15); color: var(--purple); }
.rag-source.src-doc { background: rgba(255,159,67,.15); color: var(--orange); }

/* knowledge base panel */
.window.ac-orange { --win-color: var(--orange); border-color: rgba(255,159,67,0.35); box-shadow: 0 0 12px rgba(255,159,67,0.06); }
#kb-window { flex: 0 0 auto; }

.kb-list { padding: 6px 10px; }
.kb-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 4px 0;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  font-size: 9px;
}
.kb-item:last-child { border-bottom: none; }
.kb-info { flex:1; min-width:0; }
.kb-name { color: var(--text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.kb-meta { color: var(--muted); font-size: 8px; }
.kb-del {
  color: var(--red);
  cursor: pointer;
  font-size: 9px;
  padding: 2px 4px;
  opacity: 0.5;
  flex-shrink: 0;
}
.kb-del:hover { opacity: 1; }
.kb-empty { color: var(--muted); font-size: 9px; padding: 8px 0; text-align: center; }

/* upload button */
#upload-btn {
  height: 36px;
  padding: 0 10px;
  background: var(--s2);
  border: 1px solid var(--border2);
  border-radius: 4px;
  color: var(--orange);
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  cursor: pointer;
  transition: all .15s;
  display: flex;
  align-items: center;
  justify-content: center;
}
#upload-btn:hover { border-color: var(--orange); background: rgba(255,159,67,0.06); }

/* doc badge */
.badge.doc { background:rgba(255,159,67,.1); color:var(--orange); border:1px solid rgba(255,159,67,.25); }

/* ── FOOTER / CONSOLE ──────────────────── */
#footer {
  flex-shrink: 0;
  height: 90px;
  padding: 3px;
  min-height: 30px;
  max-height: 200px;
}

#console-window { height: 100%; }

#console-body {
  display: flex;
  flex-direction: row;
  height: 100%;
  min-height: 0;
}

#console-log {
  display: flex;
  flex-direction: column;
  gap: 1px;
  padding: 4px 10px;
  overflow-y: auto;
  overflow-x: hidden;
  flex: 1;
  min-height: 0;
}
#console-log::-webkit-scrollbar { width: 2px; }
#console-log::-webkit-scrollbar-thumb { background: var(--border2); }

.clog {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 9px;
  flex-shrink: 0;
  white-space: nowrap;
}
.clog-time { color: var(--muted); }
.clog-msg { color: var(--text2); }
.clog-msg.ok { color: var(--green); }
.clog-msg.warn { color: var(--yellow); }
.clog-msg.err { color: var(--red); }
.clog-msg.info { color: var(--cyan); }

#console-right {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 4px 10px;
  font-size: 9px;
  color: var(--muted);
  flex-shrink: 0;
  border-left: 1px solid rgba(255,255,255,0.04);
  justify-content: center;
}
.cr-item { display:flex; gap:3px; align-items:center; }
.cr-val { color: var(--yellow); font-weight:600; }

/* ── SETTINGS MODAL ────────────────────── */
#modal-bg {
  display:none;
  position:fixed; inset:0;
  background: rgba(0,0,0,.8);
  z-index:500;
  align-items:center;
  justify-content:center;
}
#modal-bg.open { display:flex; }

#modal {
  background: var(--s1);
  border: 1px solid var(--cyan);
  border-radius: 4px;
  width: 440px;
  max-width: 92vw;
  box-shadow: 0 0 30px rgba(0,229,255,0.1);
}

.modal-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
}
.modal-title {
  font-size: 11px;
  font-weight: 700;
  color: var(--cyan);
  letter-spacing: 0.8px;
  text-transform: uppercase;
}

.modal-body { padding: 16px; }

.srow { margin-bottom: 12px; }
.srow label {
  display: block;
  font-size: 9px;
  color: var(--muted);
  margin-bottom: 4px;
  letter-spacing: .4px;
  text-transform: uppercase;
}
.srow input, .srow select, .srow textarea {
  width: 100%;
  background: var(--s2);
  border: 1px solid var(--border2);
  border-radius: 3px;
  padding: 7px 10px;
  color: var(--text);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  outline: none;
  transition: border-color .15s;
}
.srow input:focus, .srow select:focus, .srow textarea:focus { border-color: var(--cyan); }
.srow textarea { min-height: 70px; resize: vertical; }
.srow select option { background: var(--s2); }

.modal-actions {
  display: flex;
  gap: 6px;
  justify-content: flex-end;
  padding: 0 16px 14px;
}

.btn-cancel {
  padding: 6px 14px;
  background: transparent;
  border: 1px solid var(--border2);
  color: var(--muted);
  border-radius: 3px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  cursor: pointer;
}
.btn-save {
  padding: 6px 16px;
  background: var(--cyan);
  border: none;
  color: var(--bg);
  border-radius: 3px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 700;
  cursor: pointer;
}

/* ── ANIMATIONS ────────────────────────── */
@keyframes glow {
  0%,100% { opacity:1; } 50% { opacity:.5; }
}
</style>
</head>
<body>
<div id="app">
<div id="scanline"></div>

<!-- TITLE BAR -->
<header id="titlebar">
  <div id="tb-left">
    <span class="tb-dot r"></span>
    <span class="tb-dot y"></span>
    <span class="tb-dot g"></span>
    <span id="tb-title">MICRO GPT 2.0 &mdash; LOCAL AI WORKSTATION</span>
  </div>
  <div id="tb-right">
    <select id="model-select" onchange="switchModel(this.value)"></select>
    <button class="tb-btn" id="search-btn" onclick="toggleSearch()">SEARCH: OFF</button>
    <button class="tb-btn" onclick="openSettings()">SETTINGS</button>
    <span id="tb-clock">--:--</span>
  </div>
</header>

<!-- MAIN AREA (flex row: sidebar | splitter | center | splitter | rightbar) -->
<div id="main-area">

<!-- LEFT SIDEBAR -->
<aside id="sidebar">
  <div class="window ac-cyan" id="explorer-window">
    <div class="win-bar">
      <span class="win-title">SESSION_EXPLORER.db</span>
      <div class="win-right">
        <div class="win-dots">
          <span class="wd wd-min" onclick="toggleMin('explorer-window')"></span>
          <span class="wd wd-max"></span>
          <span class="wd wd-close"></span>
        </div>
      </div>
    </div>
    <div class="win-body">
      <button id="new-btn" onclick="newChat()">+ NEW SESSION</button>
      <div id="sessions"></div>
    </div>
    <div class="win-footer">
      <div class="sidebar-stats">
        <div class="sr"><span>Messages</span><span class="sv" id="st-msgs">0</span></div>
        <div class="sr"><span>Vectors</span><span class="sv" id="st-vecs">0</span></div>
        <div class="sr"><span>Sessions</span><span class="sv" id="st-sess">0</span></div>
      </div>
    </div>
  </div>
</aside>

<div class="splitter splitter-v" id="split-left"></div>

<!-- CENTER -->
<main id="center">
  <!-- Chat Window -->
  <div class="window ac-green" id="chat-window">
    <div class="win-bar">
      <span class="win-title">CHAT_TERMINAL.gpt</span>
      <div class="win-right">
        <span id="chat-badge" style="font-size:8px;color:var(--muted);letter-spacing:0.3px"></span>
        <div class="win-dots">
          <span class="wd wd-min" onclick="toggleMin('chat-window')"></span>
          <span class="wd wd-max"></span>
          <span class="wd wd-close"></span>
        </div>
      </div>
    </div>
    <div class="win-body" id="chat-body">
      <div id="msgs">
        <div id="empty">
          <div id="empty-icon">&#9781;</div>
          <div id="empty-title">MICRO GPT 2.0</div>
          <div class="empty-sub">100% local &middot; persistent vector memory &middot; temporal decay RAG</div>
          <div class="hints">
            <div class="hint" onclick="useHint(this)">what do you remember about me?</div>
            <div class="hint" onclick="useHint(this)">summarize our conversations</div>
            <div class="hint" onclick="useHint(this)">explain vector embeddings</div>
            <div class="hint" onclick="useHint(this)">search the web for latest news</div>
          </div>
        </div>
      </div>
    </div>
    <div class="win-input">
      <div class="chat-input-row">
        <textarea id="user-input" placeholder="Enter message..." rows="1"
          onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
        <input type="file" id="file-input" accept=".txt,.md,.pdf" style="display:none" onchange="uploadDoc(this)">
        <button id="upload-btn" onclick="document.getElementById('file-input').click()" title="Upload document (.txt, .md, .pdf)">&#128206;</button>
        <button id="send-btn" onclick="sendMsg()">SEND &#9654;</button>
      </div>
    </div>
  </div>

  <div class="splitter splitter-h" id="split-center"></div>

  <!-- Image Lab -->
  <div class="window ac-pink" id="image-window">
    <div class="win-bar">
      <span class="win-title">IMAGE_LAB.flux</span>
      <div class="win-right">
        <span style="font-size:8px;color:var(--muted)">x/flux2-klein:latest</span>
        <div class="win-dots">
          <span class="wd wd-min" onclick="toggleMin('image-window')"></span>
          <span class="wd wd-max"></span>
          <span class="wd wd-close"></span>
        </div>
      </div>
    </div>
    <div class="win-body" id="image-body">
      <div id="image-display">
        <div id="img-empty">Enter a prompt below to generate an image<br><span style="color:var(--pink)">MODEL: x/flux2-klein:latest</span></div>
        <img id="img-result" />
      </div>
    </div>
    <div class="win-footer" id="img-gallery"></div>
    <div class="win-input">
      <div class="img-input-row">
        <input id="image-prompt" type="text" placeholder="Describe your image..."
          onkeydown="if(event.key==='Enter'){event.preventDefault();generateImage()}" />
        <button id="gen-btn" onclick="generateImage()">GENERATE &#9889;</button>
      </div>
    </div>
  </div>
</main>

<div class="splitter splitter-v" id="split-right"></div>

<!-- RIGHT SIDEBAR -->
<aside id="rightbar">
  <!-- System Status -->
  <div class="window ac-yellow" id="status-window">
    <div class="win-bar">
      <span class="win-title">SYSTEM_STATUS</span>
      <div class="win-dots">
        <span class="wd wd-min" onclick="toggleMin('status-window')"></span>
        <span class="wd wd-max"></span>
        <span class="wd wd-close"></span>
      </div>
    </div>
    <div class="win-body">
      <div class="status-grid">
        <div class="status-row">
          <span class="status-dot sd-green" id="sys-ollama-dot"></span>
          <span class="status-label">Ollama</span>
          <span class="status-val" id="sys-ollama">checking...</span>
        </div>
        <div class="status-row">
          <span class="status-dot sd-cyan"></span>
          <span class="status-label">Model</span>
          <span class="status-val" id="sys-model">-</span>
        </div>
        <div class="status-row">
          <span class="status-dot sd-cyan"></span>
          <span class="status-label">Embed</span>
          <span class="status-val" id="sys-embed">-</span>
        </div>
        <div class="status-row">
          <span class="status-dot sd-off" id="sys-search-dot"></span>
          <span class="status-label">Search</span>
          <span class="status-val" id="sys-search">OFF</span>
        </div>
        <div class="status-row">
          <span class="status-dot sd-yellow"></span>
          <span class="status-label">Decay</span>
          <span class="status-val" id="sys-decay">0.96/day</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Memory Bank -->
  <div class="window ac-purple" id="memory-window">
    <div class="win-bar">
      <span class="win-title">MEMORY_BANK.vec</span>
      <div class="win-dots">
        <span class="wd wd-min" onclick="toggleMin('memory-window')"></span>
        <span class="wd wd-max"></span>
        <span class="wd wd-close"></span>
      </div>
    </div>
    <div class="win-body">
      <div class="mem-grid">
        <div class="mem-row"><span class="mem-label">Messages</span><span class="mem-val" id="mem-total">0</span></div>
        <div class="mem-row"><span class="mem-label">Vectorized</span><span class="mem-val" id="mem-vecs">0</span></div>
        <div class="mem-row"><span class="mem-label">Sessions</span><span class="mem-val" id="mem-sess">0</span></div>
        <div class="mem-bar-wrap">
          <div class="mem-bar"><div class="mem-bar-fill" id="mem-bar-fill"></div></div>
          <div class="mem-bar-label" id="mem-pct">0% vectorized</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Knowledge Base -->
  <div class="window ac-orange" id="kb-window">
    <div class="win-bar">
      <span class="win-title">KNOWLEDGE_BASE.docs</span>
      <div class="win-right">
        <span id="kb-count" style="font-size:8px;color:var(--muted)"></span>
        <div class="win-dots">
          <span class="wd wd-min" onclick="toggleMin('kb-window')"></span>
          <span class="wd wd-max"></span>
          <span class="wd wd-close"></span>
        </div>
      </div>
    </div>
    <div class="win-body">
      <div class="kb-list" id="kb-list">
        <div class="kb-empty" id="kb-empty">No documents uploaded</div>
      </div>
    </div>
  </div>

  <!-- RAG Context -->
  <div class="window ac-cyan" id="rag-window">
    <div class="win-bar">
      <span class="win-title">RAG_CONTEXT</span>
      <div class="win-right">
        <span id="rag-count" style="font-size:8px;color:var(--muted)"></span>
        <div class="win-dots">
          <span class="wd wd-min" onclick="toggleMin('rag-window')"></span>
          <span class="wd wd-max"></span>
          <span class="wd wd-close"></span>
        </div>
      </div>
    </div>
    <div class="win-body" id="rag-body">
      <div class="rag-content">
        <div id="rag-empty">Memories will appear here during chat</div>
        <div id="rag-list"></div>
      </div>
    </div>
  </div>
</aside>

</div><!-- /main-area -->

<div class="splitter splitter-h" id="split-footer"></div>

<!-- FOOTER CONSOLE -->
<footer id="footer">
  <div class="window ac-yellow" id="console-window" style="height:100%">
    <div class="win-bar">
      <span class="win-title">CONSOLE</span>
      <div class="win-right">
        <span id="console-count" style="font-size:8px;color:var(--muted)"></span>
        <div class="win-dots">
          <span class="wd wd-min" onclick="toggleMin('console-window')"></span>
          <span class="wd wd-max"></span>
          <span class="wd wd-close"></span>
        </div>
      </div>
    </div>
    <div class="win-body" id="console-body">
      <div id="console-log"></div>
      <div id="console-right">
        <span class="cr-item">VEC: <span class="cr-val" id="cr-vecs">0</span></span>
        <span class="cr-item">MEM: <span class="cr-val" id="cr-mem">0</span></span>
        <span class="cr-item" style="color:var(--green)">v2.0</span>
      </div>
    </div>
  </div>
</footer>
</div>

<!-- SETTINGS MODAL -->
<div id="modal-bg" onclick="if(event.target===this)closeSettings()">
  <div id="modal">
    <div class="modal-bar">
      <span class="modal-title">SETTINGS</span>
      <div class="win-dots">
        <span class="wd wd-close" onclick="closeSettings()"></span>
      </div>
    </div>
    <div class="modal-body">
      <div class="srow">
        <label>CHAT MODEL</label>
        <select id="s-model"></select>
      </div>
      <div class="srow">
        <label>EMBEDDING MODEL</label>
        <input id="s-embed" type="text" placeholder="nomic-embed-text">
      </div>
      <div class="srow">
        <label>MEMORY DEPTH (top-k)</label>
        <input id="s-k" type="number" min="1" max="20" value="5">
      </div>
      <div class="srow">
        <label>SYSTEM PROMPT</label>
        <textarea id="s-sys"></textarea>
      </div>
    </div>
    <div class="modal-actions">
      <button class="btn-cancel" onclick="closeSettings()">CANCEL</button>
      <button class="btn-save" onclick="saveSettings()">SAVE</button>
    </div>
  </div>
</div>

<script>
let session = null;
let streaming = false;
let imgGenerating = false;
let cfg = {};
let sessionImages = [];
const consoleLogs = [];

// ── BOOT ──────────────────────────────────────────────────────
async function init() {
  logC('MICRO GPT 2.0 — BOOT SEQUENCE', 'info');
  updateClock();
  setInterval(updateClock, 1000);

  try {
    await loadCfg();
    logC('Settings loaded', 'ok');
  } catch(e) {
    logC('Failed to load settings', 'err');
  }

  try {
    await loadModels();
    logC('Models loaded', 'ok');
  } catch(e) {
    logC('Could not reach Ollama', 'err');
  }

  await loadSessions();
  await updateAllStats();
  await loadKnowledgeBase();
  checkOllama();
  logC('System ready. All systems nominal.', 'ok');
}

async function loadCfg() {
  const r = await fetch('/api/settings');
  cfg = await r.json();
  updateStatusPanel();
  setSearchState(cfg.web_search === 'true');
}

async function loadModels() {
  const r = await fetch('/api/models');
  const models = await r.json();
  const sel = document.getElementById('model-select');
  sel.innerHTML = '';
  if (models.length) {
    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m;
      opt.textContent = m;
      if (m === cfg.model) opt.selected = true;
      sel.appendChild(opt);
    });
  } else {
    sel.innerHTML = '<option>' + (cfg.model || 'no models') + '</option>';
  }
}

async function checkOllama() {
  try {
    const r = await fetch('/api/models');
    const m = await r.json();
    document.getElementById('sys-ollama').textContent = 'Connected';
    document.getElementById('sys-ollama-dot').className = 'status-dot sd-green';
  } catch {
    document.getElementById('sys-ollama').textContent = 'Offline';
    document.getElementById('sys-ollama-dot').className = 'status-dot sd-red';
  }
}

function updateStatusPanel() {
  document.getElementById('sys-model').textContent = cfg.model || '-';
  document.getElementById('sys-embed').textContent = cfg.embed_model || '-';
  const searchOn = cfg.web_search === 'true';
  document.getElementById('sys-search').textContent = searchOn ? 'ON' : 'OFF';
  document.getElementById('sys-search-dot').className = 'status-dot ' + (searchOn ? 'sd-green' : 'sd-off');
}

function updateClock() {
  const now = new Date();
  document.getElementById('tb-clock').textContent =
    now.toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
}

// ── SESSIONS ─────────────────────────────────────────────────
async function loadSessions() {
  const r = await fetch('/api/sessions');
  const list = await r.json();
  const el = document.getElementById('sessions');
  el.innerHTML = '';
  if (!list.length) {
    el.innerHTML = '<div style="padding:8px;font-size:9px;color:var(--muted)">No sessions yet</div>';
    return;
  }
  list.forEach(s => {
    const d = document.createElement('div');
    d.className = 'sess' + (s.id === session ? ' active' : '');
    d.innerHTML = '<span class="sess-ico">&#9656;</span><span class="sess-name">' + esc(s.name) + '</span><span class="sess-del" onclick="delSession(event,' + s.id + ')">&#10005;</span>';
    d.onclick = (e) => { if (!e.target.classList.contains('sess-del')) loadSession(s.id); };
    el.appendChild(d);
  });
}

async function newChat() {
  const r = await fetch('/api/sessions', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name:'New Chat'})
  });
  const s = await r.json();
  session = s.id;
  document.getElementById('msgs').innerHTML = '';
  document.getElementById('user-input').focus();
  sessionImages = [];
  updateGallery();
  const display = document.getElementById('image-display');
  if (display) display.innerHTML = '<div id="img-empty" style="color:var(--muted);font-size:10px;padding:16px">No image yet — type a prompt and hit GENERATE</div>';
  await loadSessions();
  logC('New session #' + s.id + ' created', 'info');
}

async function loadSession(id) {
  session = id;
  const r = await fetch('/api/history/' + id);
  const hist = await r.json();
  const msgs = document.getElementById('msgs');
  msgs.innerHTML = '';
  hist.forEach(m => appendMsg(m.role, m.content, null, false));
  document.getElementById('chat-body').scrollTop = 999999;
  await loadSessionImages();
  await loadSessions();
  logC('Session #' + id + ' loaded (' + hist.length + ' msgs)', 'info');
}

async function delSession(e, id) {
  e.stopPropagation();
  if (!confirm('Delete this session?')) return;
  await fetch('/api/sessions/' + id, {method:'DELETE'});
  if (session === id) {
    session = null;
    document.getElementById('msgs').innerHTML = emptyHTML();
    sessionImages = [];
    updateGallery();
    const display = document.getElementById('image-display');
    if (display) display.innerHTML = '<div id="img-empty" style="color:var(--muted);font-size:10px;padding:16px">No image yet — type a prompt and hit GENERATE</div>';
  }
  await loadSessions();
  updateAllStats();
  logC('Session #' + id + ' deleted', 'warn');
}

// ── SEND MESSAGE ─────────────────────────────────────────────
async function sendMsg() {
  if (streaming) return;
  const inp = document.getElementById('user-input');
  const txt = inp.value.trim();
  if (!txt) return;

  if (!session) await newChat();

  const empty = document.getElementById('empty');
  if (empty) empty.remove();

  inp.value = '';
  autoResize(inp);
  appendMsg('user', txt, null, false);

  streaming = true;
  document.getElementById('send-btn').disabled = true;
  document.getElementById('chat-badge').textContent = 'STREAMING...';

  logC('Message sent to ' + (cfg.model||'llama3.2'), 'info');

  const asstEl = appendMsg('assistant', '', null, true);
  const bubble = asstEl.querySelector('.bubble');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({session_id: session, message: txt})
    });

    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    let meta = null;
    let full = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const lines = buf.split('\n');
      buf = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const d = JSON.parse(line.slice(6));
          if (d.type === 'meta') {
            meta = d;
            if (meta.memories > 0) {
              logC('RAG: ' + meta.memories + ' memories retrieved', 'ok');
              updateRagPanel(meta.snippets || []);
            }
            if (meta.searched) logC('Web search performed', 'info');
          } else if (d.type === 'chunk') {
            full += d.content;
            bubble.innerHTML = renderContent(full) + '<span class="tcursor"></span>';
          } else if (d.type === 'done') {
            bubble.innerHTML = renderContent(full);
            if (meta && (meta.memories > 0 || meta.searched)) {
              addBadges(asstEl, meta);
            }
          }
        } catch {}
      }
      document.getElementById('chat-body').scrollTop = 999999;
    }
  } catch (err) {
    bubble.innerHTML = '<span style="color:var(--red)">[Connection error - is Ollama running?]</span>';
    logC('Connection error', 'err');
  }

  bubble.querySelector('.tcursor')?.remove();
  streaming = false;
  document.getElementById('send-btn').disabled = false;
  document.getElementById('chat-badge').textContent = '';
  await loadSessions();
  updateAllStats();
}

// ── IMAGE GENERATION ─────────────────────────────────────────
const IMAGE_MODEL = 'x/flux2-klein:latest';

async function generateImage() {
  if (imgGenerating) return;
  const inp = document.getElementById('image-prompt');
  const prompt = inp.value.trim();
  if (!prompt) return;

  if (!session) await newChat();

  imgGenerating = true;
  document.getElementById('gen-btn').disabled = true;
  document.getElementById('gen-btn').textContent = 'WORKING...';

  const display = document.getElementById('image-display');
  display.innerHTML =
    '<div class="img-loading">' +
      '<div class="img-loading-text">GENERATING WITH ' + IMAGE_MODEL + '</div>' +
      '<div class="img-loading-bar"></div>' +
      '<div style="font-size:9px;color:var(--muted);margin-top:8px">' + esc(prompt) + '</div>' +
    '</div>';

  logC('Image gen started [' + IMAGE_MODEL + ']: "' + prompt.slice(0,40) + '..."', 'info');

  try {
    const start = Date.now();
    const r = await fetch('/api/generate-image', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({prompt: prompt, session_id: session})
    });

    const data = await r.json();
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);

    if (data.error) {
      display.innerHTML = '<div id="img-empty" style="color:var(--red);padding:16px;font-size:10px">' + esc(data.error) + '</div>';
      logC('Image gen failed: ' + data.error.slice(0,60), 'err');
    } else if (data.images && data.images.length > 0) {
      const imgSrc = 'data:image/png;base64,' + data.images[0];
      display.innerHTML = '<img id="img-result" src="' + imgSrc + '" style="display:block;max-width:100%;max-height:100%;border-radius:4px;border:1px solid var(--border2)" />';
      await loadSessionImages();
      logC('Image ready (' + elapsed + 's)', 'ok');
    } else if (data.response) {
      display.innerHTML = '<div id="img-empty" style="padding:16px;font-size:10px"><span style="color:var(--yellow)">Model returned text instead of image:</span><br><br>' + esc(data.response.slice(0,300)) + '</div>';
      logC('Got text response instead of image', 'warn');
    } else {
      display.innerHTML = '<div id="img-empty" style="padding:16px;font-size:10px;color:var(--yellow)">No image returned. Make sure ' + IMAGE_MODEL + ' is pulled:<br><code>ollama pull ' + IMAGE_MODEL + '</code></div>';
      logC('No image in response', 'warn');
    }
  } catch (err) {
    display.innerHTML = '<div id="img-empty" style="color:var(--red);padding:16px;font-size:10px">Error: ' + esc(err.message) + '<br><br>Make sure Ollama is running and ' + IMAGE_MODEL + ' is available.</div>';
    logC('Image gen error: ' + err.message, 'err');
  }

  imgGenerating = false;
  document.getElementById('gen-btn').disabled = false;
  document.getElementById('gen-btn').innerHTML = 'GENERATE &#9889;';
}

async function loadSessionImages() {
  if (!session) { sessionImages = []; updateGallery(); return; }
  try {
    const r = await fetch('/api/images/' + session);
    const data = await r.json();
    sessionImages = data.map(img => ({
      id: img.id,
      src: 'data:image/png;base64,' + img.image_data,
      prompt: img.prompt
    }));
  } catch(e) {
    sessionImages = [];
  }
  updateGallery();
  if (sessionImages.length > 0) {
    const last = sessionImages[sessionImages.length - 1];
    document.getElementById('image-display').innerHTML = '<img id="img-result" src="' + last.src + '" style="display:block;max-width:100%;max-height:100%;border-radius:4px;border:1px solid var(--border2)" />';
  } else {
    document.getElementById('image-display').innerHTML = '<div id="img-empty" style="color:var(--muted);font-size:10px;padding:16px">No image yet — type a prompt and hit GENERATE</div>';
  }
}

function updateGallery() {
  const gal = document.getElementById('img-gallery');
  gal.innerHTML = '';
  sessionImages.slice(-10).forEach((img, i) => {
    const thumb = document.createElement('img');
    thumb.className = 'gallery-thumb';
    thumb.src = img.src;
    thumb.title = img.prompt;
    thumb.onclick = () => {
      document.getElementById('image-display').innerHTML = '<img id="img-result" src="' + img.src + '" style="display:block;max-width:100%;max-height:100%;border-radius:4px;border:1px solid var(--border2)" />';
    };
    gal.appendChild(thumb);
  });
}

// ── RAG PANEL ────────────────────────────────────────────────
function updateRagPanel(snippets) {
  const list = document.getElementById('rag-list');
  const empty = document.getElementById('rag-empty');
  if (!snippets || !snippets.length) {
    empty.style.display = 'block';
    list.innerHTML = '';
    document.getElementById('rag-count').textContent = '';
    return;
  }
  empty.style.display = 'none';
  document.getElementById('rag-count').textContent = snippets.length + ' found';
  list.innerHTML = snippets.map(s => {
    const srcClass = s.source === 'doc' ? 'src-doc' : 'src-mem';
    const srcLabel = s.source === 'doc' ? 'DOC' : 'MEM';
    return '<div class="rag-item">' +
      '<div class="rag-head">' +
        '<span class="rag-source ' + srcClass + '">' + srcLabel + '</span>' +
        '<span class="rag-score">' + s.score.toFixed(2) + '</span>' +
        '<span class="rag-ts">' + esc(s.ts || '') + '</span>' +
        (s.session ? '<span class="rag-ts">' + esc(s.session) + '</span>' : '') +
      '</div>' +
      '<div class="rag-text">' + esc(s.text) + '</div>' +
      '<div class="rag-temporal">sim: ' + (s.similarity||0).toFixed(2) + ' | temporal: ' + (s.temporal||0).toFixed(2) + '</div>' +
    '</div>';
  }).join('');
}

// ── SEARCH TOGGLE ────────────────────────────────────────────
async function toggleSearch() {
  const newVal = cfg.web_search !== 'true';
  await fetch('/api/settings', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({web_search: newVal.toString()})
  });
  cfg.web_search = newVal.toString();
  setSearchState(newVal);
  updateStatusPanel();
  logC('Web search: ' + (newVal ? 'ON' : 'OFF'), newVal ? 'ok' : 'warn');
}

function setSearchState(on) {
  const btn = document.getElementById('search-btn');
  btn.textContent = 'SEARCH: ' + (on ? 'ON' : 'OFF');
  btn.className = 'tb-btn' + (on ? ' on' : '');
}

// ── MODEL SWITCH ─────────────────────────────────────────────
async function switchModel(model) {
  await fetch('/api/settings', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({model: model})
  });
  cfg.model = model;
  updateStatusPanel();
  logC('Model switched to ' + model, 'info');
}

// ── SETTINGS ─────────────────────────────────────────────────
async function openSettings() {
  try {
    const r = await fetch('/api/models');
    const models = await r.json();
    const sel = document.getElementById('s-model');
    sel.innerHTML = models.length
      ? models.map(m => '<option value="' + m + '"' + (m===cfg.model?' selected':'') + '>' + m + '</option>').join('')
      : '<option value="' + (cfg.model||'') + '">' + (cfg.model||'no models') + '</option>';
  } catch {}
  document.getElementById('s-embed').value = cfg.embed_model || 'nomic-embed-text';
  document.getElementById('s-k').value = cfg.memory_k || 5;
  document.getElementById('s-sys').value = cfg.system_prompt || '';
  document.getElementById('modal-bg').classList.add('open');
}

function closeSettings() {
  document.getElementById('modal-bg').classList.remove('open');
}

async function saveSettings() {
  const data = {
    model: document.getElementById('s-model').value,
    embed_model: document.getElementById('s-embed').value,
    memory_k: document.getElementById('s-k').value,
    system_prompt: document.getElementById('s-sys').value,
  };
  await fetch('/api/settings', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify(data)
  });
  Object.assign(cfg, data);

  // sync model dropdown
  const sel = document.getElementById('model-select');
  for (let opt of sel.options) {
    if (opt.value === data.model) { opt.selected = true; break; }
  }

  updateStatusPanel();
  closeSettings();
  logC('Settings saved', 'ok');
}

// ── STATS ────────────────────────────────────────────────────
async function updateAllStats() {
  try {
    const r = await fetch('/api/memory/stats');
    const s = await r.json();

    // sidebar
    document.getElementById('st-msgs').textContent = s.total;
    document.getElementById('st-vecs').textContent = s.vectorized;
    document.getElementById('st-sess').textContent = s.sessions;

    // memory bank
    document.getElementById('mem-total').textContent = s.total;
    document.getElementById('mem-vecs').textContent = s.vectorized;
    document.getElementById('mem-sess').textContent = s.sessions;

    const pct = s.total > 0 ? Math.round((s.vectorized / s.total) * 100) : 0;
    document.getElementById('mem-bar-fill').style.width = pct + '%';
    document.getElementById('mem-pct').textContent = pct + '% vectorized';

    // console footer
    document.getElementById('cr-vecs').textContent = s.vectorized;
    document.getElementById('cr-mem').textContent = s.total;
  } catch {}
}

// ── CONSOLE LOG ──────────────────────────────────────────────
function logC(msg, type) {
  type = type || 'info';
  const now = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit', second:'2-digit'});
  consoleLogs.push({time: now, msg: msg, type: type});

  const el = document.getElementById('console-log');
  const entry = document.createElement('span');
  entry.className = 'clog';
  entry.innerHTML = '<span class="clog-time">[' + now + ']</span> <span class="clog-msg ' + type + '">' + esc(msg) + '</span>';
  el.appendChild(entry);
  el.scrollTop = el.scrollHeight;

  document.getElementById('console-count').textContent = consoleLogs.length + ' events';

  // keep max 50
  while (el.children.length > 50) el.removeChild(el.firstChild);
}

// ── WINDOW MANAGEMENT ────────────────────────────────────────
function toggleMin(id) {
  document.getElementById(id).classList.toggle('minimized');
}

// ── MESSAGE RENDERING ────────────────────────────────────────
function appendMsg(role, content, meta, isStreaming) {
  const msgs = document.getElementById('msgs');
  const now = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML =
    '<div class="msg-head">' +
      '<span class="role-tag">' + role + '</span>' +
      '<span>' + now + '</span>' +
    '</div>' +
    '<div class="bubble">' + (isStreaming ? '<span class="tcursor"></span>' : renderContent(content)) + '</div>' +
    '<div class="badges"></div>';
  msgs.appendChild(div);
  document.getElementById('chat-body').scrollTop = 999999;
  return div;
}

function addBadges(el, meta) {
  const b = el.querySelector('.badges');
  const snippets = meta.snippets || [];
  const memCount = snippets.filter(s => s.source !== 'doc').length;
  const docCount = snippets.filter(s => s.source === 'doc').length;
  if (memCount > 0) {
    const span = document.createElement('span');
    span.className = 'badge mem';
    span.textContent = memCount + ' memor' + (memCount===1?'y':'ies');
    b.appendChild(span);
  }
  if (docCount > 0) {
    const span = document.createElement('span');
    span.className = 'badge doc';
    span.textContent = docCount + ' doc chunk' + (docCount===1?'':'s');
    b.appendChild(span);
  }
  if (meta.searched) {
    const span = document.createElement('span');
    span.className = 'badge web';
    span.textContent = 'web searched';
    b.appendChild(span);
  }
}

// ── CONTENT RENDERING (simple markdown) ──────────────────────
function renderContent(text) {
  if (!text) return '';
  const parts = text.split(/(```[\s\S]*?```)/g);
  return parts.map(part => {
    if (part.startsWith('```')) {
      const m = part.match(/```(\w*)\n?([\s\S]*?)```/);
      return '<pre>' + esc(m ? m[2] : part.slice(3,-3)) + '</pre>';
    }
    let h = esc(part);
    h = h.replace(/`([^`]+)`/g, '<code class="inline">$1</code>');
    h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    return h;
  }).join('');
}

// ── HELPERS ──────────────────────────────────────────────────
function esc(t) {
  return (t||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMsg(); }
}

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function useHint(el) {
  const inp = document.getElementById('user-input');
  inp.value = el.textContent;
  autoResize(inp);
  inp.focus();
}

function emptyHTML() {
  return '<div id="empty"><div id="empty-icon">&#9781;</div><div id="empty-title">MICRO GPT 2.0</div><div class="empty-sub">Select or create a session to begin</div></div>';
}

// ── DOCUMENT UPLOAD & KNOWLEDGE BASE ─────────────────────────
async function uploadDoc(input) {
  const file = input.files[0];
  if (!file) return;
  input.value = '';

  logC('Uploading ' + file.name + '...', 'info');
  const start = Date.now();

  const form = new FormData();
  form.append('file', file);

  try {
    const r = await fetch('/api/upload-doc', { method: 'POST', body: form });
    const data = await r.json();
    if (data.error) {
      logC('Upload failed: ' + data.error, 'err');
      return;
    }
    const elapsed = ((Date.now() - start) / 1000).toFixed(1);
    logC(file.name + ' — ' + data.chunks + ' chunks embedded (' + elapsed + 's)', 'ok');
    loadKnowledgeBase();
  } catch (err) {
    logC('Upload error: ' + err.message, 'err');
  }
}

async function loadKnowledgeBase() {
  try {
    const r = await fetch('/api/documents');
    const docs = await r.json();
    const list = document.getElementById('kb-list');
    const empty = document.getElementById('kb-empty');

    if (!docs.length) {
      empty.style.display = 'block';
      list.querySelectorAll('.kb-item').forEach(el => el.remove());
      document.getElementById('kb-count').textContent = '';
      return;
    }

    empty.style.display = 'none';
    const totalChunks = docs.reduce((a, d) => a + d.chunks, 0);
    document.getElementById('kb-count').textContent = docs.length + ' docs / ' + totalChunks + ' chunks';

    // remove old items, keep empty div
    list.querySelectorAll('.kb-item').forEach(el => el.remove());

    docs.forEach(d => {
      const item = document.createElement('div');
      item.className = 'kb-item';
      const date = (d.uploaded_at || '').slice(0, 10);
      item.innerHTML =
        '<div class="kb-info">' +
          '<div class="kb-name">' + esc(d.filename) + '</div>' +
          '<div class="kb-meta">' + d.chunks + ' chunks &middot; ' + date + '</div>' +
        '</div>' +
        '<span class="kb-del" title="Delete document" onclick="deleteDoc(\'' + esc(d.filename).replace(/'/g, "\\'") + '\')">&times;</span>';
      list.appendChild(item);
    });
  } catch {}
}

async function deleteDoc(filename) {
  if (!confirm('Delete "' + filename + '" and all its chunks?')) return;
  try {
    await fetch('/api/documents/' + encodeURIComponent(filename), { method: 'DELETE' });
    logC(filename + ' deleted', 'warn');
    loadKnowledgeBase();
  } catch (err) {
    logC('Delete failed: ' + err.message, 'err');
  }
}

// ── RESIZABLE SPLITTERS ──────────────────────────────────────
function initSplitters() {
  // left sidebar <-> center
  makeSplitter('split-left', 'h', function(action, delta, state) {
    var sb = document.getElementById('sidebar');
    if (action === 'start') return { w: sb.offsetWidth };
    var w = Math.max(140, Math.min(400, state.w + delta));
    sb.style.width = w + 'px';
  });

  // center <-> right sidebar
  makeSplitter('split-right', 'h', function(action, delta, state) {
    var rb = document.getElementById('rightbar');
    if (action === 'start') return { w: rb.offsetWidth };
    var w = Math.max(180, Math.min(420, state.w - delta));
    rb.style.width = w + 'px';
  });

  // chat window <-> image lab (use flex-grow ratios so they shrink with container)
  makeSplitter('split-center', 'v', function(action, delta, state) {
    var chat = document.getElementById('chat-window');
    var img = document.getElementById('image-window');
    if (action === 'start') {
      return { chatH: chat.offsetHeight, imgH: img.offsetHeight };
    }
    var chatH = Math.max(80, state.chatH + delta);
    var imgH = Math.max(80, state.imgH - delta);
    // use flex-grow ratios (not fixed px) so they stay responsive to container resize
    chat.style.flex = chatH + ' 1 0px';
    img.style.flex = imgH + ' 1 0px';
  });

  // footer height
  makeSplitter('split-footer', 'v', function(action, delta, state) {
    var ft = document.getElementById('footer');
    if (action === 'start') return { h: ft.offsetHeight };
    var h = Math.max(30, Math.min(200, state.h - delta));
    ft.style.height = h + 'px';
  });
}

function makeSplitter(id, axis, handler) {
  var el = document.getElementById(id);
  if (!el) return;

  el.addEventListener('mousedown', function(e) {
    e.preventDefault();
    var startPos = axis === 'h' ? e.clientX : e.clientY;
    var state = handler('start', 0, null);

    el.classList.add('dragging');
    document.body.style.cursor = axis === 'h' ? 'col-resize' : 'row-resize';
    document.body.style.userSelect = 'none';

    function onMouseMove(e) {
      var cur = axis === 'h' ? e.clientX : e.clientY;
      handler('move', cur - startPos, state);
    }

    function onMouseUp() {
      el.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    }

    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  });
}

// ── KEYBOARD SHORTCUTS ───────────────────────────────────────
document.addEventListener('keydown', function(e) {
  if (e.ctrlKey && e.key === 'n') { e.preventDefault(); newChat(); }
  if (e.key === 'Escape') closeSettings();
});

initSplitters();
init();
</script>
</body>
</html>
"""

# ───────────────────────────────────────────────────────────────
# ENTRYPOINT
# ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    init_db()
    print("""
╔═══════════════════════════════════════════════════╗
║     MICRO GPT 2.0 — LOCAL AI WORKSTATION          ║
║     ollama · rag · vectors · search · image gen   ║
╚═══════════════════════════════════════════════════╝

  setup:
    ollama serve
    ollama pull llama3.2
    ollama pull nomic-embed-text
    ollama pull x/flux2-klein:latest   (image gen)
    pip install flask duckduckgo-search

  → open http://localhost:5000
""")
    app.run(debug=False, port=5000, threaded=True)
