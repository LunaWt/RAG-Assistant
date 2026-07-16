import json

import requests
import streamlit as st

from components import copy_button_html, search_card_html

API = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Agent",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": None, "Report a bug": None, "About": None},
)

CHAT_BG = "#212121"
SIDEBAR_BG = "#000000"

st.markdown(f"""
<style>
/* Full-height chat zone — hide Deploy only, keep sidebar toggle */
[data-testid="stHeader"] {{
    background: {CHAT_BG};
    border-bottom: none;
    box-shadow: none;
}}
[data-testid="stAppDeployButton"] {{
    display: none !important;
}}
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] {{
    display: flex !important;
    visibility: visible !important;
    color: #ececec !important;
    opacity: 1 !important;
}}
[data-testid="stHeaderActionElements"] {{
    visibility: visible !important;
}}
footer {{
    display: none;
}}
.stApp {{
    background-color: {CHAT_BG};
}}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
section.main,
section.main > div {{
    background-color: {CHAT_BG};
}}
[data-testid="stMain"] .block-container {{
    max-width: 58rem;
    margin: 0 auto;
    padding-top: 0.75rem;
    padding-left: 1rem;
    padding-right: 1rem;
    padding-bottom: 6rem;
}}

/* Sidebar — black, darker than chat */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {{
    background-color: {SIDEBAR_BG};
}}
[data-testid="stSidebar"] {{
    border-right: 1px solid #2a2a2a;
}}
[data-testid="stSidebar"] * {{
    color: #ececec;
}}

/* Chat input — pill shape, no inner focus glow */
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] > div {{
    background-color: {CHAT_BG};
    border-top: none;
}}
[data-testid="stBottom"] {{
    background-color: {CHAT_BG};
}}
[data-testid="stChatInput"] {{
    max-width: 58rem;
    margin: 0 auto;
}}
[data-testid="stChatInput"] > div {{
    border-radius: 1.75rem !important;
    background-color: #2f2f2f !important;
    border: 1px solid #565869 !important;
    box-shadow: none !important;
}}
[data-testid="stChatInput"] > div:focus-within {{
    border-color: #565869 !important;
    box-shadow: none !important;
    outline: none !important;
}}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] {{
    background-color: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    color: #f0f0f0;
}}
[data-testid="stChatInput"] div[data-baseweb="textarea"],
[data-testid="stChatInput"] div[data-baseweb="base-input"] {{
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}
[data-testid="stChatInput"] *:focus,
[data-testid="stChatInput"] *:focus-visible {{
    outline: none !important;
    box-shadow: none !important;
}}

/* Messages — no avatars; user = bubble, assistant = plain text */
[data-testid="stChatMessage"] {{
    display: flex !important;
    align-items: flex-start;
    gap: 0;
    width: 100%;
    background: transparent !important;
    border: none !important;
    padding: 0.4rem 0;
}}
/* hide avatars entirely (elements stay in DOM so :has selectors still work) */
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"] {{
    display: none !important;
}}
/* user message → bubble hugs its text and sits on the right */
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{
    justify-content: flex-end;
}}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"])
[data-testid="stChatMessageContent"] {{
    text-align: right;
}}
.user-bubble {{
    display: inline-block;
    text-align: left;
    background: #2f2f2f;
    border: 1px solid #3d3d3d;
    border-radius: 18px;
    padding: 0.5rem 0.95rem;
    max-width: 85%;
    white-space: pre-wrap;
    word-wrap: break-word;
}}

.chat-greeting {{
    text-align: center;
    color: #ececec;
    font-size: 1.75rem;
    font-weight: 600;
    margin: 2.5rem 0 1.5rem 0;
}}
/* Reasoning / tool toggles — subtle, borderless, no big plank */
[data-testid="stChatMessage"] details {{
    background-color: transparent !important;
    border: none !important;
    border-radius: 0;
}}
[data-testid="stChatMessage"] summary {{
    color: #8b949e;
    font-size: 0.85rem;
    padding: 0.1rem 0 !important;
}}
[data-testid="stChatMessage"] summary:hover {{
    color: #c9d1d9;
}}
[data-testid="stChatMessage"] [data-testid="stExpander"] {{
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}}
/* copy button (iframe) — narrow so it never stretches the bubble; shown on hover */
[data-testid="stChatMessage"] iframe {{
    width: 2.2rem !important;
    opacity: 0;
    transition: opacity 0.15s ease;
}}
[data-testid="stChatMessage"]:hover iframe {{
    opacity: 1;
}}

.search-card {{
    border: 1px solid #30363d;
    border-radius: 12px;
    background: #161b22;
    margin: 0.5rem 0 1rem 0;
    overflow: hidden;
    animation: fadeIn 0.3s ease;
}}
.search-header {{
    padding: 10px 14px;
    font-size: 0.9rem;
    color: #c9d1d9;
    border-bottom: 1px solid #30363d;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.search-results {{
    max-height: 220px;
    overflow-y: auto;
}}
.hit-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 10px 14px;
    text-decoration: none;
    color: inherit;
    border-bottom: 1px solid #21262d;
    transition: background 0.15s ease;
}}
.hit-row:last-child {{ border-bottom: none; }}
.hit-row:hover {{ background: #21262d; }}
.hit-title {{
    flex: 1;
    font-size: 0.85rem;
    color: #e6edf3;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.hit-domain {{
    flex-shrink: 0;
    font-size: 0.75rem;
    color: #8b949e;
}}
.thinking-pulse {{ animation: pulse 1.2s ease-in-out infinite; }}
@keyframes pulse {{
    0%, 100% {{ opacity: 0.45; }}
    50% {{ opacity: 1; }}
}}
.spin {{
    display: inline-block;
    width: 0.85em;
    height: 0.85em;
    border: 2px solid #3d3d3d;
    border-top-color: #ececec;
    border-radius: 50%;
    margin-right: 0.45em;
    vertical-align: -0.12em;
    animation: spin 0.7s linear infinite;
}}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
@keyframes fadeIn {{
    from {{ opacity: 0; transform: translateY(4px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
</style>
""", unsafe_allow_html=True)

def get_active_session_id() -> int | None:
    """Active chat id from the URL (?session=N). None if absent/invalid."""
    raw = st.query_params.get("session")
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def server_message_to_ui(m: dict) -> dict:
    """Map a DB message (blocks: {type, content}) into the UI render shape."""
    if m["role"] == "user":
        text = "".join(b["content"] for b in m["blocks"] if b["type"] == "user-query")
        return {"role": "user", "content": text}
    ui_blocks: list[dict] = []
    for b in m["blocks"]:
        if b["type"] == "thought":
            ui_blocks.append({"type": "thought", "text": b["content"]})
        elif b["type"] == "answer":
            ui_blocks.append({"type": "answer", "text": b["content"]})
        elif b["type"] == "tool":
            data = json.loads(b["content"])
            ui_blocks.append({"type": "tool", "query": data["query"], "hits": data["hits"]})
    return {"role": "assistant", "blocks": ui_blocks}


def load_session_messages(session_id: int) -> list[dict]:
    """Fetch a session's history from the API, mapped to UI messages."""
    r = requests.get(f"{API}/sessions/{session_id}/messages", timeout=10)
    r.raise_for_status()
    return [server_message_to_ui(m) for m in r.json().get("messages", [])]


def load_sessions() -> list[dict]:
    """All chat sessions, newest first: [{id, title, created_at}]."""
    r = requests.get(f"{API}/sessions", timeout=10)
    r.raise_for_status()
    return r.json()


# first run only: seed the active chat from the URL (?session=N)
if "session_id" not in st.session_state:
    st.session_state.session_id = get_active_session_id()
    st.session_state.messages = []
    st.session_state.loaded_for = None

# reconcile: keep messages in sync with the active session_id.
# Handlers (switch / delete) only change session_id + rerun; reload happens here.
if st.session_state.session_id != st.session_state.loaded_for:
    if st.session_state.session_id is None:
        st.session_state.messages = []
    else:
        try:
            st.session_state.messages = load_session_messages(st.session_state.session_id)
        except requests.RequestException:
            st.query_params.clear()
            st.session_state.session_id = None
            st.session_state.messages = []
    st.session_state.loaded_for = st.session_state.session_id

if not st.session_state.messages:
    st.markdown('<div class="chat-greeting">RAG Agent</div>', unsafe_allow_html=True)


def unique_sources(docs_payload: dict) -> list[str]:
    """Chroma get() returns metadatas: [{source: filename}, ...]."""
    metadatas = docs_payload.get("metadatas") or []
    return sorted({m["source"] for m in metadatas if m and m.get("source")})


def _tool_label(name: str) -> str:
    return {
        "search_knowledge_base": "Searching knowledge base…",
        "web_search": "Searching the web…",
        "calculator": "Calculating…",
    }.get(name, f"Running {name}…")


def _render_tool_detail(name: str, args: dict | None) -> None:
    args = args or {}
    if name == "search_knowledge_base":
        st.caption("🔎 Query")
        st.code(args.get("query", ""), language=None)
        if args.get("filename"):
            st.caption(f"📄 Filename: `{args['filename']}`")
    elif name == "web_search":
        st.caption("🌐 Query")
        st.code(args.get("query", ""), language=None)
    elif name == "calculator":
        st.caption("🧮 Expression")
        st.code(args.get("expression", ""), language="python")
    else:
        st.code(str(args) or "(no args)", language="json")


def apply_event(blocks: list[dict], ev: dict) -> None:
    """Append stream event into ordered timeline blocks."""
    t = ev["type"]
    if t == "thought_delta":
        if blocks and blocks[-1]["type"] == "thought":
            blocks[-1]["text"] += ev["text"]
        else:
            blocks.append({"type": "thought", "text": ev["text"]})
    elif t == "text_delta":
        if blocks and blocks[-1]["type"] == "answer":
            blocks[-1]["text"] += ev["text"]
        else:
            blocks.append({"type": "answer", "text": ev["text"]})
    elif t == "tool_start":
        names = list(ev.get("name", []))
        args = list(ev.get("args", []))
        if blocks and blocks[-1]["type"] == "tool_spinner" and not blocks[-1].get("done"):
            blocks[-1]["names"].extend(names)
            blocks[-1]["args"].extend(args)
        else:
            blocks.append({
                "type": "tool_spinner",
                "names": names,
                "args": args,
                "done": False,
            })
    elif t == "tool_hits":
        for b in reversed(blocks):
            if b["type"] == "tool_spinner" and not b.get("done"):
                b["done"] = True
                b.setdefault("results", []).append({
                    "query": ev["query"],
                    "hits": ev["hits"],
                })
                break
        else:
            blocks.append({
                "type": "tool_spinner",
                "names": [],
                "args": [],
                "done": True,
                "results": [{"query": ev["query"], "hits": ev["hits"]}],
            })


def _render_process_block(b: dict) -> None:
    """One reasoning-timeline item (thought or tool) inside the Reasoning panel."""
    if b["type"] == "thought":
        st.markdown(b["text"])
    elif b["type"] == "tool_spinner":
        for name, args in zip(b.get("names", []), b.get("args", [])):
            _render_tool_detail(name, args)
        for res in b.get("results", []):
            st.markdown(search_card_html(res["query"], res["hits"]), unsafe_allow_html=True)
    elif b["type"] == "tool":  # from loaded history
        st.markdown(search_card_html(b["query"], b["hits"]), unsafe_allow_html=True)


def _live_activity_label(blocks: list[dict]) -> str:
    """What the agent is doing right now, for the streaming status line."""
    for b in reversed(blocks):
        if b["type"] == "tool_spinner" and not b.get("done"):
            names = b.get("names", [])
            return _tool_label(names[0]) if names else "Running tools…"
        if b["type"] == "thought":
            return "Thinking…"
    return "Thinking…"


def render_blocks(blocks: list[dict], *, streaming: bool = False) -> None:
    """Thoughts + tools tucked into one Reasoning panel; answer as plain text."""
    process = [b for b in blocks if b["type"] in ("thought", "tool_spinner", "tool")]
    answers = [b for b in blocks if b["type"] == "answer"]

    if streaming and not answers:
        # while working (before any answer): one animated status line, no panels
        label = _live_activity_label(blocks)
        st.markdown(
            f'<span class="spin"></span><span class="thinking-pulse">{label}</span>',
            unsafe_allow_html=True,
        )
    elif process:
        with st.expander("Reasoning", expanded=False):
            for b in process:
                _render_process_block(b)

    for b in answers:
        text = b["text"]
        if streaming and b is blocks[-1]:
            text += "▌"
        st.markdown(text)


def message_copy_text(message: dict) -> str:
    if message["role"] == "user":
        return message.get("content", "")
    parts: list[str] = []
    for b in message.get("blocks", []):
        if b["type"] == "thought" and b.get("text"):
            parts.append(f"[Reasoning]\n{b['text']}")
        elif b["type"] == "tool":
            parts.append(f"[Search: {b.get('query', '')}]")
        elif b["type"] == "answer" and b.get("text"):
            parts.append(b["text"])
    return "\n\n".join(parts).strip()


def message_for_history(msg: dict) -> dict:
    """Normalize session message for API history payload."""
    if msg["role"] == "user":
        return {"role": "user", "content": msg["content"]}
    return {"role": "assistant", "blocks": msg.get("blocks", [])}


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_user_bubble(text: str) -> None:
    st.markdown(
        f'<div style="display:flex; justify-content:flex-end;">'
        f'<div class="user-bubble">{_esc(text)}</div></div>',
        unsafe_allow_html=True,
    )


def render_message(message: dict) -> None:
    if message["role"] == "user":
        render_user_bubble(message["content"])
        return
    render_blocks(message.get("blocks", []))


def render_copy_button(message: dict, key: str) -> None:
    text = message_copy_text(message)
    if not text:
        return
    align = "right" if message["role"] == "user" else "left"
    copy_button_html(text, key=key, align=align)


# --- sidebar (always visible on every rerun) ---
with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.session_id = None
        st.query_params.clear()
        st.rerun()

    st.subheader("Chats")
    try:
        sessions = load_sessions()
        if sessions:
            for s in sessions:
                active = s["id"] == st.session_state.session_id
                col1, col2 = st.columns([5, 1])
                if col1.button(
                    s["title"] or "New chat",
                    key=f"sess_{s['id']}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    st.session_state.session_id = s["id"]
                    st.query_params["session"] = str(s["id"])
                    st.rerun()
                with col2.popover("⋯", use_container_width=True):
                    new_title = st.text_input(
                        "Rename chat",
                        value=s["title"] or "",
                        max_chars=60,
                        key=f"rename_input_{s['id']}",
                    )
                    if st.button("Save", key=f"rename_save_{s['id']}", use_container_width=True):
                        if new_title.strip():  # empty title → no-op, no error shown
                            try:
                                requests.put(
                                    f"{API}/sessions/{s['id']}",
                                    json={"title": new_title},
                                    timeout=10,
                                ).raise_for_status()
                                st.rerun()
                            except requests.RequestException as e:
                                st.error(f"Rename failed: {e}")
                    if st.button("🗑 Delete", key=f"delsess_{s['id']}", use_container_width=True):
                        try:
                            requests.delete(f"{API}/sessions/{s['id']}", timeout=10).raise_for_status()
                            if st.session_state.session_id == s["id"]:
                                st.session_state.session_id = None
                                st.query_params.clear()
                            st.rerun()
                        except requests.RequestException as e:
                            st.error(f"Delete failed: {e}")
        else:
            st.caption("No chats yet")
    except requests.RequestException:
        st.warning("API unavailable")

    st.divider()
    st.header("Knowledge base")
    uploaded = st.file_uploader(
        "Upload document",
        type=["pdf", "txt", "md", "docx", "xlsx", "pptx"],
    )
    if uploaded and st.button("Index", use_container_width=True):
        progress = st.progress(0, text="Uploading…")
        try:
            progress.progress(15, text="Uploading file…")
            r = requests.post(
                f"{API}/upload-document",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                timeout=120,
            )
            progress.progress(55, text="Parsing & chunking…")
            r.raise_for_status()
            data = r.json()
            progress.progress(90, text="Indexing in Chroma…")
            progress.progress(100, text="Done")
            st.success(data.get("message", "Document indexed"))
            st.rerun()
        except requests.RequestException as e:
            progress.empty()
            detail = ""
            if hasattr(e, "response") and e.response is not None:
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    detail = e.response.text[:200]
            st.error(detail or f"Upload failed: {e}")

    st.subheader("Documents")
    try:
        names = requests.get(f"{API}/documents/sources", timeout=10).json().get("sources", [])
        if names:
            for name in names:
                col1, col2 = st.columns([5, 1])
                col1.caption(f"📄 {name}")
                if col2.button("🗑", key=f"del_{name}", help=f"Delete {name}"):
                    try:
                        r = requests.delete(
                            f"{API}/documents",
                            params={"filename": name},
                            timeout=30,
                        )
                        r.raise_for_status()
                        st.toast(r.json().get("message", "Deleted"))
                        st.rerun()
                    except requests.RequestException as e:
                        st.error(f"Delete failed: {e}")
        else:
            st.caption("No documents yet")
    except requests.RequestException:
        st.warning("API unavailable")

# --- chat history ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        render_message(message)
        render_copy_button(message, key=f"copy_{i}")

# --- new message ---
if prompt := st.chat_input("Ask anything…"):
    if st.session_state.session_id is None:
        try:
            r = requests.post(f"{API}/sessions", timeout=10)
            r.raise_for_status()
            st.session_state.session_id = r.json()["id"]
        except requests.RequestException as e:
            st.error(f"Could not start chat: {e}")
            st.stop()

    history = [message_for_history(m) for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        render_user_bubble(prompt)
        render_copy_button(
            {"role": "user", "content": prompt},
            key=f"copy_live_user_{len(st.session_state.messages)}",
        )

    with st.chat_message("assistant"):
        blocks: list[dict] = []
        stream_box = st.empty()
        completed = False
        status_box = st.empty()

        try:
            with requests.post(
                f"{API}/chat",
                json={
                    "query": prompt,
                    "history": history,
                    "session_id": st.session_state.session_id,
                },
                stream=True,
                timeout=300,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    ev = json.loads(line)
                    t = ev.get("type")

                    if t == "stream_reset":
                        blocks.clear()
                        stream_box.empty()  # hard-clear partial render before retry
                        status_box.caption("Соединение оборвалось, повторяем…")
                        continue

                    if t == "error":
                        blocks.clear()
                        stream_box.empty()
                        status_box.error(ev.get("message", "Ошибка агента"))
                        break

                    if t == "done":
                        completed = True
                        status_box.empty()
                        continue

                    apply_event(blocks, ev)
                    with stream_box.container():
                        render_blocks(blocks, streaming=True)

            if completed and blocks:
                st.session_state.messages.append({
                    "role": "assistant",
                    "blocks": blocks,
                })
                st.query_params["session"] = str(st.session_state.session_id)
                st.rerun()
            elif not completed:
                stream_box.empty()
                status_box.warning(
                    "Ответ не завершён — сообщение не сохранено. Попробуйте ещё раз."
                )

        except requests.exceptions.ChunkedEncodingError:
            stream_box.empty()
            status_box.warning(
                "Соединение оборвалось — незавершённый ответ не сохранён."
            )
        except requests.RequestException as e:
            stream_box.empty()
            status_box.error(f"API error: {e}")
