import json

import streamlit.components.v1 as components
from urllib.parse import urlparse

def search_card_html(query: str, hits: list[dict]) -> str:
    rows = ""
    for h in hits:
        domain = urlparse(h["href"]).netloc.replace("www.", "")
        rows += f"""
        <a class="hit-row" href="{h['href']}" target="_blank">
          <span class="hit-title">{h.get('title','')}</span>
          <span class="hit-domain">{domain}</span>
        </a>"""
    return f"""
    <div class="search-card">
      <div class="search-header">🔍 {query}</div>
      <div class="search-results">{rows}</div>
    </div>"""


def copy_button_html(text: str, key: str, *, align: str = "left") -> None:
    """Clipboard copy without st.copy_button (needs Streamlit >= 1.29)."""
    safe_key = "".join(c if c.isalnum() else "_" for c in key)
    components.html(
        f"""
        <div style="text-align:{align}; margin:0;">
          <button id="copy_{safe_key}" title="Copy" style="
            background:transparent; color:#8b949e; border:none;
            padding:2px 4px; font-size:0.85rem; cursor:pointer; line-height:1;
            border-radius:6px; transition:color 0.15s ease, background 0.15s ease;
          "
          onmouseover="this.style.color='#ececec';this.style.background='#2f2f2f';"
          onmouseout="this.style.color='#8b949e';this.style.background='transparent';"
          >📋</button>
          <span id="ok_{safe_key}" style="display:none;color:#8b949e;font-size:0.72rem;margin-left:4px;">
            Copied
          </span>
        </div>
        <script>
          document.getElementById("copy_{safe_key}").onclick = () => {{
            navigator.clipboard.writeText({json.dumps(text)});
            const ok = document.getElementById("ok_{safe_key}");
            ok.style.display = "inline";
            setTimeout(() => {{ ok.style.display = "none"; }}, 1500);
          }};
        </script>
        """,
        height=24,
    )