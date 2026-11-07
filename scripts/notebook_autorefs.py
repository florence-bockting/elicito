"""
MkDocs hook: let autorefs resolve `[title][identifier]` in notebook pages

mkdocs-jupyter renders notebooks without Python-Markdown, so the autorefs
extension never sees their links. This hook writes the `<autoref>` tags
that the autorefs plugin resolves in `on_post_page`.
"""

from __future__ import annotations

import re
from typing import Any

from mkdocs.structure.pages import Page

SKIP_RE = re.compile(r"(<(pre|style|script)\b.*?</\2>)", flags=re.DOTALL)
LINK_RE = re.compile(r"\[(?P<title>[^\[\]]+)\]\[(?P<identifier>[A-Za-z][\w.:-]*)\]")


def on_page_content(html: str, page: Page, **kwargs: Any) -> str:
    """Change the reference links of a notebook page to autoref tags"""
    if not page.file.src_uri.endswith(".ipynb"):
        return html
    parts = SKIP_RE.split(html)
    # split() returns [text, block, tag name, text, block, tag name, ...]
    for i in range(0, len(parts), 3):
        parts[i] = LINK_RE.sub(
            r'<autoref identifier="\g<identifier>">\g<title></autoref>', parts[i]
        )
    return "".join(parts)
