"""Fill release details into the built site.

Run by build.sh after mdBook and before seo.py. Reads the version from the
workspace Cargo.toml and the newest release from CHANGELOG.md, then:

- replaces {{ft_version}} in every built .html, .txt and .xml file;
- replaces the <!-- whats-new --> marker on the landing page with a short
  summary of that release (first sentence of each entry), linked to its
  GitHub release.

Fails if the CHANGELOG's newest release is not the Cargo.toml version, so a
release can't ship with a site that describes the previous one.
"""

import html
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = "https://github.com/puranikyashaswin/ferrotensor"
MARKER = "<!-- whats-new -->"
TOKEN = "{{ft_version}}"


def workspace_version() -> str:
    with open(ROOT / "Cargo.toml", "rb") as f:
        return tomllib.load(f)["workspace"]["package"]["version"]


def latest_release(changelog: str) -> tuple[str, str, dict[str, list[str]]]:
    """Version, date and {section: [entries]} of the newest released version."""
    m = re.search(r"^## \[(\d+\.\d+\.\d+)\] - (\d{4}-\d{2}-\d{2})\n(.*?)(?=^## \[)",
                  changelog, re.M | re.S)
    if not m:
        sys.exit("release.py: no released version found in CHANGELOG.md")
    version, date, body = m.groups()
    sections: dict[str, list[str]] = {}
    current = None
    for line in body.splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            sections[current] = []
        elif line.startswith("- ") and current:
            sections[current].append(line[2:].strip())
    return version, date, sections


def first_sentence(text: str) -> str:
    """Up to the first '. ' that is outside backticks."""
    in_code = False
    for i, ch in enumerate(text):
        if ch == "`":
            in_code = not in_code
        elif ch == "." and not in_code and text[i + 1:i + 2] in (" ", ""):
            return text[:i + 1]
    return text


def inline_md(text: str) -> str:
    """Escape, then render `code`, **bold** and [text](url)."""
    out = html.escape(text, quote=False)
    out = re.sub(r"`([^`]+)`", r"<code>\1</code>", out)
    out = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", r'<a href="\2">\1</a>', out)
    return out


def whats_new_html(version: str, date: str, sections: dict[str, list[str]]) -> str:
    items = []
    for kind, entries in sections.items():
        for entry in entries:
            items.append(f'<li><span class="tag">{html.escape(kind)}</span>'
                         f'<span class="txt">{inline_md(first_sentence(entry))}</span></li>')
    return (
        '<section class="news" aria-labelledby="news-title">\n'
        '  <div class="wrap news-in">\n'
        f'    <h2 id="news-title" class="news-h">What\'s new in {version}'
        f' <time datetime="{date}">{date}</time></h2>\n'
        f'    <ul class="news-list">{"".join(items)}</ul>\n'
        f'    <p class="news-more"><a href="{REPO}/releases/tag/v{version}">Release notes</a>'
        f' · <a href="docs/changelog.html">Full changelog</a></p>\n'
        "  </div>\n"
        "</section>"
    )


def main(out: Path) -> None:
    version = workspace_version()
    rel_version, date, sections = latest_release((ROOT / "CHANGELOG.md").read_text())
    if rel_version != version:
        sys.exit(f"release.py: Cargo.toml is {version} but the newest CHANGELOG "
                 f"release is {rel_version}")

    landing = out / "index.html"
    page = landing.read_text()
    if MARKER not in page:
        sys.exit(f"release.py: {MARKER} missing from the landing page")
    landing.write_text(page.replace(MARKER, whats_new_html(version, date, sections)))

    filled = 0
    for path in out.rglob("*"):
        if path.suffix in (".html", ".txt", ".xml") and path.is_file():
            text = path.read_text()
            if TOKEN in text:
                path.write_text(text.replace(TOKEN, version))
                filled += 1
    leftover = [p for p in out.rglob("*.html") if "{{" + "ft_" in p.read_text()]
    if leftover:
        sys.exit(f"release.py: unfilled placeholder in {leftover[0]}")
    n = sum(len(v) for v in sections.values())
    print(f"release: {version} ({date}), {n} changelog entries, {filled} files filled")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
