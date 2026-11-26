#!/usr/bin/env python3
"""Point every page of the old GitHub Pages site at ferrotensor.dev. Usage: redirect_to_new_site.py <site-dir>.

The website moved to https://ferrotensor.dev. GitHub Pages cannot send a 301,
so each built .html file gets the two signals it can carry: a canonical link to
the matching new URL (search engines move the ranking there) and a
`<meta http-equiv="refresh" content="0; url=NEW">` (visitors land on the new
site). An og:url tag, if present, is set to the same URL. Run it on the final
output folder, after seo.py, right before the upload.

sitemap.xml is deleted: it would list URLs that no longer own their content,
and the new site has its own. robots.txt keeps `Allow: /` and loses its
Sitemap line. It must not disallow anything, because a crawler that cannot
fetch a page never sees the canonical tag on it.

Exit code 1 when an .html file has no <head> (nothing can be written there).
"""

import os
import re
import sys

NEW_BASE = "https://ferrotensor.dev/"

# Paths of the new site relative to NEW_BASE ("" is the home page). Copied from
# https://ferrotensor.dev/sitemap.xml on 26 November 2026 (17 pages); update it
# when the new site gains or loses a page.
NEW_PATHS = frozenset(
    {
        "",
        "about/",
        "docs/",
        "docs/architecture",
        "docs/benchmarks",
        "docs/benchmarks-mac",
        "docs/changelog",
        "docs/examples/classify",
        "docs/examples/lora-sft",
        "docs/good-first-issues",
        "docs/installation",
        "docs/python",
        "docs/python-reference",
        "docs/testing-on-your-gpu",
        "docs/use-cases",
        "docs/writing-a-kernel",
        "guide/",
    }
)

# Old pages that are the same content as a page of the new site under another name:
# seo.py already treats docs/introduction.html as a duplicate of docs/index.html.
ALIASES = {"docs/introduction": "docs/"}

HEAD_OPEN = re.compile(r"<head(?:\s[^>]*)?>", re.I)
CANONICAL = re.compile(r"[ \t]*<link\b[^>]*\brel\s*=\s*[\"']?canonical[\"']?[^>]*>[ \t]*\n?", re.I)
REFRESH = re.compile(r"[ \t]*<meta\b[^>]*\bhttp-equiv\s*=\s*[\"']?refresh[\"']?[^>]*>[ \t]*\n?", re.I)
OG_URL = re.compile(r"(<meta\b[^>]*\bproperty\s*=\s*[\"']og:url[\"'][^>]*\bcontent\s*=\s*)([\"'])[^\"']*\2", re.I)
OG_URL_CONTENT_FIRST = re.compile(r"(<meta\b[^>]*\bcontent\s*=\s*)([\"'])[^\"']*\2([^>]*\bproperty\s*=\s*[\"']og:url[\"'])", re.I)


def new_path(rel):
    """Map an output path (relative to the Pages root, with /) to (path on the new site, how)."""
    if rel == "index.html" or rel.endswith("/index.html"):
        path = rel[: -len("index.html")]
    elif rel.endswith(".html"):
        path = rel[: -len(".html")]
    else:
        path = rel
    if path in NEW_PATHS:
        return path, "exact"
    if path in ALIASES:
        return ALIASES[path], "alias"
    # An unknown docs page (toc.html, 404.html, a page that was dropped) goes to the docs index.
    if path.startswith("docs/"):
        return "docs/", "docs"
    return "", "home"


def rewrite_head(text, url):
    """Return text with one canonical, one refresh and the og:url set to url; ValueError without a <head>."""
    match = HEAD_OPEN.search(text)
    if match is None:
        raise ValueError("no <head>")
    text = CANONICAL.sub("", text)
    text = REFRESH.sub("", text)
    text = OG_URL.sub(lambda m: f'{m.group(1)}"{url}"', text)
    text = OG_URL_CONTENT_FIRST.sub(lambda m: f'{m.group(1)}"{url}"{m.group(3)}', text)
    # The removals above may have moved the opening tag, so find it again.
    match = HEAD_OPEN.search(text)
    tags = f'\n<meta http-equiv="refresh" content="0; url={url}">\n<link rel="canonical" href="{url}">'
    return text[: match.end()] + tags + text[match.end():]


def redirect_site(site):
    """Rewrite every .html file under site. Return (rewritten urls by path, list of files without <head>)."""
    done = []
    broken = []
    for dirpath, _, files in os.walk(site):
        for name in sorted(files):
            if not name.endswith(".html"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, site).replace(os.sep, "/")
            target, how = new_path(rel)
            with open(path, encoding="utf-8") as f:
                text = f.read()
            try:
                text = rewrite_head(text, NEW_BASE + target)
            except ValueError:
                broken.append(rel)
                continue
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            done.append((rel, NEW_BASE + target, how))
    return sorted(done), sorted(broken)


def drop_old_sitemap(site):
    """Delete sitemap.xml and its line in robots.txt; return what was removed."""
    removed = []
    sitemap = os.path.join(site, "sitemap.xml")
    if os.path.exists(sitemap):
        os.remove(sitemap)
        removed.append("sitemap.xml")
    robots = os.path.join(site, "robots.txt")
    if os.path.exists(robots):
        with open(robots, encoding="utf-8") as f:
            lines = f.read().splitlines()
        kept = [ln for ln in lines if not ln.lower().startswith("sitemap:")]
        if kept != lines:
            with open(robots, "w", encoding="utf-8") as f:
                f.write("\n".join(kept).rstrip("\n") + "\n")
            removed.append("the Sitemap line of robots.txt")
    return removed


def main(argv):
    if len(argv) != 2 or not os.path.isdir(argv[1]):
        print("usage: redirect_to_new_site.py <site-dir>", file=sys.stderr)
        return 2
    done, broken = redirect_site(argv[1])
    removed = drop_old_sitemap(argv[1])
    docs = [d for d in done if d[2] == "docs"]
    home = [d for d in done if d[2] == "home"]
    aliased = [d for d in done if d[2] == "alias"]
    print(f"redirect_to_new_site: {len(done)} pages rewritten, {len(docs)} sent to the docs index, {len(home)} sent to the home page")
    for rel, url, _ in aliased:
        print(f"  alias to {url}: {rel}")
    for rel, url, _ in docs:
        print(f"  fallback to {url}: {rel}")
    for rel, url, _ in home:
        print(f"  fallback to {url}: {rel}")
    if removed:
        print("  removed: " + ", ".join(removed))
    for rel in broken:
        print(f"error: {rel} has no <head>", file=sys.stderr)
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
