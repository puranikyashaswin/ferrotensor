#!/usr/bin/env python3
"""Finish a built site for search engines and check it. Usage: seo.py <site-dir>.

On the mdBook pages under docs/ it sets a unique <title> and meta description,
adds canonical, Open Graph and Twitter tags, fixes a few accessibility and loading
issues in mdBook's template (see fix_template), and marks toc and 404 pages noindex. It then
writes sitemap.xml and robots.txt and fails the build when a page is missing
from PAGES, a title or description is too long or repeated, a page has other
than one <h1>, or the landing page's JSON-LD does not parse or its FAQ answers
differ from the visible text.
"""

import html
import json
import os
import re
import sys

BASE = "https://puranikyashaswin.github.io/ferrotensor/"
OG_IMAGE = BASE + "og.png"
TITLE_MAX = 60
DESC_MAX = 155

# docs page -> (title, description). Every indexable page must be listed.
PAGES = {
    "index.html": (
        "Ferrotensor guide: pure-Rust ML on Metal, Vulkan, DX12",
        "The Ferrotensor guide: install the Rust crate, CLI or Python package, run the examples, "
        "fine-tune with LoRA and see how ops reach WGSL kernels.",
    ),
    "installation.html": (
        "Install Ferrotensor: Rust crate, CLI and Python package",
        "Install Ferrotensor with cargo add ferrotensor, cargo install ferrotensor-cli or "
        "pip install ferrotensor. Needs Rust 1.88+ or Python 3.10+; no CUDA.",
    ),
    "python.html": (
        "Ferrotensor Python quickstart: tensors, nn and training",
        "Train a model from Python with Ferrotensor: PyTorch-style tensors, autograd, nn and optim "
        "on the CPU or a Metal, Vulkan or DX12 GPU. No NumPy needed.",
    ),
    "use-cases.html": (
        "Ferrotensor use cases: runnable Rust examples",
        "Four runnable Ferrotensor programs with real output: an MLP with a DataLoader, a "
        "character-level transformer, GPU matmul and attention, similarity search.",
    ),
    "examples/lora-sft.html": (
        "LoRA fine-tuning on an 8 GB MacBook | Ferrotensor",
        "Fine-tune SmolLM2-135M with LoRA on an 8 GB Apple M2 MacBook: held-out loss 2.43 to 1.65 "
        "in 250 steps, 3.3 GB peak memory, every command shown.",
    ),
    "architecture.html": (
        "Ferrotensor architecture: from tensor op to WGSL kernel",
        "How Ferrotensor is built: the tensor API and the LLM stack, batched GPU dispatch, autograd "
        "and the CPU device, following one matmul down to its kernel.",
    ),
    "writing-a-kernel.html": (
        "Writing a WGSL GPU kernel for Ferrotensor",
        "Add a GPU op to Ferrotensor step by step: the WGSL case, a CPU reference, the public method "
        "with its gradient, and the tests that check it.",
    ),
    "benchmarks.html": (
        "Ferrotensor benchmarks: WGSL kernels on an Apple M2",
        "Ferrotensor benchmarks on an Apple M2: 611 GFLOPS f32 matmul in WGSL, flash attention, and "
        "tensor-API timings on CPU and GPU, with the method.",
    ),
    "python-reference.html": (
        "Ferrotensor Python API: coverage and PyTorch differences",
        "What the Ferrotensor Python bindings cover (tensors, autograd, nn, optim, safetensors), "
        "which devices they run on, and how they differ from PyTorch.",
    ),
    "changelog.html": (
        "Ferrotensor changelog",
        "Release notes for Ferrotensor: what was added, changed and removed in each version, from "
        "0.1.0 to the current release.",
    ),
    "good-first-issues.html": (
        "Ferrotensor good first issues for contributors",
        "Concrete starting points for contributing to Ferrotensor: each task names the files "
        "involved and how you would know it is done.",
    ),
}
# Same content as another page: canonical points there, not in the sitemap.
DUPLICATES = {"introduction.html": "index.html"}
NOINDEX = {"toc.html", "404.html"}


def page_url(rel):
    rel = "docs/" + rel
    return BASE + (rel[: -len("index.html")] if rel.endswith("index.html") else rel)


def head_tags(title, desc, url):
    t, d, u = (html.escape(x, quote=True) for x in (title, desc, url))
    tags = [
        f'<link rel="canonical" href="{u}">',
        '<meta property="og:type" content="article">',
        '<meta property="og:site_name" content="Ferrotensor">',
        f'<meta property="og:title" content="{t}">',
        f'<meta property="og:description" content="{d}">',
        f'<meta property="og:url" content="{u}">',
        f'<meta property="og:image" content="{OG_IMAGE}">',
        '<meta name="twitter:card" content="summary_large_image">',
    ]
    return "\n        ".join(tags)


def fix_template(text):
    """Accessibility and loading fixes to mdBook's page template."""
    # One <h1> per page: the menu-bar title becomes a paragraph.
    text = re.sub(r'<h1 class="menu-title">(.*?)</h1>', r'<p class="menu-title">\1</p>', text, flags=re.S)
    # A <label> may not carry aria-expanded or aria-label; make the toggle a
    # button (theme/ferrotensor.js flips the checkbox it used to label).
    text = re.sub(
        r'<label id="mdbook-sidebar-toggle" class="icon-button" for="mdbook-sidebar-toggle-anchor"([^>]*)>(.*?)</label>',
        r'<button id="mdbook-sidebar-toggle" class="icon-button" type="button"\1>\2</button>',
        text,
        count=1,
        flags=re.S,
    )
    # fonts.css is empty (the guide uses system fonts): skip the request.
    text = re.sub(r'\s*<!-- Fonts -->\s*<link rel="stylesheet" href="[^"]*fonts/fonts-[^"]*\.css">', "", text)
    # The scripts at the end of <body> do not need to block the first paint;
    # deferred scripts still run in order.
    text = re.sub(r'<script src="([^"]+\.js)"></script>(?=[\s\S]*</body>)', r'<script defer src="\1"></script>', text)
    return text


def count_h1(text):
    return len(re.findall(r"<h1[\s>]", text))


def fix_docs(site, problems):
    docs = os.path.join(site, "docs")
    seen = set()
    # book.js sets an empty unload handler, which is deprecated and keeps
    # pages out of the back/forward cache.
    for name in os.listdir(docs):
        if re.fullmatch(r"book(-[0-9a-f]+)?\.js", name):
            path = os.path.join(docs, name)
            js = open(path, encoding="utf-8").read()
            open(path, "w", encoding="utf-8").write(js.replace("window.onunload = function() { };\n", ""))
    for dirpath, _, files in os.walk(docs):
        for name in files:
            if not name.endswith(".html"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, docs).replace(os.sep, "/")
            text = open(path, encoding="utf-8").read()
            text = fix_template(text)
            if rel in NOINDEX:
                if 'name="robots"' not in text:
                    text = text.replace("</head>", '    <meta name="robots" content="noindex">\n    </head>', 1)
            else:
                key = DUPLICATES.get(rel, rel)
                if key not in PAGES:
                    problems.append(f"docs/{rel}: no entry in PAGES in site/seo.py")
                    continue
                title, desc = PAGES[key]
                seen.add(key)
                text = re.sub(r"<title>.*?</title>", f"<title>{html.escape(title)}</title>", text, count=1, flags=re.S)
                text = re.sub(
                    r'<meta name="description" content="[^"]*">',
                    f'<meta name="description" content="{html.escape(desc, quote=True)}">',
                    text,
                    count=1,
                )
                text = text.replace("</head>", "    " + head_tags(title, desc, page_url(key)) + "\n    </head>", 1)
                if count_h1(text) != 1:
                    problems.append(f"docs/{rel}: {count_h1(text)} <h1> elements")
            open(path, "w", encoding="utf-8").write(text)
    for key in sorted(set(PAGES) - seen):
        problems.append(f"PAGES lists docs/{key}, which was not built")


def visible_text(fragment):
    text = html.unescape(re.sub(r"<[^>]+>", "", fragment)).replace(" ", " ")
    return " ".join(text.split())


def check_landing(site, problems):
    text = open(os.path.join(site, "index.html"), encoding="utf-8").read()
    title = re.search(r"<title>(.*?)</title>", text, re.S).group(1)
    desc = re.search(r'<meta name="description" content="([^"]*)">', text).group(1)
    check_lengths("index.html", html.unescape(title), html.unescape(desc), problems)
    if count_h1(text) != 1:
        problems.append(f"index.html: {count_h1(text)} <h1> elements")
    blocks = re.findall(r'<script type="application/ld\+json">(.*?)</script>', text, re.S)
    if not blocks:
        problems.append("index.html: no JSON-LD")
    faq_ld = []
    for block in blocks:
        try:
            data = json.loads(block)
        except json.JSONDecodeError as err:
            problems.append(f"index.html: JSON-LD does not parse: {err}")
            continue
        for node in data.get("@graph", [data]):
            if node.get("@type") == "FAQPage":
                faq_ld = [(q["name"], q["acceptedAnswer"]["text"]) for q in node["mainEntity"]]
    shown = [
        (visible_text(q), visible_text(a))
        for q, a in re.findall(r'<div class="faq-item">\s*<h3[^>]*>(.*?)</h3>\s*<p>(.*?)</p>', text, re.S)
    ]
    if [(visible_text(q), visible_text(a)) for q, a in faq_ld] != shown or not shown:
        problems.append("index.html: FAQPage JSON-LD does not match the visible FAQ")
    return len(shown)


def check_lengths(where, title, desc, problems):
    if len(title) > TITLE_MAX:
        problems.append(f"{where}: title is {len(title)} characters: {title}")
    if len(desc) > DESC_MAX:
        problems.append(f"{where}: description is {len(desc)} characters")


def write_sitemap(site):
    urls = [BASE] + [page_url(rel) for rel in PAGES]
    entries = "\n".join(f"  <url><loc>{html.escape(u)}</loc></url>" for u in urls)
    with open(os.path.join(site, "sitemap.xml"), "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{entries}\n</urlset>\n')
    with open(os.path.join(site, "robots.txt"), "w", encoding="utf-8") as f:
        f.write(f"User-agent: *\nAllow: /\n\nSitemap: {BASE}sitemap.xml\n")
    return len(urls)


def main():
    site = sys.argv[1]
    problems = []
    for key, (title, desc) in PAGES.items():
        check_lengths("docs/" + key, title, desc, problems)
    for kind, values in (("title", [t for t, _ in PAGES.values()]), ("description", [d for _, d in PAGES.values()])):
        if len(set(values)) != len(values):
            problems.append(f"PAGES has a repeated {kind}")
    fix_docs(site, problems)
    faqs = check_landing(site, problems)
    urls = write_sitemap(site)
    for p in problems:
        print(f"seo: {p}", file=sys.stderr)
    if problems:
        sys.exit(1)
    print(f"seo: {urls} URLs in sitemap.xml, {faqs} FAQ answers match their JSON-LD")


if __name__ == "__main__":
    main()
