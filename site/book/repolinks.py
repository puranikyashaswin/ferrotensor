#!/usr/bin/env python3
"""mdBook preprocessor: fix relative links in chapters that include repository files.

A chapter such as `architecture.md` is `{{#include ../../../docs/architecture.md}}`,
so its links are written relative to `docs/`, not to the book. For every chapter
listed in `[preprocessor.repolinks.sources]`, each relative link is resolved against
the included file's place in the repository and rewritten to

- the book chapter that includes the target file, when there is one, or
- the file (or directory) on GitHub otherwise.

A page keeps one top-level heading: when an included file has a second `# `
heading (benchmarks/RESULTS.md does), that heading and every heading after it
move down one level. Anchors are made from the heading text, so links still work.

Links and headings inside fenced code blocks are left alone. Runs after
mdBook's `links` preprocessor, so the included text is already in place.
"""

import json
import os
import posixpath
import re
import sys

GITHUB = "https://github.com/puranikyashaswin/ferrotensor"
BRANCH = "main"

INLINE = re.compile(r"(\]\()([^)\s]+)(\))")
REFDEF = re.compile(r"^(\s{0,3}\[[^\]]+\]:\s*)(\S+)(.*)$")
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING = re.compile(r"^(#{1,5}) ")
EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|#|/)", re.IGNORECASE)


def rewrite_target(target, chapter, source, by_repo_path, repo_root, problems):
    if EXTERNAL.match(target):
        return target
    path, _, frag = target.partition("#")
    frag = "#" + frag if frag else ""
    repo_path = posixpath.normpath(posixpath.join(posixpath.dirname(source), path))
    if repo_path in by_repo_path:
        rel = posixpath.relpath(by_repo_path[repo_path], posixpath.dirname(chapter) or ".")
        return rel + frag
    full = os.path.join(repo_root, *repo_path.split("/"))
    if not os.path.exists(full) or repo_path.startswith(".."):
        problems.append(f"{chapter}: link to missing file {target} ({repo_path})")
        return target
    kind = "tree" if os.path.isdir(full) else "blob"
    return f"{GITHUB}/{kind}/{BRANCH}/{repo_path}{frag}"


def rewrite(content, chapter, source, by_repo_path, repo_root, problems):
    out, fence, h1s = [], None, 0
    for line in content.split("\n"):
        m = FENCE.match(line)
        if m:
            if fence is None:
                fence = m.group(1)
            elif m.group(1) == fence:
                fence = None
            out.append(line)
            continue
        if fence is None:
            h = HEADING.match(line)
            if h and len(h.group(1)) == 1:
                h1s += 1
            if h and h1s > 1:
                line = "#" + line
            fix = lambda t: rewrite_target(t, chapter, source, by_repo_path, repo_root, problems)
            line = INLINE.sub(lambda m: m.group(1) + fix(m.group(2)) + m.group(3), line)
            line = REFDEF.sub(lambda m: m.group(1) + fix(m.group(2)) + m.group(3), line)
        out.append(line)
    return "\n".join(out)


def walk(items, visit):
    for item in items:
        chapter = item.get("Chapter") if isinstance(item, dict) else None
        if chapter:
            visit(chapter)
            walk(chapter.get("sub_items", []), visit)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "supports":
        sys.exit(0)
    context, book = json.load(sys.stdin)
    root = context["root"]
    repo_root = os.path.normpath(os.path.join(root, "..", ".."))
    sources = context["config"]["preprocessor"]["repolinks"]["sources"]
    by_repo_path = {src: chapter for chapter, src in sources.items()}
    problems = []

    def visit(chapter):
        path = (chapter.get("path") or "").replace(os.sep, "/")
        if path in sources:
            chapter["content"] = rewrite(
                chapter["content"], path, sources[path], by_repo_path, repo_root, problems
            )

    walk(book.get("items", book.get("sections", [])), visit)
    for p in problems:
        print(f"repolinks: {p}", file=sys.stderr)
    if problems:
        sys.exit(1)
    json.dump(book, sys.stdout)


if __name__ == "__main__":
    main()
