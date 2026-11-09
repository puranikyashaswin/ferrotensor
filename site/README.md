# Website

Source of <https://ferrotensor.dev/>, served by GitHub Pages with ferrotensor.dev as
the repository's custom domain (Settings > Pages). The old address,
<https://puranikyashaswin.github.io/ferrotensor/>, redirects there.

- `landing/`: the static landing page (HTML, one CSS file, a little JS). No build step.
- `book/`: the mdBook guide, served under `/docs/`. Its chapters `{{#include}}`
  the files in `docs/`, `benchmarks/RESULTS.md`, `CHANGELOG.md` and
  `ferrotensor-py/`, so edit those files, not the book. `repolinks.py` rewrites
  their relative links to book pages or to GitHub.
- `seo.py`: run by `build.sh` after mdBook. Sets each guide page's title and
  description (from its `PAGES` table: add an entry when you add a chapter),
  canonical and Open Graph tags, fixes a few accessibility issues in mdBook's
  template, and writes `sitemap.xml` and `robots.txt`. The build fails if a
  page is missing from `PAGES`, a title or description is too long, a page has
  more than one `<h1>`, or the landing page's FAQ text and its `FAQPage`
  JSON-LD differ: edit both together.
- `landing/llms.txt`: summary and key links for language models
  ([llmstxt.org](https://llmstxt.org)).
- `release.py`: run by `build.sh` after mdBook. Fills `{{ft_version}}` from the
  workspace `Cargo.toml` everywhere it appears, and replaces the
  `<!-- whats-new -->` marker on the landing page with the newest
  `CHANGELOG.md` release. The build fails if the two versions disagree.
- `build.sh`: builds everything into `site/dist/` (gitignored).

```bash
site/build.sh                     # needs mdbook 0.5.4 and python3
python3 -m http.server -d site/dist 8000   # open http://localhost:8000/
```

`.github/workflows/pages.yml` builds and deploys the site on pushes to `main`.
`landing/og.svg` is the source of the social preview image `landing/og.png`.

The site is served at the domain root, so `robots.txt` and
`https://ferrotensor.dev/sitemap.xml` are where crawlers expect them. Add the
domain in Google Search Console and Bing Webmaster Tools and submit the
sitemap there.
