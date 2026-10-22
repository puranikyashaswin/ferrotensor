# Website

Source of <https://puranikyashaswin.github.io/ferrotensor/>.

- `landing/`: the static landing page (HTML, one CSS file, a little JS). No build step.
- `book/`: the mdBook guide, served under `/docs/`. Its chapters `{{#include}}`
  the files in `docs/`, `benchmarks/RESULTS.md`, `CHANGELOG.md` and
  `ferrotensor-py/`, so edit those files, not the book. `repolinks.py` rewrites
  their relative links to book pages or to GitHub.
- `build.sh`: builds both into `site/dist/` (gitignored).

```bash
site/build.sh                     # needs mdbook 0.5.4 and python3
mkdir -p /tmp/ftsite && ln -sfn "$PWD/site/dist" /tmp/ftsite/ferrotensor
python3 -m http.server -d /tmp/ftsite 8000   # open http://localhost:8000/ferrotensor/
```

`.github/workflows/pages.yml` builds and deploys the site on pushes to `main`.
`landing/og.svg` is the source of the social preview image `landing/og.png`.
