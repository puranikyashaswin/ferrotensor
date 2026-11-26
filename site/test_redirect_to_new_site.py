"""Tests for redirect_to_new_site.py; standard library only. Run: python3 -m unittest discover -s site -p 'test_*.py'."""

import importlib.util
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("redirect_to_new_site", os.path.join(HERE, "redirect_to_new_site.py"))
redirect = importlib.util.module_from_spec(spec)
spec.loader.exec_module(redirect)

OLD = "https://puranikyashaswin.github.io/ferrotensor/"


def page(extra=""):
    return f"<!doctype html>\n<html>\n<head>\n<title>T</title>\n{extra}</head>\n<body><h1>T</h1></body>\n</html>\n"


FIXTURES = {
    "index.html": page(
        f'<link rel="canonical" href="{OLD}">\n<meta property="og:url" content="{OLD}">\n'
        f'<meta property="og:image" content="{OLD}og.png">\n'
    ),
    "docs/installation.html": page(f'<link rel="canonical" href="{OLD}docs/installation.html">\n'),
    "docs/examples/lora-sft.html": page(f'<link rel="canonical" href="{OLD}docs/examples/lora-sft.html">\n'),
    "old/unknown.html": page(f'<link rel="canonical" href="{OLD}old/unknown.html">\n'),
    "docs/toc.html": page('<meta http-equiv="refresh" content="5; url=https://example.org/">\n'),
}

EXPECTED = {
    "index.html": "https://ferrotensor.dev/",
    "docs/installation.html": "https://ferrotensor.dev/docs/installation",
    "docs/examples/lora-sft.html": "https://ferrotensor.dev/docs/examples/lora-sft",
    "old/unknown.html": "https://ferrotensor.dev/",
    "docs/toc.html": "https://ferrotensor.dev/docs/",
}


class RedirectTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        for rel, text in FIXTURES.items():
            self.write(rel, text)

    def write(self, rel, text):
        path = os.path.join(self.root, *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def read(self, rel):
        with open(os.path.join(self.root, *rel.split("/")), encoding="utf-8") as f:
            return f.read()

    def test_mapping(self):
        cases = {
            "index.html": ("", "exact"),
            "guide/index.html": ("guide/", "exact"),
            "about/index.html": ("about/", "exact"),
            "docs/index.html": ("docs/", "exact"),
            "docs/introduction.html": ("docs/", "alias"),
            "docs/python-reference.html": ("docs/python-reference", "exact"),
            "docs/examples/classify.html": ("docs/examples/classify", "exact"),
            "docs/404.html": ("docs/", "docs"),
            "docs/examples/other.html": ("docs/", "docs"),
            "x/y.html": ("", "home"),
        }
        for rel, want in cases.items():
            self.assertEqual(redirect.new_path(rel), want, rel)

    def test_new_paths_are_the_seventeen_of_the_new_site(self):
        self.assertEqual(len(redirect.NEW_PATHS), 17)

    def test_exact_tags(self):
        done, broken = redirect.redirect_site(self.root)
        self.assertEqual(broken, [])
        self.assertEqual(len(done), 5)
        for rel, url in EXPECTED.items():
            text = self.read(rel)
            refresh = f'<meta http-equiv="refresh" content="0; url={url}">'
            canonical = f'<link rel="canonical" href="{url}">'
            self.assertIn("<head>\n" + refresh + "\n" + canonical + "\n<title>T</title>", text, rel)
            self.assertEqual(text.count('rel="canonical"'), 1, rel)
            self.assertEqual(text.count('http-equiv="refresh"'), 1, rel)
            self.assertNotIn(f'href="{OLD}', text, rel)
            self.assertNotIn(f'url={OLD}', text, rel)
            self.assertNotIn("example.org", text, rel)

    def test_og_url_and_other_tags(self):
        redirect.redirect_site(self.root)
        text = self.read("index.html")
        self.assertIn('<meta property="og:url" content="https://ferrotensor.dev/">', text)
        self.assertEqual(text.count('property="og:url"'), 1)
        # The image still lives on the old host, so its tag is left alone.
        self.assertIn(f'<meta property="og:image" content="{OLD}og.png">', text)
        self.assertIn("<body><h1>T</h1></body>", text)

    def test_second_run_changes_nothing(self):
        redirect.redirect_site(self.root)
        first = {rel: self.read(rel) for rel in FIXTURES}
        redirect.redirect_site(self.root)
        self.assertEqual({rel: self.read(rel) for rel in FIXTURES}, first)

    def test_missing_head_is_reported_and_file_kept(self):
        self.write("docs/fragment.html", "<p>no head here</p>\n")
        done, broken = redirect.redirect_site(self.root)
        self.assertEqual(broken, ["docs/fragment.html"])
        self.assertEqual(self.read("docs/fragment.html"), "<p>no head here</p>\n")
        self.assertEqual(redirect.main(["x", self.root]), 1)

    def test_header_element_is_not_a_head(self):
        self.assertRaises(ValueError, redirect.rewrite_head, "<header>x</header>", "https://ferrotensor.dev/")

    def test_sitemap_and_robots(self):
        self.write("sitemap.xml", "<urlset/>")
        self.write("robots.txt", f"User-agent: *\nAllow: /\n\nSitemap: {OLD}sitemap.xml\n")
        removed = redirect.drop_old_sitemap(self.root)
        self.assertEqual(removed, ["sitemap.xml", "the Sitemap line of robots.txt"])
        self.assertFalse(os.path.exists(os.path.join(self.root, "sitemap.xml")))
        self.assertEqual(self.read("robots.txt"), "User-agent: *\nAllow: /\n")


if __name__ == "__main__":
    unittest.main()
