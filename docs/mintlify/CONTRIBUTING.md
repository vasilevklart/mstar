# Mintlify documentation

This directory is the source of the M* guides, published at https://m-star-labs.mintlify.site.
The Mintlify GitHub App deploys it from `main` on every merge. `.mintignore` keeps this file off
the site.

## Preview and validate

Install Node.js 22 LTS, then run from this directory:

```bash
npx --yes mint@4.2.1184 dev
```

The preview opens at `http://localhost:3000`. Before opening a PR, run:

```bash
npx --yes mint@4.2.1184 validate
npx --yes mint@4.2.1184 broken-links
```

`.github/workflows/docs-mintlify.yml` runs both on every PR that changes this directory. The
pinned CLI is a documentation tool only; it is not a Python dependency.

## Pages and navigation

- A page is one `.mdx` file, and its path without the extension is its URL: `models/qwen.mdx`
  is `/models/qwen`.
- `docs.json` lists every page in the sidebar. Add a new page to a group there. When you remove
  or rename a page, add its old path to `redirects`.
- The frontmatter has a `title` and a `description`, one sentence saying what the page is for.
- Every command, flag, config key, environment variable and route on a page exists in the code.

## Headings and links

- Every `##` and `###` heading has an explicit ID, such as `## Serve {#serve}`. Keep the ID when
  you reword a heading, so links from other pages and from outside keep working.
- Link within the site with `/page` or `/page#id`, not with the hostname.
- Link Python symbols, configs and the design notes in `docs/design/` to their source on GitHub.
  They are not part of the site.
- Escape `{` and `<` outside code, and give every fenced code block a language.

## Python API reference

Sphinx builds the API reference from docstrings, from `docs/index.rst`, `docs/api.rst` and
`docs/conf.py`. `.github/workflows/docs.yml` publishes it to the `gh-pages` branch on every push to
`main` that touches `docs/`, `mstar/` or `pyproject.toml`, and it is served at
https://m-star.org/mstar/api.html. The navbar links to it. To document a new subpackage, add it to
the autosummary list in `docs/api.rst`.
