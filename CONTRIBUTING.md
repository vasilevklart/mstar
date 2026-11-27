# Contributing to M*

## Issues

- **Bug?** Open a [bug report](https://github.com/mstar-project/mstar/issues/new?template=bug_report.yml).
- **Idea or feature?** Open a [feature request](https://github.com/mstar-project/mstar/issues/new?template=feature_request.yml).
- **Want a model supported?** Open a [new model request](https://github.com/mstar-project/mstar/issues/new?template=new_model.yml).

General questions are welcome too: open a plain issue or email <atindra@cs.stanford.edu>.

## Pull requests

1. Fork and create a branch.
2. Make a focused change.
3. Run `ruff check .` (CI enforces it).
4. Open a PR against `main`. It runs CI and gets a review before merging. To merge, add it to the
   merge queue ("Merge when ready"): the queue runs CI on your PR merged with `main` and with every
   PR queued ahead of it, and lands it only if that passes, so two PRs that are each green alone
   can't break `main` together.

### HTTP routes and OpenAI adapters

The Python HTTP frontend is being retired in favor of the Rust one (`rust/server/`). A new route, request field, response format or model OpenAI adapter goes into both in the same PR, with a test in `test/rust/` comparing the two; see invariant 15 in [AGENTS.md](AGENTS.md).

### Documentation

The guides are the `.mdx` pages in `docs/mintlify/`, published at https://m-star-labs.mintlify.site
when they merge to `main`. Update the page your change affects in the same PR. Preview and check
the site with Node.js 22:

```bash
cd docs/mintlify
npx --yes mint@4.2.1184 dev            # preview at http://localhost:3000
npx --yes mint@4.2.1184 validate
npx --yes mint@4.2.1184 broken-links
```

CI runs the last two on every PR that touches `docs/mintlify/`. `docs/*.rst` only builds the
Python API reference from docstrings (https://m-star.org/mstar/api.html). Page conventions are in
[docs/mintlify/CONTRIBUTING.md](docs/mintlify/CONTRIBUTING.md).

### Automated review

A bot reviews a PR only when someone comments a `/review` command. Anyone who has had a PR merged here can comment `/review light` on their own PR. Someone with write access can comment `/review light` on any PR, and on a PR whose author has write access, `/review nogpu` (which also reports P3 improvements and design questions) or `/review` (which adds a live GPU test and can approve a PR that meets a fixed rule). The bot cites the invariants in [AGENTS.md](AGENTS.md) by number. Light and nogpu reviews are advisory, and a maintainer still reviews your PR. Comment the same command again to re-review after a push, and reply to disagree. Details are in [.github/review/](.github/review/).

### AI-assisted contributions

These are welcome, with two conditions: read [AGENTS.md](AGENTS.md) first (point your agent at it, or symlink it to whatever filename your tooling expects), and say in the PR description that AI assistance was used, which tests you ran, and what the results were. You need to understand and be able to defend every line you submit.

### Optional Mac checks

On Apple Silicon macOS, `./scripts/ci/mac.sh all` runs lint, part of the CPU Core suite, the Dynamo
checks and the Rust tests in ARM Linux containers. It is opt-in and does not replace GitHub CI;
setup and coverage are in [scripts/ci/README.md](scripts/ci/README.md).

### CPU core tests on GitHub

The **CPU Core** job runs every test module under the directories in
`CPU_TEST_PATHS` ([scripts/ci/select_cpu_tests.py](scripts/ci/select_cpu_tests.py)):
`test/modular/`, `test/distributed/`, the model-local `mstar/model/*/tests/`
and a few more, on a GitHub-hosted Ubuntu runner whenever a PR targeting `main`
is opened, updated or reopened. It runs as two shards on two runners, "CPU Core 1/2"
and "CPU Core 2/2", each split over the runner's 4 cores with pytest-xdist; a test
file goes whole to one shard, by a stable hash of its path
([scripts/ci/shard_plugin.py](scripts/ci/shard_plugin.py), `MSTAR_CI_SHARD=i/n`). `test_tp_async_lockstep_sim.py`
runs in a third job beside them, "CPU Core (TP lockstep sim)", and a "CPU Core" job
passes only when all three did. It needs no GPU or model weights. New updates cancel
an older CPU Core run for the same PR. The installation and invocation commands are in
[the CI workflow](.github/workflows/ci.yml).

There is no list of test files to add to: a new module under one of those
directories runs in CI as soon as it exists, so it must pass or skip on a
CPU-only machine without model weights or optional packages. A test module
anywhere else must be listed in `NOT_CPU_CORE` in the same script, with the
place it runs instead; `test/ci/test_cpu_selection.py` fails on a module in
neither, and on any module the old `test/cpu-core.txt` ran that is no longer
selected. Within the selected directories:

- a module that needs CUDA throughout sets
  `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason=...)`;
- a module that imports an optional package at module level calls
  `pytest.importorskip("<module>")` with the import name (`flashinfer`,
  `misaki`, `qwen_tts`, ...) before that import;
- a single test that needs one makes the same call at its top, and a
  real-weights test skips when the checkpoint isn't cached;
- a test that is broken on `main` gets `@pytest.mark.skip(reason=...)` in its
  own file, with a reason that says why and names the PR or issue fixing it
  when one exists, so the fix removes the skip;
- a test that reads or writes a fixed path or port uses `tmp_path` or a free
  port instead, since the workers run tests concurrently;
- a test module outside a Python package has a basename no other selected
  module has (pytest imports it by that name; the guard checks);
- a guard on a reference package named like an mstar model (`kokoro`,
  `chatterbox`, `omnivoice`, ...) imports a submodule only the reference has,
  e.g. `pytest.importorskip("kokoro.model")`. `mstar/model` is a namespace
  package, so pytest puts it on `sys.path` for the model-local tests, and from
  then on the bare name finds mstar's own model package for the rest of the
  session.

Run the new file with CI's installation before opening the PR. Check ARM
compatibility before extending the Mac exclusion list. CI prints the slowest
tests and retains JUnit reports for seven days. Which checks
must pass before a PR merges, in the queue or not, is set in `main`'s branch protection rule;
every required check has to run on the `merge_group` event too.

The randomized tests in `test/modular/test_kv_page_ownership.py` add their seed
and recent operations to a failure; rerun one with the node id pytest prints.

## Adding a model

Start with the [Adding a New Model](https://m-star-labs.mintlify.site/adding_models) guide. It
walks through the component graph, graph walks, the state a model declares and the submodule
pattern.

By contributing, you agree that your contributions are licensed under the repository's
[Apache-2.0 license](LICENSE).
