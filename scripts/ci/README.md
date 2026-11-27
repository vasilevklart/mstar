# CI scripts

Helpers for GitHub CI (`.github/workflows/ci.yml`) and for the optional Mac checks below.
`select_cpu_tests.py` defines the CPU Core test paths (`CPU_TEST_PATHS`), `shard_plugin.py` splits
them over runners, and `cpu-torch.txt` pins the CPU PyTorch the Mac checks install, the same
versions `ci.yml` installs.
[CONTRIBUTING.md](../../CONTRIBUTING.md#cpu-core-tests-on-github) describes the CPU Core job.

## Optional Mac checks

On **Apple Silicon macOS**, you can check your current worktree before
committing or pushing:

```bash
./scripts/ci/mac.sh all
```

This is opt-in: no Git hooks are installed and running it is not a contribution
requirement. It uses native `linux/arm64` Docker containers. **Passing these Mac
checks does not establish that the full x86 CPU suite or GPU tests pass.** GitHub
continues to run the full CPU Core suite on its Ubuntu runners.

### Setup

You need Git, Python 3, and an ARM Docker engine, such as Docker Desktop or
Docker with Colima on your Mac. Remote x86 Docker engines are unsupported.
Colima is not an mstar dependency. Allocate at least **8 GiB
of VM memory** for the Rust build; validation used 4 CPUs and 8 GiB. Rust release
linking can exhaust a 2 GiB VM even with one compilation job.

The runner uses your current Docker context. You can select another configured
context with `DOCKER_CONTEXT=your-context ./scripts/ci/mac.sh all`. It does not
create, resize, or reconfigure your VM. The initial run downloads images,
Python packages and Rust dependencies; no GPU or model weights are needed.

It supports Apple Silicon macOS with rootful Docker, without user-namespace
remapping. Intel Macs, other host operating systems and x86 emulation are
not supported. In particular, setting `CI_PLATFORM=linux/amd64` is rejected.

### Checks and coverage

| Command | Checks |
| --- | --- |
| `./scripts/ci/mac.sh lint` | Ruff and runner regression tests |
| `./scripts/ci/mac.sh cpu-smoke` | ARM-compatible subset of the CPU Core suite |
| `./scripts/ci/mac.sh dynamo` | Dynamo bridges and registration |
| `./scripts/ci/mac.sh rust` | Native Rust tests, server/extension builds and Python integration tests |

`all` runs these four checks sequentially and stops at the first failure.
`cpu-smoke` runs the CPU Core job's test paths without
the modules listed in [test/cpu-mac-exclude.txt](../../test/cpu-mac-exclude.txt):

- `test_waypoint_components.py`
- `test_waypoint_dit.py`
- `test_flex_attention_resource.py`
- `test_waypoint_benchmark_videos.py`

The first three invoke compiled FlexAttention, which required AVX2 in the CPU
PyTorch these exclusions were tested with (2.9.1; [cpu-torch.txt](cpu-torch.txt)
now pins 2.10.0). The video test hits an illegal
instruction in PyAV 19.0.1 encoding in the tested Apple VM; this also reproduces
without mstar. The entire modules are excluded,
including their portable structural cases. The runner prints the exclusions
and saves the exact pytest arguments in `cpu-smoke-tests.txt` with the
reports. New core modules are included automatically unless explicitly
excluded; invalid or stale exclusions fail the check. Existing GPU/optional-feature skips are reported by pytest.
No test assertions are changed for the Mac runner.

CPU smoke also disables PyTorch's oneDNN backend in its test process: the
ARM wheel of PyTorch 2.9.1 crashes on a standalone BF16 Linear operation in the
tested Apple VM (`scripts/ci/check.sh`). BF16 tests still execute through PyTorch's CPU fallback with their
original assertions. These checks do not cover optimized oneDNN kernels.
The setting does not affect GitHub or other jobs.

The local installation and check scripts follow the existing GitHub jobs, with
Python 3.12 and a separate environment for each check. The local image uses
Debian ARM while GitHub uses Ubuntu x86. Some dependencies and Rust stable are
unpinned; resolved packages and tool versions are recorded per run. When
updating CI dependencies, update `scripts/ci/install.sh` and
`scripts/ci/cpu-torch.txt` alongside `.github/workflows/ci.yml`.

### Source, reports and caches

The snapshot includes current tracked files and eligible untracked source,
including uncommitted edits. Untracked ignored files, local environments and
build outputs are excluded; tracked files are retained. Tests run on copies
inside disposable containers and do not modify your checkout. This checks the
working tree, rather than only staged changes or GitHub's PR merge commit.

Logs, timings and pytest reports go to `.ci-results/`. Download and Rust build
caches use Docker volumes named `mstar-ci-*`. Images and caches remain for later
runs; containers and temporary snapshots are removed. Ctrl-C also removes the
active container and retains its logs. There is no automatic local job timeout.
Rust defaults to two parallel compilation jobs; use `CI_BUILD_JOBS=1` to reduce
parallelism when needed.

The native Rust server links against the build interpreter's `libpython`.
When launching the binary separately, make that matching library available to
the dynamic loader. The local Rust check configures `LD_LIBRARY_PATH` for it.
