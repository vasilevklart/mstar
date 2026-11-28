"""The images ship mstar-server, which `mstar serve` runs as its HTTP frontend by
default: find_server_binary (mstar/api_server/rust_frontend.py) stops it at
startup when there is none, and an editable install does not build one."""

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKER_DIR = REPO_ROOT / "mstar/integrations/dynamo/docker"
CUDA_DOCKERFILES = [DOCKER_DIR / "Dockerfile", DOCKER_DIR / "Dockerfile.dynamo-base"]
XPU_DOCKERFILE = REPO_ROOT / "docker/Dockerfile.xpu"
# the checked-in Cargo.lock, as the wheels build it
BUILD = "cargo build --release --locked --manifest-path rust/server/Cargo.toml"
# on every base image's PATH, which find_server_binary searches
INSTALL = "install -m 755 rust/server/target/release/mstar-server /usr/local/bin/mstar-server"


def server_step(path):
    """The Dockerfile with continuation lines joined, and the RUN step that builds mstar-server."""
    text = path.read_text().replace("\\\n", " ")
    steps = [m for m in re.finditer(r"^RUN [^\n]*$", text, re.MULTILINE) if "rust/server/Cargo.toml" in m.group(0)]
    if len(steps) != 1:
        raise AssertionError(f"{path.name}: {len(steps)} RUN steps build mstar-server, not 1")
    return text, steps[0]


class DockerServerTests(unittest.TestCase):
    def test_every_image_builds_and_installs_mstar_server(self):
        for path in [*CUDA_DOCKERFILES, XPU_DOCKERFILE]:
            with self.subTest(path.name):
                step = server_step(path)[1].group(0)
                self.assertIn(BUILD, step)
                self.assertIn(INSTALL, step)
                self.assertLess(step.index(BUILD), step.index(INSTALL))
                # in the same layer, or the build tree stays in the image
                self.assertRegex(step, r"rm -rf [^&]*\brust/server/target\b")

    def test_cuda_images_build_it_in_the_serve_stage_with_the_pinned_toolchain(self):
        for path in CUDA_DOCKERFILES:
            with self.subTest(path.name):
                text, match = server_step(path)
                step = match.group(0)
                # no toolchain named: rustup installs the one rust-toolchain.toml pins
                toolchain = re.search(r"rustup toolchain install\s+&&", step)
                self.assertIsNotNone(toolchain)
                self.assertLess(toolchain.start(), step.find(BUILD))
                self.assertRegex(step, r"rm -rf /tmp/rustup /tmp/cargo\b")
                # mstar serve's own lookup finds the installed binary
                self.assertIn("found = find_server_binary(); assert found == '/usr/local/bin/mstar-server'", step)
                # the first stage, which `--target serve` builds alone
                stages = [m.start() for m in re.finditer(r"^FROM ", text, re.MULTILINE)]
                self.assertTrue(len(stages) == 1 or match.start() < stages[1], stages)


if __name__ == "__main__":
    unittest.main()
