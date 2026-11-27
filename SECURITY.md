# Security policy

## Reporting a vulnerability

Please don't open a public issue for a security problem. Report it privately through GitHub instead: on the repository's **Security** tab, choose **Report a vulnerability** ([direct link](https://github.com/mstar-project/mstar/security/advisories/new)). If you can't use the form, email <atindra@cs.stanford.edu> with "M* security" in the subject.

A useful report says which commit or release you tested, how the server was launched (command, config, which frontend), and the request or steps that show the problem.

We will confirm we received it, work on a fix in a private advisory, and credit you when it is published, unless you'd rather not be named.

## Supported versions

Fixes land on `main`. Older releases don't get backports, so please check that the problem still happens on `main`.

## What counts

In scope: anything that lets a client of the HTTP API do more than the deployment allows, for example:

- getting past the API key (`MSTAR_API_KEY`) when one is set;
- reading or writing server files outside the upload directory, or running code, through a request;
- reaching internal addresses through remote media fetch (`MSTAR_ALLOW_REMOTE=1`) despite its address checks;
- one request's prompts, media or outputs reaching another client;
- exhausting memory, disk or connections past a documented limit (body size, media files, pixels, audio seconds, concurrent requests).

By design, and not a vulnerability on its own:

- With no API key set, anyone who can reach the HTTP port can run requests. `mstar serve` listens on `0.0.0.0` by default, so set a key, put an authenticating gateway in front, or pass `--host 127.0.0.1`.
- The internal control mesh (ZeroMQ) and the Mooncake RDMA and TCP tensor-transport ports have no peer authentication. They trust their network: keep them on loopback or a private fabric.
- Model weights, configs and custom code that the operator chooses to load are trusted.
- Bugs in a dependency belong upstream. Do tell us if M* requires or pins a vulnerable version.
