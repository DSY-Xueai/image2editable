# Security policy

## Supported versions

The `0.2.x` release line receives security fixes. Older releases are not supported.

## Reporting a vulnerability

Please do not open a public issue for a security vulnerability. Send a private report to the repository maintainers with:

- a short description and impact;
- reproducible steps or a minimal proof of concept;
- affected version and operating system;
- any logs or artifacts needed to reproduce the issue, after removing private input data.

Do not include source images, PDFs, presentations, model caches, API keys, or other private data unless the maintainer explicitly requests a redacted sample.

## Local model and host data

Local model files and conversion inputs remain on the user's machine and are not part of releases. Host-agent mode may send diagnostic artifacts to the configured host service; use the local agent for sensitive files.
