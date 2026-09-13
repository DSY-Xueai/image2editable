# Security Policy

## Supported versions

Only the `0.3.x` release line receives security fixes.

## Reporting a Vulnerability

Please do not open a public issue for a security vulnerability. Use [GitHub Private Vulnerability Reporting](https://github.com/DSY-Xueai/image2editable/security/advisories/new) when it is enabled. If unavailable, open an issue requesting a private contact channel without including vulnerability details. Response times depend on maintainer availability.

Include:

- a short description and impact;
- reproducible steps or a minimal proof of concept;
- the affected version and operating system;
- sanitized logs or artifacts needed to reproduce the issue.

Do not include source images, PDFs, presentations, model caches, API keys, or other private data unless I explicitly request a redacted sample.

## Host Data

Runtime model files and conversion inputs are not included in releases. The Agent may send diagnostic artifacts to its model service. Before processing sensitive files, verify that the service's data policy and environment meet your requirements.
