# Security Policy

## Supported versions

Only the `0.2.x` release line receives security fixes.

## Reporting a Vulnerability

Please do not open a public issue for a security vulnerability. Use GitHub Private Vulnerability Reporting when it is enabled for this repository; otherwise contact the maintainer privately. I will acknowledge a complete report within 48 hours and provide an initial assessment within 7 days.

Include:

- a short description and impact;
- reproducible steps or a minimal proof of concept;
- the affected version and operating system;
- sanitized logs or artifacts needed to reproduce the issue.

Do not include source images, PDFs, presentations, model caches, API keys, or other private data unless I explicitly request a redacted sample.

## Local Model and Host Data

Local model files and conversion inputs stay on the user's machine and are not included in releases. Host Agent may send diagnostic artifacts to the configured host service. For sensitive files, use a local model service (`local-service`) that you control and verify that its endpoint stays within your environment.
