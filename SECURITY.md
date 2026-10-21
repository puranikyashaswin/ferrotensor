# Security Policy

## Supported versions

Ferrotensor is pre-1.0. Only the latest release on `main` receives security fixes.

## Reporting a vulnerability

Please do not open a public issue. Report privately through
[GitHub Security Advisories](https://github.com/puranikyashaswin/ferrotensor/security/advisories/new),
or email puranikyashaswinsharma@gmail.com.

Include a description, reproduction steps, and the affected version or commit.
You can expect an acknowledgement within 7 days. Fixes are disclosed in the
changelog once a patched release is available.

## Scope

Of particular interest: memory-safety or out-of-bounds issues in WGSL kernels or
buffer handling, and unsafe handling of untrusted model files (safetensors,
tokenizer JSON) in `ferrotensor-loader`.
