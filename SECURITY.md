# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| Latest release | ✅ Yes — fixes ship as a new release |
| Latest `master` | ✅ Yes |
| Older releases / commits | ❌ No — please update to the latest release (pre-1.0: no backports) |

Published releases are never modified; a fix always arrives as a new version, and affected
releases get a notice in their release notes (see `docs/RELEASING.md` § Maintaining published
releases).

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

To report a security vulnerability, please open a **[GitHub Security Advisory](https://github.com/shifulegend/project-zero/security/advisories/new)** (private disclosure).

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge your report within 72 hours and aim to release a fix within 14 days for critical issues.

## If a credential is exposed

A secret scan (Gitleaks) runs on every push and pull request. If a token, API key or password is
found in the repository, its history or a release asset:

1. The credential is treated as compromised and revoked at its provider within 24 hours of the
   finding being confirmed.
2. A replacement is issued and any affected automation is updated.
3. The file is fixed in a new commit; if a published release is affected, its notes gain a dated
   notice and the fix ships as a new release (published releases are never modified).

Please report a suspected exposure through the Security Advisory link above.

## Scope

Security issues relevant to this project include:

- **Command injection** via the agentic `<exec>` tool interceptor (allow-list bypass)
- **Path traversal** in model file loading or mmap paths
- **Buffer overflows** in tokenizer, weight loading, or GGUF parsing
- **Memory safety** issues (use-after-free, out-of-bounds reads/writes)
- **Integer overflows** in size calculations or offset arithmetic

## Known Security Design Decisions

- The `<exec>` agentic tool is **allow-listed** to: `echo`, `ls`, `cat`, `pwd`, `uname`, `date`, `id`. All other commands are refused. See [`WALKTHROUGH_PHASE14.md`](docs/phases/WALKTHROUGH_PHASE14.md) for design rationale.
- Model files are loaded via `mmap` with `PROT_READ` — weights are never written to.
- The engine does not make network requests at runtime.

## Security Audit

A prior independent security audit is documented in [`SECURITY_FINDINGS.md`](docs/reports/SECURITY_FINDINGS.md) and [`INDEPENDENT_CODE_AUDIT_REPORT.md`](docs/reports/INDEPENDENT_CODE_AUDIT_REPORT.md).
