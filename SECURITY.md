# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| Latest `master` | ✅ Yes |
| Older commits | ❌ No — please update to latest |

## Reporting a Vulnerability

**Please do NOT open a public GitHub issue for security vulnerabilities.**

To report a security vulnerability, please open a **[GitHub Security Advisory](https://github.com/shifulegend/project-zero/security/advisories/new)** (private disclosure).

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge your report within 72 hours and aim to release a fix within 14 days for critical issues.

## Scope

Security issues relevant to this project include:

- **Command injection** via the agentic `<exec>` tool interceptor (allow-list bypass)
- **Path traversal** in model file loading or mmap paths
- **Buffer overflows** in tokenizer, weight loading, or GGUF parsing
- **Memory safety** issues (use-after-free, out-of-bounds reads/writes)
- **Integer overflows** in size calculations or offset arithmetic

## Known Security Design Decisions

- The `<exec>` agentic tool is **allow-listed** to: `echo`, `ls`, `cat`, `pwd`, `uname`, `date`, `id`. All other commands are refused. See [`WALKTHROUGH_PHASE14.md`](WALKTHROUGH_PHASE14.md) for design rationale.
- Model files are loaded via `mmap` with `PROT_READ` — weights are never written to.
- The engine does not make network requests at runtime.

## Security Audit

A prior independent security audit is documented in [`SECURITY_FINDINGS.md`](SECURITY_FINDINGS.md) and [`INDEPENDENT_CODE_AUDIT_REPORT.md`](INDEPENDENT_CODE_AUDIT_REPORT.md).
