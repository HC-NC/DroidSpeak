# Security Policy

## Supported Versions

The following versions of DroidSpeak receive security updates.

| Version | Supported          | Notes |
| ------- | ------------------ | ----- |
| 1.0.x   | :white_check_mark: | Current stable release line. |
| < 1.0   | :x:                | Pre-release / preview builds — please upgrade to `1.0.x`. |

Only the latest patch of the supported minor version is maintained. If you are on an older build,
please reproduce the issue against the newest `1.0.x` release before reporting.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public issues, discussions, or pull
requests.**

Instead, report them privately using GitHub's
[private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
feature:

1. Go to the **Security** tab of the repository.
2. Click **Report a vulnerability** to open a private security advisory.

Please include as much of the following as possible:

- The affected version(s) and component (`DroidSpeak.Core`, `DroidSpeak.Demo`,
  `DroidSpeak.Trainer`, `DroidSpeak.Web`, neural `.dsbrain` loading, or the acoustic modem).
- A description of the vulnerability and its potential impact.
- Step-by-step instructions or a minimal proof of concept to reproduce it.
- Any known mitigations or workarounds.

We aim to acknowledge reports within **a few business days** and will keep you updated on our
progress toward a fix and disclosure.

## What to expect

- We will confirm receipt of your report and begin investigating.
- If the report is accepted, we will work on a fix and coordinate a disclosure timeline with you.
- If the report is declined, we will explain why.
- We will credit reporters in the advisory unless you prefer to remain anonymous.

## Scope notes

DroidSpeak processes untrusted input in a few places — most notably the `.dsbrain` and `.dsdata`
binary loaders, WAV/MIDI/RTTTL parsers, and the textual input to synthesis. These loaders are
designed to reject malformed data (bounds-checked, no reflection, no code execution), so
crash-on-malformed-input bugs and resource-exhaustion issues in these parsers are in scope and
very much welcome.

The project ships **no embedded neural models** and makes **no network calls** from
`DroidSpeak.Core`. Thank you for helping keep DroidSpeak and its users safe.
