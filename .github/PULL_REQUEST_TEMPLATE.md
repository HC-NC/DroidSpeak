<!--
  Thanks for contributing to DroidSpeak! Please keep pull requests focused and fill in the
  checklist below. Delete any section that does not apply.
-->

## Summary

<!-- What does this change do, and why? Link related issues, e.g. "Closes #123". -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature / voice preset / DSP or neural capability
- [ ] Modem or music enhancement
- [ ] Documentation
- [ ] Build / CI / tooling
- [ ] Refactor (no behavior change)

## How was it verified?

<!-- Commands run, tests added/updated, and any audio before/after comparisons. -->

```bash
dotnet build DroidSpeak.sln -c Release
dotnet test  DroidSpeak.sln -c Release
```

## Checklist

- [ ] **`DroidSpeak.Core` has zero third-party dependencies.** No new `PackageReference` was added to the core library (only the .NET BCL).
- [ ] **NativeAOT / trim safe.** No reflection-based serialization or dynamic code was introduced; source-generated JSON and existing format readers/writers are used where applicable.
- [ ] **Builds clean with zero warnings** in `Release` across the whole solution.
- [ ] **Unit tests pass**, and new behavior is covered by tests in `DroidSpeak.Tests` where practical.
- [ ] I did not change existing CLI archive formats or the `.dsbrain`/`.dsdata` binary layouts without a documented, versioned migration.
- [ ] Public API changes are documented (XML docs and/or the relevant `docs/*.md` guide).
- [ ] `CHANGELOG`/release notes or the README were updated if user-facing behavior changed.

## Additional notes

<!-- Anything reviewers should know: performance implications, follow-up work, screenshots, sample audio. -->