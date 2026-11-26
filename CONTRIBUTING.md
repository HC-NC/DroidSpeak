# Contributing to DroidSpeak

Thanks for your interest in improving DroidSpeak! This guide covers how to build the project, run
the tests, train custom neural voices, and submit a pull request.

By participating you agree to our [Code of Conduct](CODE_OF_CONDUCT.md).

## Prerequisites

- **.NET 10 SDK** (the solution targets `net10.0`). Check with `dotnet --version`.
- A terminal. Everything below works on Windows, Linux, and macOS.
- Optional, only for training from reference recordings: some WAV files to analyze.

No other tooling is required. `DroidSpeak.Core` and `DroidSpeak.Trainer` have **no third-party
dependencies**; `DroidSpeak.Demo` uses [SoundFlow](https://www.nuget.org/packages/SoundFlow) for
playback.

## Repository layout

```text
DroidSpeak.Core      # Core library (zero external dependencies, NativeAOT/trim safe)
DroidSpeak.Trainer   # Offline distillation / extraction / fine-tuning CLI
DroidSpeak.Demo      # Interactive CLI/REPL with live playback
DroidSpeak.Web       # Blazor WebAssembly playground (GitHub Pages)
DroidSpeak.Tests     # xUnit suite, including acoustic fingerprint regression tests
docs/                # Architecture, DSP, neural-training, and modem guides
```

## Build and test

```bash
# Restore + build the whole solution in Release
dotnet build DroidSpeak.sln -c Release

# Run the full test suite
dotnet test DroidSpeak.sln -c Release
```

**A pull request must build with zero warnings and pass all tests.** Treat warnings as errors
locally by adding `-warnaserror` if you want to be certain:

```bash
dotnet build DroidSpeak.sln -c Release -warnaserror
```

### Running the tools from source

```bash
# Interactive REPL with live playback
dotnet run --project DroidSpeak.Demo

# One-shot synthesis
dotnet run --project DroidSpeak.Demo -- "Hello there! Systems online." -v r2 --no-play -o hello.wav

# Trainer
dotnet run --project DroidSpeak.Trainer -- selftest
```

## Training a custom `.dsbrain` model

The default voice is rule-based; neural prosody is entirely optional. The `DroidSpeak.Trainer` CLI
distills the rule engine into a small MLP and can adapt it to your own recordings. See the full
[Neural Prosody Training guide](docs/neural-training.md) for the network layout, the `.dsbrain` /
`.dsdata` binary formats, and every option.

The canonical workflow is **distill → (optional) extract → finetune**:

```bash
# 1. Distill the rule-based teacher into a base model.
dotnet run --project DroidSpeak.Trainer -- distill \
  --samples 8000 --epochs 60 --arch 64,48 --activation tanh --output-activation sigmoid \
  --seed 20260107 --profile r2 --out r2-base.dsbrain

# Evaluate the result against the teacher.
dotnet run --project DroidSpeak.Trainer -- eval --brain r2-base.dsbrain --samples 1500 --profile r2

# 2. (Optional) Extract an acoustic dataset from your own reference WAVs.
dotnet run --project DroidSpeak.Trainer -- extract \
  --input ./reference-wavs --profile r2 --out r2-ref.dsdata

# 3. Fine-tune without catastrophic forgetting via procedural replay.
dotnet run --project DroidSpeak.Trainer -- finetune \
  --brain r2-base.dsbrain --data r2-ref.dsdata --out r2-neural.dsbrain \
  --epochs 60 --replay 3000 --replay-ratio 1 --profile r2
```

Use a trained model from code with
`DroidSynthesizer.FromBrainFile("r2-neural.dsbrain", profile: DroidVoiceProfile.R2)`, from the CLI
with `--brain r2-neural.dsbrain`, or in the playground by dropping the file into
`DroidSpeak.Web/wwwroot/brains/` and adding its name to `manifest.json`.

> `.dsbrain` and `.dsdata` files are git-ignored — they are developer-local artifacts and must never
> be committed.

## Design constraints

DroidSpeak is deliberately small and portable. Pull requests are reviewed against these rules:

- **Zero external dependencies in `DroidSpeak.Core`.** Use only the .NET BCL. Do not add
  `PackageReference` entries to the core project.
- **NativeAOT & trim safety.** No reflection-based serialization or dynamic code. Use
  source-generated JSON (`System.Text.Json` source generation) and explicit binary readers/writers.
  `DroidSpeak.Core` builds with `IsAotCompatible` and `EnableTrimAnalyzer`, so the analyzers will
  flag violations.
- **Determinism.** Audio must remain a pure function of text + seed + profile + predictor.
- **Do not change CLI archive formats** or the `.dsbrain` / `.dsdata` binary layouts without a
  documented, versioned migration path.
- **Stay allocation-friendly.** Keep the warm rendering path allocation-free where it already is.

## Testing guidelines

- Add or update tests in `DroidSpeak.Tests` for any behavior change.
- Prefer deterministic tests (fixed seeds, fixed sample rates) so they are stable in CI.
- Acoustic-fingerprint regression tests exist for rendering; if you intentionally change the
  generated audio, update those expectations as part of the same PR and explain why.

## Submitting a pull request

1. **Fork** the repository and create a topic branch off `main`
   (e.g. `feat/modem-turbo-mode`, `fix/parser-question-rise`).
2. Keep the change **focused** — one logical change per PR.
3. Make sure `dotnet build DroidSpeak.sln -c Release` produces **zero warnings** and
   `dotnet test DroidSpeak.sln -c Release` is **green**.
4. Fill in the [pull request template](.github/PULL_REQUEST_TEMPLATE.md), including the checklist.
5. Write commit messages in [Conventional Commits](https://www.conventionalcommits.org/) style where
   practical (e.g. `feat(dsp): add ...`, `fix(neural): ...`, `docs: ...`), since releases are
   generated from commit history.
6. Open the PR against `main` and respond to review feedback.

CI (GitHub Actions) builds and tests every pull request; CI is the source of truth for the
zero-warning requirement.

## Reporting bugs and requesting features

Use the structured issue forms under [.github/ISSUE_TEMPLATE](.github/ISSUE_TEMPLATE). For questions
and open-ended ideas, use [Discussions](https://github.com/HC-NC/DroidSpeak/discussions). For
security issues, follow [SECURITY.md](SECURITY.md) — do not open a public issue.

## License

By contributing, you agree that your contributions are licensed under the [MIT License](LICENSE).
