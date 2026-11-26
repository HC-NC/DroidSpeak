# Neural Prosody Training

DroidSpeak's default voice is rule-based. Everything in this guide is **optional**: it trains a
small MLP that *imitates* the rule engine (distillation) or adapts to reference recordings
(fine-tuning), then loads it as `NeuralProsodyPredictor`. Core never embeds a model — you ship a
`.dsbrain` yourself.

Prerequisites: .NET 10 SDK. The trainer is pure C# math with no third-party packages.

## The network (v0.6–v0.8)

| Aspect | Value |
| :--- | :--- |
| Input | `SyllableFeatureEncoder.InputSize` = **60** (20 features × a `[previous, current, next]` window) |
| Output | `NeuralProsodyPredictor.OutputSize` = **17** normalized values in `[0, 1]` |
| Default hidden stack | 60 → 64 (`ModRelu`) → 32 (`Tanh`) → 17 (`Sigmoid`) |
| Default init seed | `MlpModel.DefaultInitSeed` (`0x5EED0001`) |
| Limits | `MaxSupportedLayerWidth` = 256 per layer, `MaxLayerCount` = 32 |
| Activations persisted per layer | `Sigmoid`, `Tanh`, `Relu`, `LeakyRelu`, `ModRelu`, `Linear` (members are never renumbered) |

A loadable model must have exactly **60 inputs and 17 outputs**; `NeuralProsodyPredictor` rejects
anything else. Hidden widths are free within the 256/32 limits.

### The 17 outputs

In index order (constants on `NeuralProsodyPredictor`): start frequency `0`, mid/pitch-curve `1`,
end frequency `2`, duration `3`, gain `4`, FM depth `5`, FM ratio `6`, growl `7`, noise `8`, ring
mod (`9`), formant 1 (`10`), formant 2 (`11`), formant mix (`12`), attack (`13`), release (`14`),
saturation (`15`), pause-after (`16`). Every value is clamped before it reaches a frame, so even a
badly trained model cannot emit non-finite or out-of-range acoustics.

### What stays procedural

Timing and structure — phrase leads, word gaps, hyphen breaks, word spans, whistle/chatter events —
come from the same deterministic rules as `RuleBasedProsodyPredictor`, scaled by the model's own
duration and pause outputs. That is why `GetSampleCount` stays honest: it runs the exact `Predict`
method on pooled buffers and reads the sample counts back. The network sees `[prev, cur, next]`, so
it shapes its own contours; the rule engine's question-rise / falling-ending is deliberately *not*
applied on top.

## File formats

### `.dsbrain` — model container

Little-endian, reflection-free, NativeAOT/WASM-safe. Layout (`DsBrainFormat`):

| Part | Fields |
| :--- | :--- |
| Header (12 B) | `"DSBR"` magic, `version` (1), flags, `inputSize` u16, `outputSize` u16, `layerCount` u16 |
| Per layer (14 B + tensors) | `inputSize` u16, `outputSize` u16, `activation` u8, flags, `weightCount` u32, `biasCount` u32, then `weightCount` + `outputSize` f32 values |

Loading is the safety boundary: a file is accepted only when the header is intact, the layer chain
is continuous, every tensor has exactly the declared size, no value is NaN/∞, and there are **no
trailing bytes**. Anything else throws `InvalidDataException`. Writing is deterministic — the same
model always produces the same bytes.

### `.dsdata` — extracted acoustic dataset

Little-endian, self-describing, uncompressed (`DroidSpeak.Trainer.Datasets.DsDataFile`):

| Part | Fields |
| :--- | :--- |
| Header (26 B) | `"DSDA"` magic, `version` (1), flags, `renderRate`, `seed`, `clipCount`, `syllableCount` (five i32) |
| Per clip | `clipId` i32, path length i32, UTF-8 path bytes, syllable count i32, then that many context records + frame records |

Contexts are 52 B and frames **153 B** each (fixed-size structs, see `DsDataFile.ContextBytes` /
`FrameBytes`), storing `SyllableContext` / `SyllableFrame` verbatim. Datasets store clips rather
than encoded rows so they can be re-encoded for a different profile or sample rate without
re-reading the source WAVs.

## Step-by-step: distill → extract → finetune

All commands run through the trainer CLI:

```bash
dotnet run --project DroidSpeak.Trainer -- <command> [options]
```

### 1. Distill a teacher

Builds a procedural corpus, distills the rule-based teacher into an MLP with Adam, and saves the
best validation checkpoint.

```bash
dotnet run --project DroidSpeak.Trainer -- distill \
  --samples 8000 --epochs 60 --arch 64,48 --activation tanh --output-activation sigmoid \
  --seed 20260107 --profile r2 --out r2-base.dsbrain
```

Key options: `--samples`, `--epochs`, `--batch-size`, `--learning-rate`, `--lr-min`,
`--arch A,B,...`, `--activation`, `--output-activation`, `--seed`, `--validation`,
`--schedule {constant|cosine|step}`, `--step-size`, `--decay`, Adam `--beta1/--beta2/--epsilon`,
`--profile` (preset name or JSON path), `--sample-rate`, `--log-every`, `--out`.

Validate the result against the teacher:

```bash
dotnet run --project DroidSpeak.Trainer -- eval --brain r2-base.dsbrain --samples 1500 --profile r2
```

### 2. Extract from recordings (optional)

Segments reference WAVs into syllables and maps each pulse to a `SyllableContext` + `SyllableFrame`,
producing a `.dsdata` dataset. Use this only when you want the voice to follow *your* recordings.

```bash
dotnet run --project DroidSpeak.Trainer -- extract \
  --input ./reference-wavs --profile r2 --out r2-ref.dsdata
```

Useful options: `--render-rate` (baked into contexts/frames), `--analysis-rate` (resample rate,
default 22050), `--resampler {linear|cubic}`, `--frame-ms`, `--hop-ratio`, `--open-factor`,
`--close-factor`, `--min-pulse-ms`, `--min-gap-ms`, `--word-gap-ms`. Multiple inputs are
semicolon-separated: `--input "a.wav;b.wav;dir"`.

### 3. Fine-tune without forgetting

Seeds from a `.dsbrain`, trains on `.dsdata`, and replays the procedural corpus each epoch so the
voice adapts without catastrophic forgetting.

```bash
dotnet run --project DroidSpeak.Trainer -- finetune \
  --brain r2-base.dsbrain --data r2-ref.dsdata --out r2-neural.dsbrain \
  --epochs 60 --replay 3000 --replay-ratio 1 --profile r2
```

Key options: `--brain`, `--data` (semicolon-separated), `--out`, `--profile`, `--epochs`,
`--batch-size`, `--learning-rate`, `--lr-min`, `--schedule`, `--replay`, `--replay-ratio`,
`--validation` (held out by clip), `--seed`, `--log-every`.

### Also available

`selftest` trains a tiny XOR network to prove backprop and Adam work — a good first check on a new
machine.

## Using a trained model

```csharp
using DroidSpeak.DSP;

// One call: load the .dsbrain and build the neural predictor.
var synth = DroidSynthesizer.FromBrainFile("r2-neural.dsbrain", seed: 0, profile: DroidVoiceProfile.R2);
float[] pcm = synth.Synthesize("Beep boop! Neural prosody online.");

// Or from bytes you fetched / embedded, then inject explicitly.
var other = new DroidSynthesizer(predictor: new Neural.NeuralProsodyPredictor(MlpModel.Load(path)));
```

In the CLI demo: `--brain r2-neural.dsbrain` for one run, or `/brain r2-neural.dsbrain` in the REPL
(`/brain off` returns to the rule engine). In the Web playground, drop the file into
`wwwroot/brains/`, add it to `manifest.json`, or use the **Load .dsbrain File** control — see the
[README](../README.md#neural-prosody-in-the-playground).

When any load fails, fall back to `RuleBasedProsodyPredictor.Shared`; that is what the demo and the
playground do.
