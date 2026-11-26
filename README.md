# DroidSpeak

[![NuGet Version](https://img.shields.io/nuget/v/DroidSpeak.Core?style=flat&logo=nuget&label=NuGet)](https://nuget.org/packages/DroidSpeak.Core)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/HC-NC/DroidSpeak/release.yml)](https://github.com/HC-NC/DroidSpeak/actions)
[![MIT License](https://img.shields.io/github/license/HC-NC/DroidSpeak)](LICENSE)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/HC-NC/DroidSpeak)](#project-structure)
[![.NET](https://img.shields.io/badge/.NET-10.0-512BD4)](https://dotnet.microsoft.com)
[![AOT Compatible](https://img.shields.io/badge/NativeAOT-Compatible-blue)](#why-droidspeak)

**Text-to-Beep Engine** — a zero-dependency, NativeAOT-compatible C# library (.NET 10) for procedural
synthesis of expressive, astromech-style droid speech from arbitrary text, with polyphonic
MIDI/RTTTL music, pluggable (optionally neural) prosody, voice profiles and an acoustic
steganography modem.

[![Launch Interactive Web Playground](https://img.shields.io/badge/Launch_Playground-Live_WASM_Demo-38BDF8?style=for-the-badge&logo=webassembly&logoColor=white)](https://hc-nc.github.io/DroidSpeak/)

```bash
dotnet add package DroidSpeak.Core
```

---

## Why DroidSpeak

- **Zero dependencies.** `DroidSpeak.Core` uses only the .NET 10 BCL.
- **NativeAOT & trim safe.** Source-generated JSON, no reflection; runs on iOS, consoles, Unity/Godot
  and Blazor WebAssembly.
- **Deterministic.** Audio is a pure function of text + seed + profile + predictor.
- **Allocation-friendly.** Render straight into a pre-allocated `Span<float>`, or pull a
  `DroidSpeechStream` in real time.
- **Pluggable prosody.** Swap the rule engine for a trained MLP (`.dsbrain`) or your own predictor.
- **Extras included.** MIDI/RTTTL music, and a 4-FSK acoustic modem with FEC and CRCs.

## Quick start

```csharp
using DroidSpeak.DSP;

var synth = new DroidSynthesizer(sampleRate: 44100, seed: 0);
float[] samples = synth.Synthesize("Beep boop! System is ready.");
WavWriter.Write("greeting.wav", samples, synth.SampleRate);

// Zero-allocation rendering once the text is parsed:
using DroidSpeak.NLP;
IReadOnlyList<Phrase> phrases = TextParser.Parse("Beep boop! System is ready.");
Span<float> buffer = new float[synth.GetSampleCount(phrases)];
int written = synth.Synthesize(phrases, buffer); // no allocations on the warm path
```

### Custom voice profiles

```csharp
using DroidSpeak;
using DroidSpeak.DSP;

var grumpy = new DroidSynthesizer(seed: 1, profile: DroidVoiceProfile.Grumpy);
float[] a = grumpy.Synthesize("I really don't want to do this...");

var custom = DroidVoiceProfile.R2 with { MaxFreq = 2000f, PitchCurve = 0.7f };
string json = custom.ToJson();                       // NativeAOT-safe round trip
DroidVoiceProfile loaded = DroidVoiceProfile.FromJson(json);
```

Four presets ship built in: `R2`, `Grumpy`, `Bass`, `Tiny`. See
[DSP & Voice Profiles](https://github.com/HC-NC/DroidSpeak/blob/main/docs/dsp-and-voices.md) for the
full parameter list and the v0.5 formant / ring-mod / saturation / room / pitch-curve layer.

### Neural prosody (optional)

A trained `.dsbrain` model can replace the rule engine with one call. No model is embedded in the
library; the classic rule-based voice stays the default.

```csharp
using DroidSpeak.DSP;

var neural = DroidSynthesizer.FromBrainFile("r2-neural.dsbrain", profile: DroidVoiceProfile.R2);
float[] pcm = neural.Synthesize("Beep boop! Neural prosody online.");

// or from bytes you fetched/embedded
var other = DroidSynthesizer.FromBrainBytes(bytes, seed: 0);
```

Train your own with the `DroidSpeak.Trainer` CLI — see
[Neural Prosody Training](https://github.com/HC-NC/DroidSpeak/blob/main/docs/neural-training.md) for
the MLP architecture, `.dsbrain`/`.dsdata` formats, and the `distill` → `extract` → `finetune`
workflow.

### Acoustic modem

```csharp
using DroidSpeak.DSP;
using DroidSpeak.Modem;

var synth = new DroidSynthesizer(seed: 1);
float[] speech = synth.Synthesize("Move along, nothing to see here.");

float[] mixed = DroidModemEncoder.EncodeIntoSpeech(
    "Secret coordinates: 42.19", speech, ModemMode.HighBandOverlay, synth.SampleRate);

if (DroidModemDecoder.TryDecode(mixed, synth.SampleRate, out var result))
    Console.WriteLine($"{result.Text} · CRC OK · {result.ClockDriftPpm:+0;-0;0} ppm");
```

`HighBandOverlay` adds tones at 14.6–17 kHz without touching the voice; `AudibleChirpStream` uses
2.3–4.3 kHz chirps for narrowband channels. Hamming(7,4) FEC, CRC-16, timing pilots, clock-drift
tracking and streaming capture are covered in
[Acoustic Modem](https://github.com/HC-NC/DroidSpeak/blob/main/docs/acoustic-modem.md).

## Web playground & drop-in brains

The interactive playground runs entirely in the browser (Blazor WebAssembly) at
**[hc-nc.github.io/DroidSpeak](https://hc-nc.github.io/DroidSpeak/)**.

### Neural prosody in the playground

- A **Prosody Engine / Brain** dropdown lists `Rule-Based (Classic)` plus every model named in
  `DroidSpeak.Web/wwwroot/brains/manifest.json`.
- To host a model with no recompile: drop `my-voice.dsbrain` into `wwwroot/brains/` and add its file
  name to the manifest (a JSON array of bare file names):

  ```json
  [ "r2-neural.dsbrain" ]
  ```

  `.dsbrain` files are git-ignored; the manifest is tracked and ships as `[]` by default.
- **Load .dsbrain File** lets any visitor use a local model entirely client-side.
- Missing or malformed manifests and failed loads fall back to `Rule-Based (Classic)`.

> On a static host, publish both the model file and the updated `manifest.json` to the served
> `brains/` directory. No C# rebuild is needed, but GitHub Pages still requires a deployment.

## CLI tools

`DroidSpeak.Demo` and `DroidSpeak.Trainer` are also available as self-contained single-file binaries
for `win-x64`, `linux-x64` and `osx-arm64` from the
**[latest GitHub Release](https://github.com/HC-NC/DroidSpeak/releases/latest)** — no .NET runtime
install required.

```bash
# Interactive REPL with live playback
./DroidSpeak.Demo

# One-shot synthesis, MIDI, and export
./DroidSpeak.Demo "Hello there! Systems online." -v r2
./DroidSpeak.Demo -m melody.mid --solo -v r2
./DroidSpeak.Demo "ALERT! Reactor overheating!" -o alert.wav --no-play
./DroidSpeak.Demo "Neural voice" --brain r2-neural.dsbrain -o neural.wav --no-play

# Offline prosody training
./DroidSpeak.Trainer distill --samples 8000 --profile r2 --out r2-base.dsbrain
./DroidSpeak.Trainer eval --brain r2-base.dsbrain --samples 1500
```

REPL commands: `/voice`, `/voice json|save <file>`, `/brain [file|off]`, `/music [--solo] <file>`,
`/transmit [--mode high|audible] <text>`, `/demo`, `/save <file.wav>`, `/help`, `/exit`.

From source: `dotnet run --project DroidSpeak.Demo` / `dotnet run --project DroidSpeak.Trainer -- <command>`.

> `DroidSpeak.Demo` uses **SoundFlow** for cross-platform playback. `DroidSpeak.Core` itself has zero
> third-party packages.

## Documentation

| Guide | Contents |
| :--- | :--- |
| [Architecture](https://github.com/HC-NC/DroidSpeak/blob/main/docs/architecture.md) | Pipeline stages, `IProsodyPredictor`, `SyllableFrame`, zero-allocation guarantees, determinism |
| [DSP & Voice Profiles](https://github.com/HC-NC/DroidSpeak/blob/main/docs/dsp-and-voices.md) | Carrier/FM/envelope chain, v0.5 expressive layer, presets, JSON profiles, music, streaming |
| [Neural Prosody Training](https://github.com/HC-NC/DroidSpeak/blob/main/docs/neural-training.md) | MLP architecture, `.dsbrain`/`.dsdata` formats, `distill` → `extract` → `finetune` |
| [Acoustic Modem](https://github.com/HC-NC/DroidSpeak/blob/main/docs/acoustic-modem.md) | Modem modes, Hamming(7,4) FEC, CRC-16, pilots, streaming capture, legacy decode |

## How typography shapes speech

| Input | Acoustic Effect |
| :--- | :--- |
| **`!`** | Boosts `Arousal`: louder output, faster tempo, wider pitch leaps, stronger FM |
| **`?`** | Rising pitch glissando across the final two syllables |
| **`...`** | Slower tempo, lower pitch and `Arousal`, downward cadence |
| **`ALL CAPS`** | Local FM overdrive, soft clipping, and elevated pitch (shouting/alarm) |
| **Syllables in a word** | Legato glide with continuous phase; ADSR attack on first, release on last |
| **Fricatives (`s, sh, ch, f`)** | Short white-noise burst on syllable attack |
| **Voiced (`b, d, g, r, z`)** | Low-frequency AM/FM modulation ("growl") |

## Audio demos

<details>
<summary>🔊 <b>1. Expressive Astromech (<code>R2</code>)</b> — <i>Emotion shifts, trills & legato glides</i></summary>

> *"Beep-beep! Oh no, wait-wait, WHAT IS THAT CRITICAL ERROR?! Well... guess I'll have to fix the console again..."*

https://github.com/user-attachments/assets/1b717af0-5443-464e-9660-1f1974bcccb3

</details>

<details>
<summary>🔊 <b>2. Grumpy Unit (<code>Grumpy</code>)</b> — <i>Low pitch, heavy FM growl</i></summary>

> *"Bzzzt... I really don't want to compile this legacy code... Fine, executing now!"*

https://github.com/user-attachments/assets/9d936f29-e37c-4f2a-af88-7e1a1bbf1a22

</details>

<details>
<summary>🔊 <b>3. Rich-Spectrum Acoustic Modem (<code>HighBandOverlay</code>)</b> — <i>Full voice preserved, data above the audible band</i></summary>

> `/transmit --mode high Secret coordinates: 56.0153, 92.8932`

https://github.com/user-attachments/assets/fe18eafe-f8b1-4112-b45c-e7d553d30709

</details>

## Project structure

```text
DroidSpeak.sln
├── DroidSpeak.Core        # Core library (zero dependencies)
│   ├── NLP/               # Text, syllable, and emotion analysis
│   ├── Prosody/           # SyllableContext -> SyllableFrame (IProsodyPredictor)
│   ├── Neural/            # MLP inference, .dsbrain format, NeuralProsodyPredictor
│   ├── DSP/               # FM synthesizer, chatter, streaming, WAV writer
│   ├── Music/             # MIDI & RTTTL parsers, polyphonic droid music
│   └── Modem/             # 4-FSK encoder/decoder, FEC, CRC, streaming receiver
├── DroidSpeak.Trainer     # Offline distillation / extraction / fine-tuning CLI
├── DroidSpeak.Demo        # Interactive CLI/REPL with live playback (SoundFlow)
├── DroidSpeak.Web         # Interactive Blazor WebAssembly playground (GitHub Pages)
└── DroidSpeak.Tests       # xUnit suite incl. acoustic fingerprint regression tests
```

## Build & test

```bash
dotnet build DroidSpeak.sln -c Release
dotnet test  DroidSpeak.sln -c Release
```

## License & disclaimer

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Disclaimer:** This is an independent, non-commercial fan-made open-source project. All audio is
procedurally synthesized from scratch using mathematical DSP algorithms without any copyrighted
audio samples. *Star Wars*, *R2-D2*, and related names are trademarks of Lucasfilm Ltd. / The Walt
Disney Company. This project is not affiliated with, endorsed by, or sponsored by Lucasfilm Ltd. or
The Walt Disney Company.