# DroidSpeak

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/HC-NC/DroidSpeak/release.yml)](https://github.com/HC-NC/DroidSpeak/actions)
[![GitHub Release](https://img.shields.io/github/v/release/HC-NC/DroidSpeak)](https://github.com/HC-NC/DroidSpeak/releases/latest)
[![MIT License](https://img.shields.io/github/license/HC-NC/DroidSpeak)](LICENSE)
[![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/HC-NC/DroidSpeak)](#project-structure)
[![.NET](https://img.shields.io/badge/.NET-10.0-512BD4)](https://dotnet.microsoft.com)
[![AOT Compatible](https://img.shields.io/badge/NativeAOT-Compatible-blue)](#features)

**Text-to-Beep Engine** — a zero-dependency, NativeAOT-compatible C# library (.NET 10) for procedural synthesis of expressive, astromech-style droid speech from arbitrary text, complete with voice profiles and an acoustic steganography modem.

[![Launch Interactive Web Playground](https://img.shields.io/badge/Launch_Playground-Live_WASM_Demo-38BDF8?style=for-the-badge&logo=webassembly&logoColor=white)](https://hc-nc.github.io/DroidSpeak/)

---

## Features

- **Zero External Dependencies:** `DroidSpeak.Core` relies strictly on the .NET 10 standard library.
- **NativeAOT & Reflection-Free:** Uses source-generated `System.Text.Json` (`DroidVoiceProfileJsonContext`), making it ready for iOS, consoles, Unity/Godot, and AOT deployments.
- **Deterministic Synthesis:** Syllable variations are seeded via a cross-platform FNV-1a hash of lowercase words—the same text and seed always produce bit-identical audio.
- **Emotion & Typography NLP (`DroidSpeak.NLP`):** Extracts continuous `Valence–Arousal` vectors from punctuation (`!`, `?`, `...`), `ALL CAPS`, and a built-in bilingual (EN/RU) lexicon.
- **Procedural FM & Legato DSP (`DroidSpeak.DSP`):** Continuous-phase sine/triangle oscillators, dual FM modulators, exponential portamento, word-level ADSR envelopes, consonant noise/growl shaping, and procedural chatter (30 Hz trills, micro-arpeggios, relay clicks).
- **Polyphonic Music & Melody Engine (`DroidSpeak.Music`):** Built-in zero-dependency parsers for Standard MIDI Files (`.mid`, SMF Format 0 & 1) and RTTTL ringtones, rendered with an expressive lead droid voice (portamento glides, trills, upward chirps) and relay-click percussion, plus a `--solo` lead-only mode.
- **Customizable Voice Profiles:** Built-in presets (`R2`, `Grumpy`, `Bass`, `Tiny`) plus JSON serialization to craft custom droid personalities.
- **Acoustic Steganography Modem (`DroidSpeak.Modem`):** Hides UTF-8 text payloads inside droid speech using FSK (3400 / 4400 Hz) with Hamming(12,8) forward error correction, bit interleaving, CRC-8, and an in-house radix-2 FFT decoder.
- **Allocation-Friendly API:** Supports writing directly into pre-allocated `Span<float>` buffers alongside 16-bit mono PCM `.wav` export.

## Audio Demos

Listen to raw procedural audio generated directly by `DroidSpeak.Core` (zero samples, pure C# DSP):

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
<summary>🔊 <b>3. Acoustic Modem (<code>FSK Stego</code>)</b> — <i>Hidden UTF-8 payload embedded in speech</i></summary>

> `/transmit Secret coordinates: 56.0153, 92.8932`

https://github.com/user-attachments/assets/fe18eafe-f8b1-4112-b45c-e7d553d30709

</details>

---

## Project Structure

```text
DroidSpeak.sln
├── DroidSpeak.Core        # Core library (zero dependencies)
│   ├── DroidVoiceProfile.cs
│   ├── NLP/               # Text, syllable, and emotion analysis
│   ├── DSP/               # FM synthesizer, chatter generator, WAV writer
│   ├── Music/             # MIDI & RTTTL parsers, polyphonic droid music synthesizer
│   └── Modem/             # FSK acoustic encoder/decoder & radix-2 FFT
├── DroidSpeak.Demo        # Interactive CLI/REPL with live playback (SoundFlow)
├── DroidSpeak.Web         # Interactive Blazor WebAssembly playground (GitHub Pages)
└── DroidSpeak.Tests       # xUnit test suite
```

---

## Quick Start

### Build & Test

```bash
dotnet build
dotnet test
```

### Run the Interactive CLI / REPL

```bash
# Start interactive REPL mode
dotnet run --project DroidSpeak.Demo

# Synthesize and play a single phrase with a chosen voice profile
dotnet run --project DroidSpeak.Demo -- "Hello there! Systems online." -v r2

# Play a Standard MIDI file in solo lead-whistle mode
dotnet run --project DroidSpeak.Demo -- -m melody.mid --solo -v r2

# Export speech or music to a 16-bit PCM WAV file without playback
dotnet run --project DroidSpeak.Demo -- "ALERT! Reactor overheating!" -o alert.wav --no-play
```

#### REPL Commands

| Command | Description |
| :--- | :--- |
| `<text>` | Synthesize and play text, displaying `(Valence, Arousal)` and duration |
| `/voice [r2\|grumpy\|bass\|tiny]` | Show or switch the active droid voice profile |
| `/voice json <file.json>` | Load a custom voice profile from JSON |
| `/voice save <file.json>` | Save the current voice profile to JSON |
| `/music [--solo] <file.mid\|rtttl>` | Play a MIDI file or RTTTL ringtone using the active voice (press any key to stop) |
| `/transmit <text>` | Embed hidden UTF-8 text into droid speech, save `transmit.wav`, and decode it |
| `/demo` | Play four built-in emotional test phrases |
| `/save <file.wav>` | Save the last synthesized utterance or melody to a WAV file |
| `/help` | Show available commands |
| `/exit` | Quit the REPL |

> **Note:** `DroidSpeak.Demo` uses **SoundFlow** for cross-platform live audio playback (with instant keypress interruption). `DroidSpeak.Core` itself has zero third-party packages.

---

## Usage

### Basic Synthesis & WAV Export

```csharp
using DroidSpeak.DSP;

var synthesizer = new DroidSynthesizer(sampleRate: 44100, seed: 0);
float[] samples = synthesizer.Synthesize("Beep boop! System is ready.");

WavWriter.Write("greeting.wav", samples, synthesizer.SampleRate);
```

### Zero-Allocation Buffer Rendering

```csharp
int requiredSamples = synthesizer.GetSampleCount(text);
Span<float> buffer = new float[requiredSamples];
int written = synthesizer.Synthesize(text, buffer);
```

### Voice Profiles (Code & JSON)

```csharp
using DroidSpeak;
using DroidSpeak.DSP;

var grumpy = new DroidSynthesizer(seed: 1, profile: DroidVoiceProfile.Grumpy);
float[] complaint = grumpy.Synthesize("I really don't want to do this...");

// Override voice profile for a single utterance
float[] chirp = grumpy.Synthesize("Whoa!", DroidVoiceProfile.Tiny);

// Serialize / deserialize custom profiles (NativeAOT-safe)
string json = DroidVoiceProfile.Tiny.ToJson();
DroidVoiceProfile loaded = DroidVoiceProfile.FromJson(json);
```

| Preset | `MinFreq` (Hz) | `MaxFreq` (Hz) | `SpeedMultiplier` | `FmIntensity` | `ChatterProbability` | `TriangleMix` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R2`** | 350 | 1600 | 1.00 | 1.00 | 0.50 | 0.23 |
| **`Grumpy`** | 220 | 1100 | 0.92 | 1.45 | 0.34 | 0.36 |
| **`Bass`** | 120 | 650 | 0.82 | 1.20 | 0.26 | 0.30 |
| **`Tiny`** | 600 | 2400 | 1.38 | 1.10 | 0.78 | 0.16 |

The word-level ADSR envelope, the dynamic resonant low-pass filter and the whistle intensity are
profile-driven too. Documents written by older versions that omit these fields keep the defaults:

| Parameter | Default | Range | Effect |
| :--- | :---: | :---: | :--- |
| `AttackSeconds` | 0.010 | 0.002–0.100 | Word-level ADSR attack |
| `DecaySeconds` | 0.022 | 0.005–0.150 | Word-level ADSR decay |
| `SustainLevel` | 0.74 | 0.10–1.00 | Word-level sustain gain |
| `ReleaseSeconds` | 0.016 | 0.005–0.200 | Word-level ADSR release |
| `FilterCutoffMultiplier` | 1.65 | 0.80–4.00 | Base cutoff relative to `MaxFreq` |
| `FilterResonance` | 1.45 | 0.50–4.00 | Q of the resonant low-pass |
| `WhistleProbability` | 1.00 | 0.00–1.50 | Whistle chance multiplier (`0` disables whistles) |

Preset highlights: `Grumpy` uses a lower cutoff with a stronger resonant peak, `Bass` has a slower
attack and a longer release, and `Tiny` has snappier envelopes plus extra whistles. If a word is
too short to fit the requested attack, decay and release, the envelope is scaled down
proportionally.

### Music & Melodies (`DroidSpeak.Music`)

Render Standard MIDI files (`.mid`) or RTTTL ringtone strings using any droid voice profile:

```csharp
using DroidSpeak;
using DroidSpeak.DSP;
using DroidSpeak.Music;

var musicSynth = new DroidMusicSynthesizer(profile: DroidVoiceProfile.R2);

// Parse and render an RTTTL ringtone
DroidScore joy = RtttlParser.Shared.Parse(
    "OdeToJoy:d=4,o=5,b=125:e,e,f,g,g,f,e,d,c,c,d,e,e.,8d,2d");
float[] joySamples = musicSynth.Render(joy);
WavWriter.Write("joy.wav", joySamples, musicSynth.SampleRate);

// Load a Standard MIDI file (.mid) and render only the upper lead melody in solo droid whistle mode
DroidScore midiScore = MidiParser.Shared.Load("melody.mid");
float[] soloSamples = musicSynth.Render(midiScore, DroidVoiceProfile.R2, soloMelodyOnly: true);
```

Custom musical notations can also be plugged in by implementing `IScoreParser` or `ITextScoreParser`.

### Acoustic Steganography Modem (`DroidSpeak.Modem`)

Hide up to **255 bytes** of UTF-8 text inside droid speech (~15% carrier amplitude modulated by the speech envelope) and decode it back from raw PCM audio:

```csharp
using DroidSpeak.DSP;
using DroidSpeak.Modem;

var synthesizer = new DroidSynthesizer(seed: 1);
float[] speech = synthesizer.Synthesize("Move along, nothing to see here.");

// Embed secret text into the speech signal
float[] mixed = DroidModemEncoder.EncodeIntoSpeech("Secret coordinates: 42.19", speech, synthesizer.SampleRate);
WavWriter.Write("transmit.wav", mixed, synthesizer.SampleRate);

// Recover text from audio samples
string? decoded = DroidModemDecoder.Decode(mixed, synthesizer.SampleRate);
Console.WriteLine(decoded); // Output: Secret coordinates: 42.19
```

---

## How Typography Shapes Speech

| Input | Acoustic Effect |
| :--- | :--- |
| **`!`** | Boosts `Arousal`: louder output, faster tempo, wider pitch leaps, stronger FM |
| **`?`** | Rising pitch glissando across the final two syllables |
| **`...`** | Slower tempo, lower pitch and `Arousal`, downward cadence |
| **`ALL CAPS`** | Local FM overdrive, soft clipping, and elevated pitch (shouting/alarm) |
| **Syllables in a word** | Legato glide with continuous phase; ADSR attack on first syllable, release on last |
| **Fricatives (`s, sh, ch, f`)** | Short white-noise burst on syllable attack |
| **Voiced (`b, d, g, r, z`)** | Low-frequency AM/FM modulation ("growl") |

---

## License & Disclaimer

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Disclaimer:** This is an independent, non-commercial fan-made open-source project. All audio is procedurally synthesized from scratch using mathematical DSP algorithms without any copyrighted audio samples. *Star Wars*, *R2-D2*, and related names are trademarks of Lucasfilm Ltd. / The Walt Disney Company. This project is not affiliated with, endorsed by, or sponsored by Lucasfilm Ltd. or The Walt Disney Company.