# DroidSpeak

[![.NET](https://shields.io)](https://microsoft.com)
[![License: MIT](https://shields.io)](LICENSE)
[![Zero Dependencies](https://shields.io)](#features)
[![AOT Compatible](https://shields.io)](#features)

**Text-to-Beep Engine** — a zero-dependency, NativeAOT-compatible C# library (.NET 10) for procedural synthesis of expressive, astromech-style droid speech from arbitrary text, complete with voice profiles and an acoustic steganography modem.

---

## Features

- **Zero External Dependencies:** `DroidSpeak.Core` relies strictly on the .NET 10 standard library.
- **NativeAOT & Reflection-Free:** Uses source-generated `System.Text.Json` (`DroidVoiceProfileJsonContext`), making it ready for iOS, consoles, Unity/Godot, and AOT deployments.
- **Deterministic Synthesis:** Syllable variations are seeded via a cross-platform FNV-1a hash of lowercase words—the same text and seed always produce bit-identical audio.
- **Emotion & Typography NLP (`DroidSpeak.NLP`):** Extracts continuous `Valence–Arousal` vectors from punctuation (`!`, `?`, `...`), `ALL CAPS`, and a built-in bilingual (EN/RU) lexicon.
- **Procedural FM & Legato DSP (`DroidSpeak.DSP`):** Continuous-phase sine/triangle oscillators, dual FM modulators, exponential portamento, word-level ADSR envelopes, consonant noise/growl shaping, and procedural chatter (30 Hz trills, micro-arpeggios, relay clicks).
- **Customizable Voice Profiles:** Built-in presets (`R2`, `Grumpy`, `Bass`, `Tiny`) plus JSON serialization to craft custom droid personalities.
- **Acoustic Steganography Modem (`DroidSpeak.Modem`):** Hides UTF-8 text payloads inside droid speech using FSK (3400 / 4400 Hz) with Hamming(12,8) forward error correction, bit interleaving, CRC-8, and an in-house radix-2 FFT decoder.
- **Allocation-Friendly API:** Supports writing directly into pre-allocated `Span<float>` buffers alongside 16-bit mono PCM `.wav` export.

---

## Project Structure

```text
DroidSpeak.sln
├── DroidSpeak.Core        # Core library (zero dependencies)
│   ├── DroidVoiceProfile.cs
│   ├── NLP/               # Text, syllable, and emotion analysis
│   ├── DSP/               # FM synthesizer, chatter generator, WAV writer
│   └── Modem/             # FSK acoustic encoder/decoder & radix-2 FFT
├── DroidSpeak.Demo        # Interactive CLI/REPL with live playback (SoundFlow)
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

# Synthesize and play a single phrase
dotnet run --project DroidSpeak.Demo -- "Hello there! Systems online."

# Export to a 16-bit PCM WAV file without playback
dotnet run --project DroidSpeak.Demo -- "ALERT! Reactor overheating!" -o alert.wav --no-play
```

#### REPL Commands

| Command | Description |
| :--- | :--- |
| `<text>` | Synthesize and play text, displaying `(Valence, Arousal)` and duration |
| `/voice [r2\|grumpy\|bass\|tiny]` | Show or switch the active droid voice profile |
| `/voice json <file.json>` | Load a custom voice profile from JSON |
| `/voice save <file.json>` | Save the current voice profile to JSON |
| `/transmit <text>` | Embed hidden UTF-8 text into droid speech, save `transmit.wav`, and decode it |
| `/demo` | Play four built-in emotional test phrases |
| `/save <file.wav>` | Save the last synthesized utterance to a WAV file |
| `/help` | Show available commands |
| `/exit` | Quit the REPL |

> **Note:** `DroidSpeak.Demo` uses **SoundFlow** for cross-platform live audio playback. `DroidSpeak.Core` itself has zero third-party packages.

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
