# DSP & Voice Profiles

Everything here lives in `DroidSpeak.DSP` (renderer) and `DroidSpeak` (`DroidVoiceProfile`). The
stages that feed it are described in [architecture.md](architecture.md).

## The carrier

One continuous oscillator spans the whole utterance — phase never resets between syllables — and
the carrier is a mix of a sine and a triangle wave (`TriangleMix`). Two FM modulators are applied:
one harmonic (`FmRatio`) and one inharmonic (`FmRatio2`, ≈3.07) for the metallic edge. On top of that:

* **Growl** — low-frequency AM/FM modulation for voiced onsets (`b, d, g, r, z`).
* **Noise transients** — short white-noise bursts for sibilants (`s, sh, ch, f`).
* **Word-level ADSR** — attack/decay/sustain/release applied across each word, scaled down
  proportionally if the word is too short to fit the requested times.
* **Resonant low-pass** — dynamic cutoff relative to `MaxFreq`, with a Q peak.
* **Legato** — exponential portamento between syllables of the same word; continuous glide.
* **Whistles & chatter** — seeded 1400–2600 Hz whistles/sobs on stressed or emotional syllables,
  plus 30 Hz trills, micro-arpeggios and relay clicks, all driven by stable word hashes.

All randomness derives from the FNV-1a word hash mixed with the global seed, so output is
deterministic for a given text + seed + profile + predictor.

## v0.5 expressive layer

Five profile parameters add analog/vocal character. **All default to `0`**, which reproduces the
v0.4 renderer bit-for-bit for a hand-built profile or an older JSON document; the shipped presets
enable them.

| Parameter | Default | Range | Effect |
| :--- | :---: | :---: | :--- |
| `FormantMix` | 0.00 | 0.00–1.00 | Parallel vowel formant bank driven by the syllable nucleus (Latin/Cyrillic → F1/F2) |
| `RingModMix` | 0.00 | 0.00–1.00 | Inharmonic ring modulation for metallic sidebands |
| `SaturationDrive` | 0.00 | 0.00–4.00 | Analog soft-clipping before the resonant low-pass |
| `RoomMix` | 0.00 | 0.00–1.00 | Early-reflection ambience (7 ms / 13 ms taps) |
| `PitchCurve` | 0.00 | 0.00–1.00 | Three-point pitch arcs and energy-driven attack chirps |

`RuleBasedProsodyPredictor` scales per-syllable formant mix, ring depth and chirp strength by vowel
presence, onset type, stress, shout and phrase emotion, so the expressive layer follows the text
instead of coloring every syllable identically. In a custom predictor, the same effect is produced
through the frame fields below.

### Frame fields the renderer reads

| Frame field(s) | Effect when non-zero |
| :--- | :--- |
| `MidFreq` | Bends the start→end glide into an arc with the requested midpoint |
| `AttackChirp` | Signed 14 ms pitch-attack chirp (`0.12` = +12 %); `0` keeps the legacy onset scoop |
| `FormantF1`, `FormantF2`, `FormantMix` | Per-syllable formant bank (geometric mean of F1/F2 acts as the band floor) |
| `RingModMix` | Per-syllable ring depth |
| `Drive`, `Saturation` | Per-syllable soft-clipping drive/amount |
| `TrillDepth`, `TrillRate` | Pitch trill |

## Presets

| Preset | `MinFreq` | `MaxFreq` | `SpeedMultiplier` | `FmIntensity` | `ChatterProbability` | `TriangleMix` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`R2`** | 350 | 1600 | 1.00 | 1.00 | 0.50 | 0.23 |
| **`Grumpy`** | 220 | 1100 | 0.92 | 1.45 | 0.34 | 0.36 |
| **`Bass`** | 120 | 650 | 0.82 | 1.20 | 0.26 | 0.30 |
| **`Tiny`** | 600 | 2400 | 1.38 | 1.10 | 0.78 | 0.16 |

`Grumpy` lowers the cutoff with a stronger resonant peak; `Bass` has a slower attack and longer
release; `Tiny` has snappier envelopes plus extra whistles.

```csharp
using DroidSpeak;
using DroidSpeak.DSP;

var grumpy = new DroidSynthesizer(seed: 1, profile: DroidVoiceProfile.Grumpy);
float[] a = grumpy.Synthesize("I really don't want to do this...");

// One-shot override
float[] b = grumpy.Synthesize("Whoa!", DroidVoiceProfile.Tiny);
```

## Voice profiles as JSON

`DroidVoiceProfile` is an immutable record with `init` properties, validated on load through a
source-generated (NativeAOT-safe) `System.Text.Json` contract.

```csharp
string json = DroidVoiceProfile.Tiny.ToJson();
DroidVoiceProfile loaded = DroidVoiceProfile.FromJson(json); // revalidates every range

// Compose a custom profile with `with`; Normalized() clamps and fixes ordering.
var custom = DroidVoiceProfile.R2 with { MaxFreq = 2000f, PitchCurve = 0.7f };
var synth = new DroidSynthesizer(profile: custom);
```

Missing properties fall back to defaults, so documents written by older versions keep loading.
`Normalized()` clamps every value into its allowed range and guarantees `MinFreq < MaxFreq`; the
synthesizer normalizes the profile it is given, so out-of-range input can never reach the DMA path.

The Web playground's customizer builds the same object through `VoiceProfileSliders`
(`FromProfile` / `ToProfile` / `Normalized`), which is the integer-valued round trip the sliders bind
to — tests assert it reproduces the preset object exactly.

## Live streaming and WAV export

```csharp
DroidSpeechStream stream = new DroidSynthesizer().CreateStream("Chunked playback!");
var buffer = new float[22050];
int n;
while ((n = stream.Read(buffer)) > 0) { /* queue buffer[..n] */ }

WavWriter.Write("out.wav", samples, 44100);   // 16-bit mono PCM
```

`DroidSpeechStream` renders only the slice the caller asks for, so playback can start before the
whole utterance exists. `WavWriter` (Core) writes 16-bit mono PCM; the 16-bit mono PCM readers the
tools and the playground use for modem input live in `DroidSpeak.Web` (`WavReader`) and
`DroidSpeak.Trainer` (audio extraction), not in Core.

## Music (`DroidSpeak.Music`)

`DroidMusicSynthesizer` renders a `DroidScore` through the same voice. Built-in parsers:

* `MidiParser.Shared` — Standard MIDI Files, SMF Format 0 & 1 (`.mid`).
* `RtttlParser.Shared` — RTTTL ringtone strings.

The upper line is rendered as an expressive lead droid voice (portamento glides, trills, upward
chirps); harmony is softened so the melody stays clear. `Render(score, profile, soloMelodyOnly: true)`
keeps only the lead melody. Custom notations plug in via `IScoreParser` / `ITextScoreParser`.

```csharp
var music = new DroidMusicSynthesizer(profile: DroidVoiceProfile.R2);
DroidScore joy = RtttlParser.Shared.Parse("Ode:d=4,o=5,b=125:e,e,f,g,g,f,e,d,c,c,d,e,e.,8d,2d");
float[] pcm = music.Render(joy);
DroidScore mid = MidiParser.Shared.Load("melody.mid");
float[] solo = music.Render(mid, DroidVoiceProfile.R2, soloMelodyOnly: true);
```
