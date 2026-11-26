# Architecture

DroidSpeak renders text to mono PCM in four strictly separated stages. Every stage talks to the
next one through plain value types, which is what keeps the library zero-dependency, NativeAOT-safe
and easy to re-target.

```text
string
  └─ TextParser (DroidSpeak.NLP)        text → Phrase[] → Word[] → Syllable[] + EmotionVector
       └─ SyllableContextBuilder        Phrase[] → flat SyllableContext[] (one per syllable)
            └─ IProsodyPredictor        SyllableContext[] → SyllableFrame[]   ← the pluggable brain
                 └─ DroidSynthesizer    SyllableFrame[] → Span<float> PCM     ← pure DSP
                      └─ WavWriter      PCM → 16-bit mono .wav
```

## NLP stage (`DroidSpeak.NLP`)

`TextParser.Parse` splits input into `Phrase`s at sentence punctuation, words and syllables
(Latin and Cyrillic vowel-group rules), and derives a continuous `EmotionVector` (Valence, Arousal)
from punctuation (`!`, `?`, `...`), `ALL CAPS` and a bilingual EN/RU emotion lexicon. Parsing is the
**only** stage that allocates freely; downstream stages accept its result as `IReadOnlyList<Phrase>`.

## Prosody stage (`DroidSpeak.Prosody`)

`SyllableContextBuilder.Count` and `.Fill` flatten the parsed utterance into a contiguous array of
`SyllableContext` structs — a fixed set of reference-free fields (indices, `TempoScale`, emotion,
punctuation, shout/stress flags, onset classes, nucleus char, word hash). Nothing else crosses the
boundary.

`IProsodyPredictor` is the whole contract:

```csharp
void Predict(
    ReadOnlySpan<SyllableContext> contexts,
    Span<SyllableFrame> frames,
    DroidVoiceProfile profile);
```

Implementations must be **deterministic** — identical contexts + profile must yield identical frames —
because `DroidSynthesizer.GetSampleCount` runs the same method on pooled buffers just to read the
sample counts back before allocating the render buffer.

Two implementations ship with the library:

| Predictor | Notes |
| :--- | :--- |
| `RuleBasedProsodyPredictor.Shared` (default) | The classic procedural voice; bit-identical to v0.3/v0.4 output on a given platform. Stateless and shareable. |
| `NeuralProsodyPredictor` (v0.6+) | Wraps an `MlpModel`; see [neural-training.md](neural-training.md). Timing/structure stays procedural, acoustic values come from the network. |

Custom predictors (whisper, alarm, game NPCs) can be injected through the `DroidSynthesizer`
constructor or the `predictor:` argument of the brain factories — the renderer never knows or cares.

## Frame (`Prosody.SyllableFrame`)

One `SyllableFrame` is the complete acoustic description of one syllable: absolute placement
(`StartSample`, `LengthSamples`, pauses), two-point pitch glides plus the v0.5 `MidFreq` arc,
start/target pairs for gain, both FM depths and growl (so the renderer can ramp continuously),
whistle/chatter specs, word span and legato flags, noise mix, formant targets, and envelope times.
Ramp-valued fields (every `Start*`/`*` pair) exist because the renderer glides between them — that
mirrors the pre-0.4 internal note event exactly, which is what lets the rule engine reproduce old
audio bit-for-bit.

## DSP stage (`DroidSpeak.DSP`)

`DroidSynthesizer` flattens frames into note events and renders them through a single continuous
oscillator (phase never resets between syllables; word-internal syllables join via exponential
portamento). Timbre and effect details live in [dsp-and-voices.md](dsp-and-voices.md).
`DroidSpeechStream` exposes the same pipeline as a pull stream for real-time playback, and
`DroidMusicSynthesizer` reuses the renderer for `DroidScore` notes.

## Zero-allocation guarantees

* `Synthesize(IReadOnlyList<Phrase>, Span<float>)` and `GetSampleCount(IReadOnlyList<Phrase>)`
  allocate **nothing** on the warm path: `SyllableContext[]` and `SyllableFrame[]` come from
  `ArrayPool<T>.Shared`, the renderer works on caller-owned `Span<float>`, and the neural forward
  pass ping-pongs between two `stackalloc` activation buffers (bounded by
  `MlpModel.MaxSupportedLayerWidth = 256`, ≈2 KB of stack).
* `string` overloads allocate at `TextParser.Parse` only; parse once, then reuse the phrases.
* Model *loading* (`MlpModel.Load`, file/HTTP reads) allocates tensors once per model; `Predict`
  never does.
* The modem (`DroidSpeak.Modem`) is likewise allocation-light on the encode path; streaming capture
  is documented in [acoustic-modem.md](acoustic-modem.md).

## Determinism

Per-syllable variation is seeded from a cross-platform FNV-1a hash of the lowercase word mixed with
the global `seed`, so the same text + seed + profile + predictor always produce identical structure.
Sample values are bit-identical on a given platform; across libm implementations they may differ by
sub-ULP rounding noise that is inaudible and absorbed by the acoustic-fingerprint regression tests.

## Project layout

| Project | Role | Dependencies |
| :--- | :--- | :--- |
| `DroidSpeak.Core` | The library: NLP, Prosody, DSP, Music, Modem, Neural | none (.NET 10 BCL only) |
| `DroidSpeak.Trainer` | Offline distillation / extraction / fine-tuning CLI | Core |
| `DroidSpeak.Demo` | Interactive CLI/REPL with live playback | Core + SoundFlow |
| `DroidSpeak.Web` | Blazor WASM playground (GitHub Pages) | Core |
| `DroidSpeak.Tests` | xUnit suite incl. golden acoustic fingerprints | all of the above |
