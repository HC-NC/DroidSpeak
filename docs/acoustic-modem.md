# Acoustic Modem

`DroidSpeak.Modem` hides a short UTF-8 message inside droid audio and recovers it from a WAV file
or a live microphone — without filtering the voice. Payloads are limited to **255 bytes**
(`AcousticFrameProtocol.MaxPayloadLength`).

```csharp
using DroidSpeak.DSP;
using DroidSpeak.Modem;

var synth = new DroidSynthesizer(seed: 1);
float[] speech = synth.Synthesize("Move along, nothing to see here.");

float[] mixed = DroidModemEncoder.EncodeIntoSpeech("Secret coordinates: 42.19", speech, ModemMode.HighBandOverlay, 44100);
WavWriter.Write("transmit.wav", mixed, 44100);

if (DroidModemDecoder.TryDecode(mixed, 44100, out var result))
    Console.WriteLine($"{result.Text} · CRC {result.CrcValid} · FEC {result.CorrectedBits} · {result.ClockDriftPpm:+0;-0;0} ppm");
```

## Two physical-layer modes

`ModemMode` selects where the packet lives relative to the voice.

| | `HighBandOverlay` (default) | `AudibleChirpStream` |
| :--- | :--- | :--- |
| Sub-carriers | 4-FSK @ **14.6 / 15.4 / 16.2 / 17.0 kHz** | 4-FSK @ **2.3 / 2.9 / 3.5 / 4.3 kHz** |
| Symbol duration | 4 ms | 6 ms |
| Minimum sample rate | 38 kHz | 10 kHz |
| Voice | Entire audible band untouched; tones sit above it | Voice untouched; chirps use a natural pause or the decorative tail |
| Channel | Needs ≥38 kHz hardware, and survives only if nothing low-passes below ~18 kHz | Survives narrowband/band-limited channels (phone codecs, small speakers) |

Choose `HighBandOverlay` for full-fidelity file transmission; choose `AudibleChirpStream` whenever
the audio may pass through a codec or speaker that rolls off below 18 kHz. `DroidModemEncoder.Encode`
alone emits a clean `HighBandOverlay` packet with no speech. `maskLevel` (default
`DroidModemEncoder.DefaultMaskLevel`) scales the modem level relative to the speech peak.

## Frame format (v0.9)

`AcousticFrameProtocol` builds a little-endian frame:

| Part | Bytes | Contents |
| :--- | :--- | :--- |
| Header | 2 | `(version << 4) \| mode` (version `0x09`, mode 0/1), then `payloadLength` |
| Payload | 1–255 | UTF-8 of the message |
| CRC | 2 | CRC-16 **/CCITT-FALSE** (poly `0x1021`, init `0xFFFF`, no reflection) over header + payload |

**FEC:** every byte is split into two nibbles, each encoded as a **Hamming(7,4)** codeword, so the
physical layer corrects one flipped bit per 4-bit nibble. The syndrome also detects uncorrectable
patterns, but final acceptance is always decided by CRC-16 (a multi-bit error can masquerade as a
correctable single-bit one).

**Bits → symbols:** codeword bits are bit-packed two at a time into 4-FSK symbols. Between symbols a
**timing pilot** is inserted every **24 body symbols** (`PilotInterval`, pilot symbol values
`{0, 3}` = the extreme tones). The demodulator re-locks the symbol phase on the pilots and reports
the estimated **clock drift in ppm**, which is what lets a real speaker→microphone link with a
slightly-off sample clock still decode.

**Sync preamble:** `FF FF AA AA AA AA D5` — two bytes of 1-bits for a clean energy onset, four
alternating `0xAA` bytes to train the symbol clock (packed into `10/01`-style pilot-like symbols),
and a `0xD5` marker for the exact end of the preamble.

## Decoding

```csharp
// One-shot from a buffer (e.g. a decoded .wav)
DroidModemDecoder.TryDecode(samples, sampleRate, out ModemDecodeResult r);

// Force a specific mode when you know the channel
DroidModemDecoder.Decode(samples, sampleRate, ModemMode.AudibleChirpStream);
```

`ModemDecodeResult` carries `Text`, `Mode`, `CorrectedBits`, `CrcValid`, `ClockDriftPpm`,
`SamplesConsumed` and `Legacy`. Successful results always have `CrcValid == true`.

### Streaming capture

`StreamingAcousticDemodulator` consumes microphone chunks and yields packets as they complete. Its
buffer is bounded to the last 30 seconds, so a long-running listener does not grow without bound.

```csharp
var receiver = new StreamingAcousticDemodulator(sampleRate: 44100);
receiver.Feed(micChunk);
while (receiver.TryReadPacket(out var packet))
    Console.WriteLine(packet.Text);
receiver.Reset(); // drop buffered audio
```

Use at least 38 kHz capture for `HighBandOverlay`; the playground refuses to enable it on a
lower-rate microphone and asks you to switch to `AudibleChirpStream`.

## Legacy v0.8 compatibility

Recordings made by the older binary-FSK modem still decode: 2-FSK at **3400 Hz** (bit 0) and
**4400 Hz** (bit 1), 3 ms symbols, **Hamming(12,8)** FEC and **CRC-8/ATM** (`0x07`), with the same
`FF FF AA AA AA AA D5` preamble. Decoded legacy packets set `ModemDecodeResult.Legacy = true`. There
is no legacy *encoder* — new transmissions always use the v0.9 rich-spectrum format.

## Practical notes

* Keep payloads well under 255 bytes; longer text throws `ArgumentException` at encode time.
* `HighBandOverlay` is preserved by 44.1/48 kHz hardware but **not** guaranteed through codecs or
  speakers that low-pass below ~18 kHz; prefer `AudibleChirpStream` there.
* This is a toy/experimental steganography channel, not a secure link — it provides error
  correction for an acoustic channel, not confidentiality. Do not use it for secrets.
