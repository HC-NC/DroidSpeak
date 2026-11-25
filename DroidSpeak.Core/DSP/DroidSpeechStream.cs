using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.DSP;

/// <summary>
/// Stateful pull-based stream for real-time, chunked speech synthesis.
/// <para>
/// Each <see cref="Read"/> call synthesizes only the slice of the <see cref="DroidSynthesizer"/>
/// plan that overlaps the requested window. Oscillator phase, the two-pole resonant low-pass
/// filter and the envelope follower are all carried over between calls, so pulling the stream in
/// arbitrary chunk sizes produces the same continuous utterance as rendering it in one go.
/// </para>
/// </summary>
public sealed class DroidSpeechStream
{
    private const float TwoPi = MathF.PI * 2f;
    private const float PeakCeiling = 0.95f;

    // Soft-knee limiter: unity gain below the knee, smooth compression above it, asymptote at
    // PeakCeiling. The knee keeps normal speech untouched while taming shouted transients.
    private const float Knee = 0.57f;

    private readonly int _sampleRate;
    private readonly DroidSynthesizer.SynthesisPlan _plan;
    private readonly DroidVoiceProfile _profile;

    private int _position;
    private int _noteCursor;
    private int _whistleCursor;
    private int _chatterCursor;

    // Continuous oscillator phases across chunks.
    private double _carrierPhase;
    private double _modulatorPhase;
    private double _modulatorPhase2;

    // Continuous 2-pole SVF low-pass filter state across chunks.
    private float _filterLow;
    private float _filterBand;
    private float _envFollower;

    internal DroidSpeechStream(
        DroidSynthesizer synthesizer,
        DroidSynthesizer.SynthesisPlan plan,
        DroidVoiceProfile profile)
    {
        _sampleRate = synthesizer.SampleRate;
        _plan = plan;
        _profile = profile;
    }

    /// <summary>Output sample rate in Hz.</summary>
    public int SampleRate => _sampleRate;

    /// <summary>Total length of the utterance in samples.</summary>
    public int TotalSamples => _plan.TotalSamples;

    /// <summary>Current playback/render cursor position in samples.</summary>
    public int Position => _position;

    /// <summary>True when every sample has been pulled.</summary>
    public bool IsCompleted => _position >= _plan.TotalSamples;

    /// <summary>Resets the stream back to sample 0 and clears all oscillator/filter state.</summary>
    public void Reset()
    {
        _position = 0;
        _noteCursor = 0;
        _whistleCursor = 0;
        _chatterCursor = 0;
        _carrierPhase = 0d;
        _modulatorPhase = 0d;
        _modulatorPhase2 = 0d;
        _filterLow = 0f;
        _filterBand = 0f;
        _envFollower = 0f;
    }

    /// <summary>
    /// Synthesizes the next chunk of mono PCM samples into <paramref name="destination"/> and
    /// returns the number of samples written (0 once the utterance is complete).
    /// </summary>
    public int Read(Span<float> destination)
    {
        if (destination.IsEmpty || _position >= _plan.TotalSamples)
        {
            return 0;
        }

        var chunkStart = _position;
        var count = Math.Min(destination.Length, _plan.TotalSamples - chunkStart);
        var chunkEnd = chunkStart + count;
        var target = destination[..count];
        target.Clear();

        RenderNotesInRange(chunkStart, chunkEnd, target);
        RenderWhistlesInRange(chunkStart, chunkEnd, target);
        RenderChattersInRange(chunkStart, chunkEnd, target);

        ApplyStatefulLowPassFilter(target);
        ApplySoftCeiling(target);

        _position = chunkEnd;
        return count;
    }

    private void RenderNotesInRange(int chunkStart, int chunkEnd, Span<float> target)
    {
        var events = _plan.Events;

        // Skip notes that were fully rendered by earlier chunks.
        while (_noteCursor < events.Count &&
               events[_noteCursor].StartSample + events[_noteCursor].Length <= chunkStart)
        {
            _noteCursor++;
        }

        for (var i = _noteCursor; i < events.Count; i++)
        {
            var note = events[i];
            if (note.StartSample >= chunkEnd)
            {
                break;
            }

            var overlapStart = Math.Max(chunkStart, note.StartSample);
            var overlapEnd = Math.Min(chunkEnd, note.StartSample + note.Length);
            if (overlapEnd <= overlapStart)
            {
                continue;
            }

            var noteOffset = overlapStart - note.StartSample;
            var destOffset = overlapStart - chunkStart;
            var length = overlapEnd - overlapStart;

            RenderNoteSlice(note, noteOffset, target.Slice(destOffset, length));
        }
    }

    /// <summary>
    /// Renders <c>target.Length</c> samples of <paramref name="note"/> starting at sample
    /// <paramref name="startOffset"/> inside the note. Both the noise PRNG and the envelope are
    /// evaluated at the exact absolute sample index, so a note can be split across any number of
    /// chunks without changing the result.
    /// </summary>
    private void RenderNoteSlice(in DroidSynthesizer.NoteEvent note, int startOffset, Span<float> target)
    {
        // Fast-forward the deterministic XorShift PRNG to the requested sample offset. Noise is
        // consumed once per sample, so skipping N samples is exactly N PRNG steps.
        var noiseState = note.Seed == 0 ? 0x9E3779B9u : note.Seed;
        if (note.NoiseAmount > 0f)
        {
            for (var skip = 0; skip < startOffset; skip++)
            {
                noiseState ^= noiseState << 13;
                noiseState ^= noiseState >> 17;
                noiseState ^= noiseState << 5;
            }
        }

        var wordDuration = Math.Max(1, note.WordEndSample - note.WordStartSample) / (float)_sampleRate;
        var frequencyRatio = note.EndFrequency / note.StartFrequency;
        var lastSample = Math.Max(1, note.Length - 1);
        var triangleBase = Math.Clamp(_profile.TriangleMix, 0f, 0.60f);
        var invRate = 1f / _sampleRate;

        for (var k = 0; k < target.Length; k++)
        {
            var n = startOffset + k;
            var noteTime = n * invRate;
            var absoluteTime = (note.StartSample + n) * invRate;
            var timeInWord = (note.StartSample + n - note.WordStartSample) * invRate;
            var position = n / (float)lastSample;

            // Smoothstep makes the glide ease in and out instead of sounding like a ramp.
            var curveT = position * position * (3f - 2f * position);

            // Pitch is perceived logarithmically, so glide exponentially between frequencies.
            var frequency = note.StartFrequency * MathF.Pow(frequencyRatio, curveT);

            // Bell-shaped onset scoop (starts and ends at zero, so no click).
            if (timeInWord < 0.022f)
            {
                var scoopPhase = timeInWord / 0.022f;
                var direction = (note.Seed & 1u) == 0u ? -0.045f : 0.040f;
                frequency *= 1f + direction * MathF.Sin(MathF.PI * scoopPhase);
            }

            if (note.TrillDepth > 0f)
            {
                // Smooth 15 ms fade-in prevents an instantaneous frequency jump at onset.
                var trillFade = Math.Clamp(noteTime / 0.015f, 0f, 1f);
                frequency += note.TrillDepth * trillFade * MathF.Sin(TwoPi * note.TrillRate * absoluteTime);
            }

            var growlDepth = note.StartGrowlDepth + (note.GrowlDepth - note.StartGrowlDepth) * curveT;
            var growlWave = MathF.Sin(TwoPi * note.GrowlFrequency * absoluteTime);
            if (growlDepth > 0f)
            {
                frequency += 45f * growlDepth * growlWave;
            }

            frequency = MathF.Max(20f, frequency);

            _carrierPhase += TwoPi * frequency * invRate;
            _modulatorPhase += TwoPi * frequency * note.ModulatorRatio * invRate;
            _modulatorPhase2 += TwoPi * frequency * note.ModulatorRatio2 * invRate;

            var modulator = MathF.Sin((float)_modulatorPhase);
            var modulator2 = MathF.Sin((float)_modulatorPhase2);

            // Dynamic FM attack decay: bright consonant bite at onset, warming into a pure whistle.
            var fmBurst = 0.42f + 0.58f * MathF.Exp(-noteTime * 28f);
            var fmIndex = (note.StartFmIndex + (note.FmIndex - note.StartFmIndex) * curveT) * fmBurst;
            var fmIndex2 = (note.StartFmIndex2 + (note.FmIndex2 - note.StartFmIndex2) * curveT) * fmBurst;

            var carrier = MathF.Sin((float)_carrierPhase + fmIndex * modulator + fmIndex2 * modulator2);
            var triangleMix = triangleBase * fmBurst;
            var wave =
                (0.85f - triangleMix) * carrier +
                triangleMix * Triangle(_carrierPhase) +
                0.15f * fmBurst * MathF.Sin((float)(_carrierPhase * 3.01 + 0.55 * modulator2));

            var envelope = AdsrEnvelope(
                timeInWord,
                wordDuration,
                note.WordAttack,
                note.WordDecay,
                note.SustainLevel,
                note.WordRelease);

            var amplitude = note.StartAmplitude + (note.Amplitude - note.StartAmplitude) * curveT;
            var growl = 1f + growlDepth * growlWave;
            var sample = wave * amplitude * envelope * growl;

            if (note.NoiseAmount > 0f)
            {
                noiseState ^= noiseState << 13;
                noiseState ^= noiseState >> 17;
                noiseState ^= noiseState << 5;
                var noise = ((noiseState >> 8) * (1f / 16777216f)) * 2f - 1f;
                var noiseEnvelope = MathF.Exp(-noteTime * 110f);
                sample += noise * note.NoiseAmount * envelope * noiseEnvelope;
            }

            if (note.Saturation > 0f && note.Drive > 1f)
            {
                var shaped = MathF.Tanh(sample * note.Drive);
                sample += (shaped - sample) * Math.Clamp(note.Saturation, 0f, 0.9f);
            }

            target[k] += sample;
        }
    }

    private void RenderWhistlesInRange(int chunkStart, int chunkEnd, Span<float> target)
    {
        var whistles = _plan.Whistles;

        while (_whistleCursor < whistles.Count &&
               whistles[_whistleCursor].StartSample + whistles[_whistleCursor].Length <= chunkStart)
        {
            _whistleCursor++;
        }

        for (var i = _whistleCursor; i < whistles.Count; i++)
        {
            var whistle = whistles[i];
            if (whistle.StartSample >= chunkEnd)
            {
                break;
            }

            var overlapStart = Math.Max(chunkStart, whistle.StartSample);
            var overlapEnd = Math.Min(chunkEnd, whistle.StartSample + whistle.Length);
            if (overlapEnd <= overlapStart)
            {
                continue;
            }

            var offset = overlapStart - whistle.StartSample;
            var slice = target.Slice(overlapStart - chunkStart, overlapEnd - overlapStart);

            var duration = whistle.Length / (float)_sampleRate;
            var attack = MathF.Min(0.004f, duration * 0.22f);
            var decay = MathF.Min(0.006f, duration * 0.20f);
            var release = MathF.Min(0.018f, duration * 0.45f);
            const float sustain = 0.78f;

            var envelopeTotal = attack + decay + release;
            if (envelopeTotal > duration * 0.95f)
            {
                var scale = duration * 0.95f / envelopeTotal;
                attack *= scale;
                decay *= scale;
                release *= scale;
            }

            // Whistles are short, so re-running the deterministic oscillator/PRNG from the event
            // start keeps the state local and makes every chunk boundary invisible.
            var noiseState = whistle.Seed == 0 ? 0x9E3779B9u : whistle.Seed;
            var phase = 0d;
            var frequencyRatio = whistle.EndFrequency / whistle.StartFrequency;
            var lastSample = Math.Max(1, whistle.Length - 1);
            var invRate = 1f / _sampleRate;

            for (var n = 0; n < offset + slice.Length; n++)
            {
                var time = n * invRate;
                var position = n / (float)lastSample;
                var smooth = position * position * (3f - 2f * position);

                var frequency = whistle.StartFrequency * MathF.Pow(frequencyRatio, smooth);
                frequency += whistle.VibratoDepth * MathF.Sin(TwoPi * whistle.VibratoRate * time);
                phase += TwoPi * frequency * invRate;

                noiseState ^= noiseState << 13;
                noiseState ^= noiseState >> 17;
                noiseState ^= noiseState << 5;

                if (n < offset)
                {
                    continue;
                }

                var tone = MathF.Sin((float)phase) + MathF.Sin((float)(phase * 2.01)) * 0.18f;
                var envelope = AdsrEnvelope(time, duration, attack, decay, sustain, release);
                var sob = 1f + whistle.SobDepth * MathF.Sin(TwoPi * whistle.VibratoRate * 0.35f * time);
                var air = (((noiseState >> 8) * (1f / 16777216f)) * 2f - 1f) * 0.020f;

                slice[n - offset] += ((tone * sob) + air) * whistle.Amplitude * envelope;
            }
        }
    }

    private void RenderChattersInRange(int chunkStart, int chunkEnd, Span<float> target)
    {
        var chatters = _plan.Chatters;

        while (_chatterCursor < chatters.Count &&
               chatters[_chatterCursor].StartSample + chatters[_chatterCursor].Length <= chunkStart)
        {
            _chatterCursor++;
        }

        for (var i = _chatterCursor; i < chatters.Count; i++)
        {
            var chatter = chatters[i];
            if (chatter.StartSample >= chunkEnd)
            {
                break;
            }

            var overlapStart = Math.Max(chunkStart, chatter.StartSample);
            var overlapEnd = Math.Min(chunkEnd, chatter.StartSample + chatter.Length);
            if (overlapEnd <= overlapStart)
            {
                continue;
            }

            var offset = overlapStart - chatter.StartSample;
            var slice = target.Slice(overlapStart - chunkStart, overlapEnd - overlapStart);
            var noiseState = chatter.Seed == 0 ? 0x9E3779B9u : chatter.Seed;
            var phase = 0d;
            var lastSample = Math.Max(1, chatter.Length - 1);
            var invRate = 1f / _sampleRate;
            var isArpeggio = chatter.Kind == ChatterKind.Arpeggio;
            var stepCount = Math.Clamp(chatter.StepCount, 2, 3);

            for (var n = 0; n < offset + slice.Length; n++)
            {
                var position = n / (float)lastSample;

                noiseState ^= noiseState << 13;
                noiseState ^= noiseState >> 17;
                noiseState ^= noiseState << 5;
                var random = (noiseState >> 8) * (1f / 16777216f);

                if (isArpeggio)
                {
                    var step = Math.Min(stepCount - 1, (int)(position * stepCount));
                    var frequency = step switch
                    {
                        0 => chatter.Frequency1,
                        1 => chatter.Frequency2,
                        _ => chatter.Frequency3,
                    };
                    phase += TwoPi * frequency * invRate;

                    if (n < offset)
                    {
                        continue;
                    }

                    var envelope = MathF.Sin(MathF.PI * MathF.Pow(position, 0.65f));
                    var tone = MathF.Sin((float)phase) + 0.28f * Triangle(phase);
                    var click = (random * 2f - 1f) * MathF.Exp(-position * 24f) * 0.25f;
                    var raw = (tone + click) * envelope * chatter.Amplitude;
                    slice[n - offset] += SoftSaturate(raw, 1.35f, 0.18f);
                }
                else
                {
                    var smooth = position * position * (3f - 2f * position);
                    var frequency = chatter.StartFrequency *
                        MathF.Pow(chatter.EndFrequency / chatter.StartFrequency, smooth);
                    phase += TwoPi * frequency * invRate;

                    if (n < offset)
                    {
                        continue;
                    }

                    var envelope = MathF.Exp(-position * 9f);
                    var contact = (random * 2f - 1f) * MathF.Exp(-position * 32f);
                    var raw = (MathF.Sin((float)phase) * 0.65f + contact) * envelope * chatter.Amplitude;
                    slice[n - offset] += SoftSaturate(raw, 2.10f, 0.42f);
                }
            }
        }
    }

    /// <summary>
    /// 2-pole resonant state-variable low-pass filter with an envelope follower. The filter state
    /// lives on the stream, so the cutoff breathes continuously across chunk boundaries.
    /// </summary>
    private void ApplyStatefulLowPassFilter(Span<float> buffer)
    {
        if (buffer.IsEmpty)
        {
            return;
        }

        var baseCutoff = Math.Clamp(
            _profile.MaxFreq * _profile.FilterCutoffMultiplier,
            1100f,
            5200f);
        var maxCutoff = Math.Min(_sampleRate * 0.42f, baseCutoff * 2.6f);
        var damping = 1f / _profile.FilterResonance;
        var attackCoeff = 1f - MathF.Exp(-1f / (0.004f * _sampleRate));
        var releaseCoeff = 1f - MathF.Exp(-1f / (0.045f * _sampleRate));
        var invRate = 1f / _sampleRate;

        for (var i = 0; i < buffer.Length; i++)
        {
            var input = buffer[i];
            var absIn = MathF.Abs(input);

            _envFollower += (absIn > _envFollower ? attackCoeff : releaseCoeff) * (absIn - _envFollower);

            var dynamicCutoff = Math.Clamp(
                baseCutoff + (maxCutoff - baseCutoff) * MathF.Min(1f, _envFollower * 2.2f),
                400f,
                _sampleRate * 0.45f);

            var f = 2f * MathF.Sin(MathF.PI * dynamicCutoff * invRate);
            f = Math.Clamp(f, 0.01f, 0.85f);

            // Two-pass oversampled SVF step for stability at high frequencies.
            for (var step = 0; step < 2; step++)
            {
                var high = input - _filterLow - damping * _filterBand;
                _filterBand += (f * 0.5f) * high;
                _filterLow += (f * 0.5f) * _filterBand;
            }

            if (MathF.Abs(_filterLow) < 1e-7f && absIn == 0f)
            {
                _filterLow = 0f;
                _filterBand = 0f;
            }

            buffer[i] = _filterLow;
        }
    }

    private static void ApplySoftCeiling(Span<float> buffer)
    {
        for (var i = 0; i < buffer.Length; i++)
        {
            var sample = buffer[i];
            if (!float.IsFinite(sample))
            {
                buffer[i] = 0f;
                continue;
            }

            buffer[i] = Math.Clamp(SoftLimit(sample), -1f, 1f);
        }
    }

    private static float SoftLimit(float sample)
    {
        var magnitude = MathF.Abs(sample);
        if (magnitude <= Knee)
        {
            return sample;
        }

        var headroom = PeakCeiling - Knee;
        var limited = Knee + headroom * MathF.Tanh((magnitude - Knee) / headroom);
        return sample < 0f ? -limited : limited;
    }

    private static float SoftSaturate(float sample, float drive, float amount)
    {
        if (amount <= 0f || drive <= 1f)
        {
            return sample;
        }

        var shaped = MathF.Tanh(sample * drive);
        return sample + (shaped - sample) * Math.Clamp(amount, 0f, 0.9f);
    }

    private static float AdsrEnvelope(
        float time,
        float duration,
        float attack,
        float decay,
        float sustain,
        float release)
    {
        if (time <= 0f || time >= duration)
        {
            return 0f;
        }

        if (time < attack)
        {
            return time / attack;
        }

        var decayEnd = attack + decay;
        if (time < decayEnd)
        {
            return 1f - (1f - sustain) * ((time - attack) / decay);
        }

        var releaseStart = duration - release;
        if (time >= releaseStart)
        {
            return sustain * ((duration - time) / release);
        }

        return sustain;
    }

    private static float Triangle(double phase)
    {
        var normalized = phase / (2d * Math.PI);
        normalized -= Math.Floor(normalized);
        return (float)(4d * Math.Abs(normalized - 0.5d) - 1d);
    }
}