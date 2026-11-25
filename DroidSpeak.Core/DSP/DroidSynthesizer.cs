using System.Buffers;
using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.DSP;

/// <summary>
/// Procedural Text-to-Beep engine. Turns parsed text into a mono <see cref="float"/> PCM buffer.
/// <para>
/// The voice is built from a sine + triangle carrier with two FM modulators (one harmonic and one
/// inharmonic, for a metallic timbre), white-noise transients for sibilants, low-frequency "growl"
/// modulation for voiced consonants and a word-level ADSR envelope. A single continuous oscillator
/// spans the whole utterance: phase never resets between syllables, and syllables inside a word are
/// joined by exponential portamento (legato). Stressed or emotional syllables may add short
/// 1400вЂ“2600 Hz whistles and sobs, while shouted words and exclamations are driven into a
/// soft-clipping stage. All randomness is derived from stable word hashes, so the same input always
/// produces exactly the same audio.
/// </para>
/// <para>
/// Prosody (timing, pitch, envelope and event decisions) is produced by an
/// <see cref="IProsodyPredictor"/>; this class only flattens the resulting
/// <see cref="SyllableFrame"/> values and renders them.
/// </para>
/// </summary>
public sealed class DroidSynthesizer
{
    /// <summary>The default output sample rate (Hz).</summary>
    public const int DefaultSampleRate = 44100;

    private const float TwoPi = MathF.PI * 2f;
    private const float PeakCeiling = 0.95f;

    private readonly int _sampleRate;
    private readonly uint _seed;
    private readonly DroidVoiceProfile _profile;
    private readonly IProsodyPredictor _prosody;

    /// <summary>Creates a synthesizer.</summary>
    /// <param name="sampleRate">Output sample rate in Hz (8000вЂ“192000).</param>
    /// <param name="seed">Global seed mixed into every syllable seed. Defaults to <c>0</c>.</param>
    /// <param name="profile">Voice profile. Defaults to <see cref="DroidVoiceProfile.R2"/>.</param>
    /// <param name="predictor">
    /// Prosody model. Defaults to <see cref="RuleBasedProsodyPredictor"/>, which reproduces the
    /// classic procedural droid voice.
    /// </param>
    public DroidSynthesizer(
        int sampleRate = DefaultSampleRate,
        int seed = 0,
        DroidVoiceProfile? profile = null,
        IProsodyPredictor? predictor = null)
    {
        if (sampleRate is < 8000 or > 192000)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sampleRate),
                sampleRate,
                "Sample rate must be between 8000 and 192000 Hz.");
        }

        _sampleRate = sampleRate;
        _seed = unchecked((uint)seed);
        _profile = (profile ?? DroidVoiceProfile.R2).Normalized();
        _prosody = predictor ?? RuleBasedProsodyPredictor.Shared;
    }

    /// <summary>The output sample rate in Hz.</summary>
    public int SampleRate => _sampleRate;

    /// <summary>The default voice profile used when a call does not supply one.</summary>
    public DroidVoiceProfile Profile => _profile;

    /// <summary>Synthesizes text into a newly allocated mono buffer.</summary>
    /// <param name="text">Text to speak.</param>
    /// <param name="profile">Optional voice profile override. Defaults to the synthesizer profile.</param>
    public float[] Synthesize(string text, DroidVoiceProfile? profile = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        return Synthesize(TextParser.Parse(text), profile);
    }

    /// <summary>
    /// Creates a real-time pull stream that synthesizes <paramref name="text"/> in arbitrary chunks
    /// without allocating a full utterance buffer upfront.
    /// </summary>
    public DroidSpeechStream CreateStream(string text, DroidVoiceProfile? profile = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        return CreateStream(TextParser.Parse(text), profile);
    }

    /// <summary>
    /// Creates a real-time pull stream from already parsed <paramref name="phrases"/>.
    /// </summary>
    public DroidSpeechStream CreateStream(IReadOnlyList<Phrase> phrases, DroidVoiceProfile? profile = null)
    {
        ArgumentNullException.ThrowIfNull(phrases);
        var activeProfile = (profile ?? _profile).Normalized();
        var plan = BuildPlan(phrases, activeProfile);
        return new DroidSpeechStream(this, plan, activeProfile);
    }

    /// <summary>Synthesizes already parsed phrases into a newly allocated mono buffer.</summary>
    /// <param name="phrases">Parsed phrases.</param>
    /// <param name="profile">Optional voice profile override. Defaults to the synthesizer profile.</param>
    public float[] Synthesize(IReadOnlyList<Phrase> phrases, DroidVoiceProfile? profile = null)
    {
        ArgumentNullException.ThrowIfNull(phrases);

        var plan = BuildPlan(phrases, profile ?? _profile);
        var buffer = new float[plan.TotalSamples];
        Render(plan.Frames, buffer, profile ?? _profile);
        return buffer;
    }

    /// <summary>
    /// Returns the exact number of samples <see cref="Synthesize(string, Span{float})"/> will write
    /// for <paramref name="text"/>.
    /// </summary>
    public int GetSampleCount(string text, DroidVoiceProfile? profile = null)
    {
        ArgumentNullException.ThrowIfNull(text);
        return GetSampleCount(TextParser.Parse(text), profile);
    }

    /// <summary>Returns the exact number of samples the synthesizer will write for <paramref name="phrases"/>.</summary>
    public int GetSampleCount(IReadOnlyList<Phrase> phrases, DroidVoiceProfile? profile = null)
    {
        ArgumentNullException.ThrowIfNull(phrases);

        var count = SyllableContextBuilder.Count(phrases);
        if (count == 0)
        {
            return 0;
        }

        var contexts = ArrayPool<SyllableContext>.Shared.Rent(count);
        var frames = ArrayPool<SyllableFrame>.Shared.Rent(count);
        try
        {
            return PredictFrames(
                phrases,
                profile ?? _profile,
                contexts.AsSpan(0, count),
                frames.AsSpan(0, count));
        }
        finally
        {
            ArrayPool<SyllableContext>.Shared.Return(contexts);
            ArrayPool<SyllableFrame>.Shared.Return(frames);
        }
    }

    /// <summary>
    /// Synthesizes text directly into <paramref name="destination"/> and returns the number of
    /// samples written. If the destination is too small, the audio is truncated.
    /// </summary>
    public int Synthesize(string text, Span<float> destination)
    {
        ArgumentNullException.ThrowIfNull(text);
        return Synthesize(TextParser.Parse(text), destination);
    }

    /// <summary>
    /// Synthesizes parsed phrases directly into <paramref name="destination"/> and returns the
    /// number of samples written. If the destination is too small, the audio is truncated.
    /// </summary>
    public int Synthesize(IReadOnlyList<Phrase> phrases, Span<float> destination)
    {
        ArgumentNullException.ThrowIfNull(phrases);

        var count = SyllableContextBuilder.Count(phrases);
        if (count == 0)
        {
            return 0;
        }

        var contexts = ArrayPool<SyllableContext>.Shared.Rent(count);
        var frames = ArrayPool<SyllableFrame>.Shared.Rent(count);
        try
        {
            var frameSpan = frames.AsSpan(0, count);
            var totalSamples = PredictFrames(
                phrases,
                _profile,
                contexts.AsSpan(0, count),
                frameSpan);

            var countToWrite = Math.Min(destination.Length, totalSamples);
            if (countToWrite == 0)
            {
                return 0;
            }

            var target = destination[..countToWrite];
            target.Clear();
            Render(frameSpan, target, _profile);
            return countToWrite;
        }
        finally
        {
            ArrayPool<SyllableContext>.Shared.Return(contexts);
            ArrayPool<SyllableFrame>.Shared.Return(frames);
        }
    }

    internal SynthesisPlan BuildPlan(IReadOnlyList<Phrase> phrases) =>
        BuildPlan(phrases, _profile);

    internal SynthesisPlan BuildPlan(IReadOnlyList<Phrase> phrases, DroidVoiceProfile profile)
    {
        profile = profile.Normalized();

        var plan = new SynthesisPlan();
        var syllableCount = SyllableContextBuilder.Count(phrases);
        if (syllableCount == 0)
        {
            plan.TotalSamples = 0;
            return plan;
        }

        var contexts = new SyllableContext[syllableCount];
        var frames = new SyllableFrame[syllableCount];

        plan.TotalSamples = PredictFrames(phrases, profile, contexts, frames);
        plan.Frames = frames;
        Flatten(frames, plan);

        return plan;
    }

    /// <summary>
    /// Builds contexts, runs the predictor and returns the exact utterance length in samples.
    /// Both spans must be at least as long as the phrase syllable count.
    /// </summary>
    private int PredictFrames(
        IReadOnlyList<Phrase> phrases,
        DroidVoiceProfile profile,
        Span<SyllableContext> contexts,
        Span<SyllableFrame> frames)
    {
        SyllableContextBuilder.Build(phrases, _sampleRate, _seed, contexts);
        _prosody.Predict(contexts, frames, profile);
        return TotalSamples(frames);
    }

    /// <summary>
    /// Derives the exact output length from the predicted frames: the last syllable's end, the
    /// trailing phrase pause, and the fixed utterance tail.
    /// </summary>
    private int TotalSamples(ReadOnlySpan<SyllableFrame> frames)
    {
        if (frames.IsEmpty)
        {
            return 0;
        }

        var last = frames[^1];
        return last.StartSample
            + last.LengthSamples
            + last.PauseAfterSamples
            + Samples(0.15f);
    }

    /// <summary>
    /// Turns the predictor's per-syllable frames into the flat render commands consumed by
    /// <see cref="Render"/>. The order matches the v0.3.0 planner: gap chatter first, then the
    /// syllable note, then its whistle.
    /// </summary>
    private static void Flatten(ReadOnlySpan<SyllableFrame> frames, SynthesisPlan plan)
    {
        foreach (var frame in frames)
        {
            if (frame.Chatter.Present)
            {
                plan.Chatters.Add(new ChatterEvent
                {
                    StartSample = frame.StartSample + frame.Chatter.StartOffsetSamples,
                    Length = frame.Chatter.LengthSamples,
                    Kind = frame.Chatter.Kind,
                    Frequency1 = frame.Chatter.Frequency1,
                    Frequency2 = frame.Chatter.Frequency2,
                    Frequency3 = frame.Chatter.Frequency3,
                    StepCount = frame.Chatter.StepCount,
                    StartFrequency = frame.Chatter.StartFrequency,
                    EndFrequency = frame.Chatter.EndFrequency,
                    Amplitude = frame.Chatter.Amplitude,
                    Seed = frame.Chatter.Seed,
                });
            }

            plan.Events.Add(new NoteEvent
            {
                StartSample = frame.StartSample,
                Length = frame.LengthSamples,
                StartFrequency = frame.StartFreq,
                MidFrequency = frame.MidFreq,
                EndFrequency = frame.EndFreq,
                StartAmplitude = frame.StartGain,
                Amplitude = frame.Gain,
                StartFmIndex = frame.StartFmDepth,
                FmIndex = frame.FmDepth,
                StartFmIndex2 = frame.StartFmDepth2,
                FmIndex2 = frame.FmDepth2,
                StartGrowlDepth = frame.StartGrowlIntensity,
                GrowlDepth = frame.GrowlIntensity,
                GrowlFrequency = frame.GrowlFrequency,
                NoiseAmount = frame.NoiseMix,
                HasVoicedOnset = frame.HasVoicedOnset,
                Seed = frame.Seed,
                ModulatorRatio = frame.FmRatio,
                ModulatorRatio2 = frame.FmRatio2,
                RingModMix = frame.RingModMix,
                FormantF1 = frame.FormantF1,
                FormantF2 = frame.FormantF2,
                FormantMix = frame.FormantMix,
                AttackChirp = frame.AttackChirp,
                Drive = frame.Drive,
                Saturation = frame.Saturation,
                WordStartSample = frame.WordStartSample,
                WordEndSample = frame.WordEndSample,
                WordAttack = frame.AttackTime,
                WordDecay = frame.DecayTime,
                WordRelease = frame.ReleaseTime,
                SustainLevel = frame.SustainLevel,
                TrillDepth = frame.TrillDepth,
                TrillRate = frame.TrillRate,
            });

            if (frame.Whistle.Present)
            {
                plan.Whistles.Add(new WhistleEvent
                {
                    StartSample = frame.StartSample + frame.Whistle.StartOffsetSamples,
                    Length = frame.Whistle.LengthSamples,
                    StartFrequency = frame.Whistle.StartFrequency,
                    EndFrequency = frame.Whistle.EndFrequency,
                    Amplitude = frame.Whistle.Amplitude,
                    VibratoDepth = frame.Whistle.VibratoDepth,
                    VibratoRate = frame.Whistle.VibratoRate,
                    SobDepth = frame.Whistle.SobDepth,
                    Seed = frame.Whistle.Seed,
                });
            }
        }
    }

    private void Render(ReadOnlySpan<SyllableFrame> frames, Span<float> buffer, DroidVoiceProfile profile)
    {
        // One oscillator for the whole utterance: phase is never reset between syllables.
        var voice = default(VoiceState);

        // Three ordered passes preserve the v0.3.0 mix order: every note first, then every
        // whistle, then every chatter. Overlapping layers therefore sum in the same order and
        // produce bit-identical floating-point results.
        foreach (var frame in frames)
        {
            var available = buffer.Length - frame.StartSample;
            if (available <= 0)
            {
                continue;
            }

            var count = Math.Min(frame.LengthSamples, available);
            RenderNote(
                frame,
                buffer.Slice(frame.StartSample, count),
                ref voice,
                profile);
        }

        foreach (var frame in frames)
        {
            if (!frame.Whistle.Present)
            {
                continue;
            }

            var startSample = frame.StartSample + frame.Whistle.StartOffsetSamples;
            var available = buffer.Length - startSample;
            if (available <= 0)
            {
                continue;
            }

            var count = Math.Min(frame.Whistle.LengthSamples, available);
            RenderWhistle(frame, buffer.Slice(startSample, count));
        }

        foreach (var frame in frames)
        {
            if (!frame.Chatter.Present)
            {
                continue;
            }

            var startSample = frame.StartSample + frame.Chatter.StartOffsetSamples;
            var available = buffer.Length - startSample;
            if (available <= 0)
            {
                continue;
            }

            var count = Math.Min(frame.Chatter.LengthSamples, available);
            RenderChatter(frame, buffer.Slice(startSample, count));
        }

        ApplyDynamicLowPassFilter(buffer, profile);
        ApplyPeakCeiling(buffer);
    }

    private void RenderNote(
        in SyllableFrame frame,
        Span<float> target,
        ref VoiceState voice,
        DroidVoiceProfile profile)
    {
        var note = new NoteEvent
        {
            StartSample = frame.StartSample,
            Length = frame.LengthSamples,
            StartFrequency = frame.StartFreq,
            MidFrequency = frame.MidFreq,
            EndFrequency = frame.EndFreq,
            StartAmplitude = frame.StartGain,
            Amplitude = frame.Gain,
            StartFmIndex = frame.StartFmDepth,
            FmIndex = frame.FmDepth,
            StartFmIndex2 = frame.StartFmDepth2,
            FmIndex2 = frame.FmDepth2,
            StartGrowlDepth = frame.StartGrowlIntensity,
            GrowlDepth = frame.GrowlIntensity,
            GrowlFrequency = frame.GrowlFrequency,
            NoiseAmount = frame.NoiseMix,
            HasVoicedOnset = frame.HasVoicedOnset,
            Seed = frame.Seed,
            ModulatorRatio = frame.FmRatio,
            ModulatorRatio2 = frame.FmRatio2,
            Drive = frame.Drive,
            Saturation = frame.Saturation,
            WordStartSample = frame.WordStartSample,
            WordEndSample = frame.WordEndSample,
            WordAttack = frame.AttackTime,
            WordDecay = frame.DecayTime,
            WordRelease = frame.ReleaseTime,
            SustainLevel = frame.SustainLevel,
            TrillDepth = frame.TrillDepth,
            TrillRate = frame.TrillRate,
            RingModMix = frame.RingModMix,
            FormantF1 = frame.FormantF1,
            FormantF2 = frame.FormantF2,
            FormantMix = frame.FormantMix,
            AttackChirp = frame.AttackChirp,
        };

        NoteRenderer.Render(note, 0, target, ref voice, profile.TriangleMix, _sampleRate);
    }


    private void RenderWhistle(in SyllableFrame frame, Span<float> target)
    {
        var whistle = frame.Whistle;
        var random = new XorShift32(whistle.Seed);
        var duration = whistle.LengthSamples / (float)_sampleRate;

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

        var phase = 0d;
        var frequencyRatio = whistle.EndFrequency / whistle.StartFrequency;
        var lastSample = Math.Max(1, whistle.LengthSamples - 1);

        for (var n = 0; n < target.Length; n++)
        {
            var time = n / (float)_sampleRate;
            var position = n / (float)lastSample;
            var smooth = position * position * (3f - 2f * position);

            var frequency = whistle.StartFrequency * MathF.Pow(frequencyRatio, smooth);
            frequency += whistle.VibratoDepth * MathF.Sin(TwoPi * whistle.VibratoRate * time);
            phase += TwoPi * frequency / _sampleRate;

            var tone = MathF.Sin((float)phase);
            var overtone = MathF.Sin((float)(phase * 2.01)) * 0.18f;
            var envelope = AdsrEnvelope(time, duration, attack, decay, sustain, release);
            var sob = 1f + whistle.SobDepth * MathF.Sin(TwoPi * whistle.VibratoRate * 0.35f * time);
            var air = (random.NextFloat() * 2f - 1f) * 0.020f;

            target[n] += ((tone + overtone) * sob + air) * whistle.Amplitude * envelope;
        }
    }

    private void RenderChatter(in SyllableFrame frame, Span<float> target)
    {
        var chatter = frame.Chatter;
        var random = new XorShift32(chatter.Seed);
        var phase = 0d;
        var lastSample = Math.Max(1, chatter.LengthSamples - 1);

        if (chatter.Kind == ChatterKind.Arpeggio)
        {
            var stepCount = Math.Clamp(chatter.StepCount, 2, 3);

            for (var n = 0; n < target.Length; n++)
            {
                var position = n / (float)lastSample;
                var step = Math.Min(stepCount - 1, (int)(position * stepCount));
                var frequency = step switch
                {
                    0 => chatter.Frequency1,
                    1 => chatter.Frequency2,
                    _ => chatter.Frequency3,
                };

                phase += TwoPi * frequency / _sampleRate;

                // A bell-like attack and a quick mechanical decay.
                var envelope = MathF.Sin(MathF.PI * MathF.Pow(position, 0.65f));
                var tone = MathF.Sin((float)phase) + 0.28f * Triangle(phase);
                var click = (random.NextFloat() * 2f - 1f) * MathF.Exp(-position * 24f) * 0.25f;

                target[n] += SoftSaturate(
                    (tone + click) * envelope * chatter.Amplitude,
                    1.35f,
                    0.18f);
            }

            return;
        }

        var ratio = chatter.EndFrequency / chatter.StartFrequency;
        for (var n = 0; n < target.Length; n++)
        {
            var position = n / (float)lastSample;
            var smooth = position * position * (3f - 2f * position);
            var frequency = chatter.StartFrequency * MathF.Pow(ratio, smooth);
            phase += TwoPi * frequency / _sampleRate;

            // Relay click: an almost instant decay plus a noisy contact transient.
            var envelope = MathF.Exp(-position * 9f);
            var contact = (random.NextFloat() * 2f - 1f) * MathF.Exp(-position * 32f);
            var tone = MathF.Sin((float)phase) * 0.65f + contact;

            target[n] += SoftSaturate(
                tone * envelope * chatter.Amplitude,
                2.10f,
                0.42f);
        }
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

    /// <summary>
    /// 2-pole resonant state-variable low-pass filter with an envelope follower.
    /// Eliminates static high-frequency harmonic "rails" and emulates an analog ARP 2600 filter.
    /// </summary>
    private void ApplyDynamicLowPassFilter(Span<float> buffer, DroidVoiceProfile profile)
    {
        if (buffer.IsEmpty)
        {
            return;
        }

        // Base cutoff scales with the droid's vocal range (e.g. ~2600 Hz for R2, ~1800 Hz for Grumpy)
        var baseCutoff = Math.Clamp(
            profile.MaxFreq * profile.FilterCutoffMultiplier,
            1100f,
            5200f);
        var maxCutoff = Math.Min(_sampleRate * 0.42f, baseCutoff * 2.6f);

        // Q comes from the profile; higher values give that warm, vocal analog whistle character.
        var damping = 1f / profile.FilterResonance;

        var low = 0f;
        var band = 0f;
        var envFollower = 0f;

        var attackCoeff = 1f - MathF.Exp(-1f / (0.004f * _sampleRate));
        var releaseCoeff = 1f - MathF.Exp(-1f / (0.045f * _sampleRate));

        for (var i = 0; i < buffer.Length; i++)
        {
            var input = buffer[i];
            var absIn = MathF.Abs(input);

            // Track signal loudness envelope smoothly
            envFollower += (absIn > envFollower ? attackCoeff : releaseCoeff) * (absIn - envFollower);

            // Cutoff frequency breathes dynamically with signal energy
            var dynamicCutoff = Math.Clamp(
                baseCutoff + (maxCutoff - baseCutoff) * MathF.Min(1f, envFollower * 2.2f),
                400f,
                _sampleRate * 0.45f);

            var f = 2f * MathF.Sin(MathF.PI * dynamicCutoff / _sampleRate);
            f = Math.Clamp(f, 0.01f, 0.85f);

            // Two-pass oversampled SVF step for rock-solid stability at high frequencies
            for (var step = 0; step < 2; step++)
            {
                var high = input - low - damping * band;
                band += (f * 0.5f) * high;
                low += (f * 0.5f) * band;
            }

            if (MathF.Abs(low) < 1e-7f && absIn == 0f)
            {
                low = 0f;
                band = 0f;
            }

            buffer[i] = low;
        }
    }

    private static void ApplyPeakCeiling(Span<float> buffer)
    {
        var peak = 0f;
        foreach (var sample in buffer)
        {
            if (!float.IsFinite(sample))
            {
                continue;
            }

            var magnitude = MathF.Abs(sample);
            if (magnitude > peak)
            {
                peak = magnitude;
            }
        }

        if (peak > PeakCeiling)
        {
            var gain = PeakCeiling / peak;
            for (var i = 0; i < buffer.Length; i++)
            {
                buffer[i] *= gain;
            }
        }

        for (var i = 0; i < buffer.Length; i++)
        {
            buffer[i] = float.IsFinite(buffer[i])
                ? Math.Clamp(buffer[i], -1f, 1f)
                : 0f;
        }
    }

    private int Samples(float seconds) =>
        Math.Max(0, (int)MathF.Round(seconds * _sampleRate));

    internal sealed class SynthesisPlan
    {
        public SyllableFrame[] Frames { get; set; } = Array.Empty<SyllableFrame>();

        public List<NoteEvent> Events { get; } = new();

        public List<WhistleEvent> Whistles { get; } = new();

        public List<ChatterEvent> Chatters { get; } = new();

        public int TotalSamples { get; set; }
    }

    internal struct NoteEvent
    {
        public int StartSample;
        public int Length;
        public float StartFrequency;
        public float MidFrequency;
        public float EndFrequency;
        public float StartAmplitude;
        public float Amplitude;
        public float StartFmIndex;
        public float FmIndex;
        public float StartFmIndex2;
        public float FmIndex2;
        public float StartGrowlDepth;
        public float GrowlDepth;
        public float GrowlFrequency;
        public float NoiseAmount;
        public bool HasVoicedOnset;
        public uint Seed;
        public float ModulatorRatio;
        public float ModulatorRatio2;
        public float RingModMix;
        public float FormantF1;
        public float FormantF2;
        public float FormantMix;
        public float AttackChirp;
        public float Drive;
        public float Saturation;
        public int WordStartSample;
        public int WordEndSample;
        public float WordAttack;
        public float WordDecay;
        public float WordRelease;
        public float SustainLevel;
        public float TrillDepth;
        public float TrillRate;
    }

    internal struct ChatterEvent
    {
        public int StartSample;
        public int Length;
        public ChatterKind Kind;
        public float Frequency1;
        public float Frequency2;
        public float Frequency3;
        public int StepCount;
        public float StartFrequency;
        public float EndFrequency;
        public float Amplitude;
        public uint Seed;
    }

    internal struct WhistleEvent
    {
        public int StartSample;
        public int Length;
        public float StartFrequency;
        public float EndFrequency;
        public float Amplitude;
        public float VibratoDepth;
        public float VibratoRate;
        public float SobDepth;
        public uint Seed;
    }
}
