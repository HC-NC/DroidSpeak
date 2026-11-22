using DroidSpeak.NLP;

namespace DroidSpeak.DSP;

/// <summary>
/// Procedural Text-to-Beep engine. Turns parsed text into a mono <see cref="float"/> PCM buffer.
/// <para>
/// The voice is built from a sine + triangle carrier with two FM modulators (one harmonic and one
/// inharmonic, for a metallic timbre), white-noise transients for sibilants, low-frequency "growl"
/// modulation for voiced consonants and a word-level ADSR envelope. A single continuous oscillator
/// spans the whole utterance: phase never resets between syllables, and syllables inside a word are
/// joined by exponential portamento (legato). Stressed or emotional syllables may add short
/// 1400–2600 Hz whistles and sobs, while shouted words and exclamations are driven into a
/// soft-clipping stage. All randomness is derived from stable word hashes, so the same input always
/// produces exactly the same audio.
/// </para>
/// </summary>
public sealed class DroidSynthesizer
{
    /// <summary>The default output sample rate (Hz).</summary>
    public const int DefaultSampleRate = 44100;

    private const float TwoPi = MathF.PI * 2f;
    private const float MinFrequency = 350f;
    private const float MaxFrequency = 1600f;
    private const float WhistleMinFrequency = 1400f;
    private const float WhistleMaxFrequency = 2600f;
    private const float PeakCeiling = 0.95f;

    private readonly int _sampleRate;
    private readonly uint _seed;
    private readonly DroidVoiceProfile _profile;

    /// <summary>Creates a synthesizer.</summary>
    /// <param name="sampleRate">Output sample rate in Hz (8000–192000).</param>
    /// <param name="seed">Global seed mixed into every syllable seed. Defaults to <c>0</c>.</param>
    /// <param name="profile">Voice profile. Defaults to <see cref="DroidVoiceProfile.R2"/>.</param>
    public DroidSynthesizer(
        int sampleRate = DefaultSampleRate,
        int seed = 0,
        DroidVoiceProfile? profile = null)
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

    /// <summary>Synthesizes already parsed phrases into a newly allocated mono buffer.</summary>
    /// <param name="phrases">Parsed phrases.</param>
    /// <param name="profile">Optional voice profile override. Defaults to the synthesizer profile.</param>
    public float[] Synthesize(IReadOnlyList<Phrase> phrases, DroidVoiceProfile? profile = null)
    {
        ArgumentNullException.ThrowIfNull(phrases);

        var plan = BuildPlan(phrases, profile ?? _profile);
        var buffer = new float[plan.TotalSamples];
        Render(plan, buffer, profile ?? _profile);
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
        return BuildPlan(phrases, profile ?? _profile).TotalSamples;
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

        var plan = BuildPlan(phrases, _profile);
        var count = Math.Min(destination.Length, plan.TotalSamples);
        if (count == 0)
        {
            return 0;
        }

        var target = destination[..count];
        target.Clear();
        Render(plan, target, _profile);
        return count;
    }

    internal SynthesisPlan BuildPlan(IReadOnlyList<Phrase> phrases) =>
        BuildPlan(phrases, _profile);

    internal SynthesisPlan BuildPlan(IReadOnlyList<Phrase> phrases, DroidVoiceProfile profile)
    {
        profile = profile.Normalized();
        var plan = new SynthesisPlan();
        var legatoRanges = new List<(int Start, int End)>();
        var cursor = Samples(0.04f);
        var firstPhrase = true;

        foreach (var phrase in phrases)
        {
            if (phrase.Words.Count == 0)
            {
                continue;
            }

            if (!firstPhrase)
            {
                cursor += Samples(0.09f);
            }

            firstPhrase = false;
            var phraseEventStart = plan.Events.Count;
            var previousWordHadComma = false;
            var isExclamation = phrase.Punctuation == PhrasePunctuation.Exclamation;
            var valence = MathF.Max(0f, phrase.Emotion.Valence);
            var arousal = MathF.Max(0f, phrase.Emotion.Arousal);
            var pitchSpread = 1f + 0.45f * arousal;

            for (var wordIndex = 0; wordIndex < phrase.Words.Count; wordIndex++)
            {
                var word = phrase.Words[wordIndex];

                if (wordIndex > 0)
                {
                    // Word gaps are jittered too, so the rhythm never feels metronomic.
                    var wordGapRandom = new XorShift32(MixSeed(_seed, word.Hash, 0x51ED270Bu));
                    var wordGap = previousWordHadComma
                        ? 0.060f + wordGapRandom.NextFloat() * 0.045f
                        : 0.016f + wordGapRandom.NextFloat() * 0.028f;
                    var gapSamples = Samples(wordGap);
                    var gapStart = cursor;
                    cursor += gapSamples;
                    TryAddChatter(plan, phrase, word, gapStart, gapSamples, profile);
                }

                previousWordHadComma = word.IsFollowedByComma;
                var wordLengthFactor = Math.Clamp(
                    1f + 0.05f * (word.Text.Length - 4),
                    0.82f,
                    1.35f);
                var wordPitch = (word.Hash % 161u) - 80f;

                // All syllables of a word (or of a hyphen-separated part) are one legato
                // segment: no silence between notes, one attack and one release.
                var segmentStartSample = cursor;
                var segmentNoteStart = plan.Events.Count;
                var segmentGrowlFrequency = 26f + (word.Hash & 0x0Fu);

                for (var syllableIndex = 0; syllableIndex < word.Syllables.Count; syllableIndex++)
                {
                    var syllable = word.Syllables[syllableIndex];

                    // A hyphen stays a real pause, but everything inside each part is legato.
                    if (syllableIndex > 0 && syllable.Text.Contains('-'))
                    {
                        FinishLegatoSegment(
                            plan,
                            legatoRanges,
                            segmentNoteStart,
                            segmentStartSample,
                            cursor,
                            segmentGrowlFrequency);

                        var hyphenRandom = new XorShift32(MixSeed(_seed, word.Hash, (uint)(0x900 + syllableIndex)));
                        var hyphenGap = Samples(0.018f + hyphenRandom.NextFloat() * 0.014f);
                        var hyphenGapStart = cursor;
                        cursor += hyphenGap;
                        TryAddChatter(plan, phrase, word, hyphenGapStart, hyphenGap, profile);

                        segmentStartSample = cursor;
                        segmentNoteStart = plan.Events.Count;
                        segmentGrowlFrequency = 26f + ((word.Hash >> 4) & 0x0Fu);
                    }

                    var syllableSeed = MixSeed(_seed, word.Hash, (uint)syllableIndex);
                    var random = new XorShift32(syllableSeed);
                    var jitter = 0.90f + random.NextFloat() * 0.25f;

                    var duration = 0.105f * phrase.TempoScale * wordLengthFactor * jitter;
                    duration /= profile.SpeedMultiplier;
                    duration *= syllable.IsStressed ? 1.16f : 0.93f;
                    if (word.IsShout)
                    {
                        duration *= 0.92f;
                    }

                    if (word.Syllables.Count > 1)
                    {
                        // Multi-syllable words chatter a little faster, but without gaps.
                        duration *= 0.94f + random.NextFloat() * 0.04f;
                    }

                    duration = Math.Clamp(duration, 0.045f, 0.50f);
                    var length = Math.Max(1, Samples(duration));

                    var baseFrequency =
                        640f +
                        70f * phrase.Emotion.Valence +
                        150f * phrase.Emotion.Arousal +
                        wordPitch +
                        (jitter - 1f) * 120f * pitchSpread;

                    if (word.IsShout)
                    {
                        baseFrequency += 130f;
                    }

                    if (syllable.IsStressed)
                    {
                        baseFrequency *= 1.05f;
                    }

                    baseFrequency = MapFrequency(baseFrequency, profile);

                    var amplitude = 0.36f + 0.16f * phrase.Emotion.Arousal + (word.IsShout ? 0.10f : 0f);
                    amplitude *= syllable.IsStressed ? 1.08f : 0.95f;
                    amplitude = Math.Clamp(amplitude, 0.18f, 0.62f);

                    var fmIndex =
                        1.35f +
                        1.15f * arousal +
                        (word.IsShout ? 1.85f : 0f) +
                        (syllable.HasVoicedOnset ? 0.55f : 0f) +
                        (isExclamation ? 0.50f : 0f);
                    fmIndex *= profile.FmIntensity;
                    fmIndex = Math.Clamp(fmIndex, 0.20f, 6.50f);
                    var fmIndex2 = fmIndex * (0.35f + 0.12f * jitter);

                    var noise = syllable.HasSibilantOnset
                        ? 0.09f + 0.05f * arousal
                        : 0f;
                    if (word.IsShout)
                    {
                        noise *= 1.25f;
                    }

                    var drive = word.IsShout
                        ? 2.00f + 0.90f * arousal
                        : 1f + 0.85f * arousal + (isExclamation ? 0.55f : 0f);
                    var saturation = word.IsShout
                        ? 0.50f + 0.15f * arousal
                        : 0.10f + 0.28f * arousal + (isExclamation ? 0.16f : 0f);
                    saturation = Math.Clamp(saturation, 0f, 0.72f);

                    var trillDepth = 0f;
                    var trillRate = 0f;
                    var chatter = profile.ChatterProbability;
                    if (chatter > 0f)
                    {
                        var energy =
                            0.55f * valence +
                            0.90f * arousal +
                            (syllable.IsStressed ? 0.22f : 0f) +
                            (word.IsShout ? 0.32f : 0f) +
                            (isExclamation ? 0.22f : 0f);
                        var trillChance = chatter * Math.Clamp(
                            0.06f + 0.34f * energy + (duration >= 0.10f ? 0.08f : 0f),
                            0f,
                            0.75f);

                        if (random.NextFloat() < trillChance)
                        {
                            // The signature "tr-r-rill!": a fast 30 Hz frequency wobble.
                            trillDepth = 16f + 34f * Math.Clamp(energy, 0f, 1f) + random.NextFloat() * 10f;
                            trillRate = 27f + random.NextFloat() * 8f;
                        }
                    }

                    var growlDepth = syllable.HasVoicedOnset ? 0.20f : 0f;

                    plan.Events.Add(new NoteEvent
                    {
                        StartSample = cursor,
                        Length = length,
                        StartFrequency = ClampFrequency(baseFrequency * 0.97f, profile),
                        EndFrequency = baseFrequency,
                        StartAmplitude = amplitude,
                        Amplitude = amplitude,
                        StartFmIndex = fmIndex,
                        FmIndex = fmIndex,
                        StartFmIndex2 = fmIndex2,
                        FmIndex2 = fmIndex2,
                        StartGrowlDepth = growlDepth,
                        GrowlDepth = growlDepth,
                        NoiseAmount = noise,
                        HasVoicedOnset = syllable.HasVoicedOnset,
                        Seed = syllableSeed,
                        ModulatorRatio = 2f + (jitter - 1f) * 0.10f,
                        ModulatorRatio2 = 3.07f + (jitter - 1f) * 0.35f,
                        Drive = drive,
                        Saturation = saturation,
                        TrillDepth = trillDepth,
                        TrillRate = trillRate,
                    });

                    TryAddWhistle(plan, phrase, word, syllable, syllableIndex, cursor, length, isExclamation, profile);

                    cursor += length;
                }

                FinishLegatoSegment(
                    plan,
                    legatoRanges,
                    segmentNoteStart,
                    segmentStartSample,
                    cursor,
                    segmentGrowlFrequency);
            }

            ApplyContour(plan.Events, phraseEventStart, phrase, profile);
            cursor += Samples(phrase.Punctuation == PhrasePunctuation.Ellipsis ? 0.18f : 0.12f);
        }

        foreach (var (start, end) in legatoRanges)
        {
            LinkLegatoSegment(plan.Events, start, end);
        }

        plan.TotalSamples = plan.Events.Count == 0 ? 0 : cursor + Samples(0.15f);
        return plan;
    }

    private void FinishLegatoSegment(
        SynthesisPlan plan,
        List<(int Start, int End)> ranges,
        int noteStart,
        int sampleStart,
        int sampleEnd,
        float growlFrequency)
    {
        if (plan.Events.Count <= noteStart)
        {
            return;
        }

        ranges.Add((noteStart, plan.Events.Count));
        ApplyWordEnvelope(plan.Events, noteStart, plan.Events.Count, sampleStart, sampleEnd, growlFrequency);
    }

    private void ApplyWordEnvelope(
        List<NoteEvent> events,
        int startIndex,
        int endIndex,
        int wordStartSample,
        int wordEndSample,
        float growlFrequency)
    {
        var duration = Math.Max(1, wordEndSample - wordStartSample) / (float)_sampleRate;

        // Attack only on the first syllable, release only on the last one.
        var attack = Math.Clamp(duration * 0.14f, 0.009f, 0.016f);
        var decay = Math.Clamp(duration * 0.20f, 0.010f, 0.030f);
        var release = Math.Clamp(duration * 0.22f, 0.010f, 0.020f);
        const float sustain = 0.74f;

        var envelopeTotal = attack + decay + release;
        if (envelopeTotal > duration * 0.90f)
        {
            var scale = duration * 0.90f / envelopeTotal;
            attack *= scale;
            decay *= scale;
            release *= scale;
        }

        for (var i = startIndex; i < endIndex; i++)
        {
            var note = events[i];
            note.WordStartSample = wordStartSample;
            note.WordEndSample = wordEndSample;
            note.WordAttack = attack;
            note.WordDecay = decay;
            note.WordRelease = release;
            note.SustainLevel = sustain;
            note.GrowlFrequency = growlFrequency;
            events[i] = note;
        }
    }

    private static void LinkLegatoSegment(List<NoteEvent> events, int startIndex, int endIndex)
    {
        for (var i = startIndex + 1; i < endIndex; i++)
        {
            var previous = events[i - 1];
            var note = events[i];

            // The next syllable starts exactly where the previous one ended: frequency,
            // amplitude and FM depth are all continuous across the joint.
            note.StartFrequency = previous.EndFrequency;
            note.StartAmplitude = previous.Amplitude;
            note.StartFmIndex = previous.FmIndex;
            note.StartFmIndex2 = previous.FmIndex2;
            note.StartGrowlDepth = previous.GrowlDepth;
            events[i] = note;
        }
    }

    /// <summary>
    /// Adds a short high whistle or sob to stressed / emotional syllables. The probability and
    /// intensity grow with positive valence, arousal, shouting and exclamation marks.
    /// </summary>
    private void TryAddWhistle(
        SynthesisPlan plan,
        Phrase phrase,
        Word word,
        Syllable syllable,
        int syllableIndex,
        int noteStart,
        int noteLength,
        bool isExclamation,
        DroidVoiceProfile profile)
    {
        var valence = MathF.Max(0f, phrase.Emotion.Valence);
        var arousal = MathF.Max(0f, phrase.Emotion.Arousal);
        var energy =
            0.55f * valence +
            0.90f * arousal +
            (syllable.IsStressed ? 0.22f : 0f) +
            (word.IsShout ? 0.32f : 0f) +
            (isExclamation ? 0.22f : 0f);

        if (!syllable.IsStressed && energy < 0.48f)
        {
            return;
        }

        var chance = Math.Clamp(0.05f + 0.50f * energy, 0.05f, 0.78f);
        var random = new XorShift32(MixSeed(_seed, word.Hash, (uint)(0x500 + syllableIndex)));
        if (random.NextFloat() >= chance)
        {
            return;
        }

        var duration = 0.032f + random.NextFloat() * 0.042f;
        duration /= profile.SpeedMultiplier;
        if (syllable.IsStressed)
        {
            duration *= 1.12f;
        }

        // Keep whistles inside the profile's high band while staying below Nyquist.
        var whistleScale = Math.Clamp(profile.MaxFreq / 1600f, 0.55f, 1.60f);
        var whistleMin = MathF.Max(600f, WhistleMinFrequency * whistleScale);
        var whistleMax = MathF.Min(
            WhistleMaxFrequency * whistleScale,
            _sampleRate * 0.45f);
        if (whistleMax <= whistleMin)
        {
            return;
        }

        var length = Math.Max(1, Samples(duration));
        var rising = (random.NextUInt() & 1u) == 0u;
        var low = (1440f + random.NextFloat() * 420f) * whistleScale;
        var high = (1980f + random.NextFloat() * 560f) * whistleScale;
        var startFrequency = rising ? low : high;
        var endFrequency = rising ? high : low;

        plan.Whistles.Add(new WhistleEvent
        {
            StartSample = noteStart + (int)(noteLength * (0.62f + 0.18f * random.NextFloat())),
            Length = length,
            StartFrequency = Math.Clamp(startFrequency, whistleMin, whistleMax),
            EndFrequency = Math.Clamp(endFrequency, whistleMin, whistleMax),
            Amplitude = 0.09f + 0.11f * Math.Clamp(energy, 0f, 1.1f),
            VibratoDepth = rising
                ? 12f + random.NextFloat() * 18f
                : 28f + random.NextFloat() * 34f,
            VibratoRate = rising
                ? 18f + random.NextFloat() * 12f
                : 7f + random.NextFloat() * 7f,
            SobDepth = rising ? 0.06f : 0.18f,
            Seed = MixSeed(_seed, word.Hash, (uint)(0x700 + syllableIndex)),
        });
    }

    private void TryAddChatter(
        SynthesisPlan plan,
        Phrase phrase,
        Word word,
        int gapStart,
        int gapLength,
        DroidVoiceProfile profile)
    {
        if (gapLength <= 0 || profile.ChatterProbability <= 0f)
        {
            return;
        }

        var arousal = MathF.Max(0f, phrase.Emotion.Arousal);
        var fast = arousal > 0.18f || phrase.SyllableCount >= 12 || word.IsShout;
        if (!fast)
        {
            return;
        }

        var random = new XorShift32(MixSeed(_seed, word.Hash, 0xCA77E4u));
        var chance = profile.ChatterProbability * (0.30f + 0.60f * arousal);
        if (random.NextFloat() >= chance)
        {
            return;
        }

        var maxLength = Math.Min(gapLength, Samples(0.025f));
        var minLength = Math.Min(maxLength, Samples(0.015f));
        if (maxLength < minLength || maxLength <= 0)
        {
            return;
        }

        var length = minLength + (int)(random.NextUInt() % (uint)Math.Max(1, maxLength - minLength + 1));

        if ((random.NextUInt() & 1u) == 0u)
        {
            // A tiny stepped arpeggio, like a relay bank cycling.
            var baseFrequency = 700f + random.NextFloat() * 900f;
            plan.Chatters.Add(new ChatterEvent
            {
                StartSample = gapStart,
                Length = length,
                Kind = ChatterKind.Arpeggio,
                Frequency1 = baseFrequency,
                Frequency2 = baseFrequency * (1.20f + random.NextFloat() * 0.25f),
                Frequency3 = baseFrequency * (1.55f + random.NextFloat() * 0.35f),
                StepCount = 2 + (int)(random.NextUInt() % 2u),
                Amplitude = 0.09f + 0.06f * arousal,
                Seed = MixSeed(_seed, word.Hash, 0xC0DEu),
            });
        }
        else
        {
            // A mechanical relay click with a sharp downward frequency drop.
            plan.Chatters.Add(new ChatterEvent
            {
                StartSample = gapStart,
                Length = length,
                Kind = ChatterKind.RelayClick,
                StartFrequency = 1800f + random.NextFloat() * 1200f,
                EndFrequency = 180f + random.NextFloat() * 220f,
                Amplitude = 0.10f + 0.08f * arousal,
                Seed = MixSeed(_seed, word.Hash, 0xC11Cu),
            });
        }
    }

    private void ApplyContour(
        List<NoteEvent> events,
        int phraseStart,
        Phrase phrase,
        DroidVoiceProfile profile)
    {
        if (events.Count <= phraseStart)
        {
            return;
        }

        if (phrase.Punctuation == PhrasePunctuation.Question)
        {
            var from = Math.Max(phraseStart, events.Count - 2);
            for (var i = from; i < events.Count; i++)
            {
                var note = events[i];
                var isLast = i == events.Count - 1;
                note.StartFrequency = ClampFrequency(
                    note.StartFrequency * (isLast ? 1.10f : 1.00f),
                    profile);
                note.EndFrequency = ClampFrequency(
                    note.EndFrequency * (isLast ? 1.42f : 1.18f),
                    profile);

                if (note.EndFrequency <= note.StartFrequency)
                {
                    note.EndFrequency = ClampFrequency(note.StartFrequency * 1.12f, profile);
                }

                events[i] = note;
            }
        }
        else if (phrase.Punctuation == PhrasePunctuation.Ellipsis)
        {
            var from = Math.Max(phraseStart, events.Count - 2);
            for (var i = from; i < events.Count; i++)
            {
                var note = events[i];
                var isLast = i == events.Count - 1;
                note.StartFrequency = ClampFrequency(
                    note.StartFrequency * (isLast ? 0.90f : 0.97f),
                    profile);
                note.EndFrequency = ClampFrequency(
                    note.EndFrequency * (isLast ? 0.68f : 0.82f),
                    profile);
                events[i] = note;
            }
        }
        else
        {
            var last = events[^1];
            last.EndFrequency = ClampFrequency(last.EndFrequency * 0.92f, profile);
            events[^1] = last;
        }
    }

    private void Render(SynthesisPlan plan, Span<float> buffer, DroidVoiceProfile profile)
    {
        // One oscillator for the whole utterance: phase is never reset between syllables.
        var carrierPhase = 0d;
        var modulatorPhase = 0d;
        var modulatorPhase2 = 0d;

        foreach (var note in plan.Events)
        {
            var available = buffer.Length - note.StartSample;
            if (available <= 0)
            {
                continue;
            }

            var count = Math.Min(note.Length, available);
            RenderNote(
                note,
                buffer.Slice(note.StartSample, count),
                ref carrierPhase,
                ref modulatorPhase,
                ref modulatorPhase2,
                profile);
        }

        foreach (var whistle in plan.Whistles)
        {
            var available = buffer.Length - whistle.StartSample;
            if (available <= 0)
            {
                continue;
            }

            var count = Math.Min(whistle.Length, available);
            RenderWhistle(whistle, buffer.Slice(whistle.StartSample, count));
        }

        foreach (var chatter in plan.Chatters)
        {
            var available = buffer.Length - chatter.StartSample;
            if (available <= 0)
            {
                continue;
            }

            var count = Math.Min(chatter.Length, available);
            RenderChatter(chatter, buffer.Slice(chatter.StartSample, count));
        }

        ApplyDynamicLowPassFilter(buffer, profile);
        ApplyPeakCeiling(buffer);
    }

    private void RenderNote(
        in NoteEvent note,
        Span<float> target,
        ref double carrierPhase,
        ref double modulatorPhase,
        ref double modulatorPhase2,
        DroidVoiceProfile profile)
    {
        var random = new XorShift32(note.Seed);
        var wordDuration = Math.Max(1, note.WordEndSample - note.WordStartSample) / (float)_sampleRate;
        var frequencyRatio = note.EndFrequency / note.StartFrequency;
        var lastSample = Math.Max(1, note.Length - 1);

        for (var n = 0; n < target.Length; n++)
        {
            var noteTime = n / (float)_sampleRate;
            var absoluteTime = (note.StartSample + n) / (float)_sampleRate;
            var timeInWord = (note.StartSample + n - note.WordStartSample) / (float)_sampleRate;
            var position = n / (float)lastSample;

            // Smoothstep makes the glide ease in and out instead of sounding like a ramp.
            var curveT = position * position * (3f - 2f * position);

            // Pitch is perceived logarithmically, so glide exponentially between frequencies.
            var frequency = note.StartFrequency * MathF.Pow(frequencyRatio, curveT);

            // 1. Smooth bell-shaped whistle transient (starts and ends at 0, zero phase/freq click)
            if (timeInWord < 0.022f)
            {
                var scoopPhase = timeInWord / 0.022f;
                var direction = (note.Seed & 1u) == 0u ? -0.045f : 0.040f;
                frequency *= 1f + direction * MathF.Sin(MathF.PI * scoopPhase);
            }


            if (note.TrillDepth > 0f)
            {
                // Smooth 15ms fade-in prevents an instantaneous frequency jump (click) at syllable onset
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

            carrierPhase += TwoPi * frequency / _sampleRate;
            modulatorPhase += TwoPi * frequency * note.ModulatorRatio / _sampleRate;
            modulatorPhase2 += TwoPi * frequency * note.ModulatorRatio2 / _sampleRate;

            var modulator = MathF.Sin((float)modulatorPhase);
            var modulator2 = MathF.Sin((float)modulatorPhase2);

            // 2. Dynamic FM attack decay: bright consonant bite at onset, warming into a pure whistle
            var fmBurst = 0.42f + 0.58f * MathF.Exp(-noteTime * 28f);
            var fmIndex = (note.StartFmIndex + (note.FmIndex - note.StartFmIndex) * curveT) * fmBurst;
            var fmIndex2 = (note.StartFmIndex2 + (note.FmIndex2 - note.StartFmIndex2) * curveT) * fmBurst;

            var carrier = MathF.Sin((float)carrierPhase + fmIndex * modulator + fmIndex2 * modulator2);
            var triangleMix = Math.Clamp(profile.TriangleMix, 0f, 0.60f) * fmBurst;
            var wave =
                (0.85f - triangleMix) * carrier +
                triangleMix * Triangle(carrierPhase) +
                0.15f * fmBurst * MathF.Sin((float)(carrierPhase * 3.01 + 0.55 * modulator2));

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
                var noise = random.NextFloat() * 2f - 1f;
                var noiseEnvelope = MathF.Exp(-noteTime * 110f);
                sample += noise * note.NoiseAmount * envelope * noiseEnvelope;
            }

            target[n] += SoftSaturate(sample, note.Drive, note.Saturation);
        }
    }

    private void RenderWhistle(in WhistleEvent whistle, Span<float> target)
    {
        var random = new XorShift32(whistle.Seed);
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

        var phase = 0d;
        var frequencyRatio = whistle.EndFrequency / whistle.StartFrequency;
        var lastSample = Math.Max(1, whistle.Length - 1);

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

    private void RenderChatter(in ChatterEvent chatter, Span<float> target)
    {
        var random = new XorShift32(chatter.Seed);
        var phase = 0d;
        var lastSample = Math.Max(1, chatter.Length - 1);

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
        var baseCutoff = Math.Clamp(profile.MaxFreq * 1.65f, 1100f, 5200f);
        var maxCutoff = Math.Min(_sampleRate * 0.42f, baseCutoff * 2.6f);

        // Resonance Q ≈ 1.45 gives that warm, vocal analog whistle character
        const float damping = 1f / 1.45f;

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

    private static float ClampFrequency(float frequency, DroidVoiceProfile profile) =>
        Math.Clamp(frequency, profile.MinFreq, profile.MaxFreq);

    /// <summary>
    /// Maps a frequency expressed in the classic R2 range into the profile's own range.
    /// The mapping is linear, so relative pitch relationships are preserved.
    /// </summary>
    private static float MapFrequency(float frequency, DroidVoiceProfile profile)
    {
        var normalized = Math.Clamp(
            (frequency - MinFrequency) / (MaxFrequency - MinFrequency),
            0f,
            1f);

        return Math.Clamp(
            profile.MinFreq + normalized * (profile.MaxFreq - profile.MinFreq),
            profile.MinFreq,
            profile.MaxFreq);
    }

    private int Samples(float seconds) =>
        Math.Max(0, (int)MathF.Round(seconds * _sampleRate));

    private static uint MixSeed(uint globalSeed, uint wordHash, uint syllableIndex)
    {
        var value = globalSeed ^ (wordHash * 0x9E3779B1u) ^ ((syllableIndex + 1u) * 0x85EBCA77u);
        value ^= value >> 16;
        value *= 0x7FEB352Du;
        value ^= value >> 15;
        value *= 0x846CA68Bu;
        value ^= value >> 16;
        return value;
    }

    internal sealed class SynthesisPlan
    {
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

    internal enum ChatterKind
    {
        Arpeggio,
        RelayClick,
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

    private struct XorShift32
    {
        private uint _state;

        public XorShift32(uint seed)
        {
            _state = seed == 0 ? 0x9E3779B9u : seed;
        }

        public uint NextUInt()
        {
            var value = _state;
            value ^= value << 13;
            value ^= value >> 17;
            value ^= value << 5;
            _state = value;
            return value;
        }

        public float NextFloat() => (NextUInt() >> 8) * (1f / 16777216f);
    }
}
