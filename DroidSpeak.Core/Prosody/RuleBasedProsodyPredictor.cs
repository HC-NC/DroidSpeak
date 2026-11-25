using DroidSpeak.DSP;
using DroidSpeak.NLP;

namespace DroidSpeak.Prosody;

/// <summary>
/// The v0.3.0 procedural prosody model, expressed as an <see cref="IProsodyPredictor"/>.
/// <para>
/// It reproduces the original timing, contour, envelope, trill, whistle and chatter rules exactly,
/// including the order in which deterministic random values are drawn, so the rendered PCM is
/// bit-identical to the pre-v0.4.0 engine for the same text, seed and profile.
/// </para>
/// </summary>
public sealed class RuleBasedProsodyPredictor : IProsodyPredictor
{
    private const float MinFrequency = 350f;
    private const float MaxFrequency = 1600f;
    private const float WhistleMinFrequency = 1400f;
    private const float WhistleMaxFrequency = 2600f;

    /// <inheritdoc />
    public void Predict(
        ReadOnlySpan<SyllableContext> contexts,
        Span<SyllableFrame> frames,
        DroidVoiceProfile profile)
    {
        if (contexts.IsEmpty)
        {
            return;
        }

        if (frames.Length < contexts.Length)
        {
            throw new ArgumentException(
                "The frame buffer must be at least as long as the context buffer.",
                nameof(frames));
        }

        profile = profile.Normalized();

        var sampleRate = contexts[0].SampleRate;
        var seed = contexts[0].GlobalSeed;

        var frameCount = 0;
        var cursor = 0;
        var previousPhraseIndex = -1;
        var phraseStartFrame = 0;
        var previousWordIndex = -1;
        var segmentStartFrame = 0;
        var segmentStartSample = 0;
        var segmentGrowlFrequency = 0f;
        var pitchSpread = 1f;
        var pauseBeforeSamples = 0;
        var pendingChatter = default(ChatterSpec);
        var pendingChatterStart = 0;

        for (var i = 0; i < contexts.Length; i++)
        {
            var context = contexts[i];

            if (context.PhraseIndex != previousPhraseIndex)
            {
                if (previousPhraseIndex >= 0)
                {
                    ApplyWordEnvelope(
                        frames,
                        segmentStartFrame,
                        frameCount,
                        segmentStartSample,
                        cursor,
                        segmentGrowlFrequency,
                        sampleRate,
                        profile);

                    var previousPunctuation = contexts[i - 1].Punctuation;
                    ApplyContour(frames, phraseStartFrame, frameCount, previousPunctuation, profile);

                    var phrasePause = Samples(
                        previousPunctuation == PhrasePunctuation.Ellipsis ? 0.18f : 0.12f,
                        sampleRate);
                    cursor += phrasePause;
                    frames[frameCount - 1].PauseAfterSamples = phrasePause;
                    frames[frameCount - 1].PauseAfterSeconds = phrasePause / (float)sampleRate;
                }

                cursor += Samples(context.IsFirstSpokenPhrase ? 0.04f : 0.09f, sampleRate);
                phraseStartFrame = frameCount;
                previousWordIndex = -1;
                previousPhraseIndex = context.PhraseIndex;
                pitchSpread = 1f + 0.45f * MathF.Max(0f, context.EmotionArousal);
                pendingChatter = default;
            }

            var wordLengthFactor = Math.Clamp(
                1f + 0.05f * (context.WordTextLength - 4),
                0.82f,
                1.35f);
            var wordPitch = (context.WordHash % 161u) - 80f;

            pauseBeforeSamples = 0;

            if (context.WordIndexInPhrase != previousWordIndex)
            {
                if (previousWordIndex >= 0)
                {
                    ApplyWordEnvelope(
                        frames,
                        segmentStartFrame,
                        frameCount,
                        segmentStartSample,
                        cursor,
                        segmentGrowlFrequency,
                        sampleRate,
                        profile);
                }

                if (context.WordIndexInPhrase > 0)
                {
                    // Word gaps are jittered too, so the rhythm never feels metronomic.
                    var wordGapRandom = new XorShift32(MixSeed(seed, context.WordHash, 0x51ED270Bu));
                    var wordGap = context.PreviousWordHadComma
                        ? 0.060f + wordGapRandom.NextFloat() * 0.045f
                        : 0.016f + wordGapRandom.NextFloat() * 0.028f;
                    var gapSamples = Samples(wordGap, sampleRate);
                    pendingChatterStart = cursor;
                    pendingChatter = BuildChatter(context, gapSamples, sampleRate, seed, profile);
                    cursor += gapSamples;
                    pauseBeforeSamples = gapSamples;
                }

                segmentStartFrame = frameCount;
                segmentStartSample = cursor;
                segmentGrowlFrequency = 26f + (context.WordHash & 0x0Fu);
                previousWordIndex = context.WordIndexInPhrase;
            }
            else if (context.SyllableIndex > 0 && context.IsHyphenBoundary)
            {
                ApplyWordEnvelope(
                    frames,
                    segmentStartFrame,
                    frameCount,
                    segmentStartSample,
                    cursor,
                    segmentGrowlFrequency,
                    sampleRate,
                    profile);

                var hyphenRandom = new XorShift32(
                    MixSeed(seed, context.WordHash, (uint)(0x900 + context.SyllableIndex)));
                var hyphenGap = Samples(0.018f + hyphenRandom.NextFloat() * 0.014f, sampleRate);
                pendingChatterStart = cursor;
                pendingChatter = BuildChatter(context, hyphenGap, sampleRate, seed, profile);
                cursor += hyphenGap;
                pauseBeforeSamples = hyphenGap;

                segmentStartFrame = frameCount;
                segmentStartSample = cursor;
                segmentGrowlFrequency = 26f + ((context.WordHash >> 4) & 0x0Fu);
            }

            var syllableSeed = MixSeed(seed, context.WordHash, (uint)context.SyllableIndex);
            var random = new XorShift32(syllableSeed);
            var jitter = 0.90f + random.NextFloat() * 0.25f;

            var duration = 0.105f * context.TempoScale * wordLengthFactor * jitter;
            duration /= profile.SpeedMultiplier;
            duration *= context.IsStressed ? 1.16f : 0.93f;
            if (context.IsShout)
            {
                duration *= 0.92f;
            }

            if (context.WordSyllableCount > 1)
            {
                // Multi-syllable words chatter a little faster, but without gaps.
                duration *= 0.94f + random.NextFloat() * 0.04f;
            }

            duration = Math.Clamp(duration, 0.045f, 0.50f);
            var length = Math.Max(1, Samples(duration, sampleRate));

            var valence = MathF.Max(0f, context.EmotionValence);
            var arousal = MathF.Max(0f, context.EmotionArousal);
            var isExclamation = context.Punctuation == PhrasePunctuation.Exclamation;

            var baseFrequency =
                640f +
                70f * context.EmotionValence +
                150f * context.EmotionArousal +
                wordPitch +
                (jitter - 1f) * 120f * pitchSpread;

            if (context.IsShout)
            {
                baseFrequency += 130f;
            }

            if (context.IsStressed)
            {
                baseFrequency *= 1.05f;
            }

            baseFrequency = MapFrequency(baseFrequency, profile);

            var amplitude = 0.36f + 0.16f * context.EmotionArousal + (context.IsShout ? 0.10f : 0f);
            amplitude *= context.IsStressed ? 1.08f : 0.95f;
            amplitude = Math.Clamp(amplitude, 0.18f, 0.62f);

            var fmIndex =
                1.35f +
                1.15f * arousal +
                (context.IsShout ? 1.85f : 0f) +
                (context.HasVoicedOnset ? 0.55f : 0f) +
                (isExclamation ? 0.50f : 0f);
            fmIndex *= profile.FmIntensity;
            fmIndex = Math.Clamp(fmIndex, 0.20f, 6.50f);
            var fmIndex2 = fmIndex * (0.35f + 0.12f * jitter);

            var noise = context.HasSibilantOnset
                ? 0.09f + 0.05f * arousal
                : 0f;
            if (context.IsShout)
            {
                noise *= 1.25f;
            }

            var drive = context.IsShout
                ? 2.00f + 0.90f * arousal
                : 1f + 0.85f * arousal + (isExclamation ? 0.55f : 0f);
            var saturation = context.IsShout
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
                    (context.IsStressed ? 0.22f : 0f) +
                    (context.IsShout ? 0.32f : 0f) +
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

            var growlDepth = context.HasVoicedOnset ? 0.20f : 0f;

            var frame = new SyllableFrame
            {
                StartSample = cursor,
                LengthSamples = length,
                DurationSeconds = length / (float)sampleRate,
                PauseBeforeSamples = pauseBeforeSamples,
                PauseBeforeSeconds = pauseBeforeSamples / (float)sampleRate,
                StartFreq = ClampFrequency(baseFrequency * 0.97f, profile),
                EndFreq = baseFrequency,
                StartGain = amplitude,
                Gain = amplitude,
                StartFmDepth = fmIndex,
                FmDepth = fmIndex,
                StartFmDepth2 = fmIndex2,
                FmDepth2 = fmIndex2,
                FmRatio = 2f + (jitter - 1f) * 0.10f,
                FmRatio2 = 3.07f + (jitter - 1f) * 0.35f,
                StartGrowlIntensity = growlDepth,
                GrowlIntensity = growlDepth,
                NoiseMix = noise,
                HasVoicedOnset = context.HasVoicedOnset,
                Seed = syllableSeed,
                Drive = drive,
                Saturation = saturation,
                TrillDepth = trillDepth,
                TrillRate = trillRate,
                StartsLegatoSegment = frameCount == segmentStartFrame,
            };

            if (pendingChatter.Present)
            {
                pendingChatter.StartOffsetSamples = pendingChatterStart - cursor;
                frame.Chatter = pendingChatter;
                pendingChatter = default;
            }

            frame.Whistle = BuildWhistle(context, length, isExclamation, sampleRate, seed, profile);

            frames[frameCount] = frame;
            frameCount++;
            cursor += length;
        }

        if (frameCount == 0)
        {
            return;
        }

        ApplyWordEnvelope(
            frames,
            segmentStartFrame,
            frameCount,
            segmentStartSample,
            cursor,
            segmentGrowlFrequency,
            sampleRate,
            profile);

        var lastPunctuation = contexts[^1].Punctuation;
        ApplyContour(frames, phraseStartFrame, frameCount, lastPunctuation, profile);

        var finalPause = Samples(
            lastPunctuation == PhrasePunctuation.Ellipsis ? 0.18f : 0.12f,
            sampleRate);
        frames[frameCount - 1].PauseAfterSamples = finalPause;
        frames[frameCount - 1].PauseAfterSeconds = finalPause / (float)sampleRate;

        LinkLegatoSegments(frames, frameCount);
    }

    private static void ApplyWordEnvelope(
        Span<SyllableFrame> frames,
        int startIndex,
        int endIndex,
        int wordStartSample,
        int wordEndSample,
        float growlFrequency,
        int sampleRate,
        DroidVoiceProfile profile)
    {
        if (endIndex <= startIndex)
        {
            return;
        }

        var duration = Math.Max(1, wordEndSample - wordStartSample) / (float)sampleRate;

        // The ADSR shape comes from the voice profile. Attack only on the first syllable,
        // release only on the last one; shrink everything proportionally when a word is too
        // short to fit the requested attack, decay and release.
        var attack = profile.AttackSeconds;
        var decay = profile.DecaySeconds;
        var release = profile.ReleaseSeconds;
        var sustain = profile.SustainLevel;

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
            frames[i].WordStartSample = wordStartSample;
            frames[i].WordEndSample = wordEndSample;
            frames[i].AttackTime = attack;
            frames[i].DecayTime = decay;
            frames[i].ReleaseTime = release;
            frames[i].SustainLevel = sustain;
            frames[i].GrowlFrequency = growlFrequency;
        }
    }

    private static void LinkLegatoSegments(Span<SyllableFrame> frames, int count)
    {
        for (var i = 1; i < count; i++)
        {
            if (frames[i].StartsLegatoSegment)
            {
                continue;
            }

            var previous = frames[i - 1];

            // The next syllable starts exactly where the previous one ended: frequency,
            // amplitude and FM depth are all continuous across the joint.
            frames[i].StartFreq = previous.EndFreq;
            frames[i].StartGain = previous.Gain;
            frames[i].StartFmDepth = previous.FmDepth;
            frames[i].StartFmDepth2 = previous.FmDepth2;
            frames[i].StartGrowlIntensity = previous.GrowlIntensity;
        }
    }

    private static void ApplyContour(
        Span<SyllableFrame> frames,
        int phraseStart,
        int phraseEnd,
        PhrasePunctuation punctuation,
        DroidVoiceProfile profile)
    {
        if (phraseEnd <= phraseStart)
        {
            return;
        }

        if (punctuation == PhrasePunctuation.Question)
        {
            var from = Math.Max(phraseStart, phraseEnd - 2);
            for (var i = from; i < phraseEnd; i++)
            {
                var isLast = i == phraseEnd - 1;
                frames[i].StartFreq = ClampFrequency(
                    frames[i].StartFreq * (isLast ? 1.10f : 1.00f),
                    profile);
                frames[i].EndFreq = ClampFrequency(
                    frames[i].EndFreq * (isLast ? 1.42f : 1.18f),
                    profile);

                if (frames[i].EndFreq <= frames[i].StartFreq)
                {
                    frames[i].EndFreq = ClampFrequency(frames[i].StartFreq * 1.12f, profile);
                }
            }
        }
        else if (punctuation == PhrasePunctuation.Ellipsis)
        {
            var from = Math.Max(phraseStart, phraseEnd - 2);
            for (var i = from; i < phraseEnd; i++)
            {
                var isLast = i == phraseEnd - 1;
                frames[i].StartFreq = ClampFrequency(
                    frames[i].StartFreq * (isLast ? 0.90f : 0.97f),
                    profile);
                frames[i].EndFreq = ClampFrequency(
                    frames[i].EndFreq * (isLast ? 0.68f : 0.82f),
                    profile);
            }
        }
        else
        {
            frames[phraseEnd - 1].EndFreq = ClampFrequency(
                frames[phraseEnd - 1].EndFreq * 0.92f,
                profile);
        }
    }

    /// <summary>
    /// Builds a short high whistle or sob for stressed / emotional syllables. The probability and
    /// intensity grow with positive valence, arousal, shouting and exclamation marks.
    /// </summary>
    private static WhistleSpec BuildWhistle(
        in SyllableContext context,
        int noteLength,
        bool isExclamation,
        int sampleRate,
        uint seed,
        DroidVoiceProfile profile)
    {
        var spec = default(WhistleSpec);
        if (profile.WhistleProbability <= 0f)
        {
            return spec;
        }

        var valence = MathF.Max(0f, context.EmotionValence);
        var arousal = MathF.Max(0f, context.EmotionArousal);
        var energy =
            0.55f * valence +
            0.90f * arousal +
            (context.IsStressed ? 0.22f : 0f) +
            (context.IsShout ? 0.32f : 0f) +
            (isExclamation ? 0.22f : 0f);

        if (!context.IsStressed && energy < 0.48f)
        {
            return spec;
        }

        var chance = Math.Clamp(
            (0.05f + 0.50f * energy) * profile.WhistleProbability,
            0f,
            1f);
        var random = new XorShift32(MixSeed(seed, context.WordHash, (uint)(0x500 + context.SyllableIndex)));
        if (random.NextFloat() >= chance)
        {
            return spec;
        }

        var duration = 0.032f + random.NextFloat() * 0.042f;
        duration /= profile.SpeedMultiplier;
        if (context.IsStressed)
        {
            duration *= 1.12f;
        }

        // Keep whistles inside the profile's high band while staying below Nyquist.
        var whistleScale = Math.Clamp(profile.MaxFreq / 1600f, 0.55f, 1.60f);
        var whistleMin = MathF.Max(600f, WhistleMinFrequency * whistleScale);
        var whistleMax = MathF.Min(
            WhistleMaxFrequency * whistleScale,
            sampleRate * 0.45f);
        if (whistleMax <= whistleMin)
        {
            return spec;
        }

        var length = Math.Max(1, Samples(duration, sampleRate));
        var rising = (random.NextUInt() & 1u) == 0u;
        var low = (1440f + random.NextFloat() * 420f) * whistleScale;
        var high = (1980f + random.NextFloat() * 560f) * whistleScale;
        var startFrequency = rising ? low : high;
        var endFrequency = rising ? high : low;

        return new WhistleSpec
        {
            Present = true,
            StartOffsetSamples = (int)(noteLength * (0.62f + 0.18f * random.NextFloat())),
            LengthSamples = length,
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
            Seed = MixSeed(seed, context.WordHash, (uint)(0x700 + context.SyllableIndex)),
        };
    }

    private static ChatterSpec BuildChatter(
        in SyllableContext context,
        int gapLength,
        int sampleRate,
        uint seed,
        DroidVoiceProfile profile)
    {
        var spec = default(ChatterSpec);
        if (gapLength <= 0 || profile.ChatterProbability <= 0f)
        {
            return spec;
        }

        var arousal = MathF.Max(0f, context.EmotionArousal);
        var fast = arousal > 0.18f || context.PhraseSyllableCount >= 12 || context.IsShout;
        if (!fast)
        {
            return spec;
        }

        var random = new XorShift32(MixSeed(seed, context.WordHash, 0xCA77E4u));
        var chance = profile.ChatterProbability * (0.30f + 0.60f * arousal);
        if (random.NextFloat() >= chance)
        {
            return spec;
        }

        var maxLength = Math.Min(gapLength, Samples(0.025f, sampleRate));
        var minLength = Math.Min(maxLength, Samples(0.015f, sampleRate));
        if (maxLength < minLength || maxLength <= 0)
        {
            return spec;
        }

        var length = minLength + (int)(random.NextUInt() % (uint)Math.Max(1, maxLength - minLength + 1));

        if ((random.NextUInt() & 1u) == 0u)
        {
            // A tiny stepped arpeggio, like a relay bank cycling.
            var baseFrequency = 700f + random.NextFloat() * 900f;
            return new ChatterSpec
            {
                Present = true,
                LengthSamples = length,
                Kind = ChatterKind.Arpeggio,
                Frequency1 = baseFrequency,
                Frequency2 = baseFrequency * (1.20f + random.NextFloat() * 0.25f),
                Frequency3 = baseFrequency * (1.55f + random.NextFloat() * 0.35f),
                StepCount = 2 + (int)(random.NextUInt() % 2u),
                Amplitude = 0.09f + 0.06f * arousal,
                Seed = MixSeed(seed, context.WordHash, 0xC0DEu),
            };
        }

        // A mechanical relay click with a sharp downward frequency drop.
        return new ChatterSpec
        {
            Present = true,
            LengthSamples = length,
            Kind = ChatterKind.RelayClick,
            StartFrequency = 1800f + random.NextFloat() * 1200f,
            EndFrequency = 180f + random.NextFloat() * 220f,
            Amplitude = 0.10f + 0.08f * arousal,
            Seed = MixSeed(seed, context.WordHash, 0xC11Cu),
        };
    }

    private static int Samples(float seconds, int sampleRate) =>
        Math.Max(0, (int)MathF.Round(seconds * sampleRate));

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
}
