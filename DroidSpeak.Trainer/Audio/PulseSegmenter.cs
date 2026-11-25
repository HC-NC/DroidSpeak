namespace DroidSpeak.Trainer.Audio;

/// <summary>One contiguous burst of energy inside a clip: a droid "syllable".</summary>
/// <param name="StartSample">First sample of the burst, inclusive.</param>
/// <param name="EndSample">One past the last sample of the burst.</param>
/// <param name="DurationSeconds">Sounded length in seconds.</param>
/// <param name="PauseAfterSeconds">Silence between this burst and the next one, in seconds.</param>
/// <param name="PeakRms">Loudest analysis-frame RMS inside the burst.</param>
/// <param name="MeanRms">Mean analysis-frame RMS over the burst.</param>
internal readonly record struct AcousticPulse(
    int StartSample,
    int EndSample,
    float DurationSeconds,
    float PauseAfterSeconds,
    float PeakRms,
    float MeanRms)
{
    /// <summary>Sounded length in samples.</summary>
    public int LengthSamples => EndSample - StartSample;
}

/// <summary>A run of pulses that the gap heuristic decided belongs to one word.</summary>
/// <param name="FirstPulse">Index of the first pulse in the clip's pulse list.</param>
/// <param name="PulseCount">Number of pulses in the word.</param>
/// <param name="StartSample">Sample position of the first pulse's onset.</param>
/// <param name="EndSample">Sample position of the last pulse's end.</param>
internal readonly record struct AcousticWord(int FirstPulse, int PulseCount, int StartSample, int EndSample);

/// <summary>The segmentation of one clip: the pulses plus the word grouping over them.</summary>
/// <param name="Pulses">Detected pulses in time order.</param>
/// <param name="Words">Word groups over <paramref name="Pulses"/>.</param>
internal readonly record struct Segmentation(
    AcousticPulse[] Pulses,
    AcousticWord[] Words);

/// <summary>
/// Splits a mono clip into discrete acoustic pulses with an RMS envelope follower, adaptive noise
/// floor and hysteresis, then groups pulses into words by measuring the silence between them.
/// <para>
/// Energy segmentation is the only approach that survives the kind of audio this trainer is fed.
/// Transcription is not available, the clips are noisy reference rips, and a droid beep has no
/// phonetic cues a spectral segmenter could trust — but it does have an unambiguous on/off envelope.
/// Everything downstream (pitch, formants, timbre) assumes one pulse is one syllable, so the
/// thresholds here are the single most consequential tuning surface in extraction.
/// </para>
/// <para>
/// Hysteresis matters twice over. The open threshold is set well above the noise floor so room tone
/// cannot start a pulse, and the close threshold sits lower so a syllable that softens mid-way is not
/// chopped into fragments. The floor itself is estimated from the quiet frames of the clip instead of
/// being a fixed constant, which is what lets one set of defaults work on both a clean game rip and a
/// phone recording.
/// </para>
/// </summary>
internal static class PulseSegmenter
{
    /// <summary>Length of one RMS analysis frame in seconds.</summary>
    public const float DefaultFrameSeconds = 0.011f;

    /// <summary>Fraction of a frame the analysis window advances between RMS values.</summary>
    public const float DefaultHopFraction = 0.5f;

    /// <summary>Multiplier over the estimated noise floor that opens a pulse.</summary>
    public const float DefaultOpenFactor = 3.0f;

    /// <summary>Multiplier over the estimated noise floor that closes a pulse.</summary>
    public const float DefaultCloseFactor = 1.6f;

    /// <summary>Shortest accepted pulse in seconds; shorter blips are treated as clicks.</summary>
    public const float DefaultMinPulseSeconds = 0.025f;

    /// <summary>Shortest silence that can end a pulse, in seconds.</summary>
    public const float DefaultMinGapSeconds = 0.008f;

    /// <summary>Gap above which a new pulse starts a new word, in seconds.</summary>
    public const float DefaultWordGapSeconds = 0.100f;

    /// <summary>Absolute RMS floor so a digitally silent clip never opens a pulse on a stray byte.</summary>
    public const float AbsoluteFloor = 1e-4f;

    /// <summary>Tuning for <see cref="Segment"/>; the defaults are the documented CLI defaults.</summary>
    /// <param name="FrameSeconds">RMS analysis frame length.</param>
    /// <param name="HopFraction">Hop as a fraction of the frame length.</param>
    /// <param name="OpenFactor">Multiplier over the noise floor that opens a pulse.</param>
    /// <param name="CloseFactor">Multiplier over the noise floor that closes a pulse.</param>
    /// <param name="MinPulseSeconds">Shortest accepted pulse.</param>
    /// <param name="MinGapSeconds">Shortest silence that ends a pulse.</param>
    /// <param name="WordGapSeconds">Gap above which pulses belong to different words.</param>
    public readonly record struct Options(
        float FrameSeconds,
        float HopFraction,
        float OpenFactor,
        float CloseFactor,
        float MinPulseSeconds,
        float MinGapSeconds,
        float WordGapSeconds)
    {
        /// <summary>The documented defaults that the <c>extract</c> command exposes as flags.</summary>
        public static Options Default => new(
            DefaultFrameSeconds,
            DefaultHopFraction,
            DefaultOpenFactor,
            DefaultCloseFactor,
            DefaultMinPulseSeconds,
            DefaultMinGapSeconds,
            DefaultWordGapSeconds);

        /// <summary>Rejects values that cannot produce a sensible envelope.</summary>
        public string? Validate()
        {
            if (FrameSeconds is < 0.002f or > 0.100f)
            {
                return "--frame-ms must sit between 2 and 100 ms.";
            }

            if (HopFraction is < 0.10f or > 1.00f)
            {
                return "--hop-ratio must sit between 0.1 and 1.0.";
            }

            if (OpenFactor <= 1f || CloseFactor <= 1f || CloseFactor >= OpenFactor)
            {
                return "--open-factor and --close-factor must be above 1 and close must be below open.";
            }

            if (MinPulseSeconds is < 0.005f or > 0.500f)
            {
                return "--min-pulse-ms must sit between 5 and 500 ms.";
            }

            if (MinGapSeconds is < 0f or > 0.200f)
            {
                return "--min-gap-ms must sit between 0 and 200 ms.";
            }

            if (WordGapSeconds is < 0.010f or > 1.000f)
            {
                return "--word-gap-ms must sit between 10 and 1000 ms.";
            }

            return null;
        }
    }

    /// <summary>
    /// Segments <paramref name="samples"/> into pulses and words.
    /// </summary>
    /// <param name="samples">Mono samples normalised to roughly <c>[-1, 1]</c>.</param>
    /// <param name="sampleRate">Rate of <paramref name="samples"/> in Hz.</param>
    /// <param name="options">Tuning; <c>null</c> uses <see cref="Options.Default"/>.</param>
    /// <returns>The pulses and their word grouping; both empty for a silent clip.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="sampleRate"/> is not positive.</exception>
    public static Segmentation Segment(
        ReadOnlySpan<float> samples,
        int sampleRate,
        Options? options = null)
    {
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Rate must be positive.");
        }

        var tuning = options ?? Options.Default;

        if (samples.IsEmpty)
        {
            return new Segmentation([], []);
        }

        var frameLength = Math.Max(4, (int)MathF.Round(tuning.FrameSeconds * sampleRate));
        var hop = Math.Max(1, (int)MathF.Round(frameLength * tuning.HopFraction));
        var envelope = Envelope(samples, frameLength, hop);
        var floor = NoiseFloor(envelope);

        var openThreshold = Math.Max(AbsoluteFloor, floor * tuning.OpenFactor);
        var closeThreshold = Math.Max(AbsoluteFloor * 0.5f, floor * tuning.CloseFactor);
        var minPulseSamples = (int)MathF.Round(tuning.MinPulseSeconds * sampleRate);
        var minGapSamples = (int)MathF.Round(tuning.MinGapSeconds * sampleRate);

        var pulses = new List<AcousticPulse>();
        var active = false;
        var startFrame = 0;
        var lastAboveFrame = 0;

        for (var frame = 0; frame < envelope.Length; frame++)
        {
            var level = envelope[frame];

            if (!active)
            {
                if (level >= openThreshold)
                {
                    active = true;
                    startFrame = frame;
                    lastAboveFrame = frame;
                }

                continue;
            }

            if (level >= closeThreshold)
            {
                lastAboveFrame = frame;
                continue;
            }

            // The silence has to be long enough to count, otherwise a brief dip inside a vowel would
            // split one syllable into two.
            var silenceSamples = (frame - lastAboveFrame) * hop;
            if (silenceSamples < minGapSamples && frame + 1 < envelope.Length)
            {
                continue;
            }

            Emit(pulses, envelope, startFrame, lastAboveFrame, frameLength, hop, samples.Length);
            active = false;
        }

        if (active)
        {
            Emit(pulses, envelope, startFrame, lastAboveFrame, frameLength, hop, samples.Length);
        }

        // Pulses shorter than the minimum are clicks, relays and breath noise, not syllables.
        var kept = new List<AcousticPulse>(pulses.Count);
        foreach (var pulse in pulses)
        {
            if (pulse.LengthSamples >= minPulseSamples)
            {
                kept.Add(pulse);
            }
        }

        var pulseArray = kept.ToArray();
        var withPauses = ApplyPauses(pulseArray, sampleRate);
        var words = GroupWords(withPauses, sampleRate, tuning.WordGapSeconds);

        return new Segmentation(withPauses, words);
    }

    /// <summary>
    /// Computes the per-frame RMS envelope with a half-frame hop by default.
    /// </summary>
    /// <param name="samples">Mono samples.</param>
    /// <param name="frameLength">Window length in samples.</param>
    /// <param name="hop">Advance between windows in samples.</param>
    /// <returns>One RMS value per frame; never empty for a non-empty clip.</returns>
    public static float[] Envelope(ReadOnlySpan<float> samples, int frameLength, int hop)
    {
        if (samples.IsEmpty)
        {
            return [];
        }

        frameLength = Math.Max(1, frameLength);
        hop = Math.Max(1, hop);

        var frames = Math.Max(1, (samples.Length - frameLength) / hop + 1);
        if (samples.Length < frameLength)
        {
            frames = 1;
        }

        var envelope = new float[frames];

        for (var frame = 0; frame < frames; frame++)
        {
            var start = frame * hop;
            var end = Math.Min(samples.Length, start + frameLength);
            var sum = 0f;

            for (var i = start; i < end; i++)
            {
                sum += samples[i] * samples[i];
            }

            envelope[frame] = end > start ? MathF.Sqrt(sum / (end - start)) : 0f;
        }

        return envelope;
    }

    /// <summary>
    /// Estimates the noise floor as a low percentile of the frame RMS values.
    /// </summary>
    /// <param name="envelope">Frame RMS values.</param>
    /// <returns>A floor no lower than <see cref="AbsoluteFloor"/>.</returns>
    /// <remarks>
    /// The 20th percentile is used rather than the minimum: a clip trimmed to begin exactly on a beep
    /// has no silence at all, and the minimum would then be a loud frame, which would set a threshold
    /// high enough to lose the quiet syllable tails. A percentile keeps the estimate anchored to the
    /// quietest fifth of the clip regardless of how it was trimmed.
    /// </remarks>
    public static float NoiseFloor(ReadOnlySpan<float> envelope)
    {
        if (envelope.IsEmpty)
        {
            return AbsoluteFloor;
        }

        var sorted = envelope.ToArray();
        Array.Sort(sorted);
        var index = Math.Clamp((int)(sorted.Length * 0.20f), 0, sorted.Length - 1);
        return Math.Max(AbsoluteFloor, sorted[index]);
    }

    private static void Emit(
        List<AcousticPulse> pulses,
        float[] envelope,
        int startFrame,
        int lastAboveFrame,
        int frameLength,
        int hop,
        int clipLength)
    {
        var start = startFrame * hop;
        var end = Math.Min(clipLength, (lastAboveFrame * hop) + frameLength);

        if (end <= start)
        {
            return;
        }

        var peak = 0f;
        var sum = 0f;
        var count = 0;

        for (var frame = startFrame; frame <= lastAboveFrame && frame < envelope.Length; frame++)
        {
            var level = envelope[frame];
            peak = Math.Max(peak, level);
            sum += level;
            count++;
        }

        var mean = count > 0 ? sum / count : 0f;

        // Duration and pause are filled in by ApplyPauses once every pulse is known.
        pulses.Add(new AcousticPulse(start, end, 0f, 0f, peak, mean));
    }

    private static AcousticPulse[] ApplyPauses(AcousticPulse[] pulses, int sampleRate)
    {
        if (pulses.Length == 0)
        {
            return pulses;
        }

        var result = new AcousticPulse[pulses.Length];

        for (var i = 0; i < pulses.Length; i++)
        {
            var pulse = pulses[i];
            var duration = pulse.LengthSamples / (float)sampleRate;
            var pause = i + 1 < pulses.Length
                ? Math.Max(0, pulses[i + 1].StartSample - pulse.EndSample) / (float)sampleRate
                : 0f;

            result[i] = pulse with { DurationSeconds = duration, PauseAfterSeconds = pause };
        }

        return result;
    }

    /// <summary>
    /// Groups consecutive pulses into words, starting a new word whenever the silence between them
    /// exceeds <paramref name="wordGapSeconds"/>.
    /// </summary>
    /// <param name="pulses">Pulses in time order.</param>
    /// <param name="sampleRate">Rate the pulse positions are expressed in.</param>
    /// <param name="wordGapSeconds">Gap threshold in seconds.</param>
    public static AcousticWord[] GroupWords(
        AcousticPulse[] pulses,
        int sampleRate,
        float wordGapSeconds = DefaultWordGapSeconds)
    {
        if (pulses.Length == 0)
        {
            return [];
        }

        var threshold = wordGapSeconds * sampleRate;
        var words = new List<AcousticWord>();
        var first = 0;

        for (var i = 1; i <= pulses.Length; i++)
        {
            var boundary = i == pulses.Length ||
                pulses[i].StartSample - pulses[i - 1].EndSample > threshold;

            if (!boundary)
            {
                continue;
            }

            words.Add(new AcousticWord(
                first,
                i - first,
                pulses[first].StartSample,
                pulses[i - 1].EndSample));
            first = i;
        }

        return words.ToArray();
    }
}
