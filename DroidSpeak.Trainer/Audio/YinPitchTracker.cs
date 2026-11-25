namespace DroidSpeak.Trainer.Audio;

/// <summary>The result of one YIN pitch measurement.</summary>
/// <param name="FrequencyHz">Estimated fundamental in Hz, or <c>0</c> when the frame is unvoiced.</param>
/// <param name="Clarity">
/// Aperiodicity complement in <c>[0, 1]</c>: <c>1</c> is a perfectly periodic signal and <c>0</c> is
/// noise or silence. Meaningless when <paramref name="FrequencyHz"/> is <c>0</c>.
/// </param>
internal readonly record struct PitchEstimate(float FrequencyHz, float Clarity)
{
    /// <summary>Whether the estimate carries a usable fundamental.</summary>
    public bool IsVoiced => FrequencyHz > 0f;
}

/// <summary>
/// YIN fundamental-frequency estimator, used to recover the carrier pitch of a droid pulse from its
/// rendered audio.
/// <para>
/// The implementation follows the original YIN paper: a difference function, its cumulative mean
/// normalization, an absolute threshold that picks the first plausible period instead of the global
/// minimum, and a parabolic interpolation around that dip. The first-dip rule is what keeps an
/// octave error out of the dataset: the global minimum of a droid's FM-heavy tone often lands on
/// twice the carrier period, and training on that would teach the network to sing an octave low.
/// </para>
/// <para>
/// Pitch is measured on the pulse itself, which is short (tens of milliseconds), so the estimator
/// works on the whole slice rather than sliding frames. Non-finite samples are treated as silence so
/// one bad byte cannot poison a clip.
/// </para>
/// </summary>
internal static class YinPitchTracker
{
    /// <summary>Default aperiodicity threshold above which a frame counts as unvoiced.</summary>
    public const float DefaultThreshold = 0.15f;

    /// <summary>Lowest fundamental searched by default, in Hz.</summary>
    public const float DefaultMinFrequency = 40f;

    /// <summary>Highest fundamental searched by default, in Hz.</summary>
    public const float DefaultMaxFrequency = 2000f;

    /// <summary>
    /// Estimates the fundamental of <paramref name="samples"/>.
    /// </summary>
    /// <param name="samples">Mono samples normalized to roughly <c>[-1, 1]</c>.</param>
    /// <param name="sampleRate">Rate of <paramref name="samples"/> in Hz.</param>
    /// <param name="threshold">YIN absolute threshold; lower values reject more frames.</param>
    /// <param name="minFrequency">Lowest fundamental to consider, in Hz.</param>
    /// <param name="maxFrequency">Highest fundamental to consider, in Hz.</param>
    /// <returns>
    /// The frequency and clarity, or <c>(0, 0)</c> when the slice is too short, silent, digital
    /// garbage, or does not contain a periodic dip below <paramref name="threshold"/>.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="sampleRate"/> is not positive, or the frequency band is empty.
    /// </exception>
    public static PitchEstimate Detect(
        ReadOnlySpan<float> samples,
        int sampleRate,
        float threshold = DefaultThreshold,
        float minFrequency = DefaultMinFrequency,
        float maxFrequency = DefaultMaxFrequency)
    {
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Rate must be positive.");
        }

        if (!(minFrequency > 0f) || !(maxFrequency > minFrequency))
        {
            throw new ArgumentOutOfRangeException(
                nameof(minFrequency),
                minFrequency,
                "The frequency band must be positive and non-empty.");
        }

        var minLag = Math.Max(2, (int)MathF.Floor(sampleRate / MathF.Min(maxFrequency, sampleRate * 0.45f)));
        var maxLag = (int)MathF.Ceiling(sampleRate / minFrequency);

        if (maxLag <= minLag || samples.Length < (maxLag * 2) + 2)
        {
            return default;
        }

        // The longest lag determines how much signal the difference function can reach. One window of
        // that size is enough for a pulse; a longer window would smooth over the glide inside it.
        var window = Math.Min(samples.Length - maxLag, Math.Max(maxLag, sampleRate / 40));
        if (window < 8)
        {
            return default;
        }

        var needed = window + maxLag + 1;
        var signal = new float[needed];
        var mean = 0f;
        var finiteCount = 0;

        for (var i = 0; i < samples.Length && i < needed; i++)
        {
            var value = samples[i];
            if (!float.IsFinite(value))
            {
                value = 0f;
            }

            signal[i] = value;
            mean += value;
            finiteCount++;
        }

        if (finiteCount == 0)
        {
            return default;
        }

        mean /= finiteCount;

        var energy = 0f;
        for (var i = 0; i < needed; i++)
        {
            signal[i] -= mean;
            energy += signal[i] * signal[i];
        }

        if (energy <= 1e-12f)
        {
            return default;
        }

        var difference = new float[maxLag + 1];
        for (var lag = 0; lag <= maxLag; lag++)
        {
            var sum = 0f;
            for (var i = 0; i < window; i++)
            {
                var delta = signal[i] - signal[i + lag];
                sum += delta * delta;
            }

            difference[lag] = sum;
        }

        // Cumulative mean normalization: d'(0) is forced to 1 so the threshold has a stable meaning.
        var normalized = new float[maxLag + 1];
        normalized[0] = 1f;
        var running = 0f;

        for (var lag = 1; lag <= maxLag; lag++)
        {
            running += difference[lag];
            normalized[lag] = running > 0f
                ? difference[lag] * lag / running
                : 1f;
        }

        var bestLag = -1;
        for (var lag = minLag; lag <= maxLag; lag++)
        {
            if (normalized[lag] >= threshold)
            {
                continue;
            }

            // Descend into the dip: the first lag under the threshold can sit on the dip's shoulder.
            while (lag + 1 <= maxLag && normalized[lag + 1] < normalized[lag])
            {
                lag++;
            }

            bestLag = lag;
            break;
        }

        if (bestLag < 0)
        {
            return default;
        }

        var refined = RefinePeriod(normalized, bestLag, minLag, maxLag);
        if (!(refined > 0f))
        {
            return default;
        }

        var frequency = sampleRate / refined;
        if (!float.IsFinite(frequency) || frequency < minFrequency || frequency > maxFrequency)
        {
            return default;
        }

        var clarity = Math.Clamp(1f - normalized[bestLag], 0f, 1f);
        return new PitchEstimate(frequency, clarity);
    }

    /// <summary>
    /// Measures period jitter over half-overlapping windows of <paramref name="samples"/>.
    /// </summary>
    /// <param name="samples">Mono samples normalized to roughly <c>[-1, 1]</c>.</param>
    /// <param name="sampleRate">Rate of <paramref name="samples"/> in Hz.</param>
    /// <param name="threshold">YIN absolute threshold passed to <see cref="Detect"/>.</param>
    /// <param name="minFrequency">Lowest fundamental to consider, in Hz.</param>
    /// <param name="maxFrequency">Highest fundamental to consider, in Hz.</param>
    /// <returns>
    /// Mean absolute period change in seconds between consecutive voiced windows, or <c>0</c> when
    /// fewer than two windows are voiced. Unvoiced windows are skipped without resetting the
    /// previous period, so a single glottal dropout does not halve the measured jitter.
    /// </returns>
    public static float DetectJitter(
        ReadOnlySpan<float> samples,
        int sampleRate,
        float threshold = DefaultThreshold,
        float minFrequency = DefaultMinFrequency,
        float maxFrequency = DefaultMaxFrequency)
    {
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Rate must be positive.");
        }

        var window = samples.Length / 2;
        if (window < 16)
        {
            return 0f;
        }

        var previousPeriod = 0f;
        var voiced = 0;
        var jitter = 0f;
        var transitions = 0;

        for (var start = 0; start + window <= samples.Length; start += window)
        {
            var estimate = Detect(
                samples.Slice(start, window),
                sampleRate,
                threshold,
                minFrequency,
                maxFrequency);

            if (!estimate.IsVoiced)
            {
                continue;
            }

            var period = 1f / estimate.FrequencyHz;
            if (voiced > 0 && previousPeriod > 0f)
            {
                jitter += MathF.Abs(period - previousPeriod);
                transitions++;
            }

            previousPeriod = period;
            voiced++;
        }

        return transitions > 0 ? jitter / transitions : 0f;
    }

    /// <summary>
    /// Fits a parabola through the normalized difference at <paramref name="lag"/> and its two
    /// neighbours, then returns the interpolated period in samples.
    /// </summary>
    private static float RefinePeriod(float[] normalized, int lag, int minLag, int maxLag)
    {
        if (lag <= minLag || lag >= maxLag)
        {
            return lag;
        }

        var left = normalized[lag - 1];
        var center = normalized[lag];
        var right = normalized[lag + 1];
        var denominator = left - (2f * center) + right;

        if (MathF.Abs(denominator) <= 1e-9f)
        {
            return lag;
        }

        var offset = 0.5f * (left - right) / denominator;
        if (!float.IsFinite(offset))
        {
            return lag;
        }

        return lag + Math.Clamp(offset, -0.5f, 0.5f);
    }
}
