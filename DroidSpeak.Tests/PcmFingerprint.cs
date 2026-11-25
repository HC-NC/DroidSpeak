using System.Globalization;
using System.Text;

namespace DroidSpeak.Tests;

/// <summary>
/// A platform-stable acoustic signature of a rendered buffer, used by the golden PCM snapshots.
/// <para>
/// <c>MathF.Sin</c>, <c>MathF.Cos</c>, <c>MathF.Exp</c> and <c>MathF.Tanh</c> delegate to the host
/// math library (<c>ucrtbase</c> on Windows, <c>glibc</c> on Linux, <c>libm</c> on macOS) and can
/// disagree by one unit in the last place. The renderer integrates that difference through its
/// continuous-phase oscillators and resonant filters, so by the end of a syllable an individual
/// sample can drift well past a single ULP. Hashing raw float bytes — or a 16-bit quantization of
/// them, which has roughly 3e-5 resolution and therefore sits *below* the drift — yields a
/// different digest per platform for acoustically identical audio.
/// </para>
/// <para>
/// Every metric below is either computed from integer arithmetic (buffer length, sample rate) or
/// is an aggregate that averages the drift out (energy, spectral and envelope statistics), so it
/// stays far inside its tolerance on any platform while still moving by orders of magnitude more
/// than the tolerance when the DSP math or timing genuinely changes. Byte-exact coverage is kept
/// by the same-machine comparisons in <see cref="PcmSnapshotTests"/>, where both buffers are
/// rendered by one platform and must match bit for bit.
/// </para>
/// </summary>
internal static class PcmFingerprint
{
    /// <summary>Number of sequential RMS buckets in the temporal envelope.</summary>
    private const int EnvelopeWindows = 16;

    /// <summary>Number of log-spaced Goertzel bins used for the spectral centroid.</summary>
    private const int SpectralBins = 24;

    private const double AmplitudeRelativeTolerance = 0.005;
    private const double AmplitudeFloorTolerance = 1e-4;
    private const double CentroidRelativeTolerance = 0.01;
    private const double CentroidFloorTolerance = 1.0;

    /// <summary>
    /// Crossings of zero and of the clipping ceiling may flip for a sample that sits within drift
    /// of the threshold, but any real frequency, timing or gain change moves these counts far more.
    /// </summary>
    private const double CountRelativeTolerance = 0.005;
    private const int CountFloorTolerance = 8;

    /// <summary>
    /// A syllable duration is a float converted to a sample count, so a one-ULP math difference can
    /// in principle flip the rounding boundary and change the length by a sample or two. Seed and
    /// text changes move lengths by hundreds of samples, so this tolerance costs no detection power.
    /// </summary>
    private const double LengthRelativeTolerance = 0.0005;
    private const int LengthFloorTolerance = 1;

    /// <summary>Serializes a buffer into its compact golden-signature form.</summary>
    public static string Format(ReadOnlySpan<float> samples, int sampleRate)
    {
        if (samples.IsEmpty)
        {
            throw new ArgumentException("Cannot fingerprint an empty buffer.", nameof(samples));
        }

        var culture = CultureInfo.InvariantCulture;
        var envelope = Envelope(samples);
        var builder = new StringBuilder(320);
        builder.Append("len=").Append(samples.Length.ToString(culture));
        builder.Append("|rate=").Append(sampleRate.ToString(culture));
        builder.Append("|rms=").Append(Rms(samples).ToString("F6", culture));
        builder.Append("|peak=").Append(Peak(samples).ToString("F6", culture));
        builder.Append("|mean=").Append(MeanAbs(samples).ToString("F6", culture));
        builder.Append("|dc=").Append(Mean(samples).ToString("F6", culture));
        builder.Append("|zc=").Append(ZeroCrossings(samples).ToString(culture));
        builder.Append("|clipped=").Append(ClippedSamples(samples).ToString(culture));
        builder.Append("|cen=").Append(CentroidHz(samples, sampleRate).ToString("F1", culture));
        builder.Append("|env=");
        for (var i = 0; i < envelope.Length; i++)
        {
            if (i > 0)
            {
                builder.Append(',');
            }

            builder.Append(envelope[i].ToString("F6", culture));
        }

        return builder.ToString();
    }

    /// <summary>
    /// Asserts that <paramref name="samples"/> matches the golden <paramref name="expected"/>
    /// signature within the acoustic tolerances, listing every field that drifted.
    /// </summary>
    public static void AssertMatches(string expected, ReadOnlySpan<float> samples, int sampleRate)
    {
        var violations = Violations(expected, samples, sampleRate);
        Assert.True(
            violations.Count == 0,
            "PCM signature drifted from the golden snapshot:" + Environment.NewLine +
            "  golden: " + expected + Environment.NewLine +
            "  actual: " + Format(samples, sampleRate) + Environment.NewLine +
            "  " + string.Join(Environment.NewLine + "  ", violations));
    }

    /// <summary>
    /// Returns one message per metric that left its tolerance, and nothing when the buffer is
    /// acoustically identical to the golden signature.
    /// </summary>
    public static IReadOnlyList<string> Violations(string expected, ReadOnlySpan<float> samples, int sampleRate)
    {
        var actual = Format(samples, sampleRate);
        var violations = new List<string>();

        // Sample rate is integer-exact; length is tolerant only to the rounding boundary a ULP can
        // flip, so a real timing change still trips it.
        if (!string.Equals(Field(expected, "rate"), Field(actual, "rate"), StringComparison.Ordinal))
        {
            violations.Add(
                $"rate: golden {Field(expected, "rate")}, actual {Field(actual, "rate")} (exact match required).");
        }

        CompareCounts(expected, actual, "len", violations, LengthRelativeTolerance, LengthFloorTolerance);

        CompareField(expected, actual, "rms", violations, AmplitudeRelativeTolerance, AmplitudeFloorTolerance);
        CompareField(expected, actual, "peak", violations, AmplitudeRelativeTolerance, AmplitudeFloorTolerance);
        CompareField(expected, actual, "mean", violations, AmplitudeRelativeTolerance, AmplitudeFloorTolerance);
        CompareField(expected, actual, "dc", violations, AmplitudeRelativeTolerance, AmplitudeFloorTolerance);
        CompareField(expected, actual, "cen", violations, CentroidRelativeTolerance, CentroidFloorTolerance);
        CompareCounts(expected, actual, "zc", violations);
        CompareCounts(expected, actual, "clipped", violations);

        var goldenEnvelope = Field(expected, "env").Split(',');
        var currentEnvelope = Field(actual, "env").Split(',');
        if (goldenEnvelope.Length != currentEnvelope.Length)
        {
            violations.Add($"env: golden has {goldenEnvelope.Length} windows, actual has {currentEnvelope.Length}.");
        }
        else
        {
            for (var i = 0; i < goldenEnvelope.Length; i++)
            {
                CompareValues(
                    goldenEnvelope[i],
                    currentEnvelope[i],
                    $"env[{i}]",
                    violations,
                    AmplitudeRelativeTolerance,
                    AmplitudeFloorTolerance);
            }
        }

        return violations;
    }

    private static void CompareField(string expected, string actual, string key, List<string> violations, double relative, double floor) =>
        CompareValues(Field(expected, key), Field(actual, key), key, violations, relative, floor);

    private static void CompareValues(
        string expectedValue,
        string actualValue,
        string key,
        List<string> violations,
        double relative,
        double floor)
    {
        var golden = double.Parse(expectedValue, CultureInfo.InvariantCulture);
        var current = double.Parse(actualValue, CultureInfo.InvariantCulture);
        var tolerance = Math.Max(floor, Math.Abs(golden) * relative);

        if (Math.Abs(golden - current) > tolerance)
        {
            violations.Add(
                $"{key}: golden {expectedValue}, actual {actualValue}, allowed ±" +
                $"{tolerance.ToString("0.######", CultureInfo.InvariantCulture)}.");
        }
    }

    private static void CompareCounts(string expected, string actual, string key, List<string> violations) =>
        CompareCounts(expected, actual, key, violations, CountRelativeTolerance, CountFloorTolerance);

    private static void CompareCounts(
        string expected,
        string actual,
        string key,
        List<string> violations,
        double relative,
        double floor)
    {
        var golden = int.Parse(Field(expected, key), CultureInfo.InvariantCulture);
        var current = int.Parse(Field(actual, key), CultureInfo.InvariantCulture);
        var tolerance = Math.Max(floor, (int)(Math.Abs(golden) * relative));

        if (Math.Abs(golden - current) > tolerance)
        {
            violations.Add($"{key}: golden {golden}, actual {current}, allowed ±{tolerance}.");
        }
    }

    private static string Field(string signature, string key)
    {
        foreach (var part in signature.Split('|'))
        {
            var separator = part.IndexOf('=');
            if (separator > 0 && part.AsSpan(0, separator).SequenceEqual(key.AsSpan()))
            {
                return part[(separator + 1)..];
            }
        }

        throw new ArgumentException($"Signature is missing the '{key}' field: {signature}", nameof(signature));
    }

    private static double Rms(ReadOnlySpan<float> samples)
    {
        double sum = 0;
        foreach (var sample in samples)
        {
            sum += sample * (double)sample;
        }

        return Math.Sqrt(sum / samples.Length);
    }

    private static double Peak(ReadOnlySpan<float> samples)
    {
        double peak = 0;
        foreach (var sample in samples)
        {
            var magnitude = Math.Abs(sample);
            if (magnitude > peak)
            {
                peak = magnitude;
            }
        }

        return peak;
    }

    private static double MeanAbs(ReadOnlySpan<float> samples)
    {
        double sum = 0;
        foreach (var sample in samples)
        {
            sum += Math.Abs(sample);
        }

        return sum / samples.Length;
    }

    private static double Mean(ReadOnlySpan<float> samples)
    {
        double sum = 0;
        foreach (var sample in samples)
        {
            sum += sample;
        }

        return sum / samples.Length;
    }

    private static int ZeroCrossings(ReadOnlySpan<float> samples)
    {
        var crossings = 0;
        var previous = 0f;

        foreach (var sample in samples)
        {
            // An exact zero (a pause) has no polarity, so carry the previous sign forward.
            if (sample == 0f)
            {
                continue;
            }

            if (previous != 0f && ((previous < 0f) != (sample < 0f)))
            {
                crossings++;
            }

            previous = sample;
        }

        return crossings;
    }

    private static int ClippedSamples(ReadOnlySpan<float> samples)
    {
        var clipped = 0;
        foreach (var sample in samples)
        {
            if (Math.Abs(sample) >= 0.999999f)
            {
                clipped++;
            }
        }

        return clipped;
    }

    private static double[] Envelope(ReadOnlySpan<float> samples)
    {
        var result = new double[EnvelopeWindows];
        var window = Math.Max(1, samples.Length / EnvelopeWindows);

        for (var i = 0; i < EnvelopeWindows; i++)
        {
            var start = i * window;
            if (start >= samples.Length)
            {
                continue;
            }

            var length = Math.Min(window, samples.Length - start);
            result[i] = Rms(samples.Slice(start, length));
        }

        return result;
    }

    /// <summary>
    /// Energy-weighted spectral centroid in Hz, estimated with a Goertzel power filter over
    /// log-spaced bins. It tracks the formant and ring-modulation balance, so any real timbre
    /// change moves it.
    /// </summary>
    private static double CentroidHz(ReadOnlySpan<float> samples, int sampleRate)
    {
        const double minFrequency = 120.0;
        var maxFrequency = Math.Min(sampleRate * 0.45, 9000.0);
        var nyquistFrequency = sampleRate * 0.5;

        double weightedSum = 0;
        double energySum = 0;

        for (var bin = 0; bin < SpectralBins; bin++)
        {
            var frequency = minFrequency *
                Math.Pow(maxFrequency / minFrequency, bin / (double)(SpectralBins - 1));
            var energy = GoertzelPower(samples, frequency, nyquistFrequency);
            weightedSum += frequency * energy;
            energySum += energy;
        }

        return energySum > 0 ? weightedSum / energySum : 0;
    }

    /// <summary>Standard Goertzel second-order recursion returning |X(k)| squared.</summary>
    private static double GoertzelPower(ReadOnlySpan<float> samples, double frequency, double nyquistFrequency)
    {
        var normalized = frequency / nyquistFrequency;
        if (normalized >= 1.0)
        {
            return 0;
        }

        var coefficient = 2.0 * Math.Cos(Math.PI * normalized);
        double state = 0;
        double previousState = 0;

        foreach (var sample in samples)
        {
            var current = (double)sample + (coefficient * state) - previousState;
            previousState = state;
            state = current;
        }

        return (state * state)
            + (previousState * previousState)
            - (coefficient * state * previousState);
    }
}
