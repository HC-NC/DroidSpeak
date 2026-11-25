using DroidSpeak;
using DroidSpeak.DSP;

namespace DroidSpeak.Tests;

/// <summary>
/// Validates the golden snapshot mechanism itself: the acoustic tolerances must absorb the
/// cross-platform floating-point noise (libm ULP differences integrated through the oscillator
/// phase) that made the old raw-float SHA-256 hashes fail on Linux CI, while still catching every
/// change a snapshot is supposed to catch.
/// </summary>
public class PcmFingerprintTests
{
    private const string Text = "Hello, world!";

    [Fact]
    public void Format_IsStableAcrossIdenticalRenders()
    {
        var first = Render(seed: 0);
        var second = Render(seed: 0);

        Assert.Equal(PcmFingerprint.Format(first.samples, first.rate), PcmFingerprint.Format(second.samples, second.rate));
    }

    [Fact]
    public void Violations_AreEmptyForTheRenderTheGoldenWasSignedFrom()
    {
        var (samples, rate) = Render(seed: 0);

        var violations = PcmFingerprint.Violations(PcmSnapshotTests.Golden["hello|R2|0"], samples, rate);

        Assert.Empty(violations);
    }

    [Theory]
    // 1, 8 and 128 ULP of per-sample noise: far beyond what ucrtbase/glibc disagree by before the
    // renderer integrates it, yet invisible to the acoustic metrics.
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(128)]
    public void Violations_AreEmptyUnderUlpLevelNoise(int ulps)
    {
        var (samples, rate) = Render(seed: 0);

        var golden = PcmFingerprint.Format(samples, rate);
        var perturbed = (float[])samples.Clone();
        PerturbLikeCrossPlatformMathDrift(perturbed, ulps);

        Assert.Empty(PcmFingerprint.Violations(golden, perturbed, rate));
    }

    [Fact]
    public void Violations_CatchAGainChange()
    {
        var (samples, rate) = Render(seed: 0);
        var golden = PcmFingerprint.Format(samples, rate);

        var louder = new float[samples.Length];
        for (var i = 0; i < samples.Length; i++)
        {
            // A 2% level change: inaudible as a quality problem, but a real regression if the
            // peak limiter or envelope math moved.
            louder[i] = samples[i] * 1.02f;
        }

        Assert.NotEmpty(PcmFingerprint.Violations(golden, louder, rate));
    }

    [Fact]
    public void Violations_CatchATimingChange()
    {
        var (samples, rate) = Render(seed: 0);
        var golden = PcmFingerprint.Format(samples, rate);

        var shortened = samples[..(samples.Length - 100)];

        var violations = PcmFingerprint.Violations(golden, shortened, rate);
        Assert.NotEmpty(violations);
        Assert.Contains(violations, static v => v.StartsWith("len:", StringComparison.Ordinal));
    }

    [Fact]
    public void Violations_CatchAnEnvelopeOrderChange()
    {
        var (samples, rate) = Render(seed: 0);
        var golden = PcmFingerprint.Format(samples, rate);

        // Same total energy and zero-crossing count, reversed in time: only the temporal envelope
        // fingerprint can see this, which is exactly what it is there for.
        var reversed = (float[])samples.Clone();
        Array.Reverse(reversed);

        Assert.NotEmpty(PcmFingerprint.Violations(golden, reversed, rate));
    }

    [Fact]
    public void Violations_CatchASpectralBalanceChange()
    {
        var (samples, rate) = Render(seed: 0);
        var golden = PcmFingerprint.Format(samples, rate);

        // A non-trivial low-pass: a formant, filter or ring-modulation change looks like this to
        // the centroid metric.
        const int kernel = 9;
        var smoothed = new float[samples.Length];
        for (var i = 0; i < samples.Length; i++)
        {
            var start = Math.Max(0, i - (kernel / 2));
            var end = Math.Min(samples.Length, i + (kernel / 2) + 1);
            var sum = 0f;
            for (var j = start; j < end; j++)
            {
                sum += samples[j];
            }

            smoothed[i] = sum / (end - start);
        }

        Assert.NotEmpty(PcmFingerprint.Violations(golden, smoothed, rate));
    }

    [Fact]
    public void Violations_CatchADifferentSeed()
    {
        var (samples, rate) = Render(seed: 0);

        var otherSeed = Render(seed: 12345);

        Assert.NotEmpty(PcmFingerprint.Violations(PcmFingerprint.Format(samples, rate), otherSeed.samples, otherSeed.rate));
    }

    [Fact]
    public void Violations_CatchACompileTimeConstantChange()
    {
        var (samples, rate) = Render(seed: 0);
        var golden = PcmFingerprint.Format(samples, rate);

        var clipped = (float[])samples.Clone();
        for (var i = 0; i < clipped.Length; i++)
        {
            // Simulates the peak ceiling being raised from 0.95 to full scale.
            clipped[i] = Math.Clamp(clipped[i] * 1.4f, -1f, 1f);
        }

        Assert.NotEmpty(PcmFingerprint.Violations(golden, clipped, rate));
    }

    /// <summary>
    /// Adds up to <paramref name="ulps"/> units in the last place to every sample, growing with the
    /// square root of the index, which mimics how a per-operation ULP disagreement random-walks
    /// through the continuous phase accumulators. Exact zeros are preserved: silence comes from an
    /// exactly-zero gain, so no platform's math library turns it into a denormal.
    /// </summary>
    private static void PerturbLikeCrossPlatformMathDrift(float[] samples, int ulps)
    {
        for (var i = 0; i < samples.Length; i++)
        {
            var value = samples[i];
            if (value == 0f)
            {
                continue;
            }

            var magnitude = (int)Math.Round(Math.Sqrt(i + 1) / Math.Sqrt(samples.Length) * ulps);
            for (var u = 0; u < magnitude; u++)
            {
                value = (i & 1) == 0 ? MathF.BitIncrement(value) : MathF.BitDecrement(value);
            }

            samples[i] = value;
        }
    }

    private static (float[] samples, int rate) Render(int seed)
    {
        var synthesizer = new DroidSynthesizer(seed: seed, profile: DroidVoiceProfile.R2);
        return (synthesizer.Synthesize(Text), synthesizer.SampleRate);
    }
}
