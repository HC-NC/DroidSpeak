using DroidSpeak.Trainer.Audio;

namespace DroidSpeak.Tests;

public class PulseSegmenterTests
{
    private const int Rate = 22050;

    /// <summary>Builds a clip of <paramref name="tones"/> bursts separated by silence blocks.</summary>
    private static float[] BuildBursts(params (double Seconds, float Amplitude)[] segments)
    {
        var total = segments.Sum(segment => Math.Max(0, (int)(segment.Seconds * Rate)));
        var samples = new float[total];
        var offset = 0;
        var phase = 0f;

        foreach (var (seconds, amplitude) in segments)
        {
            var length = Math.Max(0, (int)(seconds * Rate));

            for (var i = 0; i < length; i++)
            {
                // A 220 Hz carrier is well above the 40 Hz frame rate of the envelope follower, so
                // each analysis frame sees a stable RMS rather than one carrier cycle.
                phase += 2f * MathF.PI * 220f / Rate;
                samples[offset + i] = amplitude * MathF.Sin(phase);
            }

            offset += length;
        }

        return samples;
    }

    [Fact]
    public void Segment_FindsSinglePulse()
    {
        var samples = BuildBursts((0.20, 0f), (0.15, 0.6f), (0.20, 0f));

        var segmentation = PulseSegmenter.Segment(samples, Rate);

        var pulse = Assert.Single(segmentation.Pulses);
        Assert.InRange(pulse.DurationSeconds, 0.13f, 0.17f);
        Assert.True(pulse.PeakRms > 0.1f);
        Assert.Single(segmentation.Words);
    }

    [Fact]
    public void Segment_SplitsTwoPulsesSeparatedByALongSilence()
    {
        var samples = BuildBursts((0.1, 0f), (0.12, 0.6f), (0.25, 0f), (0.12, 0.6f), (0.1, 0f));

        var segmentation = PulseSegmenter.Segment(samples, Rate);

        Assert.Equal(2, segmentation.Pulses.Length);
        Assert.Equal(2, segmentation.Words.Length);
        // Frame-quantised edges shrink the measured gap slightly, so the tolerance is one frame wide.
        Assert.InRange(segmentation.Pulses[0].PauseAfterSeconds, 0.20f, 0.30f);
        Assert.Equal(0f, segmentation.Pulses[^1].PauseAfterSeconds);
    }

    [Fact]
    public void Segment_KeepsShortGapsInsideOneWord()
    {
        var samples = BuildBursts((0.1, 0f), (0.10, 0.6f), (0.04, 0f), (0.10, 0.6f), (0.1, 0f));

        var segmentation = PulseSegmenter.Segment(samples, Rate);

        Assert.Equal(2, segmentation.Pulses.Length);
        var word = Assert.Single(segmentation.Words);
        Assert.Equal(2, word.PulseCount);
        Assert.Equal(0, word.FirstPulse);
    }

    [Fact]
    public void Segment_ReturnsNothingForSilence()
    {
        var samples = new float[Rate];

        var segmentation = PulseSegmenter.Segment(samples, Rate);

        Assert.Empty(segmentation.Pulses);
        Assert.Empty(segmentation.Words);
    }

    [Fact]
    public void Segment_ReturnsNothingForEmptyInput()
    {
        var segmentation = PulseSegmenter.Segment(ReadOnlySpan<float>.Empty, Rate);

        Assert.Empty(segmentation.Pulses);
        Assert.Empty(segmentation.Words);
    }

    [Fact]
    public void Segment_IgnoresBlipsShorterThanTheMinimum()
    {
        var samples = BuildBursts((0.1, 0f), (0.005, 0.90f), (0.1, 0f), (0.15, 0.60f), (0.1, 0f));

        var segmentation = PulseSegmenter.Segment(samples, Rate);

        var pulse = Assert.Single(segmentation.Pulses);
        Assert.InRange(pulse.DurationSeconds, 0.13f, 0.17f);
    }

    [Fact]
    public void Segment_AdaptsToAQuietRoomTone()
    {
        // A steady -46 dBFS hiss should not be segmented as speech.
        var samples = BuildBursts((0.3, 0f));
        var random = new Random(1234);
        for (var i = 0; i < samples.Length; i++)
        {
            samples[i] = (float)((random.NextDouble() - 0.5) * 0.01);
        }

        var segmentation = PulseSegmenter.Segment(samples, Rate);

        Assert.Empty(segmentation.Pulses);
    }

    [Fact]
    public void Segment_AppliesHysteresisSoATaperingTailStaysOnePulse()
    {
        // A long fade leaves the burst above the close threshold well after its peak.
        var length = (int)(0.30f * Rate);
        var samples = new float[length];

        for (var i = 0; i < length; i++)
        {
            var seconds = i / (float)Rate;
            var envelope = seconds < 0.10f
                ? seconds / 0.10f
                : MathF.Max(0f, 1f - ((seconds - 0.10f) / 0.20f));
            samples[i] = 0.7f * envelope * MathF.Sin(2f * MathF.PI * 220f * i / Rate);
        }

        var segmentation = PulseSegmenter.Segment(samples, Rate);

        Assert.Single(segmentation.Pulses);
    }

    [Fact]
    public void Envelope_ProducesOneValuePerFrame()
    {
        var samples = new float[1000];

        var envelope = PulseSegmenter.Envelope(samples, frameLength: 100, hop: 50);

        Assert.Equal(19, envelope.Length);
    }

    [Fact]
    public void NoiseFloor_UsesLowPercentile()
    {
        float[] envelope = [0.001f, 0.001f, 0.5f, 0.6f, 0.7f];

        var floor = PulseSegmenter.NoiseFloor(envelope);

        Assert.Equal(0.001f, floor);
    }

    [Fact]
    public void NoiseFloor_NeverDropsBelowAbsoluteFloor()
    {
        var floor = PulseSegmenter.NoiseFloor([0f, 0f, 0f]);

        Assert.Equal(PulseSegmenter.AbsoluteFloor, floor);
    }

    [Fact]
    public void GroupWords_StartsANewWordPastTheThreshold()
    {
        AcousticPulse[] pulses =
        [
            new(0, 1000, 0.045f, 0.004f, 0.5f, 0.4f),
            new(1100, 2200, 0.050f, 0.100f, 0.5f, 0.4f),
            new(5000, 6000, 0.045f, 0f, 0.5f, 0.4f),
        ];

        var words = PulseSegmenter.GroupWords(pulses, Rate, wordGapSeconds: 0.050f);

        Assert.Equal(2, words.Length);
        Assert.Equal(2, words[0].PulseCount);
        Assert.Single(new[] { words[1] }.Select(word => word.PulseCount));
    }

    [Fact]
    public void Segment_RejectsZeroRate()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => PulseSegmenter.Segment(new float[10], 0));
    }

    [Fact]
    public void Options_ValidateRejectsCrossedThresholds()
    {
        var options = PulseSegmenter.Options.Default with { CloseFactor = 4f };

        Assert.NotNull(options.Validate());
    }

    [Fact]
    public void Options_DefaultIsValid()
    {
        Assert.Null(PulseSegmenter.Options.Default.Validate());
    }
}