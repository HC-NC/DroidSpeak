using DroidSpeak.NLP;
using DroidSpeak.Neural;
using DroidSpeak.Trainer.Audio;

namespace DroidSpeak.Tests;

/// <summary>
/// Covers the measurement half of extraction: the pitch tracker, the spectral helpers, the frame
/// extractor and the file-name-driven context mapper.
/// </summary>
public class AcousticExtractionTests
{
    private const int Rate = 44100;

    /// <summary>Builds a mono sine tone.</summary>
    private static float[] Sine(double seconds, float frequency, float amplitude = 0.5f, int rate = Rate)
    {
        var samples = new float[Math.Max(0, (int)(seconds * rate))];
        var phase = 0f;

        for (var i = 0; i < samples.Length; i++)
        {
            phase += 2f * MathF.PI * frequency / rate;
            samples[i] = amplitude * MathF.Sin(phase);
        }

        return samples;
    }

    /// <summary>Writes a windowed tone into a larger buffer.</summary>
    private static void WriteTone(
        float[] buffer,
        int rate,
        double start,
        double seconds,
        float frequency,
        float amplitude)
    {
        var from = Math.Max(0, (int)(start * rate));
        var to = Math.Min(buffer.Length, from + (int)(seconds * rate));
        var phase = 0f;

        for (var i = from; i < to; i++)
        {
            phase += 2f * MathF.PI * frequency / rate;
            buffer[i] = amplitude * MathF.Sin(phase);
        }
    }

    /// <summary>Two 150 ms tone bursts separated by enough silence to read as two words.</summary>
    private static float[] BuildTwoPulses(int rate)
    {
        var samples = new float[rate];
        WriteTone(samples, rate, 0.10, 0.15, 440f, 0.6f);
        WriteTone(samples, rate, 0.40, 0.15, 440f, 0.6f);
        return samples;
    }

    [Fact]
    public void Detect_FindsPureTone()
    {
        var samples = Sine(0.5, 220f);

        var estimate = YinPitchTracker.Detect(samples, Rate);

        Assert.True(estimate.IsVoiced);
        Assert.InRange(estimate.FrequencyHz, 215f, 225f);
        Assert.True(estimate.Clarity > 0.8f);
    }

    [Fact]
    public void Detect_ReturnsUnvoicedForSilence()
    {
        var estimate = YinPitchTracker.Detect(new float[4096], Rate);

        Assert.False(estimate.IsVoiced);
        Assert.Equal(0f, estimate.FrequencyHz);
        Assert.Equal(0f, estimate.Clarity);
    }

    [Fact]
    public void DetectJitter_IsNearZeroForSteadyTone()
    {
        var samples = Sine(0.6, 220f);

        var jitter = YinPitchTracker.DetectJitter(samples, Rate);

        // A perfectly periodic tone should not move by even a tenth of a millisecond.
        Assert.InRange(jitter, 0f, 0.0001f);
    }

    [Fact]
    public void Transform_DcInputLandsInFirstBin()
    {
        var real = new float[8];
        var imaginary = new float[8];
        Array.Fill(real, 1f);

        SpectralAnalyzer.Transform(real, imaginary);

        Assert.Equal(8f, real[0], 3);
        for (var k = 1; k < real.Length; k++)
        {
            Assert.Equal(0f, real[k], 3);
            Assert.Equal(0f, imaginary[k], 3);
        }
    }

    [Fact]
    public void Transform_RejectsNonPowerOfTwo()
    {
        var real = new float[6];
        var imaginary = new float[6];

        Assert.Throws<ArgumentException>(() => SpectralAnalyzer.Transform(real, imaginary));
    }

    [Fact]
    public void ZeroCrossingRate_CountsTwiceTheFrequency()
    {
        var samples = Sine(0.5, 200f, rate: 8000);

        var rate = SpectralAnalyzer.ZeroCrossingRate(samples, 8000);

        Assert.InRange(rate, 380f, 420f);
    }

    [Fact]
    public void HighFrequencyRatio_SeparatesBands()
    {
        var sampleRate = 22050;
        var high = new float[513];
        var low = new float[513];
        high[300] = 1f;
        low[10] = 1f;

        Assert.Equal(
            1f,
            SpectralAnalyzer.HighFrequencyRatio(high, sampleRate, 4000f),
            3);
        Assert.Equal(
            0f,
            SpectralAnalyzer.HighFrequencyRatio(low, sampleRate, 4000f),
            3);
    }

    [Fact]
    public void Formants_FindsTwoPeaks()
    {
        var sampleRate = 22050;
        var magnitudes = new float[513];
        var binHz = sampleRate / (2f * (magnitudes.Length - 1));
        magnitudes[(int)MathF.Round(500f / binHz)] = 1f;
        magnitudes[(int)MathF.Round(1500f / binHz)] = 1f;

        var pair = SpectralAnalyzer.Formants(magnitudes, sampleRate);

        Assert.InRange(pair.FirstHz, 470f, 530f);
        Assert.InRange(pair.SecondHz, 1470f, 1530f);
    }

    [Fact]
    public void ExtractFrame_StaysInsideDecoderRanges()
    {
        var samples = Sine(0.2, 440f, 0.6f);
        var pulse = new AcousticPulse(0, samples.Length, 0.2f, 0f, 0.6f, 0.42f);

        var frame = AcousticFeatureExtractor.ExtractFrame(
            samples,
            Rate,
            pulse,
            DroidVoiceProfile.R2,
            isSegmentStart: true);

        Assert.Equal(samples.Length, frame.LengthSamples);
        Assert.InRange(frame.Gain, SyllableTargetDecoder.MinGain, SyllableTargetDecoder.MaxGain);
        Assert.InRange(frame.FmDepth, SyllableTargetDecoder.MinFmDepth, SyllableTargetDecoder.MaxFmDepth);
        Assert.InRange(frame.FmRatio, SyllableTargetDecoder.MinFmRatio, SyllableTargetDecoder.MaxFmRatio);
        Assert.InRange(frame.NoiseMix, 0f, SyllableTargetDecoder.MaxNoise);
        Assert.InRange(frame.GrowlIntensity, 0f, SyllableTargetDecoder.MaxGrowl);
        Assert.InRange(frame.Saturation, 0f, SyllableTargetDecoder.MaxSaturation);
        Assert.InRange(frame.FormantF1, SyllableTargetDecoder.MinFormantHz, SyllableTargetDecoder.FormantCeiling(Rate) * 0.70f);
        Assert.InRange(frame.FormantF2, frame.FormantF1 + SyllableTargetDecoder.FormantGuardHz, SyllableTargetDecoder.FormantCeiling(Rate));
    }

    [Fact]
    public void Mapper_UsesBracketTagAndPunctuation()
    {
        var rate = 22050;
        var clip = new WavData(rate, BuildTwoPulses(rate), rate, 1);
        var segmentation = PulseSegmenter.Segment(clip.Samples, clip.SampleRate);

        Assert.Equal(2, segmentation.Pulses.Length);

        var (contexts, frames) = AcousticContextMapper.Extract(
            clip,
            segmentation,
            DroidVoiceProfile.R2,
            "hello[alert]!.wav",
            rate,
            17u);

        Assert.Equal(2, contexts.Length);
        Assert.Equal(contexts.Length, frames.Length);

        Span<float> targets = stackalloc float[SyllableTargetEncoder.TargetSize];
        for (var i = 0; i < contexts.Length; i++)
        {
            var context = contexts[i];
            Assert.Equal(PhrasePunctuation.Exclamation, context.Punctuation);
            Assert.True(context.EmotionValence < 0f);
            Assert.True(context.EmotionArousal > 0.5f);
            Assert.Equal(rate, context.SampleRate);

            SyllableTargetEncoder.EncodeTargets(frames[i], context, DroidVoiceProfile.R2, targets);
            Assert.Equal(SyllableTargetEncoder.TargetSize, targets.Length);
            foreach (var value in targets)
            {
                Assert.InRange(value, 0f, 1f);
            }
        }
    }

    [Fact]
    public void Mapper_ReadsTrailingQuestionMark()
    {
        var rate = 22050;
        var clip = new WavData(rate, BuildTwoPulses(rate), rate, 1);
        var segmentation = PulseSegmenter.Segment(clip.Samples, clip.SampleRate);

        var (contexts, _) = AcousticContextMapper.Extract(
            clip,
            segmentation,
            DroidVoiceProfile.R2,
            "hello?.wav",
            rate,
            3u);

        Assert.NotEmpty(contexts);
        Assert.All(contexts, context => Assert.Equal(PhrasePunctuation.Question, context.Punctuation));
    }

    [Fact]
    public void Mapper_ReadsTagFromContainingFolder()
    {
        var rate = 22050;
        var clip = new WavData(rate, BuildTwoPulses(rate), rate, 1);
        var segmentation = PulseSegmenter.Segment(clip.Samples, clip.SampleRate);

        var (contexts, _) = AcousticContextMapper.Extract(
            clip,
            segmentation,
            DroidVoiceProfile.R2,
            Path.Combine("corpus", "[sad]", "line.wav"),
            rate,
            11u);

        Assert.NotEmpty(contexts);
        Assert.All(contexts, context =>
        {
            Assert.Equal(PhrasePunctuation.Statement, context.Punctuation);
            Assert.True(context.EmotionValence < 0f);
        });
    }

    [Fact]
    public void Mapper_ReturnsNothingForSilence()
    {
        var rate = 22050;
        var clip = new WavData(rate, new float[rate / 4], rate, 1);
        var segmentation = PulseSegmenter.Segment(clip.Samples, clip.SampleRate);

        var (contexts, frames) = AcousticContextMapper.Extract(
            clip,
            segmentation,
            DroidVoiceProfile.R2,
            "silence.wav",
            rate,
            0u);

        Assert.Empty(contexts);
        Assert.Empty(frames);
    }

    [Fact]
    public void Mapper_RejectsIllegalRenderRate()
    {
        var rate = 22050;
        var clip = new WavData(rate, BuildTwoPulses(rate), rate, 1);
        var segmentation = PulseSegmenter.Segment(clip.Samples, clip.SampleRate);

        Assert.Throws<ArgumentOutOfRangeException>(() => AcousticContextMapper.Extract(
            clip,
            segmentation,
            DroidVoiceProfile.R2,
            "hello.wav",
            4000,
            0u));
    }
}