using DroidSpeak;
using DroidSpeak.Trainer;
using DroidSpeak.Trainer.Audio;
using DroidSpeak.Trainer.Datasets;

namespace DroidSpeak.Tests;

/// <summary>
/// Deterministic synthetic reference clips and the extraction pipeline around them. The tones are
/// pure sines separated by a silence well above the word gap, so the segmenter sees exactly the
/// number of pulses the fixture asked for on every platform.
/// </summary>
internal static class AcousticFixtures
{
    /// <summary>Analysis and render rate used by the acoustic tests.</summary>
    public const int Rate = 22050;

    private static readonly double[] Frequencies = [400.0, 500.0, 650.0, 800.0, 380.0];

    /// <summary>Builds a mono clip of <paramref name="pulseCount"/> tone bursts separated by silence.</summary>
    public static float[] BuildTone(
        int sampleRate,
        int pulseCount,
        uint seed,
        int pulseMilliseconds = 90,
        int gapMilliseconds = 130,
        int leadMilliseconds = 60,
        int trailMilliseconds = 180)
    {
        var random = new RandomSource(seed | 1u);
        var pulseSamples = sampleRate * pulseMilliseconds / 1000;
        var gapSamples = sampleRate * gapMilliseconds / 1000;
        var leadSamples = sampleRate * leadMilliseconds / 1000;
        var trailSamples = sampleRate * trailMilliseconds / 1000;
        var total = leadSamples + (pulseCount * pulseSamples) +
            ((pulseCount - 1) * gapSamples) + trailSamples;
        var samples = new float[total];
        var fade = Math.Max(1, sampleRate / 200);
        var offset = leadSamples;

        for (var pulse = 0; pulse < pulseCount; pulse++)
        {
            var frequency = Frequencies[random.NextInt(Frequencies.Length)];
            var amplitude = random.NextFloat(0.35f, 0.50f);

            for (var i = 0; i < pulseSamples; i++)
            {
                var envelope = 1.0;
                if (i < fade)
                {
                    envelope = i / (double)fade;
                }
                else if (i >= pulseSamples - fade)
                {
                    envelope = (pulseSamples - i) / (double)fade;
                }

                var time = i / (double)sampleRate;
                samples[offset + i] =
                    (float)(amplitude * envelope * Math.Sin(2.0 * Math.PI * frequency * time));
            }

            offset += pulseSamples + gapSamples;
        }

        return samples;
    }

    /// <summary>Wraps raw mono samples as a decoded clip.</summary>
    public static WavData ToAudio(float[] samples, int sampleRate = Rate) =>
        new(sampleRate, samples, sampleRate, 1);

    /// <summary>Segments and extracts one clip through the real pipeline.</summary>
    public static AcousticClip ExtractClip(
        string path,
        float[] samples,
        DroidVoiceProfile profile,
        int renderRate = Rate,
        uint seed = 20260107u,
        PulseSegmenter.Options? segmenter = null)
    {
        var audio = ToAudio(samples);
        var segmentation = PulseSegmenter.Segment(audio.Samples, audio.SampleRate, segmenter);
        var (contexts, frames) = AcousticContextMapper.Extract(
            audio, segmentation, profile, path, renderRate, seed);
        return new AcousticClip(AcousticContextMapper.ClipId(path), path, contexts, frames);
    }

    /// <summary>Builds a dataset of <paramref name="clipCount"/> synthetic clips.</summary>
    public static AcousticDataset BuildDataset(
        int clipCount,
        DroidVoiceProfile profile,
        int renderRate = Rate,
        uint seed = 20260107u,
        int pulseCount = 3)
    {
        var clips = new List<AcousticClip>(clipCount);

        for (var clip = 0; clip < clipCount; clip++)
        {
            var samples = BuildTone(Rate, pulseCount, seed + (uint)clip);
            clips.Add(ExtractClip($"clip-{clip:D2}.wav", samples, profile, renderRate, seed));
        }

        return new AcousticDataset(clips, renderRate, seed);
    }
}