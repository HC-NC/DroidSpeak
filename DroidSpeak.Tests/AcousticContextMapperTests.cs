using DroidSpeak;
using DroidSpeak.Neural;
using DroidSpeak.Trainer;
using DroidSpeak.Trainer.Audio;

namespace DroidSpeak.Tests;

/// <summary>
/// The acoustic front end: a clip's segmentation must become exactly one context and one frame per
/// pulse, the result must be reproducible, and every target derived from it must stay inside the
/// network's [0, 1] output contract.
/// </summary>
public class AcousticContextMapperTests
{
    private static readonly DroidVoiceProfile Profile = DroidVoiceProfile.R2;

    [Fact]
    public void Extract_ProducesOneContextAndFramePerPulse()
    {
        var samples = AcousticFixtures.BuildTone(AcousticFixtures.Rate, pulseCount: 3, seed: 1);
        var clip = AcousticFixtures.ExtractClip("clip.wav", samples, Profile, AcousticFixtures.Rate, 7);

        Assert.Equal(3, clip.SyllableCount);
        Assert.Equal(clip.Contexts.Length, clip.Frames.Length);

        for (var i = 0; i < clip.SyllableCount; i++)
        {
            var context = clip.Contexts[i];
            var frame = clip.Frames[i];

            Assert.Equal(AcousticFixtures.Rate, context.SampleRate);
            Assert.Equal(clip.SyllableCount, context.PhraseSyllableCount);
            Assert.Equal(7u, context.GlobalSeed);
            Assert.True(context.WordSyllableCount >= 1);
            Assert.InRange(context.SyllableIndex, 0, context.WordSyllableCount - 1);

            Assert.True(frame.LengthSamples >= 1);
            Assert.True(frame.DurationSeconds > 0f);
            Assert.True(frame.StartSample >= 0);
            Assert.True(frame.WordStartSample <= frame.WordEndSample);
            Assert.InRange(frame.Gain, SyllableTargetDecoder.MinGain, SyllableTargetDecoder.MaxGain);
            Assert.InRange(frame.StartFreq, Profile.MinFreq, Profile.MaxFreq);
            Assert.InRange(frame.EndFreq, Profile.MinFreq, Profile.MaxFreq);
        }

        // Each pulse is separated by more than the word gap, so every pulse is its own single-syllable
        // word, and only the final pulse of the clip may carry the phrase-final pause.
        for (var i = 0; i < clip.SyllableCount; i++)
        {
            Assert.Equal(1, clip.Contexts[i].WordSyllableCount);
            Assert.Equal(i, clip.Contexts[i].WordIndexInPhrase);
        }

        Assert.Equal(0, clip.Frames[0].PauseAfterSamples);
        Assert.Equal(0, clip.Frames[1].PauseAfterSamples);
    }

    [Fact]
    public void Extract_IsDeterministic()
    {
        var samples = AcousticFixtures.BuildTone(AcousticFixtures.Rate, pulseCount: 4, seed: 21);
        var first = AcousticFixtures.ExtractClip("same.wav", samples, Profile, AcousticFixtures.Rate, 99);
        var second = AcousticFixtures.ExtractClip("same.wav", samples, Profile, AcousticFixtures.Rate, 99);

        Assert.Equal(first.SyllableCount, second.SyllableCount);
        Assert.Equal(first.ClipId, second.ClipId);

        for (var i = 0; i < first.SyllableCount; i++)
        {
            AcousticAssert.ContextEqual(first.Contexts[i], second.Contexts[i]);
            AcousticAssert.FrameEqual(first.Frames[i], second.Frames[i]);
        }

        Assert.Equal(
            AcousticContextMapper.ClipId("same.wav"),
            AcousticContextMapper.ClipId("same.wav"));
    }

    [Fact]
    public void Extract_SilentClip_ReturnsEmptyArrays()
    {
        var audio = AcousticFixtures.ToAudio(new float[AcousticFixtures.Rate]);
        var segmentation = PulseSegmenter.Segment(audio.Samples, audio.SampleRate);

        var (contexts, frames) = AcousticContextMapper.Extract(
            audio, segmentation, Profile, "silent.wav", AcousticFixtures.Rate, 1);

        Assert.Empty(contexts);
        Assert.Empty(frames);
    }

    [Fact]
    public void ExtractedTargets_AreFiniteAndInsideTheUnitRange()
    {
        var samples = AcousticFixtures.BuildTone(AcousticFixtures.Rate, pulseCount: 4, seed: 5);
        var clip = AcousticFixtures.ExtractClip("targets.wav", samples, Profile, AcousticFixtures.Rate, 5);

        Assert.True(clip.SyllableCount > 0);
        Span<float> targets = stackalloc float[SyllableTargetEncoder.TargetSize];

        for (var i = 0; i < clip.SyllableCount; i++)
        {
            SyllableTargetEncoder.EncodeTargets(
                clip.Frames[i], clip.Contexts[i], Profile, targets);

            foreach (var target in targets)
            {
                Assert.True(float.IsFinite(target));
                Assert.InRange(target, 0f, 1f);
            }
        }
    }

    [Fact]
    public void Extract_RejectsOutOfRangeRenderRate()
    {
        var samples = AcousticFixtures.BuildTone(AcousticFixtures.Rate, pulseCount: 1, seed: 1);
        var audio = AcousticFixtures.ToAudio(samples);
        var segmentation = PulseSegmenter.Segment(audio.Samples, audio.SampleRate);

        Assert.Throws<ArgumentOutOfRangeException>(() => AcousticContextMapper.Extract(
            audio, segmentation, Profile, "clip.wav", renderRate: 4000, seed: 1));
    }

    [Fact]
    public void SplitByClip_KeepsEveryClipWhole()
    {
        var dataset = AcousticFixtures.BuildDataset(6, Profile, AcousticFixtures.Rate, 99);
        var random = new RandomSource(123);
        dataset.SplitByClip(
            Profile, 0.34f, ref random, out var training, out var validation);

        Assert.Equal(dataset.SyllableCount, training.Count + validation.Count);
        Assert.True(training.Count > 0);
        Assert.True(validation.Count > 0);

        // round(6 * 0.34) = 2 clips held out, so the validation row count must be the sum of exactly
        // two clip row counts. A row-level split would almost never satisfy this.
        var counts = dataset.ClipRowCounts.ToArray();
        var formedByTwoClips = false;
        for (var a = 0; a < counts.Length; a++)
        {
            for (var b = a + 1; b < counts.Length; b++)
            {
                if (counts[a] + counts[b] == validation.Count)
                {
                    formedByTwoClips = true;
                }
            }
        }

        Assert.True(formedByTwoClips, "The validation rows do not line up with a pair of whole clips.");
    }

    [Fact]
    public void SplitByClip_IsReproducibleFromTheSeed()
    {
        var dataset = AcousticFixtures.BuildDataset(5, Profile, AcousticFixtures.Rate, 7);

        var firstRandom = new RandomSource(0x5A17C0DEu);
        dataset.SplitByClip(Profile, 0.4f, ref firstRandom, out _, out var firstValidation);

        var secondRandom = new RandomSource(0x5A17C0DEu);
        dataset.SplitByClip(Profile, 0.4f, ref secondRandom, out _, out var secondValidation);

        Assert.Equal(firstValidation.Count, secondValidation.Count);
        for (var row = 0; row < firstValidation.Count; row++)
        {
            Assert.Equal(
                firstValidation.Features(row).ToArray(),
                secondValidation.Features(row).ToArray());
        }
    }
}