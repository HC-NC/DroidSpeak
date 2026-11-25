using DroidSpeak.DSP;
using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.Tests;

public class ProsodyTests
{
    private const string SampleText = "Hello, world! R2-D2 is... fine? BEEP BOOP!";

    [Fact]
    public void SpanSynthesis_IsAllocationFreeOnWarmPath()
    {
        var synthesizer = new DroidSynthesizer(seed: 7, profile: DroidVoiceProfile.R2);
        var phrases = TextParser.Parse(SampleText);
        var destination = new float[synthesizer.GetSampleCount(phrases)];

        // Warm up the array pool, tier-0 JIT and static caches.
        synthesizer.Synthesize(phrases, destination);
        synthesizer.Synthesize(phrases, destination);

        var before = GC.GetAllocatedBytesForCurrentThread();
        var written = synthesizer.Synthesize(phrases, destination);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(destination.Length, written);
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void GetSampleCount_IsAllocationFreeOnWarmPath()
    {
        var synthesizer = new DroidSynthesizer(seed: 7, profile: DroidVoiceProfile.R2);
        var phrases = TextParser.Parse(SampleText);

        synthesizer.GetSampleCount(phrases);
        synthesizer.GetSampleCount(phrases);

        var before = GC.GetAllocatedBytesForCurrentThread();
        var count = synthesizer.GetSampleCount(phrases);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.True(count > 0);
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void SpanOverload_MatchesAllocatingPhraseOverload()
    {
        var synthesizer = new DroidSynthesizer(seed: 11, profile: DroidVoiceProfile.Grumpy);
        var phrases = TextParser.Parse(SampleText);

        var expected = synthesizer.Synthesize(phrases);
        var destination = new float[expected.Length];

        var written = synthesizer.Synthesize(phrases, destination);

        Assert.Equal(expected.Length, written);
        Assert.Equal(expected, destination);
    }

    [Fact]
    public void Frames_ContainOneEntryPerSyllableInSpeakingOrder()
    {
        var synthesizer = new DroidSynthesizer(seed: 3);
        var phrases = TextParser.Parse(SampleText);
        var expected = SyllableContextBuilder.Count(phrases);

        var plan = synthesizer.BuildPlan(phrases);

        Assert.True(expected > 0);
        Assert.Equal(expected, plan.Frames.Length);
        Assert.Equal(expected, plan.Events.Count);

        for (var i = 0; i < plan.Frames.Length; i++)
        {
            var frame = plan.Frames[i];
            Assert.True(frame.LengthSamples > 0);
            Assert.InRange(frame.StartFreq, synthesizer.Profile.MinFreq, synthesizer.Profile.MaxFreq);
            Assert.InRange(frame.EndFreq, synthesizer.Profile.MinFreq, synthesizer.Profile.MaxFreq);

            if (i > 0)
            {
                var previous = plan.Frames[i - 1];
                Assert.True(frame.StartSample >= previous.StartSample + previous.LengthSamples);
            }
        }
    }

    [Fact]
    public void CustomPredictor_ReplacesTheDefaultProsody()
    {
        var predictor = new FixedPredictor();
        var synthesizer = new DroidSynthesizer(seed: 1, predictor: predictor);
        var phrases = TextParser.Parse("Hello there");
        var syllables = SyllableContextBuilder.Count(phrases);
        var expected = syllables * FixedPredictor.SyllableSamples
            + (int)MathF.Round(0.15f * synthesizer.SampleRate);

        var audio = synthesizer.Synthesize(phrases);
        var count = synthesizer.GetSampleCount(phrases);

        Assert.Equal(expected, audio.Length);
        Assert.Equal(expected, count);
        Assert.Equal(2, predictor.Calls);
    }

    [Fact]
    public void Predictor_RejectsFrameBufferSmallerThanContexts()
    {
        var contexts = new SyllableContext[2];
        var frames = new SyllableFrame[1];

        Assert.Throws<ArgumentException>(() =>
            RuleBasedProsodyPredictor.Shared.Predict(contexts, frames, DroidVoiceProfile.R2));
    }

    [Fact]
    public void Normalized_ReturnsSameInstanceWhenAlreadyNormalized()
    {
        var profile = DroidVoiceProfile.R2;

        Assert.Same(profile, profile.Normalized());
    }

    private sealed class FixedPredictor : IProsodyPredictor
    {
        public const int SyllableSamples = 100;

        public int Calls { get; private set; }

        public void Predict(
            ReadOnlySpan<SyllableContext> contexts,
            Span<SyllableFrame> frames,
            DroidVoiceProfile profile)
        {
            Calls++;

            for (var i = 0; i < contexts.Length; i++)
            {
                var start = i * SyllableSamples;
                frames[i] = new SyllableFrame
                {
                    StartSample = start,
                    LengthSamples = SyllableSamples,
                    StartFreq = 440f,
                    EndFreq = 440f,
                    StartGain = 0.2f,
                    Gain = 0.2f,
                    FmRatio = 2f,
                    FmRatio2 = 3.07f,
                    AttackTime = 0.01f,
                    DecayTime = 0.02f,
                    ReleaseTime = 0.016f,
                    SustainLevel = 0.74f,
                    WordStartSample = start,
                    WordEndSample = start + SyllableSamples,
                    StartsLegatoSegment = true,
                };
            }
        }
    }
}
