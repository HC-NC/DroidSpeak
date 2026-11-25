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
    public void V05Pipeline_StaysAllocationFreeWithEveryStageMaxed()
    {
        // Saturate every v0.5 stage at once: formant bank, ring modulator, soft clipper and the
        // stack-allocated early-reflection line must all stay off the heap on the warm path.
        var profile = new DroidVoiceProfile
        {
            FormantMix = 1f,
            RingModMix = 1f,
            SaturationDrive = 4f,
            RoomMix = 1f,
            PitchCurve = 1f,
        };
        var synthesizer = new DroidSynthesizer(seed: 13, profile: profile);
        var phrases = TextParser.Parse(SampleText);
        var destination = new float[synthesizer.GetSampleCount(phrases)];

        synthesizer.Synthesize(phrases, destination);
        synthesizer.Synthesize(phrases, destination);

        var before = GC.GetAllocatedBytesForCurrentThread();
        var written = synthesizer.Synthesize(phrases, destination);
        var count = synthesizer.GetSampleCount(phrases);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(destination.Length, written);
        Assert.Equal(destination.Length, count);
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

    [Fact]
    public void Predictor_KeepsTheLegacyPathWhenV05ParametersAreZero()
    {
        // Formants, ring modulation and pitch curvature must all stay silent for a neutral
        // profile, because that is what reproduces the v0.4 audio bit for bit.
        var profile = DroidVoiceProfile.R2 with
        {
            FormantMix = 0f,
            RingModMix = 0f,
            SaturationDrive = 0f,
            RoomMix = 0f,
            PitchCurve = 0f,
        };
        var contexts = new[] { CreateContext('и') };
        var frames = new SyllableFrame[contexts.Length];

        RuleBasedProsodyPredictor.Shared.Predict(contexts, frames, profile);

        Assert.Equal(0f, frames[0].MidFreq);
        Assert.Equal(0f, frames[0].FormantMix);
        Assert.Equal(0f, frames[0].RingModMix);
        Assert.Equal(0f, frames[0].AttackChirp);
    }

    [Fact]
    public void Predictor_MapsVowelFormantsToTheNucleus()
    {
        var contexts = new[] { CreateContext('и'), CreateContext('о') };
        var frames = new SyllableFrame[contexts.Length];

        RuleBasedProsodyPredictor.Shared.Predict(contexts, frames, DroidVoiceProfile.R2);

        // A close front vowel has a low F1 and a high F2; a back rounded vowel is the opposite.
        Assert.True(frames[0].FormantF1 < frames[1].FormantF1);
        Assert.True(frames[0].FormantF2 > frames[1].FormantF2);
        Assert.True(frames[0].FormantMix > 0f);
    }

    [Fact]
    public void Predictor_DucksFormantsForSibilantAndConsonantOnlySyllables()
    {
        var contexts = new[]
        {
            CreateContext('а'),
            CreateContext('а') with { HasSibilantOnset = true },
            CreateContext('\0'),
        };
        var frames = new SyllableFrame[contexts.Length];

        RuleBasedProsodyPredictor.Shared.Predict(contexts, frames, DroidVoiceProfile.R2);

        Assert.True(frames[1].FormantMix < frames[0].FormantMix);
        Assert.True(frames[2].FormantMix < frames[0].FormantMix);
    }

    [Fact]
    public void Predictor_CurvesThePitchContourWhenEnabled()
    {
        var contexts = new[] { CreateContext('а') };
        var frames = new SyllableFrame[contexts.Length];

        RuleBasedProsodyPredictor.Shared.Predict(contexts, frames, DroidVoiceProfile.R2);

        var frame = frames[0];
        var geometric = MathF.Sqrt(frame.StartFreq * frame.EndFreq);

        Assert.NotEqual(0f, frame.MidFreq);
        Assert.True(MathF.Abs(frame.MidFreq - geometric) > 1f);
        Assert.InRange(frame.MidFreq, DroidVoiceProfile.R2.MinFreq, DroidVoiceProfile.R2.MaxFreq);
    }

    [Fact]
    public void Predictor_ChirpIsDeterministicAndSeedDriven()
    {
        var contexts = new[] { CreateContext('а', wordHash: 0x51ED_270Bu) };

        var first = new SyllableFrame[1];
        var second = new SyllableFrame[1];
        RuleBasedProsodyPredictor.Shared.Predict(contexts, first, DroidVoiceProfile.R2);
        RuleBasedProsodyPredictor.Shared.Predict(contexts, second, DroidVoiceProfile.R2);

        Assert.NotEqual(0f, first[0].AttackChirp);
        Assert.Equal(first[0].AttackChirp, second[0].AttackChirp);
        Assert.Equal(first[0].Seed, second[0].Seed);
    }

    [Fact]
    public void Predictor_KeepsV05FieldsBoundedAcrossARichUtterance()
    {
        var synthesizer = new DroidSynthesizer(seed: 42, profile: DroidVoiceProfile.Grumpy);
        var plan = synthesizer.BuildPlan(TextParser.Parse(SampleText));

        foreach (var frame in plan.Frames)
        {
            Assert.InRange(frame.FormantMix, 0f, 1f);
            Assert.InRange(frame.RingModMix, 0f, 1f);
            Assert.InRange(frame.AttackChirp, -0.25f, 0.25f);
            Assert.True(frame.FormantF1 > 0f);
            Assert.True(frame.FormantF2 > frame.FormantF1);
            if (frame.MidFreq != 0f)
            {
                Assert.InRange(
                    frame.MidFreq,
                    synthesizer.Profile.MinFreq,
                    synthesizer.Profile.MaxFreq);
            }
        }
    }

    private static SyllableContext CreateContext(
        char nucleus,
        uint wordHash = 0x1234_5678u,
        uint seed = 99u) => new()
        {
            SampleRate = 24000,
            GlobalSeed = seed,
            WordHash = wordHash,
            PhraseIndex = 0,
            WordIndexInPhrase = 0,
            SyllableIndex = 0,
            WordSyllableCount = 1,
            PhraseSyllableCount = 1,
            WordTextLength = 4,
            TempoScale = 1f,
            EmotionValence = 0f,
            EmotionArousal = 0f,
            Punctuation = PhrasePunctuation.Statement,
            IsFirstSpokenPhrase = true,
            Nucleus = nucleus,
        };

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
