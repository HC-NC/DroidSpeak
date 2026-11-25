using DroidSpeak;
using DroidSpeak.DSP;
using DroidSpeak.Neural;
using DroidSpeak.NLP;
using DroidSpeak.Prosody;
using DroidSpeak.Trainer.Datasets;

namespace DroidSpeak.Tests;

/// <summary>
/// Guards the properties the distillation corpus has to keep: it is reproducible from its seed, the
/// requested row count is exact, the fixed coverage phase really spans every phonetic and emotional
/// case, and every encoded target survives a decode/encode round trip.
/// </summary>
public class DistillationCorpusTests
{
    private const int SampleRate = DroidSynthesizer.DefaultSampleRate;

    [Fact]
    public void Build_IsReproducibleFromTheSeed()
    {
        var first = DistillationCorpus.Build(300, DroidVoiceProfile.R2, 1234u, SampleRate);
        var second = DistillationCorpus.Build(300, DroidVoiceProfile.R2, 1234u, SampleRate);

        Assert.Equal(first.Count, second.Count);
        Assert.Equal(first.FeatureStorage, second.FeatureStorage);
        Assert.Equal(first.TargetStorage, second.TargetStorage);

        var other = DistillationCorpus.Build(300, DroidVoiceProfile.R2, 4321u, SampleRate);
        Assert.NotEqual(first.TargetStorage, other.TargetStorage);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(97)]
    [InlineData(500)]
    public void Build_ReturnsExactlyTheRequestedRowCount(int samples)
    {
        var data = DistillationCorpus.BuildDetailed(samples, DroidVoiceProfile.R2, 7u, SampleRate);

        Assert.Equal(samples, data.Set.Count);
        Assert.Equal(samples, data.Contexts.Length);
        Assert.Equal(SyllableFeatureEncoder.InputSize, data.Set.InputSize);
        Assert.Equal(SyllableTargetEncoder.TargetSize, data.Set.OutputSize);
        Assert.Equal(samples * SyllableFeatureEncoder.InputSize, data.Set.FeatureStorage.Length);
        Assert.Equal(samples * SyllableTargetEncoder.TargetSize, data.Set.TargetStorage.Length);
    }

    [Fact]
    public void Build_CoversEveryPhoneticAndEmotionalCase()
    {
        var prefixRows = CorpusVocabulary.CoverageUtterances.Sum(text =>
            SyllableContextBuilder.Count(TextParser.Parse(text)));
        var data = DistillationCorpus.BuildDetailed(prefixRows, DroidVoiceProfile.R2, 9u, SampleRate);

        var vowelClasses = new HashSet<int>();
        var punctuations = new HashSet<PhrasePunctuation>();
        var cells = new HashSet<(int Valence, int Arousal)>();
        var shout = false;
        var hyphen = false;
        var comma = false;
        var apostrophe = CorpusVocabulary.CoverageUtterances.Any(u => u.Contains('\''));

        foreach (var context in data.Contexts)
        {
            vowelClasses.Add(SyllableFeatureEncoder.VowelClass(context.Nucleus));
            punctuations.Add(context.Punctuation);
            shout |= context.IsShout;
            hyphen |= context.IsHyphenBoundary;
            comma |= context.PreviousWordHadComma;

            cells.Add((
                context.EmotionValence >= 0.10f ? 1 : context.EmotionValence <= -0.10f ? -1 : 0,
                context.EmotionArousal >= 0.15f ? 1 : context.EmotionArousal <= -0.15f ? -1 : 0));
        }

        Assert.Equal(6, vowelClasses.Count);
        Assert.Equal(4, punctuations.Count);
        Assert.True(shout, "No shout reached the corpus.");
        Assert.True(hyphen, "No hyphen boundary reached the corpus.");
        Assert.True(comma, "No comma pause reached the corpus.");
        Assert.True(apostrophe, "The coverage list lost its apostrophe contractions.");
        Assert.Equal(9, cells.Count);
    }

    [Fact]
    public void Build_KeepsEveryValueFiniteAndNormalized()
    {
        var data = DistillationCorpus.Build(400, DroidVoiceProfile.R2, 11u, SampleRate);

        foreach (var value in data.FeatureStorage)
        {
            Assert.True(float.IsFinite(value));
            Assert.InRange(value, 0f, 1f);
        }

        foreach (var value in data.TargetStorage)
        {
            Assert.True(float.IsFinite(value));
            Assert.InRange(value, 0f, 1f);
        }
    }

    [Fact]
    public void Targets_DecodeBackToTheTeacherFrames()
    {
        var data = DistillationCorpus.BuildDetailed(250, DroidVoiceProfile.R2, 13u, SampleRate);

        Span<float> roundTrip = stackalloc float[SyllableTargetEncoder.TargetSize];

        for (var row = 0; row < data.Set.Count; row++)
        {
            var context = data.Contexts[row];
            var targets = data.Set.Targets(row);

            SyllableTargetDecoder.DecodeTargets(targets, context, DroidVoiceProfile.R2, out var decoded);
            var frame = Frame(decoded, context);
            SyllableTargetEncoder.EncodeTargets(frame, context, DroidVoiceProfile.R2, roundTrip);

            for (var i = 0; i < roundTrip.Length; i++)
            {
                Assert.True(
                    MathF.Abs(targets[i] - roundTrip[i]) <= 3e-3f,
                    $"Row {row}, output {i}: {targets[i]:F5} re-encoded to {roundTrip[i]:F5}.");
            }
        }
    }

    [Fact]
    public void Build_RejectsUnusableArguments()
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => DistillationCorpus.Build(0, DroidVoiceProfile.R2, 1u, SampleRate));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => DistillationCorpus.Build(10, DroidVoiceProfile.R2, 1u, 4000));
        Assert.Throws<ArgumentNullException>(
            () => DistillationCorpus.Build(10, null!, 1u, SampleRate));
    }

    private static SyllableFrame Frame(in PredictedFrame decoded, in SyllableContext context) =>
        new()
        {
            LengthSamples = decoded.LengthSamples,
            DurationSeconds = decoded.LengthSamples / (float)context.SampleRate,
            StartFreq = decoded.StartFrequency,
            MidFreq = decoded.MidFrequency,
            EndFreq = decoded.EndFrequency,
            Gain = decoded.Gain,
            FmDepth = decoded.FmDepth,
            FmRatio = decoded.FmRatio,
            GrowlIntensity = decoded.Growl,
            NoiseMix = decoded.Noise,
            RingModMix = decoded.RingMod,
            FormantF1 = decoded.Formant1,
            FormantF2 = decoded.Formant2,
            FormantMix = decoded.FormantBlend,
            AttackTime = decoded.AttackSeconds,
            ReleaseTime = decoded.ReleaseSeconds,
            Saturation = decoded.Saturation,
            Drive = decoded.Drive,
            PauseAfterSamples = decoded.PauseAfterSamples,
            PauseAfterSeconds = decoded.PauseAfterSamples / (float)context.SampleRate,
        };
}
