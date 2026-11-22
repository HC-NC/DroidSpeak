using DroidSpeak.DSP;
using DroidSpeak.NLP;

namespace DroidSpeak.Tests;

public class DroidSynthesizerTests
{
    private const string SampleText = "Привет, R2! Как дела?";

    [Fact]
    public void SameInput_ProducesBitIdenticalAudio()
    {
        var synthesizer = new DroidSynthesizer(seed: 12345);

        var first = synthesizer.Synthesize(SampleText);
        var second = synthesizer.Synthesize(SampleText);

        Assert.Equal(first.Length, second.Length);
        Assert.True(first.AsSpan().SequenceEqual(second));
    }

    [Fact]
    public void DifferentSeed_ChangesAudio()
    {
        var first = new DroidSynthesizer(seed: 1).Synthesize(SampleText);
        var second = new DroidSynthesizer(seed: 2).Synthesize(SampleText);

        Assert.False(first.AsSpan().SequenceEqual(second));
    }

    [Theory]
    [InlineData("Привет, R2! Как дела?")]
    [InlineData("ТРЕВОГА! Перегрев реактора!")]
    [InlineData("Ну вот... опять все сломалось...")]
    [InlineData("Beep boop! System is ready.")]
    public void Output_IsFiniteAndNeverClips(string text)
    {
        var samples = new DroidSynthesizer().Synthesize(text);

        var peak = 0f;
        foreach (var sample in samples)
        {
            Assert.True(float.IsFinite(sample), "Audio contains NaN or infinity.");
            Assert.InRange(sample, -1f, 1f);
            peak = MathF.Max(peak, MathF.Abs(sample));
        }

        Assert.True(peak > 0.05f, "Audio is effectively silent.");
        Assert.True(peak <= 0.9501f, $"Peak {peak} exceeds the limiter ceiling.");
    }

    [Fact]
    public void SpanOverload_MatchesAllocatingOverload()
    {
        var synthesizer = new DroidSynthesizer(seed: 7);
        var required = synthesizer.GetSampleCount(SampleText);
        var destination = new float[required + 64];
        Array.Fill(destination, 7f);

        var written = synthesizer.Synthesize(SampleText, destination);
        var expected = synthesizer.Synthesize(SampleText);

        Assert.Equal(required, written);
        Assert.Equal(required, expected.Length);
        Assert.True(expected.AsSpan().SequenceEqual(destination.AsSpan(0, written)));

        for (var i = written; i < destination.Length; i++)
        {
            Assert.Equal(7f, destination[i]);
        }
    }

    [Fact]
    public void SpanOverload_TruncatesGracefullyWhenTooSmall()
    {
        var destination = new float[128];

        var written = new DroidSynthesizer().Synthesize(SampleText, destination);

        Assert.Equal(destination.Length, written);
        Assert.All(destination, sample =>
        {
            Assert.True(float.IsFinite(sample));
            Assert.InRange(sample, -1f, 1f);
        });
    }

    [Fact]
    public void SampleRate_IsRespected()
    {
        var lowRate = new DroidSynthesizer(sampleRate: 22050);
        var highRate = new DroidSynthesizer(sampleRate: 44100);

        var audio = lowRate.Synthesize("beep");

        Assert.Equal(lowRate.GetSampleCount("beep"), audio.Length);
        Assert.True(lowRate.GetSampleCount("beep") < highRate.GetSampleCount("beep"));
    }

    [Fact]
    public void EmptyText_ProducesEmptyBuffer()
    {
        Assert.Empty(new DroidSynthesizer().Synthesize(string.Empty));
        Assert.Empty(new DroidSynthesizer().Synthesize("   "));
    }

    [Fact]
    public void Audio_StartsAndEndsInSilence()
    {
        var samples = new DroidSynthesizer().Synthesize("beep");

        Assert.NotEmpty(samples);
        Assert.Equal(0f, samples[0]);
        Assert.Equal(0f, samples[^1]);
    }

    [Fact]
    public void InvalidSampleRate_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DroidSynthesizer(sampleRate: 1000));
    }

    [Fact]
    public void NullText_Throws()
    {
        var synthesizer = new DroidSynthesizer();

        Assert.Throws<ArgumentNullException>(() => synthesizer.Synthesize((string)null!));
        Assert.Throws<ArgumentNullException>(() => synthesizer.GetSampleCount((string)null!));
    }

    [Fact]
    public void EmotionalSpeech_AddsWhistlesInTheHighBand()
    {
        var synthesizer = new DroidSynthesizer(seed: 12345);
        var plan = synthesizer.BuildPlan(TextParser.Parse("ТРЕВОГА! Перегрев реактора!"));

        Assert.NotEmpty(plan.Whistles);
        foreach (var whistle in plan.Whistles)
        {
            Assert.InRange(whistle.StartFrequency, 1400f, 2600f);
            Assert.InRange(whistle.EndFrequency, 1400f, 2600f);
            Assert.InRange(whistle.Amplitude, 0.05f, 0.25f);
            Assert.InRange(
                whistle.Length / (double)synthesizer.SampleRate,
                0.03,
                0.09);
        }
    }

    [Fact]
    public void HighEmotion_ProducesMoreWhistlesThanCalmSpeech()
    {
        var synthesizer = new DroidSynthesizer(seed: 7);
        var excited = synthesizer.BuildPlan(TextParser.Parse("ТРЕВОГА! Перегрев реактора!"));
        var calm = synthesizer.BuildPlan(TextParser.Parse("Ну вот... опять все сломалось..."));

        Assert.True(
            excited.Whistles.Count > calm.Whistles.Count,
            $"Expected more whistles for excited speech ({excited.Whistles.Count}) than calm speech ({calm.Whistles.Count}).");
    }

    [Fact]
    public void ShoutedExclamation_UsesStrongerFmAndSoftClipping()
    {
        var synthesizer = new DroidSynthesizer(seed: 42);
        var shouted = synthesizer.BuildPlan(TextParser.Parse("ТРЕВОГА!"));
        var calm = synthesizer.BuildPlan(TextParser.Parse("тревога"));

        Assert.NotEmpty(shouted.Events);
        Assert.All(shouted.Events, note =>
        {
            Assert.True(note.Drive >= 2f, $"Shout drive was {note.Drive}.");
            Assert.True(note.Saturation >= 0.5f, $"Shout saturation was {note.Saturation}.");
            Assert.True(note.FmIndex >= 4f, $"Shout FM index was {note.FmIndex}.");
            Assert.True(note.FmIndex2 > 0f);
        });

        Assert.NotEmpty(calm.Events);
        Assert.All(calm.Events, note =>
        {
            Assert.True(note.Drive < 1.5f, $"Calm drive was {note.Drive}.");
            Assert.True(note.Saturation < 0.3f, $"Calm saturation was {note.Saturation}.");
        });
    }

    [Fact]
    public void LongWord_IsLegatoAndKeepsPhaseContinuity()
    {
        var synthesizer = new DroidSynthesizer(seed: 42);
        var plan = synthesizer.BuildPlan(TextParser.Parse("абракадабрабракадабра"));

        Assert.True(plan.Events.Count >= 8, "Expected a long multi-syllable word.");

        for (var i = 1; i < plan.Events.Count; i++)
        {
            var previous = plan.Events[i - 1];
            var next = plan.Events[i];

            Assert.Equal(previous.StartSample + previous.Length, next.StartSample);
            Assert.Equal(previous.EndFrequency, next.StartFrequency);
            Assert.Equal(previous.Amplitude, next.StartAmplitude);
            Assert.Equal(previous.FmIndex, next.StartFmIndex);
        }
    }

    [Fact]
    public void WordEnvelope_AttacksOnceAndReleasesOnce()
    {
        var synthesizer = new DroidSynthesizer(seed: 42);
        var plan = synthesizer.BuildPlan(TextParser.Parse("абракадабра"));
        var first = plan.Events[0];
        var last = plan.Events[^1];

        Assert.Equal(first.StartSample, first.WordStartSample);
        Assert.Equal(last.StartSample + last.Length, last.WordEndSample);

        Assert.All(plan.Events, note =>
        {
            Assert.Equal(first.WordStartSample, note.WordStartSample);
            Assert.Equal(last.WordEndSample, note.WordEndSample);
            Assert.InRange(note.WordAttack, 0.005f, 0.016f);
            Assert.InRange(note.WordRelease, 0.010f, 0.020f);
        });
    }

    [Fact]
    public void DemoPhrase_KeepsLegatoInsideWordsAndPausesBetweenThem()
    {
        var synthesizer = new DroidSynthesizer(seed: 0);
        var plan = synthesizer.BuildPlan(
            TextParser.Parse("Бип-буп. Не хочу я это делать... Ладно, уже выполняю."));

        var legatoJoints = 0;
        var pauseJoints = 0;

        for (var i = 1; i < plan.Events.Count; i++)
        {
            var previous = plan.Events[i - 1];
            var next = plan.Events[i];
            var gap = next.StartSample - (previous.StartSample + previous.Length);

            if (previous.WordStartSample == next.WordStartSample)
            {
                legatoJoints++;
                Assert.Equal(0, gap);
                Assert.Equal(previous.EndFrequency, next.StartFrequency);
            }
            else
            {
                pauseJoints++;
                Assert.True(gap > 0, "Expected a pause between words or hyphen parts.");
            }
        }

        Assert.True(legatoJoints > 0);
        Assert.True(pauseJoints > 0);
    }

    [Fact]
    public void HyphenatedWord_KeepsPauseBetweenParts()
    {
        var synthesizer = new DroidSynthesizer(seed: 42);
        var plan = synthesizer.BuildPlan(TextParser.Parse("Бип-буп"));

        Assert.True(plan.Events.Count >= 2, "Expected both hyphenated parts.");

        var first = plan.Events[0];
        var second = plan.Events[1];
        var gapSamples = second.StartSample - (first.StartSample + first.Length);

        Assert.True(
            gapSamples > synthesizer.SampleRate * 0.010,
            $"Expected a hyphen pause, got {gapSamples} samples.");
        Assert.NotEqual(first.WordStartSample, second.WordStartSample);
    }
}
