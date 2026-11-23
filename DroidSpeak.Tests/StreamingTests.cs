using DroidSpeak.DSP;
using DroidSpeak.Music;
using DroidSpeak.NLP;

namespace DroidSpeak.Tests;

public sealed class StreamingTests
{
    private const string SampleText =
        "Beep-beep! Hello there, systems are online. ТРЕВОГА! Перегрев реактора! Oh no... everything is broken again...";

    private static readonly DroidScore SampleScore = new(
    [
        new ScoreNote(0.0f, 0.5f, 523.25f, 0.90f),                      // Lead C5
        new ScoreNote(0.0f, 0.5f, 261.63f, 0.80f),                      // Harmony C4
        new ScoreNote(0.0f, 0.20f, 0f, 0.70f, IsPercussion: true),      // Relay click
        new ScoreNote(0.55f, 0.60f, 587.33f, 0.85f),                    // Lead D5 (portamento)
        new ScoreNote(0.55f, 0.60f, 293.66f, 0.70f),                    // Harmony D4
        new ScoreNote(1.20f, 0.40f, 659.25f, 0.80f),                    // Lead E5
    ]);

    [Fact]
    public void SpeechStream_ProducesExactLengthAndIsDeterministicAfterReset()
    {
        var synthesizer = new DroidSynthesizer(seed: 4242, profile: DroidVoiceProfile.R2);
        var expected = synthesizer.GetSampleCount(SampleText);
        var stream = synthesizer.CreateStream(SampleText);

        Assert.Equal(expected, stream.TotalSamples);
        Assert.Equal(DroidSynthesizer.DefaultSampleRate, stream.SampleRate);
        Assert.Equal(0, stream.Position);
        Assert.False(stream.IsCompleted);

        var first = ReadAll(stream.TotalSamples, 256, buffer => stream.Read(buffer));
        Assert.Equal(expected, first.Length);
        Assert.Equal(expected, stream.Position);
        Assert.True(stream.IsCompleted);
        Assert.Equal(0, stream.Read(new float[256]));
        AssertAudioIsFiniteAndBounded(first);

        stream.Reset();
        Assert.Equal(0, stream.Position);
        Assert.False(stream.IsCompleted);

        var second = ReadAll(stream.TotalSamples, 256, buffer => stream.Read(buffer));
        Assert.Equal(first, second);
    }

    [Fact]
    public void SpeechStream_MatchesPhraseOverloadAndIsChunkSizeInvariant()
    {
        var synthesizer = new DroidSynthesizer(seed: 17);
        var phrases = TextParser.Parse(SampleText);

        var fromText = synthesizer.CreateStream(SampleText);
        var fromPhrases = synthesizer.CreateStream(phrases);

        Assert.Equal(synthesizer.GetSampleCount(phrases), fromPhrases.TotalSamples);
        Assert.Equal(fromText.TotalSamples, fromPhrases.TotalSamples);

        // Chunk boundaries must be invisible: every chunk size yields the same continuous audio.
        var small = ReadAll(fromText.TotalSamples, 128, buffer => fromText.Read(buffer));
        var large = ReadAll(fromPhrases.TotalSamples, 4096, buffer => fromPhrases.Read(buffer));

        Assert.Equal(small, large);
    }

    [Fact]
    public void SpeechStream_EmptyTextCompletesImmediately()
    {
        var stream = new DroidSynthesizer().CreateStream("   ");

        Assert.Equal(0, stream.TotalSamples);
        Assert.True(stream.IsCompleted);
        Assert.Equal(0, stream.Read(new float[64]));
    }

    [Fact]
    public void MusicStream_Polyphonic_ProducesExactLengthAndIsDeterministicAfterReset()
    {
        var synthesizer = new DroidMusicSynthesizer(profile: DroidVoiceProfile.R2);
        var expected = synthesizer.GetSampleCount(SampleScore);
        var stream = synthesizer.CreateStream(SampleScore);

        Assert.Equal(expected, stream.TotalSamples);
        Assert.False(stream.IsCompleted);

        var first = ReadAll(stream.TotalSamples, 512, buffer => stream.Read(buffer));
        Assert.Equal(expected, first.Length);
        Assert.True(stream.IsCompleted);
        Assert.Equal(0, stream.Read(new float[512]));
        AssertAudioIsFiniteAndBounded(first);

        stream.Reset();
        Assert.Equal(0, stream.Position);
        Assert.False(stream.IsCompleted);

        var second = ReadAll(stream.TotalSamples, 512, buffer => stream.Read(buffer));
        Assert.Equal(first, second);
    }

    [Fact]
    public void MusicStream_SoloMelodyOnly_ProducesExactLengthBoundedAudio()
    {
        var synthesizer = new DroidMusicSynthesizer(profile: DroidVoiceProfile.R2);
        var expected = synthesizer.GetSampleCount(SampleScore);

        var polyphonic = synthesizer.CreateStream(SampleScore, soloMelodyOnly: false);
        var solo = synthesizer.CreateStream(SampleScore, soloMelodyOnly: true);

        Assert.Equal(expected, solo.TotalSamples);

        var polySamples = ReadAll(polyphonic.TotalSamples, 512, buffer => polyphonic.Read(buffer));
        var soloSamples = ReadAll(solo.TotalSamples, 512, buffer => solo.Read(buffer));

        AssertAudioIsFiniteAndBounded(soloSamples);
        Assert.NotEqual(polySamples, soloSamples);

        solo.Reset();
        Assert.Equal(0, solo.Position);
        var repeated = ReadAll(solo.TotalSamples, 512, buffer => solo.Read(buffer));
        Assert.Equal(soloSamples, repeated);
    }

    [Fact]
    public void MusicStream_IsChunkSizeInvariant()
    {
        var synthesizer = new DroidMusicSynthesizer(profile: DroidVoiceProfile.R2);

        var smallStream = synthesizer.CreateStream(SampleScore);
        var largeStream = synthesizer.CreateStream(SampleScore);

        var small = ReadAll(smallStream.TotalSamples, 64, buffer => smallStream.Read(buffer));
        var large = ReadAll(largeStream.TotalSamples, 8192, buffer => largeStream.Read(buffer));

        Assert.Equal(small, large);
    }

    [Fact]
    public void MusicStream_EmptyScoreCompletesImmediately()
    {
        var stream = new DroidMusicSynthesizer().CreateStream(new DroidScore([]));

        Assert.Equal(0, stream.TotalSamples);
        Assert.True(stream.IsCompleted);
        Assert.Equal(0, stream.Read(new float[64]));
    }

    private static float[] ReadAll(int totalSamples, int chunkSize, Func<float[], int> read)
    {
        Assert.True(totalSamples > 0);
        Assert.True(chunkSize > 0);

        var result = new float[totalSamples];
        var buffer = new float[chunkSize];
        var offset = 0;

        while (offset < totalSamples)
        {
            var written = read(buffer);
            Assert.InRange(written, 1, chunkSize);
            Assert.True(offset + written <= totalSamples);
            buffer.AsSpan(0, written).CopyTo(result.AsSpan(offset));
            offset += written;
        }

        Assert.Equal(totalSamples, offset);
        return result;
    }

    private static void AssertAudioIsFiniteAndBounded(float[] samples)
    {
        Assert.NotEmpty(samples);

        var peak = 0f;
        foreach (var sample in samples)
        {
            Assert.True(float.IsFinite(sample), "Audio contains NaN or infinity.");
            Assert.InRange(sample, -1f, 1f);
            peak = MathF.Max(peak, MathF.Abs(sample));
        }

        Assert.True(peak > 0.01f, "Expected non-silent audio.");
        Assert.True(peak <= 0.9501f, $"Peak {peak} exceeds the soft ceiling.");
    }
}