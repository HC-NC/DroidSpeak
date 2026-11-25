using DroidSpeak.DSP;

namespace DroidSpeak.Tests;

/// <summary>
/// Unit coverage for the v0.5 DSP building blocks: the band-pass biquad, the parallel formant
/// bank and the ring-modulation stage.
/// </summary>
public class DspTests
{
    private const int SampleRate = 44100;

    [Theory]
    [InlineData(8000)]
    [InlineData(22050)]
    [InlineData(44100)]
    [InlineData(48000)]
    [InlineData(192000)]
    public void BandPass_PassesItsCenterFrequencyAndRejectsFarBands(int sampleRate)
    {
        var center = 1000f;
        var onCenter = ResponseRms(center, center, sampleRate);
        var farAbove = ResponseRms(center, 4000f, sampleRate);
        var farBelow = ResponseRms(center, 120f, sampleRate);

        Assert.True(onCenter > 0.20f, $"Center gain was {onCenter} at {sampleRate} Hz.");
        Assert.True(onCenter > farAbove * 4f, "The band-pass should reject a 4 kHz tone.");
        Assert.True(onCenter > farBelow * 4f, "The band-pass should reject a 120 Hz tone.");
    }

    [Theory]
    [InlineData(20f, 12f)]
    [InlineData(10_000f, 0.35f)]
    [InlineData(float.NaN, float.NaN)]
    [InlineData(float.PositiveInfinity, 6f)]
    public void BandPass_StaysFiniteForPathologicalSettings(float frequency, float q)
    {
        var filter = default(BiquadBandPass);
        filter.SetBandPass(frequency, q, SampleRate);

        for (var i = 0; i < 4096; i++)
        {
            var output = filter.Process(i == 0 ? 1f : 0f);
            Assert.True(float.IsFinite(output), "Filter produced a non-finite sample.");
        }
    }

    [Fact]
    public void FormantBank_ChangesTheTimbreWhenMixedIn()
    {
        var neutral = RenderNote(Note(formantMix: 0f));
        var formant = RenderNote(Note(formantMix: 0.8f));

        Assert.False(neutral.AsSpan().SequenceEqual(formant));
        AssertAudioIsFinite(formant);
    }

    [Fact]
    public void FormantBank_IsInertWhenTheMixIsZero()
    {
        var withoutFrequencies = RenderNote(Note(formantMix: 0f));
        var withFrequencies = RenderNote(Note(formantMix: 0f, formantF1: 300f, formantF2: 2300f));

        Assert.True(withoutFrequencies.AsSpan().SequenceEqual(withFrequencies));
    }

    [Fact]
    public void RingModulation_AddsSidebandsWhenMixedIn()
    {
        var neutral = RenderNote(Note(ringModMix: 0f));
        var ring = RenderNote(Note(ringModMix: 0.7f));

        Assert.False(neutral.AsSpan().SequenceEqual(ring));
        AssertAudioIsFinite(ring);
    }

    [Fact]
    public void NoteRenderer_IsChunkSizeInvariantWithFormantsAndRingMod()
    {
        var note = Note(formantMix: 0.75f, ringModMix: 0.55f, length: 6000);
        var whole = new float[note.Length];
        var state = default(VoiceState);
        NoteRenderer.Render(note, 0, whole, ref state, triangleMix: 0.23f, SampleRate);

        var chunked = new float[note.Length];
        var chunkState = default(VoiceState);
        const int chunkSize = 97;
        for (var offset = 0; offset < note.Length; offset += chunkSize)
        {
            var length = Math.Min(chunkSize, note.Length - offset);
            NoteRenderer.Render(
                note,
                offset,
                chunked.AsSpan(offset, length),
                ref chunkState,
                triangleMix: 0.23f,
                SampleRate);
        }

        Assert.Equal(whole, chunked);
    }

    [Fact]
    public void NoteRenderer_FormantsRemainBoundedAtEverySupportedSampleRate()
    {
        foreach (var sampleRate in new[] { 8000, 22050, 44100, 48000, 96000, 192000 })
        {
            var note = Note(formantMix: 1f, formantF1: 700f, formantF2: 2200f, length: 2400);
            var target = new float[note.Length];
            var state = default(VoiceState);

            NoteRenderer.Render(note, 0, target, ref state, triangleMix: 0.23f, sampleRate);

            AssertAudioIsFinite(target);
        }
    }

    [Fact]
    public void PitchCurve_PassesThroughMidFrequency()
    {
        var legacy = Note(length: 8820);
        var curved = Note(length: 8820);
        curved.MidFrequency = 1500f;
        legacy.StartFrequency = 600f;
        legacy.EndFrequency = 600f;
        curved.StartFrequency = 600f;
        curved.EndFrequency = 600f;

        var flatAudio = RenderNote(legacy);
        var curvedAudio = RenderNote(curved);

        var flatMiddle = ZeroCrossings(flatAudio, 3600, 5200);
        var curvedMiddle = ZeroCrossings(curvedAudio, 3600, 5200);

        Assert.True(
            curvedMiddle > flatMiddle * 1.5f,
            $"Expected a higher midpoint pitch ({curvedMiddle}) than the flat glide ({flatMiddle}).");
    }

    [Fact]
    public void PitchCurve_GeometricMidpointKeepsTheLegacyGlide()
    {
        var legacy = Note(length: 4410);
        legacy.StartFrequency = 500f;
        legacy.EndFrequency = 900f;

        var neutral = Note(length: 4410);
        neutral.StartFrequency = 500f;
        neutral.EndFrequency = 900f;
        neutral.MidFrequency = MathF.Sqrt(500f * 900f);

        Assert.Equal(RenderNote(legacy), RenderNote(neutral));
    }

    [Fact]
    public void AttackChirp_RaisesTheOnsetPitchAndLeavesTheTailAlone()
    {
        var legacy = Note(length: 8820);
        var chirped = Note(length: 8820);
        chirped.AttackChirp = 0.35f;

        var legacyAudio = RenderNote(legacy);
        var chirpedAudio = RenderNote(chirped);

        var onsetLegacy = ZeroCrossings(legacyAudio, 0, 400);
        var onsetChirped = ZeroCrossings(chirpedAudio, 0, 400);
        var tailLegacy = ZeroCrossings(legacyAudio, 4410, 8820);
        var tailChirped = ZeroCrossings(chirpedAudio, 4410, 8820);

        Assert.True(onsetChirped > onsetLegacy, "The chirp should raise the onset pitch.");
        Assert.Equal(tailLegacy, tailChirped);
    }

    [Fact]
    public void AnalogSaturation_IsANoOpAtZeroAndBoundsTheSignal()
    {
        var dry = new float[2048];
        for (var i = 0; i < dry.Length; i++)
        {
            dry[i] = MathF.Sin(i * 0.05f) * 0.8f;
        }

        var untouched = (float[])dry.Clone();
        VoicePostProcessor.ApplyAnalogSaturation(untouched, 0f);
        Assert.Equal(dry, untouched);

        var driven = (float[])dry.Clone();
        VoicePostProcessor.ApplyAnalogSaturation(driven, 2f);

        Assert.False(dry.AsSpan().SequenceEqual(driven));
        foreach (var sample in driven)
        {
            Assert.True(float.IsFinite(sample));
            Assert.InRange(sample, -1f, 1f);
        }
    }

    [Fact]
    public void EarlyReflections_ProduceTapsAtSevenAndThirteenMilliseconds()
    {
        var buffer = new float[4096];
        buffer[100] = 1f;

        var line = new float[VoicePostProcessor.ReflectionLineLength(SampleRate)];
        var index = 0;
        var first = VoicePostProcessor.FirstTapSamples(SampleRate);
        var second = VoicePostProcessor.SecondTapSamples(SampleRate);

        VoicePostProcessor.ApplyEarlyReflections(buffer, line, ref index, 1f, first, second);

        Assert.Equal(0.38f, buffer[100 + first], 3);
        Assert.Equal(0.24f, buffer[100 + second], 3);
        Assert.Equal(1f, buffer[100]);
    }

    [Fact]
    public void EarlyReflections_AreANoOpAtZeroRoomMix()
    {
        var buffer = new float[1024];
        buffer[10] = 0.5f;

        var line = new float[VoicePostProcessor.ReflectionLineLength(SampleRate)];
        var index = 0;
        VoicePostProcessor.ApplyEarlyReflections(
            buffer,
            line,
            ref index,
            0f,
            VoicePostProcessor.FirstTapSamples(SampleRate),
            VoicePostProcessor.SecondTapSamples(SampleRate));

        Assert.Equal(0.5f, buffer[10]);
        Assert.Equal(1, buffer.Count(sample => sample != 0f));
    }

    [Fact]
    public void EarlyReflections_AreChunkSizeInvariant()
    {
        var whole = new float[6000];
        for (var i = 0; i < whole.Length; i++)
        {
            whole[i] = MathF.Sin(i * 0.02f) * 0.4f;
        }

        var chunked = (float[])whole.Clone();
        var first = VoicePostProcessor.FirstTapSamples(SampleRate);
        var second = VoicePostProcessor.SecondTapSamples(SampleRate);

        var wholeLine = new float[VoicePostProcessor.ReflectionLineLength(SampleRate)];
        var wholeIndex = 0;
        VoicePostProcessor.ApplyEarlyReflections(whole, wholeLine, ref wholeIndex, 0.8f, first, second);

        var chunkLine = new float[VoicePostProcessor.ReflectionLineLength(SampleRate)];
        var chunkIndex = 0;
        for (var offset = 0; offset < chunked.Length; offset += 211)
        {
            var length = Math.Min(211, chunked.Length - offset);
            VoicePostProcessor.ApplyEarlyReflections(
                chunked.AsSpan(offset, length),
                chunkLine,
                ref chunkIndex,
                0.8f,
                first,
                second);
        }

        Assert.Equal(whole, chunked);
    }

    private static int ZeroCrossings(float[] samples, int from, int to)
    {
        var count = 0;
        for (var i = Math.Max(1, from); i < Math.Min(to, samples.Length); i++)
        {
            if ((samples[i - 1] <= 0f && samples[i] > 0f) ||
                (samples[i - 1] >= 0f && samples[i] < 0f))
            {
                count++;
            }
        }

        return count;
    }

    private static float ResponseRms(float center, float tone, int sampleRate)    {
        var filter = default(BiquadBandPass);
        filter.SetBandPass(center, 6f, sampleRate);

        var phase = 0f;
        var step = MathF.Tau * tone / sampleRate;
        var sum = 0f;
        var count = 0;

        for (var i = 0; i < 8192; i++)
        {
            phase += step;
            var output = filter.Process(MathF.Sin(phase));
            if (i >= 4096)
            {
                sum += output * output;
                count++;
            }
        }

        return MathF.Sqrt(sum / count);
    }

    private static float[] RenderNote(DroidSynthesizer.NoteEvent note)
    {
        var target = new float[note.Length];
        var state = default(VoiceState);
        NoteRenderer.Render(note, 0, target, ref state, triangleMix: 0.23f, SampleRate);
        return target;
    }

    private static DroidSynthesizer.NoteEvent Note(
        float formantMix = 0f,
        float ringModMix = 0f,
        float formantF1 = 0f,
        float formantF2 = 0f,
        int length = 4410) => new()
        {
            StartSample = 0,
            Length = length,
            StartFrequency = 620f,
            EndFrequency = 620f,
            StartAmplitude = 0.4f,
            Amplitude = 0.4f,
            StartFmIndex = 1.4f,
            FmIndex = 1.4f,
            StartFmIndex2 = 0.6f,
            FmIndex2 = 0.6f,
            ModulatorRatio = 2f,
            ModulatorRatio2 = 3.07f,
            GrowlFrequency = 30f,
            Seed = 0x5EEDu,
            WordStartSample = 0,
            WordEndSample = length,
            WordAttack = 0.010f,
            WordDecay = 0.022f,
            WordRelease = 0.016f,
            SustainLevel = 0.74f,
            Drive = 1f,
            Saturation = 0f,
            FormantMix = formantMix,
            FormantF1 = formantF1,
            FormantF2 = formantF2,
            RingModMix = ringModMix,
        };

    private static void AssertAudioIsFinite(float[] samples)
    {
        foreach (var sample in samples)
        {
            Assert.True(float.IsFinite(sample), "Audio contains NaN or infinity.");
        }
    }
}
