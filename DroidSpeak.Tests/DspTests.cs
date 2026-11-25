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

    private static float ResponseRms(float center, float tone, int sampleRate)
    {
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
