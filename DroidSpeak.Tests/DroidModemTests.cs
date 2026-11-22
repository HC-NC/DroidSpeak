using System.Text;
using DroidSpeak.DSP;
using DroidSpeak.Modem;

namespace DroidSpeak.Tests;

public class DroidModemTests
{
    private const int SampleRate = 44100;

    [Theory]
    [InlineData("Привет")]
    [InlineData("Hello")]
    [InlineData("Beep boop!")]
    public void CleanPacket_RoundTripsExactly(string text)
    {
        var encoded = DroidModemEncoder.Encode(text, SampleRate);
        var decoded = DroidModemDecoder.Decode(encoded, SampleRate);

        Assert.Equal(text, decoded);
    }

    [Theory]
    [InlineData("Привет")]
    [InlineData("Hello")]
    [InlineData("Тревога!")]
    public void MaskedInSpeech_RoundTripsExactly(string text)
    {
        var synthesizer = new DroidSynthesizer(SampleRate, seed: 1);
        var speech = synthesizer.Synthesize(text);

        var mixed = DroidModemEncoder.EncodeIntoSpeech(text, speech, SampleRate);
        var decoded = DroidModemDecoder.Decode(mixed, SampleRate);

        Assert.Equal(text, decoded);
    }

    [Theory]
    [InlineData("Привет", "Tiny")]
    [InlineData("Hello", "Tiny")]
    [InlineData("Бип-буп", "Bass")]
    [InlineData("Тревога!", "Grumpy")]
    public void MaskedInSpeech_SurvivesEveryVoiceProfile(string text, string profileName)
    {
        var profile = profileName switch
        {
            "Tiny" => DroidVoiceProfile.Tiny,
            "Bass" => DroidVoiceProfile.Bass,
            "Grumpy" => DroidVoiceProfile.Grumpy,
            _ => DroidVoiceProfile.R2,
        };

        var synthesizer = new DroidSynthesizer(SampleRate, seed: 3, profile: profile);
        var speech = synthesizer.Synthesize(text);

        var mixed = DroidModemEncoder.EncodeIntoSpeech(text, speech, SampleRate);
        var decoded = DroidModemDecoder.Decode(mixed, SampleRate);

        Assert.Equal(text, decoded);
    }

    [Fact]
    public void EncodedPacket_KeepsEnergyInsideTheModemBand()
    {
        var encoded = DroidModemEncoder.Encode("Beep boop!", SampleRate);

        const int fftSize = 4096;
        var real = new float[fftSize];
        var imaginary = new float[fftSize];
        var start = Math.Min(encoded.Length - fftSize, fftSize);
        Assert.True(start >= 0);

        for (var i = 0; i < fftSize; i++)
        {
            var hann = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / (fftSize - 1));
            real[i] = encoded[start + i] * hann;
        }

        FastFourierTransform.Forward(real, imaginary);

        var peakBin = 0;
        var peakMagnitude = 0f;
        for (var bin = 1; bin < fftSize / 2; bin++)
        {
            var magnitude = real[bin] * real[bin] + imaginary[bin] * imaginary[bin];
            if (magnitude > peakMagnitude)
            {
                peakMagnitude = magnitude;
                peakBin = bin;
            }
        }

        var peakFrequency = peakBin * (float)SampleRate / fftSize;
        Assert.InRange(peakFrequency, 3200f, 4800f);
    }

    [Fact]
    public void Encode_RejectsPayloadLongerThanOnePacket()
    {
        var tooLong = new string('я', 128); // 256 UTF-8 bytes

        Assert.Throws<ArgumentException>(() => DroidModemEncoder.Encode(tooLong, SampleRate));
    }

    [Fact]
    public void Decode_SilenceReturnsNull()
    {
        var silence = new float[SampleRate];

        Assert.Null(DroidModemDecoder.Decode(silence, SampleRate));
    }

    [Fact]
    public void Decode_SpeechWithoutPacketReturnsNull()
    {
        var speech = new DroidSynthesizer(SampleRate, seed: 1).Synthesize("Привет");

        Assert.Null(DroidModemDecoder.Decode(speech, SampleRate));
    }

    [Fact]
    public void EncodeIntoSpeech_ExtendsShortSpeechToFitPacket()
    {
        var synthesizer = new DroidSynthesizer(SampleRate, seed: 1);
        var shortSpeech = synthesizer.Synthesize("бип");
        var modemLength = DroidModemEncoder.Encode("Бип-буп. Долгая передача.", SampleRate).Length;

        var mixed = DroidModemEncoder.EncodeIntoSpeech(
            "Бип-буп. Долгая передача.",
            shortSpeech,
            SampleRate);

        Assert.True(mixed.Length > shortSpeech.Length);
        Assert.True(mixed.Length >= modemLength);

        var decoded = DroidModemDecoder.Decode(mixed, SampleRate);
        Assert.Equal("Бип-буп. Долгая передача.", decoded);
    }

    [Fact]
    public void Crc8_MatchesKnownTestVector()
    {
        var data = Encoding.ASCII.GetBytes("123456789");

        // CRC-8/SMBUS: poly 0x07, init 0x00, no reflection, no final XOR.
        Assert.Equal(0xF4, DroidModemProtocol.ComputeCrc8(data));
    }

    [Fact]
    public void Hamming_CorrectsAnySingleBitError()
    {
        const byte value = 0xA7;
        var bits = new List<int>();
        DroidModemProtocol.AppendHammingByte(bits, value);
        Assert.Equal(12, bits.Count);

        for (var position = 0; position < bits.Count; position++)
        {
            var corrupted = bits.ToArray();
            corrupted[position] ^= 1;

            Assert.True(DroidModemProtocol.TryDecodeHammingByte(corrupted, out var recovered));
            Assert.Equal(value, recovered);
        }
    }

    [Theory]
    [InlineData("Привет")]
    [InlineData("Hello")]
    public void EncodeIntoSpeech_OutputIsFiniteAndBounded(string text)
    {
        var synthesizer = new DroidSynthesizer(SampleRate, seed: 1);
        var speech = synthesizer.Synthesize(text);
        var mixed = DroidModemEncoder.EncodeIntoSpeech(text, speech, SampleRate);

        Assert.NotEmpty(mixed);
        foreach (var sample in mixed)
        {
            Assert.True(float.IsFinite(sample));
            Assert.InRange(sample, -1f, 1f);
        }
    }

    [Fact]
    public void DefaultMaskLevel_StaysWithinTheHiddenRange()
    {
        Assert.InRange(DroidModemEncoder.DefaultMaskLevel, 0.10f, 0.15f);
    }

    [Fact]
    public void EncodeIntoSpeech_EmptySpeechFallsBackToTheCleanPacket()
    {
        var baked = DroidModemEncoder.Encode("Привет", SampleRate);
        var mixed = DroidModemEncoder.EncodeIntoSpeech("Привет", ReadOnlySpan<float>.Empty, SampleRate);

        Assert.Equal(baked.Length, mixed.Length);
        Assert.True(baked.AsSpan().SequenceEqual(mixed));
    }

    [Theory]
    [InlineData("Привет")]
    [InlineData("Hello")]
    [InlineData("Тревога!")]
    public void MaskedInSpeech_SurvivesWavQuantization(string text)
    {
        // A real transmission is recorded, i.e. reduced to 16-bit PCM, before anyone decodes it.
        // This test makes sure quantization alone cannot break the packet.
        var synthesizer = new DroidSynthesizer(SampleRate, seed: 1);
        var mixed = DroidModemEncoder.EncodeIntoSpeech(text, synthesizer.Synthesize(text), SampleRate);

        using var stream = new MemoryStream();
        WavWriter.Write(stream, mixed, SampleRate);
        stream.Position = 0;

        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
        reader.ReadBytes(36); // RIFF header + fmt chunk
        Assert.Equal("data", Encoding.ASCII.GetString(reader.ReadBytes(4)));
        var dataSize = reader.ReadInt32();
        Assert.Equal(mixed.Length * 2, dataSize);

        var recorded = new float[dataSize / 2];
        for (var i = 0; i < recorded.Length; i++)
        {
            recorded[i] = reader.ReadInt16() / 32768f;
        }

        Assert.Equal(text, DroidModemDecoder.Decode(recorded, SampleRate));
    }
}