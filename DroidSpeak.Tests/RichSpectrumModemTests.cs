using System.Text;
using DroidSpeak.DSP;
using DroidSpeak.Modem;

namespace DroidSpeak.Tests;

public class RichSpectrumModemTests
{
    private const int SampleRate = 44100;
    private const int FftSize = 4096;

    [Fact]
    public void Crc16_MatchesKnownTestVector()
    {
        var data = Encoding.ASCII.GetBytes("123456789");

        Assert.Equal(0x29B1, AcousticFrameProtocol.ComputeCrc16(data));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(5)]
    [InlineData(0xA)]
    [InlineData(0xF)]
    public void Hamming74_CorrectsEverySingleBit(int nibble)
    {
        Span<int> codeword = stackalloc int[AcousticFrameProtocol.HammingCodewordBits];
        Span<int> corrupted = stackalloc int[AcousticFrameProtocol.HammingCodewordBits];
        AcousticFrameProtocol.EncodeNibble(nibble, codeword);

        Assert.True(AcousticFrameProtocol.TryDecodeNibble(codeword, out var clean, out var corrected));
        Assert.Equal(nibble, clean);
        Assert.False(corrected);

        for (var position = 0; position < codeword.Length; position++)
        {
            codeword.CopyTo(corrupted);
            corrupted[position] ^= 1;

            Assert.True(AcousticFrameProtocol.TryDecodeNibble(corrupted, out var recovered, out var fixedBit));
            Assert.Equal(nibble, recovered);
            Assert.True(fixedBit);
        }
    }

    [Fact]
    public void FrameHeader_CarriesVersionAndMode()
    {
        var frame = AcousticFrameProtocol.BuildFrame("beep", ModemMode.AudibleChirpStream);

        Assert.True(AcousticFrameProtocol.TryReadHeader(frame, out var mode, out var length));
        Assert.Equal(ModemMode.AudibleChirpStream, mode);
        Assert.Equal(4, length);
    }

    [Theory]
    [InlineData(ModemMode.HighBandOverlay, 44100)]
    [InlineData(ModemMode.HighBandOverlay, 48000)]
    [InlineData(ModemMode.AudibleChirpStream, 44100)]
    [InlineData(ModemMode.AudibleChirpStream, 48000)]
    public void CleanPacket_RoundTripsInBothModes(ModemMode mode, int sampleRate)
    {
        var encoded = DroidModemEncoder.Encode("Hello rich spectrum!", mode, sampleRate);

        Assert.True(DroidModemDecoder.TryDecode(encoded, sampleRate, out var result));
        Assert.Equal("Hello rich spectrum!", result.Text);
        Assert.Equal(mode, result.Mode);
        Assert.True(result.CrcValid);
        Assert.False(result.Legacy);
    }

    [Theory]
    [InlineData(ModemMode.HighBandOverlay, "Привет")]
    [InlineData(ModemMode.HighBandOverlay, "Tiny unit")]
    [InlineData(ModemMode.AudibleChirpStream, "Привет")]
    [InlineData(ModemMode.AudibleChirpStream, "Tiny unit")]
    public void SpeechPacket_RoundTripsInBothModes(ModemMode mode, string text)
    {
        var synthesizer = new DroidSynthesizer(SampleRate, seed: 1);
        var speech = synthesizer.Synthesize(text);

        var mixed = DroidModemEncoder.EncodeIntoSpeech(text, speech, mode, SampleRate);

        Assert.True(DroidModemDecoder.TryDecode(mixed, SampleRate, out var result));
        Assert.Equal(text, result.Text);
        Assert.Equal(mode, result.Mode);
    }

    [Theory]
    [InlineData("Tiny")]
    [InlineData("Bass")]
    [InlineData("Grumpy")]
    public void HighBandOverlay_SurvivesEveryVoiceProfile(string profileName)
    {
        var profile = profileName switch
        {
            "Tiny" => DroidVoiceProfile.Tiny,
            "Bass" => DroidVoiceProfile.Bass,
            "Grumpy" => DroidVoiceProfile.Grumpy,
            _ => DroidVoiceProfile.R2,
        };

        var synthesizer = new DroidSynthesizer(SampleRate, seed: 3, profile: profile);
        var speech = synthesizer.Synthesize("Full spectrum test");
        var mixed = DroidModemEncoder.EncodeIntoSpeech(
            "Full spectrum test",
            speech,
            ModemMode.HighBandOverlay,
            SampleRate);

        Assert.Equal("Full spectrum test", DroidModemDecoder.Decode(mixed, SampleRate));
    }

    [Theory]
    [InlineData(ModemMode.HighBandOverlay)]
    [InlineData(ModemMode.AudibleChirpStream)]
    public void AwgnNoise_StillRoundTrips(ModemMode mode)
    {
        var encoded = DroidModemEncoder.Encode("Noise resilient packet", mode, SampleRate);
        var noisy = AddNoise(encoded, amplitude: 0.04f, seed: 11);

        Assert.True(DroidModemDecoder.TryDecode(noisy, SampleRate, out var result));
        Assert.Equal("Noise resilient packet", result.Text);
    }

    [Theory]
    [InlineData(ModemMode.HighBandOverlay)]
    [InlineData(ModemMode.AudibleChirpStream)]
    public void Fec_CorrectsOneDamagedSymbol(ModemMode mode)
    {
        var encoded = DroidModemEncoder.Encode("FEC repair", mode, SampleRate);
        var period = (int)Math.Round(AcousticFrameProtocol.SymbolDuration(mode) * SampleRate);
        var bodyStart = 28 + AcousticFrameProtocol.HeaderSymbolCount();
        var damaged = (float[])encoded.Clone();
        Array.Clear(damaged, bodyStart * period + period * 5, period);

        Assert.True(DroidModemDecoder.TryDecode(damaged, SampleRate, out var result));
        Assert.Equal("FEC repair", result.Text);
        Assert.True(result.CorrectedBits >= 1);
    }

    [Theory]
    [InlineData(ModemMode.HighBandOverlay, 1.001)]
    [InlineData(ModemMode.HighBandOverlay, 0.999)]
    [InlineData(ModemMode.AudibleChirpStream, 1.001)]
    [InlineData(ModemMode.AudibleChirpStream, 0.999)]
    public void ClockDrift_UpToThousandPpm_RoundTrips(ModemMode mode, double ratio)
    {
        var encoded = DroidModemEncoder.Encode("Clock drift", mode, SampleRate);
        var stretched = ResampleLinear(encoded, ratio);

        Assert.True(DroidModemDecoder.TryDecode(stretched, SampleRate, out var result));
        Assert.Equal("Clock drift", result.Text);
    }

    [Theory]
    [InlineData(ModemMode.HighBandOverlay)]
    [InlineData(ModemMode.AudibleChirpStream)]
    public void SampleRateConversion_44100To48000_RoundTrips(ModemMode mode)
    {
        var encoded = DroidModemEncoder.Encode("Sample rate", mode, SampleRate);
        var converted = ResampleLinear(encoded, 48000.0 / SampleRate);

        Assert.True(DroidModemDecoder.TryDecode(converted, 48000, out var result));
        Assert.Equal("Sample rate", result.Text);
    }

    [Theory]
    [InlineData(ModemMode.HighBandOverlay, 1)]
    [InlineData(ModemMode.HighBandOverlay, 17)]
    [InlineData(ModemMode.HighBandOverlay, 4096)]
    [InlineData(ModemMode.AudibleChirpStream, 1)]
    [InlineData(ModemMode.AudibleChirpStream, 113)]
    public void StreamingDemodulator_HandlesChunkBoundaries(ModemMode mode, int chunkSize)
    {
        var encoded = DroidModemEncoder.Encode("Streaming packet", mode, SampleRate);
        var receiver = new StreamingAcousticDemodulator(SampleRate);

        for (var offset = 0; offset < encoded.Length; offset += chunkSize)
        {
            receiver.Feed(encoded.AsSpan(offset, Math.Min(chunkSize, encoded.Length - offset)));
        }

        Assert.True(receiver.TryReadPacket(out var result));
        Assert.Equal("Streaming packet", result.Text);
        Assert.Equal(mode, result.Mode);
    }

    [Fact]
    public void StreamingDemodulator_ReadsBackToBackPackets()
    {
        var first = DroidModemEncoder.Encode("first", ModemMode.HighBandOverlay, SampleRate);
        var second = DroidModemEncoder.Encode("second", ModemMode.HighBandOverlay, SampleRate);
        var combined = new float[first.Length + second.Length];
        first.CopyTo(combined, 0);
        second.CopyTo(combined, first.Length);

        var receiver = new StreamingAcousticDemodulator(SampleRate);
        receiver.Feed(combined);

        Assert.True(receiver.TryReadPacket(out var result));
        Assert.Equal("first", result.Text);
        Assert.True(receiver.TryReadPacket(out result));
        Assert.Equal("second", result.Text);
        Assert.False(receiver.TryReadPacket(out _));
    }

    [Fact]
    public void HighBandOverlay_PreservesVoiceBelowTenKilohertz()
    {
        var speech = BuildVoiceFixture(seconds: 3);
        var mixed = DroidModemEncoder.EncodeIntoSpeech(
            "spectral preservation",
            speech,
            ModemMode.HighBandOverlay,
            SampleRate);

        Assert.Equal(speech.Length, mixed.Length);

        var residual = new float[speech.Length];
        for (var i = 0; i < residual.Length; i++)
        {
            residual[i] = mixed[i] - speech[i];
        }

        var voiceEnergy = BandEnergy(speech, 20f, 10000f);
        var residualEnergy = BandEnergy(residual, 20f, 10000f);

        Assert.True(voiceEnergy > 0f);
        Assert.True(
            residualEnergy <= voiceEnergy * 1e-4f,
            $"Voice-band residual was {residualEnergy / voiceEnergy:P2} of the speech energy.");
    }

    [Fact]
    public void AudibleChirpStream_AppendsWithoutFilteringTheVoice()
    {
        var speech = BuildVoiceFixture(seconds: 2);
        var mixed = DroidModemEncoder.EncodeIntoSpeech(
            "audible chirps",
            speech,
            ModemMode.AudibleChirpStream,
            SampleRate);

        Assert.True(mixed.Length > speech.Length);
        Assert.True(speech.AsSpan().SequenceEqual(mixed.AsSpan(0, speech.Length)));
    }

    [Theory]
    [InlineData(ModemMode.HighBandOverlay)]
    [InlineData(ModemMode.AudibleChirpStream)]
    public void MaximumPayload_RoundTrips(ModemMode mode)
    {
        var text = new string('A', AcousticFrameProtocol.MaxPayloadLength);
        var encoded = DroidModemEncoder.Encode(text, mode, SampleRate);

        Assert.Equal(text, DroidModemDecoder.Decode(encoded, SampleRate));
    }

    [Fact]
    public void Decode_SpecificModeRejectsTheOtherMode()
    {
        var high = DroidModemEncoder.Encode("high only", ModemMode.HighBandOverlay, SampleRate);

        Assert.Null(DroidModemDecoder.Decode(high, SampleRate, ModemMode.AudibleChirpStream));
    }

    [Fact]
    public void Decode_HeavilyCorruptedPacketReturnsNull()
    {
        var encoded = DroidModemEncoder.Encode("corrupted packet", ModemMode.HighBandOverlay, SampleRate);
        var period = (int)Math.Round(AcousticFrameProtocol.SymbolDuration(ModemMode.HighBandOverlay) * SampleRate);
        var bodyStart = 28 + AcousticFrameProtocol.HeaderSymbolCount();
        Array.Clear(encoded, bodyStart * period + period * 10, period * 120);

        Assert.Null(DroidModemDecoder.Decode(encoded, SampleRate));
    }

    [Fact]
    public void LegacyV08Packet_StillDecodesThroughTheCompatibilityPath()
    {
        // The v0.8 format is retained in DroidModemProtocol for recordings made before v0.9.
        var bits = new List<int>();
        foreach (var value in DroidModemProtocol.Preamble)
        {
            DroidModemProtocol.AppendByteBits(bits, value);
        }

        var payload = DroidModemProtocol.EncodeUtf8("legacy");
        var frame = new List<byte> { (byte)payload.Length };
        frame.AddRange(payload);
        frame.Add(DroidModemProtocol.ComputeCrc8(frame.ToArray()));

        var frameBits = new List<int>();
        foreach (var value in frame)
        {
            DroidModemProtocol.AppendHammingByte(frameBits, value);
        }

        for (var i = 0; i < 12; i++)
        {
            bits.Add(frameBits[i]);
        }

        var dataCodewords = frame.Count - 1;
        for (var bitPosition = 0; bitPosition < 12; bitPosition++)
        {
            for (var codeword = 0; codeword < dataCodewords; codeword++)
            {
                bits.Add(frameBits[(codeword + 1) * 12 + bitPosition]);
            }
        }

        var bitSamples = DroidModemProtocol.BitDuration * SampleRate;
        var samples = new float[(int)Math.Ceiling(bits.Count * bitSamples) + SampleRate / 100];
        var phase = 0d;
        for (var bitIndex = 0; bitIndex < bits.Count; bitIndex++)
        {
            var start = (int)Math.Round(bitIndex * bitSamples);
            var end = Math.Min(samples.Length, (int)Math.Round((bitIndex + 1) * bitSamples));
            var frequency = bits[bitIndex] == 0
                ? DroidModemProtocol.Bit0Frequency
                : DroidModemProtocol.Bit1Frequency;

            for (var n = start; n < end; n++)
            {
                phase += Math.PI * 2 * frequency / SampleRate;
                samples[n] = MathF.Sin((float)phase);
            }
        }

        Assert.True(DroidModemDecoder.TryDecode(samples, SampleRate, out var result));
        Assert.Equal("legacy", result.Text);
        Assert.True(result.Legacy);
    }

    private static float[] BuildVoiceFixture(int seconds)
    {
        var samples = new float[SampleRate * seconds];
        var frequencies = new[] { 220f, 700f, 1800f, 3500f, 6500f, 9000f };
        var amplitudes = new[] { 0.10f, 0.09f, 0.08f, 0.07f, 0.05f, 0.04f };

        for (var i = 0; i < samples.Length; i++)
        {
            var t = i / (double)SampleRate;
            var value = 0f;
            for (var tone = 0; tone < frequencies.Length; tone++)
            {
                value += amplitudes[tone] * MathF.Sin((float)(Math.PI * 2 * frequencies[tone] * t));
            }

            samples[i] = value;
        }

        return samples;
    }

    private static float BandEnergy(float[] samples, float low, float high)
    {
        var real = new float[FftSize];
        var imaginary = new float[FftSize];
        Assert.True(samples.Length >= FftSize);
        const int start = 0;

        for (var i = 0; i < FftSize; i++)
        {
            var hann = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / (FftSize - 1));
            real[i] = samples[start + i] * hann;
        }

        FastFourierTransform.Forward(real, imaginary);

        var lowBin = Math.Max(1, (int)MathF.Ceiling(low * FftSize / SampleRate));
        var highBin = Math.Min(FftSize / 2 - 1, (int)MathF.Floor(high * FftSize / SampleRate));
        var energy = 0f;
        for (var bin = lowBin; bin <= highBin; bin++)
        {
            energy += real[bin] * real[bin] + imaginary[bin] * imaginary[bin];
        }

        return energy;
    }

    private static float[] AddNoise(float[] input, float amplitude, int seed)
    {
        var random = new Random(seed);
        var output = new float[input.Length];
        for (var i = 0; i < input.Length; i++)
        {
            output[i] = input[i] + (float)((random.NextDouble() * 2.0 - 1.0) * amplitude);
        }

        return output;
    }

    private static float[] ResampleLinear(float[] input, double ratio)
    {
        var length = (int)Math.Floor(input.Length * ratio);
        var output = new float[length];

        for (var i = 0; i < length; i++)
        {
            var source = i / ratio;
            var index = (int)source;
            var fraction = (float)(source - index);
            var first = input[Math.Min(index, input.Length - 1)];
            var second = input[Math.Min(index + 1, input.Length - 1)];
            output[i] = first + (second - first) * fraction;
        }

        return output;
    }
}
