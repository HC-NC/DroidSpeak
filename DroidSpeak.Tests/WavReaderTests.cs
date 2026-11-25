using System.Buffers.Binary;
using System.Text;
using DroidSpeak.Trainer.Audio;

namespace DroidSpeak.Tests;

/// <summary>
/// Builds small WAVE containers by hand so each decoding path can be exercised without shipping
/// binary fixtures. Everything here mirrors what <c>WavReader</c> is supposed to accept.
/// </summary>
internal static class WavFixtures
{
    public static byte[] Build(
        int formatTag,
        int channels,
        int sampleRate,
        int bitsPerSample,
        ReadOnlySpan<float> samples,
        string? extraChunkId = null)
    {
        var bytesPerSample = bitsPerSample / 8;
        var blockAlign = channels * bytesPerSample;
        var dataSize = samples.Length * blockAlign;

        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);

        var extraSize = extraChunkId is null ? 0 : 8 + 3 + 1;

        writer.Write("RIFF"u8);
        writer.Write((uint)(4 + 8 + 16 + extraSize + 8 + dataSize));
        writer.Write("WAVE"u8);

        writer.Write("fmt "u8);
        writer.Write(16u);
        writer.Write((ushort)formatTag);
        writer.Write((ushort)channels);
        writer.Write(sampleRate);
        writer.Write(sampleRate * blockAlign);
        writer.Write((ushort)blockAlign);
        writer.Write((ushort)bitsPerSample);

        if (extraChunkId is not null)
        {
            // An odd-sized chunk, so the reader also has to honour the word-alignment pad byte.
            writer.Write(Encoding.ASCII.GetBytes(extraChunkId));
            writer.Write(3u);
            writer.Write(new byte[] { 1, 2, 3 });
            writer.Write((byte)0);
        }

        writer.Write("data"u8);
        writer.Write((uint)dataSize);

        for (var frame = 0; frame < samples.Length / channels; frame++)
        {
            for (var channel = 0; channel < channels; channel++)
            {
                WriteSample(writer, formatTag, bitsPerSample, samples[(frame * channels) + channel]);
            }
        }

        writer.Flush();
        return stream.ToArray();
    }

    public static byte[] BuildPcm16(int sampleRate, params float[] samples) =>
        Build(1, 1, sampleRate, 16, samples);

    private static void WriteSample(BinaryWriter writer, int formatTag, int bitsPerSample, float sample)
    {
        if (formatTag == 3)
        {
            if (bitsPerSample == 64)
            {
                writer.Write((double)sample);
            }
            else
            {
                writer.Write(sample);
            }

            return;
        }

        switch (bitsPerSample)
        {
            case 8:
                writer.Write((byte)Math.Clamp((int)MathF.Round(sample * 127f) + 128, 0, 255));
                break;

            case 16:
                writer.Write((short)Math.Clamp((int)MathF.Round(sample * 32767f), short.MinValue, short.MaxValue));
                break;

            case 24:
            {
                var scaled = Math.Clamp((int)MathF.Round(sample * 8388607f), -8388608, 8388607);
                writer.Write((byte)(scaled & 0xFF));
                writer.Write((byte)((scaled >> 8) & 0xFF));
                writer.Write((byte)((scaled >> 16) & 0xFF));
                break;
            }

            case 32:
                writer.Write((int)Math.Clamp(sample * 2147483647f, int.MinValue, int.MaxValue));
                break;

            default:
                throw new ArgumentOutOfRangeException(nameof(bitsPerSample));
        }
    }

    /// <summary>Reads the sample rate out of a container without going through the reader.</summary>
    public static int RateOf(byte[] bytes) => BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(24));
}

public class WavReaderTests
{
    [Fact]
    public void Parse_ReadsMonoPcm16()
    {
        var bytes = WavFixtures.BuildPcm16(22050, 0f, 0.5f, -0.5f, 1f);

        var data = WavReader.Parse(bytes, targetSampleRate: null);

        Assert.Equal(22050, data.SampleRate);
        Assert.Equal(22050, data.SourceSampleRate);
        Assert.Equal(1, data.SourceChannels);
        Assert.Equal(4, data.Samples.Length);
        Assert.Equal(0f, data.Samples[0]);
        Assert.Equal(0.5f, data.Samples[1], 3);
        Assert.Equal(-0.5f, data.Samples[2], 3);
        Assert.Equal(1f, data.Samples[3], 4);
    }

    [Fact]
    public void Parse_Reads8BitPcmAsUnsigned()
    {
        var bytes = WavFixtures.Build(1, 1, 8000, 8, [0f, 1f, -1f]);

        var data = WavReader.Parse(bytes, targetSampleRate: null);

        Assert.Equal(3, data.Samples.Length);
        Assert.Equal(0f, data.Samples[0], 2);
        Assert.True(data.Samples[1] > 0.98f);
        Assert.True(data.Samples[2] < -0.98f);
    }

    [Fact]
    public void Parse_Reads24BitPcmWithSignExtension()
    {
        var bytes = WavFixtures.Build(1, 1, 8000, 24, [0.25f, -0.25f]);

        var data = WavReader.Parse(bytes, targetSampleRate: null);

        Assert.Equal(2, data.Samples.Length);
        Assert.Equal(0.25f, data.Samples[0], 3);
        Assert.Equal(-0.25f, data.Samples[1], 3);
    }

    [Theory]
    [InlineData(32)]
    [InlineData(64)]
    public void Parse_ReadsFloatPcm(int bitsPerSample)
    {
        var bytes = WavFixtures.Build(3, 1, 22050, bitsPerSample, [0.1f, -0.1f, 0.75f]);

        var data = WavReader.Parse(bytes, targetSampleRate: null);

        Assert.Equal(3, data.Samples.Length);
        Assert.Equal(0.1f, data.Samples[0], 4);
        Assert.Equal(-0.1f, data.Samples[1], 4);
        Assert.Equal(0.75f, data.Samples[2], 4);
    }

    [Fact]
    public void Parse_DownmixesStereoByAveraging()
    {
        var bytes = WavFixtures.Build(1, 2, 22050, 16, [1f, -1f, 0.5f, 0.5f]);

        var data = WavReader.Parse(bytes, targetSampleRate: null);

        Assert.Equal(2, data.SourceChannels);
        Assert.Equal(2, data.Samples.Length);
        Assert.Equal(0f, data.Samples[0], 3);
        Assert.Equal(0.5f, data.Samples[1], 3);
    }

    [Fact]
    public void Parse_SkipsUnknownChunksBeforeData()
    {
        var bytes = WavFixtures.Build(1, 1, 8000, 16, [0.5f], extraChunkId: "JUNK");

        var data = WavReader.Parse(bytes, targetSampleRate: null);

        Assert.Equal(0.5f, Assert.Single(data.Samples), 3);
    }

    [Fact]
    public void Parse_ResamplesToRequestedRate()
    {
        var samples = new float[1000];
        for (var i = 0; i < samples.Length; i++)
        {
            samples[i] = MathF.Sin(i * 0.1f) * 0.5f;
        }

        var bytes = WavFixtures.Build(1, 1, 44100, 32, samples);

        var data = WavReader.Parse(bytes, targetSampleRate: 22050);

        Assert.Equal(22050, data.SampleRate);
        Assert.Equal(44100, data.SourceSampleRate);
        Assert.Equal(500, data.Samples.Length);
    }

    [Fact]
    public void Resample_ReturnsCopyWhenRatesMatch()
    {
        float[] samples = [0.1f, 0.2f, 0.3f];

        var result = WavReader.Resample(samples, 22050, 22050);

        Assert.Equal(samples, result);
        Assert.NotSame(samples, result);
    }

    [Fact]
    public void Resample_LinearInterpolatesMidpoints()
    {
        float[] samples = [0f, 1f, 0f, -1f];

        var result = WavReader.Resample(samples, 4, 8);

        Assert.Equal(8, result.Length);
        Assert.Equal(0f, result[0], 4);
        Assert.Equal(0.5f, result[1], 4);
        Assert.Equal(1f, result[2], 4);
        Assert.Equal(0.5f, result[3], 4);
    }

    [Fact]
    public void Resample_CubicStaysInsideInputRangeOnSmoothSignal()
    {
        var samples = new float[256];
        for (var i = 0; i < samples.Length; i++)
        {
            samples[i] = MathF.Sin(i * 0.05f) * 0.4f;
        }

        var result = WavReader.Resample(samples, 8000, 16000, WavResampler.Cubic);

        Assert.Equal(512, result.Length);
        Assert.All(result, value => Assert.InRange(value, -0.5f, 0.5f));
    }

    [Fact]
    public void Parse_RejectsNonWaveBytes()
    {
        var bytes = Encoding.ASCII.GetBytes("NOT A WAVE FILE AT ALL");

        Assert.Throws<InvalidDataException>(() => WavReader.Parse(bytes));
    }

    [Fact]
    public void Parse_RejectsMissingFmtChunk()
    {
        using var stream = new MemoryStream();
        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);
        writer.Write("RIFF"u8);
        writer.Write(12u);
        writer.Write("WAVE"u8);
        writer.Write("data"u8);
        writer.Write(0u);
        writer.Flush();

        var bytes = stream.ToArray();

        Assert.Throws<InvalidDataException>(() => WavReader.Parse(bytes));
    }

    [Fact]
    public void Parse_RejectsUnsupportedFormatTag()
    {
        var bytes = WavFixtures.Build(0x0011, 1, 8000, 16, [0.1f]);

        Assert.Throws<InvalidDataException>(() => WavReader.Parse(bytes));
    }

    [Fact]
    public void Parse_RejectsOutOfRangeTargetRate()
    {
        var bytes = WavFixtures.BuildPcm16(8000, 0.1f);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => WavReader.Parse(bytes, targetSampleRate: 1000));
    }

    [Fact]
    public void Parse_HandlesEmptyDataChunk()
    {
        var bytes = WavFixtures.Build(1, 1, 8000, 16, []);

        var data = WavReader.Parse(bytes, targetSampleRate: null);

        Assert.Empty(data.Samples);
    }

    [Fact]
    public void Parse_ToleratesDataChunkLongerThanFile()
    {
        var bytes = WavFixtures.BuildPcm16(8000, 0.25f, 0.25f);
        // Claim two extra samples that are not present, as a truncated rip would. The data chunk
        // header sits at offset 36 with its size at 40.
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(40), 8);

        var data = WavReader.Parse(bytes, targetSampleRate: null);

        Assert.Equal(2, data.Samples.Length);
    }

    [Fact]
    public void Read_DecodesFileFromDisk()
    {
        var directory = Path.Combine(Path.GetTempPath(), "DroidSpeak.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "clip.wav");

        try
        {
            File.WriteAllBytes(path, WavFixtures.BuildPcm16(22050, 0.5f, -0.5f));

            var data = WavReader.Read(path);

            Assert.Equal(22050, data.SampleRate);
            Assert.Equal(2, data.Samples.Length);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}