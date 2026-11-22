using System.Text;
using DroidSpeak.DSP;

namespace DroidSpeak.Tests;

public class WavWriterTests
{
    [Fact]
    public void Write_EmitsValidRiffHeaderAndPcmSamples()
    {
        float[] samples = [0f, 0.5f, -0.5f, 1f, -1f, float.NaN];
        using var stream = new MemoryStream();

        WavWriter.Write(stream, samples);

        Assert.Equal(44 + samples.Length * 2, stream.Length);
        stream.Position = 0;

        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);
        Assert.Equal("RIFF", new string(reader.ReadChars(4)));
        Assert.Equal(36 + samples.Length * 2, reader.ReadInt32());
        Assert.Equal("WAVE", new string(reader.ReadChars(4)));

        Assert.Equal("fmt ", new string(reader.ReadChars(4)));
        Assert.Equal(16, reader.ReadInt32());
        Assert.Equal(1, reader.ReadInt16());   // PCM
        Assert.Equal(1, reader.ReadInt16());   // mono
        Assert.Equal(44100, reader.ReadInt32());
        Assert.Equal(88200, reader.ReadInt32());
        Assert.Equal(2, reader.ReadInt16());
        Assert.Equal(16, reader.ReadInt16());

        Assert.Equal("data", new string(reader.ReadChars(4)));
        Assert.Equal(samples.Length * 2, reader.ReadInt32());

        Assert.Equal(0, reader.ReadInt16());
        Assert.Equal(16384, reader.ReadInt16());
        Assert.Equal(-16384, reader.ReadInt16());
        Assert.Equal(32767, reader.ReadInt16());
        Assert.Equal(-32767, reader.ReadInt16());
        Assert.Equal(0, reader.ReadInt16());
        Assert.Equal(stream.Length, stream.Position);
    }

    [Fact]
    public void Write_ToPath_CreatesReadableFile()
    {
        var directory = Path.Combine(Path.GetTempPath(), "DroidSpeak.Tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        var path = Path.Combine(directory, "test.wav");

        try
        {
            WavWriter.Write(path, [0.25f, -0.25f], 22050);

            var bytes = File.ReadAllBytes(path);
            Assert.Equal(44 + 4, bytes.Length);
            Assert.Equal((byte)'R', bytes[0]);
            Assert.Equal((byte)'I', bytes[1]);
            Assert.Equal((byte)'F', bytes[2]);
            Assert.Equal((byte)'F', bytes[3]);
            Assert.Equal(22050, BitConverter.ToInt32(bytes, 24));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Theory]
    [InlineData(1f, 32767)]
    [InlineData(-1f, -32767)]
    [InlineData(0f, 0)]
    [InlineData(2f, 32767)]
    [InlineData(-2f, -32767)]
    [InlineData(float.NaN, 0)]
    [InlineData(float.PositiveInfinity, 0)]
    public void ToPcm16_ClampsAndSanitizes(float sample, short expected)
    {
        Assert.Equal(expected, WavWriter.ToPcm16(sample));
    }

    [Fact]
    public void Write_RejectsInvalidSampleRate()
    {
        using var stream = new MemoryStream();

        Assert.Throws<ArgumentOutOfRangeException>(
            () => WavWriter.Write(stream, ReadOnlySpan<float>.Empty, 4000));
    }
}
