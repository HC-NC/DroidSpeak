using System.Text;

namespace DroidSpeak.DSP;

/// <summary>
/// Writes mono 16-bit PCM RIFF/WAVE files from floating-point samples in <c>[-1, 1]</c>.
/// The header is generated manually, so no external audio libraries are required.
/// </summary>
public static class WavWriter
{
    /// <summary>The default sample rate used when none is specified.</summary>
    public const int DefaultSampleRate = 44100;

    private const short PcmFormat = 1;
    private const short MonoChannels = 1;
    private const short BitsPerSample = 16;

    /// <summary>Writes <paramref name="samples"/> to a new 16-bit PCM WAV file.</summary>
    /// <param name="path">Destination file path. An existing file is overwritten.</param>
    /// <param name="samples">Mono samples in the range <c>[-1, 1]</c>.</param>
    /// <param name="sampleRate">Sample rate in Hz (8000–192000).</param>
    public static void Write(
        string path,
        ReadOnlySpan<float> samples,
        int sampleRate = DefaultSampleRate)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ValidateSampleRate(sampleRate);

        using var stream = new FileStream(
            path,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 64 * 1024,
            FileOptions.SequentialScan);

        Write(stream, samples, sampleRate);
    }

    /// <summary>
    /// Writes <paramref name="samples"/> to <paramref name="stream"/> as a 16-bit PCM WAV file.
    /// The stream is left open.
    /// </summary>
    /// <param name="stream">A writable stream positioned at the start of the file.</param>
    /// <param name="samples">Mono samples in the range <c>[-1, 1]</c>.</param>
    /// <param name="sampleRate">Sample rate in Hz (8000–192000).</param>
    public static void Write(
        Stream stream,
        ReadOnlySpan<float> samples,
        int sampleRate = DefaultSampleRate)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ValidateSampleRate(sampleRate);

        if (!stream.CanWrite)
        {
            throw new ArgumentException("The stream must be writable.", nameof(stream));
        }

        var bytesPerSample = BitsPerSample / 8;
        var dataSize = (long)samples.Length * bytesPerSample;
        if (dataSize > uint.MaxValue - 36L)
        {
            throw new ArgumentException(
                "The audio is too long for a 32-bit RIFF/WAVE file.",
                nameof(samples));
        }

        using var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true);

        writer.Write("RIFF"u8);
        writer.Write((uint)(36L + dataSize));
        writer.Write("WAVE"u8);

        writer.Write("fmt "u8);
        writer.Write(16u); // PCM fmt chunk size
        writer.Write(PcmFormat);
        writer.Write(MonoChannels);
        writer.Write((uint)sampleRate);
        writer.Write((uint)(sampleRate * MonoChannels * bytesPerSample));
        writer.Write((short)(MonoChannels * bytesPerSample));
        writer.Write(BitsPerSample);

        writer.Write("data"u8);
        writer.Write((uint)dataSize);

        foreach (var sample in samples)
        {
            writer.Write(ToPcm16(sample));
        }

        writer.Flush();
    }

    /// <summary>
    /// Converts a float sample to 16-bit PCM. Non-finite values become silence and out-of-range
    /// values are clamped, so the writer never wraps around.
    /// </summary>
    public static short ToPcm16(float sample)
    {
        if (!float.IsFinite(sample))
        {
            return 0;
        }

        var clamped = Math.Clamp(sample, -1f, 1f);
        return (short)MathF.Round(clamped * 32767f, MidpointRounding.AwayFromZero);
    }

    private static void ValidateSampleRate(int sampleRate)
    {
        if (sampleRate is < 8000 or > 192000)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sampleRate),
                sampleRate,
                "Sample rate must be between 8000 and 192000 Hz.");
        }
    }
}
