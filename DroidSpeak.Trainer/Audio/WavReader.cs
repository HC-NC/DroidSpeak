using System.Buffers.Binary;

namespace DroidSpeak.Trainer.Audio;

/// <summary>Interpolation kernel used when a file is resampled to the analysis rate.</summary>
internal enum WavResampler
{
    /// <summary>Straight linear interpolation. Cheap, deterministic and enough for pitch tracking.</summary>
    Linear = 0,

    /// <summary>Catmull-Rom cubic interpolation. Slightly sharper transients for short clips.</summary>
    Cubic = 1,
}

/// <summary>
/// A decoded, mono, float-valued audio clip together with the rate its samples are expressed in.
/// </summary>
/// <param name="SampleRate">Sample rate of <paramref name="Samples"/> in Hz.</param>
/// <param name="Samples">Mono samples normalised to <c>[-1, 1]</c>.</param>
/// <param name="SourceSampleRate">Rate the file was stored at, before any resampling.</param>
/// <param name="SourceChannels">Channel count the file was stored with.</param>
internal readonly record struct WavData(
    int SampleRate,
    float[] Samples,
    int SourceSampleRate,
    int SourceChannels);

/// <summary>
/// A dependency-free RIFF/WAVE reader: 8/16/24-bit PCM, 32-bit float and 64-bit float, any channel
/// count downmixed to mono, with optional linear or cubic resampling to a canonical analysis rate.
/// <para>
/// The trainer owns this instead of the library on purpose. Decoding a reference clip is an offline
/// developer concern, and pulling a codec package into the solution would break the zero-dependency
/// promise that lets a checkout build with nothing but the SDK. The subset below is exactly what
/// jingle rips, game audio and phone recordings produce; anything outside it is rejected loudly
/// rather than decoded into garbage.
/// </para>
/// <para>
/// Parsing walks the chunk list rather than assuming a 44-byte header, because real files carry
/// <c>LIST</c>, <c>fact</c>, <c>JUNK</c> and odd-sized chunks between <c>fmt </c> and <c>data</c>.
/// </para>
/// </summary>
internal static class WavReader
{
    /// <summary>Canonical analysis rate the extractor uses when no rate is requested.</summary>
    public const int DefaultAnalysisRate = 22050;

    private const int FormatPcm = 1;
    private const int FormatFloat = 3;
    private const int FormatExtensible = 0xFFFE;

    /// <summary>
    /// Reads and decodes a WAV file from disk.
    /// </summary>
    /// <param name="path">File to decode.</param>
    /// <param name="targetSampleRate">
    /// Analysis rate to resample to, or <c>null</c> to keep the file's own rate.
    /// </param>
    /// <param name="resampler">Interpolation kernel used when a resample is required.</param>
    /// <exception cref="IOException">The file cannot be read.</exception>
    /// <exception cref="InvalidDataException">The bytes are not a supported WAVE file.</exception>
    public static WavData Read(
        string path,
        int? targetSampleRate = DefaultAnalysisRate,
        WavResampler resampler = WavResampler.Linear)
    {
        ArgumentNullException.ThrowIfNull(path);

        using var stream = File.OpenRead(path);
        return Read(stream, targetSampleRate, resampler);
    }

    /// <summary>
    /// Reads and decodes a WAV file from <paramref name="stream"/>, which is not closed.
    /// </summary>
    /// <param name="stream">Readable source positioned at the start of the container.</param>
    /// <param name="targetSampleRate">Analysis rate to resample to, or <c>null</c> to keep it.</param>
    /// <param name="resampler">Interpolation kernel used when a resample is required.</param>
    /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
    /// <exception cref="InvalidDataException">The bytes are not a supported WAVE file.</exception>
    public static WavData Read(
        Stream stream,
        int? targetSampleRate = DefaultAnalysisRate,
        WavResampler resampler = WavResampler.Linear)
    {
        ArgumentNullException.ThrowIfNull(stream);

        if (!stream.CanRead)
        {
            throw new ArgumentException("The stream is not readable.", nameof(stream));
        }

        // The whole file is buffered: reference clips are tens of seconds at most, and a single span
        // keeps the chunk walk a pure index walk instead of a seek dance on an unseekable stream.
        using var buffer = new MemoryStream();
        stream.CopyTo(buffer);
        return Parse(buffer.GetBuffer().AsSpan(0, (int)buffer.Length), targetSampleRate, resampler);
    }

    /// <summary>
    /// Decodes an in-memory WAVE container, the form tests use so no temp file is needed.
    /// </summary>
    /// <param name="bytes">Complete container bytes.</param>
    /// <param name="targetSampleRate">Analysis rate to resample to, or <c>null</c> to keep it.</param>
    /// <param name="resampler">Interpolation kernel used when a resample is required.</param>
    /// <exception cref="InvalidDataException">The bytes are not a supported WAVE file.</exception>
    public static WavData Parse(
        ReadOnlySpan<byte> bytes,
        int? targetSampleRate = DefaultAnalysisRate,
        WavResampler resampler = WavResampler.Linear)
    {
        if (bytes.Length < 12 ||
            !bytes[..4].SequenceEqual("RIFF"u8) ||
            !bytes[8..12].SequenceEqual("WAVE"u8))
        {
            throw new InvalidDataException("The data is not a RIFF/WAVE file.");
        }

        var declaredSize = BinaryPrimitives.ReadUInt32LittleEndian(bytes[4..]);
        var riffEnd = Math.Min(bytes.Length, (long)declaredSize + 8);

        var format = default(WaveFormat);
        var haveFormat = false;
        var dataStart = -1;
        var dataLength = 0;

        var position = 12;
        while (position + 8 <= riffEnd)
        {
            var chunkId = bytes.Slice(position, 4);
            var chunkSize = (long)BinaryPrimitives.ReadUInt32LittleEndian(bytes[(position + 4)..]);
            var payload = position + 8;

            if (payload + chunkSize > riffEnd)
            {
                // A truncated final chunk is common in streamed rips; a truncated fmt chunk is not
                // recoverable, so only the data chunk is allowed to be short.
                chunkSize = riffEnd - payload;
                if (chunkSize < 0)
                {
                    throw new InvalidDataException("The WAVE chunk extends past the end of the file.");
                }
            }

            if (chunkId.SequenceEqual("fmt "u8))
            {
                format = ReadFormat(bytes.Slice(payload, (int)chunkSize));
                haveFormat = true;
            }
            else if (chunkId.SequenceEqual("data"u8))
            {
                dataStart = payload;
                dataLength = (int)chunkSize;
            }

            // Chunks are word aligned: an odd payload is followed by one pad byte.
            position = payload + (int)chunkSize + ((int)chunkSize & 1);
        }

        if (!haveFormat)
        {
            throw new InvalidDataException("The WAVE file has no 'fmt ' chunk.");
        }

        if (dataStart < 0)
        {
            throw new InvalidDataException("The WAVE file has no 'data' chunk.");
        }

        var samples = Decode(bytes.Slice(dataStart, dataLength), format);
        var sourceRate = format.SampleRate;
        var actualRate = targetSampleRate ?? sourceRate;

        if (actualRate is < 8000 or > 192000)
        {
            throw new ArgumentOutOfRangeException(
                nameof(targetSampleRate),
                targetSampleRate,
                "The analysis sample rate must be between 8000 and 192000 Hz.");
        }

        if (actualRate != sourceRate && sourceRate > 0 && samples.Length > 1)
        {
            samples = Resample(samples, sourceRate, actualRate, resampler);
        }

        return new WavData(actualRate, samples, sourceRate, format.Channels);
    }

    /// <summary>
    /// Linear interpolation resampling, exposed for tests that want to check the exact kernel.
    /// </summary>
    /// <param name="samples">Source samples.</param>
    /// <param name="fromRate">Source rate in Hz; must be positive.</param>
    /// <param name="toRate">Destination rate in Hz; must be positive.</param>
    /// <param name="resampler">Kernel to use.</param>
    /// <exception cref="ArgumentOutOfRangeException">A rate is not positive.</exception>
    public static float[] Resample(
        ReadOnlySpan<float> samples,
        int fromRate,
        int toRate,
        WavResampler resampler = WavResampler.Linear)
    {
        if (fromRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(fromRate), fromRate, "Rate must be positive.");
        }

        if (toRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(toRate), toRate, "Rate must be positive.");
        }

        if (samples.IsEmpty || fromRate == toRate)
        {
            return samples.ToArray();
        }

        var length = Math.Max(1, (int)Math.Round(samples.Length * (double)toRate / fromRate));
        var result = new float[length];
        var step = (double)fromRate / toRate;

        for (var i = 0; i < length; i++)
        {
            var source = i * step;
            var index = (int)source;
            var fraction = (float)(source - index);

            if (index >= samples.Length - 1)
            {
                result[i] = samples[^1];
                continue;
            }

            result[i] = resampler == WavResampler.Cubic
                ? Cubic(samples, index, fraction)
                : Linear(samples[index], samples[index + 1], fraction);
        }

        return result;
    }

    private static float Linear(float a, float b, float fraction) => a + ((b - a) * fraction);

    private static float Cubic(ReadOnlySpan<float> samples, int index, float fraction)
    {
        // Catmull-Rom with edge clamps: the neighbours outside the clip fold onto the boundary sample.
        var p0 = samples[Math.Max(0, index - 1)];
        var p1 = samples[index];
        var p2 = samples[index + 1];
        var p3 = samples[Math.Min(samples.Length - 1, index + 2)];
        var t = fraction;
        var t2 = t * t;
        var t3 = t2 * t;

        return 0.5f * ((2f * p1) +
            ((-p0 + p2) * t) +
            (((2f * p0) - (5f * p1) + (4f * p2) - p3) * t2) +
            (((-p0) + (3f * p1) - (3f * p2) + p3) * t3));
    }

    private readonly record struct WaveFormat(
        int FormatTag,
        int Channels,
        int SampleRate,
        int BlockAlign,
        int BitsPerSample);

    private static WaveFormat ReadFormat(ReadOnlySpan<byte> chunk)
    {
        if (chunk.Length < 16)
        {
            throw new InvalidDataException("The 'fmt ' chunk is shorter than 16 bytes.");
        }

        var formatTag = BinaryPrimitives.ReadUInt16LittleEndian(chunk);
        var channels = BinaryPrimitives.ReadUInt16LittleEndian(chunk[2..]);
        var sampleRate = BinaryPrimitives.ReadInt32LittleEndian(chunk[4..]);
        var blockAlign = BinaryPrimitives.ReadUInt16LittleEndian(chunk[12..]);
        var bitsPerSample = BinaryPrimitives.ReadUInt16LittleEndian(chunk[14..]);

        var effectiveTag = formatTag;
        if (formatTag == FormatExtensible)
        {
            // WAVE_FORMAT_EXTENSIBLE keeps the real tag in the first two bytes of the SubFormat GUID
            // (cbSize at 16, valid bits at 18, channel mask at 20, GUID from 24).
            if (chunk.Length < 26)
            {
                throw new InvalidDataException("The WAVE_FORMAT_EXTENSIBLE chunk is truncated.");
            }

            effectiveTag = BinaryPrimitives.ReadUInt16LittleEndian(chunk[24..]);
        }

        if (effectiveTag is not (FormatPcm or FormatFloat))
        {
            throw new InvalidDataException(
                $"Unsupported WAVE format tag {formatTag}; only PCM and IEEE float are decoded.");
        }

        if (channels < 1)
        {
            throw new InvalidDataException("The WAVE file declares zero channels.");
        }

        if (sampleRate < 1000)
        {
            throw new InvalidDataException($"The WAVE file declares an implausible rate of {sampleRate} Hz.");
        }

        if (bitsPerSample is not (8 or 16 or 24 or 32 or 64))
        {
            throw new InvalidDataException($"Unsupported sample width of {bitsPerSample} bits.");
        }

        if (effectiveTag == FormatFloat && bitsPerSample is not (32 or 64))
        {
            throw new InvalidDataException("A float WAVE file must store 32 or 64 bits per sample.");
        }

        var minimumBlock = channels * (bitsPerSample / 8);
        if (blockAlign < minimumBlock)
        {
            throw new InvalidDataException(
                $"Block alignment {blockAlign} is smaller than {minimumBlock} bytes per frame.");
        }

        return new WaveFormat(effectiveTag, channels, sampleRate, blockAlign, bitsPerSample);
    }

    private static float[] Decode(ReadOnlySpan<byte> data, in WaveFormat format)
    {
        var frames = data.Length / format.BlockAlign;
        if (frames == 0)
        {
            return [];
        }

        var samples = new float[frames];
        var bytesPerSample = format.BitsPerSample / 8;
        var frameOffset = 0;

        for (var frame = 0; frame < frames; frame++)
        {
            var sum = 0f;

            for (var channel = 0; channel < format.Channels; channel++)
            {
                var offset = frameOffset + (channel * bytesPerSample);
                sum += format.FormatTag == FormatFloat
                    ? ReadFloat(data, offset, bytesPerSample)
                    : ReadPcm(data, offset, bytesPerSample);
            }

            samples[frame] = sum / format.Channels;
            frameOffset += format.BlockAlign;
        }

        return samples;
    }

    private static float ReadPcm(ReadOnlySpan<byte> data, int offset, int bytesPerSample) =>
        bytesPerSample switch
        {
            // 8-bit PCM is unsigned with 128 as silence, unlike every wider width.
            1 => (data[offset] - 128) / 128f,
            2 => BinaryPrimitives.ReadInt16LittleEndian(data[offset..]) / 32768f,
            3 => ReadInt24(data, offset) / 8388608f,
            4 => BinaryPrimitives.ReadInt32LittleEndian(data[offset..]) / 2147483648f,
            _ => 0f,
        };

    private static float ReadFloat(ReadOnlySpan<byte> data, int offset, int bytesPerSample)
    {
        var value = bytesPerSample == 8
            ? (float)BinaryPrimitives.ReadDoubleLittleEndian(data[offset..])
            : BinaryPrimitives.ReadSingleLittleEndian(data[offset..]);

        return float.IsFinite(value) ? Math.Clamp(value, -1f, 1f) : 0f;
    }

    private static int ReadInt24(ReadOnlySpan<byte> data, int offset)
    {
        // Little-endian 24-bit two's complement, sign-extended into an int.
        var value = data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16);
        return (value & 0x800000) != 0 ? value | unchecked((int)0xFF000000) : value;
    }
}