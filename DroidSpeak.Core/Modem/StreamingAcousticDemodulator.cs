namespace DroidSpeak.Modem;

/// <summary>
/// Incremental v0.9 receiver for live microphone buffers. Feed arbitrary chunks, then call
/// <see cref="TryReadPacket"/> until it returns <see langword="false"/>; the next call resumes
/// exactly after the last accepted packet.
/// </summary>
public sealed class StreamingAcousticDemodulator
{
    private const int MaximumBufferedSeconds = 30;

    private float[] _buffer;
    private int _count;
    private readonly int _maximumBufferedSamples;

    /// <summary>Creates a receiver for one capture device sample rate.</summary>
    /// <param name="sampleRate">Capture sample rate in Hz (10000–192000).</param>
    public StreamingAcousticDemodulator(int sampleRate = 44100)
    {
        if (sampleRate < DroidModemProtocol.MinimumSampleRate || sampleRate > 192000)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sampleRate),
                sampleRate,
                "The acoustic modem needs a sample rate between 10000 and 192000 Hz.");
        }

        SampleRate = sampleRate;
        _maximumBufferedSamples = sampleRate * MaximumBufferedSeconds;
        _buffer = new float[Math.Max(4096, Math.Min(sampleRate, _maximumBufferedSamples))];
    }

    /// <summary>Capture sample rate in Hz.</summary>
    public int SampleRate { get; }

    /// <summary>Number of samples currently waiting for a packet.</summary>
    public int BufferedSamples => _count;

    /// <summary>Appends a microphone chunk. The buffer is bounded to the last 30 seconds.</summary>
    public void Feed(ReadOnlySpan<float> samples)
    {
        if (samples.IsEmpty)
        {
            return;
        }

        if (samples.Length >= _maximumBufferedSamples)
        {
            if (_buffer.Length < _maximumBufferedSamples)
            {
                _buffer = new float[_maximumBufferedSamples];
            }

            _count = _maximumBufferedSamples;
            samples[^_count..].CopyTo(_buffer.AsSpan(0, _count));
            return;
        }

        if (_count + samples.Length > _buffer.Length)
        {
            var required = Math.Min(_maximumBufferedSamples, _count + samples.Length);
            var capacity = _buffer.Length;
            while (capacity < required)
            {
                capacity = Math.Min(_maximumBufferedSamples, Math.Max(capacity * 2, required));
            }

            var grown = new float[capacity];
            _buffer.AsSpan(0, _count).CopyTo(grown);
            _buffer = grown;
        }

        samples.CopyTo(_buffer.AsSpan(_count));
        _count += samples.Length;

        if (_count > _maximumBufferedSamples)
        {
            var drop = _count - _maximumBufferedSamples;
            _buffer.AsSpan(drop, _maximumBufferedSamples).CopyTo(_buffer);
            _count = _maximumBufferedSamples;
        }
    }

    /// <summary>
    /// Attempts to decode the next complete packet from the buffered audio.
    /// </summary>
    public bool TryReadPacket(out ModemDecodeResult result)
    {
        if (_count == 0)
        {
            result = default;
            return false;
        }

        if (!AcousticDemodulator.TryDecode(_buffer.AsSpan(0, _count), SampleRate, out result))
        {
            return false;
        }

        var consumed = Math.Clamp(result.SamplesConsumed, 1, _count);
        _buffer.AsSpan(consumed, _count - consumed).CopyTo(_buffer);
        _count -= consumed;
        return true;
    }

    /// <summary>Discards all buffered audio.</summary>
    public void Reset() => _count = 0;
}
