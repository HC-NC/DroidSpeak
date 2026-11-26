namespace DroidSpeak.Modem;

/// <summary>
/// Encodes arbitrary UTF-8 text as an FSK acoustic packet in the 3200–4800 Hz band and can
/// hide that packet inside droid speech.
/// <para>
/// Packet layout: <c>[preamble][length:1][UTF-8 payload][CRC8:1]</c>. In masked mode the FSK
/// carrier follows the speech envelope, so it is only audible while the droid is talking.
/// </para>
/// </summary>
public static class DroidModemEncoder
{
    /// <summary>Default modem amplitude relative to the speech peak.</summary>
    public const float DefaultMaskLevel = 0.15f;

    private const float TwoPi = MathF.PI * 2f;
    private const float ModemPeak = 1f;

    /// <summary>
    /// Encodes <paramref name="text"/> with a v0.9 rich-spectrum mode.
    /// </summary>
    /// <param name="text">Text to transmit (UTF-8, at most 255 bytes).</param>
    /// <param name="mode">Physical-layer mode.</param>
    /// <param name="sampleRate">Output sample rate in Hz.</param>
    public static float[] Encode(string text, ModemMode mode, int sampleRate = 44100) =>
        AcousticModulator.Encode(text, mode, sampleRate);

    /// <summary>
    /// Encodes <paramref name="text"/> as a clean FSK signal.</summary>
    /// <param name="text">Text to transmit (UTF-8, at most 255 bytes).</param>
    /// <param name="sampleRate">Output sample rate in Hz (at least 10000).</param>
    public static float[] Encode(string text, int sampleRate = 44100)
    {
        ArgumentNullException.ThrowIfNull(text);
        ValidateSampleRate(sampleRate);

        var payload = DroidModemProtocol.EncodeUtf8(text);
        var frame = new List<byte>(payload.Length + 2)
        {
            (byte)payload.Length,
        };
        frame.AddRange(payload);
        frame.Add(DroidModemProtocol.ComputeCrc8(frame.ToArray()));

        var bits = new List<int>(DroidModemProtocol.Preamble.Length * 8 + frame.Count * 12);
        foreach (var value in DroidModemProtocol.Preamble)
        {
            DroidModemProtocol.AppendByteBits(bits, value);
        }

        var frameBits = new List<int>(frame.Count * 12);
        foreach (var value in frame)
        {
            DroidModemProtocol.AppendHammingByte(frameBits, value);
        }

        // The length codeword is sent contiguously so the receiver knows how many codewords
        // follow. Payload and CRC codewords are bit-interleaved: a short burst of voice
        // interference then damages at most one bit per byte, which Hamming can repair.
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

        var bitSamples = DroidModemProtocol.BitDuration * sampleRate;
        var totalSamples = (int)Math.Ceiling(bits.Count * bitSamples) + sampleRate / 100;
        var buffer = new float[totalSamples];
        var phase = 0d;

        for (var bitIndex = 0; bitIndex < bits.Count; bitIndex++)
        {
            var start = (int)Math.Round(bitIndex * bitSamples);
            var end = (int)Math.Round((bitIndex + 1) * bitSamples);
            end = Math.Min(end, totalSamples);
            var frequency = bits[bitIndex] == 0
                ? DroidModemProtocol.Bit0Frequency
                : DroidModemProtocol.Bit1Frequency;

            for (var n = start; n < end; n++)
            {
                phase += TwoPi * frequency / sampleRate;
                buffer[n] = MathF.Sin((float)phase) * ModemPeak;
            }
        }

        ApplyEdgeFade(buffer, sampleRate);
        return buffer;
    }

    /// <summary>
    /// Encodes <paramref name="text"/> and mixes the FSK packet into <paramref name="speech"/>.
    /// The carrier follows the speech envelope and is scaled to roughly 13% of the speech peak.
    /// If the speech is shorter than the packet, active droid-like babble is tiled to cover the
    /// remaining transmission.
    /// </summary>
    /// <param name="text">Text to hide in the speech.</param>
    /// <param name="speech">Mono speech samples in <c>[-1, 1]</c>.</param>
    /// <param name="sampleRate">Sample rate in Hz (at least 10000).</param>
    /// <param name="maskLevel">Modem amplitude relative to the speech peak (0.10–0.15 is recommended).</param>
    public static float[] EncodeIntoSpeech(
        string text,
        ReadOnlySpan<float> speech,
        int sampleRate = 44100,
        float maskLevel = DefaultMaskLevel)
    {
        ArgumentNullException.ThrowIfNull(text);
        ValidateSampleRate(sampleRate);

        var modem = Encode(text, sampleRate);
        if (speech.IsEmpty)
        {
            return modem;
        }

        var level = Math.Clamp(maskLevel, 0.05f, 0.30f);
        var source = speech.ToArray();

        // Keep the droid's low-frequency character but clear the 3200–4800 Hz band for the
        // modem. Cascaded stages give enough stop-band rejection that even bright, whistling
        // speech (the Tiny profile) cannot flip modem bits.
        LowPassSpeech(source, sampleRate, 3200f, stages: 5);

        var peak = 0f;
        foreach (var sample in source)
        {
            if (float.IsFinite(sample))
            {
                peak = MathF.Max(peak, MathF.Abs(sample));
            }
        }

        if (peak < 1e-5f)
        {
            return modem;
        }

        var firstActive = FindFirstActive(source, peak);
        var lastActive = FindLastActive(source, peak);
        var activeStart = Math.Max(0, firstActive);
        var activeLength = Math.Max(1, lastActive - activeStart + 1);
        var packetStart = Math.Max(0, firstActive);
        var requiredLength = packetStart + modem.Length + sampleRate / 50;

        if (source.Length < requiredLength)
        {
            //source = TileActiveSpeech(source, activeStart, activeLength, requiredLength, sampleRate);
            Array.Resize(ref source, requiredLength);
        }

        var envelope = BuildEnvelope(source, peak, sampleRate);
        var result = new float[requiredLength];
        Array.Copy(source, result, Math.Min(source.Length, requiredLength));

        var modemGain = level * peak;
        var packetEnd = Math.Min(requiredLength, packetStart + modem.Length);
        var fadeSamples = Math.Max(1, sampleRate / 200);

        for (var i = 0; i < modem.Length; i++)
        {
            var index = packetStart + i;
            if (index >= result.Length)
            {
                break;
            }

            // A quiet activity floor bridges short pauses and keeps the carrier decodable.
            // At 16% of the 15% mask level it stays inaudible, while the envelope above
            // the floor makes the packet rise and fall with the droid's speech.
            var activity = MathF.Max(envelope[index], 0.16f);

            var edge = 1f;
            if (i < fadeSamples)
            {
                edge = i / (float)fadeSamples;
            }
            else if (index >= packetEnd - fadeSamples)
            {
                edge = MathF.Max(0f, (packetEnd - index) / (float)fadeSamples);
            }

            result[index] += modem[i] * modemGain * activity * edge;
        }

        NormalizePeak(result);
        return result;
    }

    /// <summary>
    /// Encodes <paramref name="text"/> with a v0.9 rich-spectrum mode and places it in
    /// <paramref name="speech"/> without filtering or normalizing the voice.
    /// <para>
    /// <see cref="ModemMode.HighBandOverlay"/> adds the packet above the audible voice band.
    /// <see cref="ModemMode.AudibleChirpStream"/> uses a natural pause when one is long enough
    /// and otherwise appends the chirps after the utterance.
    /// </para>
    /// </summary>
    /// <param name="text">Text to transmit (UTF-8, at most 255 bytes).</param>
    /// <param name="speech">Mono speech samples in <c>[-1, 1]</c>.</param>
    /// <param name="mode">Physical-layer mode.</param>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="maskLevel">Relative modem level (0.05–0.30).</param>
    public static float[] EncodeIntoSpeech(
        string text,
        ReadOnlySpan<float> speech,
        ModemMode mode,
        int sampleRate = 44100,
        float maskLevel = DefaultMaskLevel) =>
        AcousticModulator.MixIntoSpeech(text, speech, mode, sampleRate, maskLevel);

    private static void ValidateSampleRate(int sampleRate)
    {
        if (sampleRate < DroidModemProtocol.MinimumSampleRate || sampleRate > 192000)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sampleRate),
                sampleRate,
                $"The modem needs a sample rate between {DroidModemProtocol.MinimumSampleRate} and 192000 Hz.");
        }
    }

    private static int FindFirstActive(float[] samples, float peak)
    {
        var threshold = peak * 0.05f;
        for (var i = 0; i < samples.Length; i++)
        {
            if (MathF.Abs(samples[i]) > threshold)
            {
                return i;
            }
        }

        return 0;
    }

    private static int FindLastActive(float[] samples, float peak)
    {
        var threshold = peak * 0.05f;
        for (var i = samples.Length - 1; i >= 0; i--)
        {
            if (MathF.Abs(samples[i]) > threshold)
            {
                return i;
            }
        }

        return samples.Length - 1;
    }

    private static float[] TileActiveSpeech(
        float[] speech,
        int activeStart,
        int activeLength,
        int requiredLength,
        int sampleRate)
    {
        var result = new float[requiredLength];
        Array.Copy(speech, result, Math.Min(speech.Length, requiredLength));

        if (activeLength <= 1)
        {
            return result;
        }

        var fade = Math.Min(sampleRate / 200, Math.Max(1, activeLength / 4));
        var position = speech.Length;

        while (position < requiredLength)
        {
            var copyStart = position;
            var count = Math.Min(activeLength, requiredLength - position);

            for (var i = 0; i < count; i++)
            {
                var sample = speech[activeStart + i];

                if (i < fade && copyStart >= fade)
                {
                    var blend = i / (float)fade;
                    var previous = result[copyStart - fade + i];
                    sample = previous * (1f - blend) + sample * blend;
                }

                result[position++] = sample;
            }
        }

        return result;
    }

    private static float[] BuildEnvelope(float[] samples, float peak, int sampleRate)
    {
        var envelope = new float[samples.Length];
        var attack = MathF.Exp(-1f / (0.008f * sampleRate));
        var release = MathF.Exp(-1f / (0.120f * sampleRate));
        var level = 0f;

        for (var i = 0; i < samples.Length; i++)
        {
            var target = MathF.Abs(samples[i]);
            var coefficient = target > level ? attack : release;
            level = coefficient * level + (1f - coefficient) * target;
            envelope[i] = Math.Clamp(level / peak, 0f, 1f);
        }

        return envelope;
    }

    private static void LowPassSpeech(float[] buffer, int sampleRate, float cutoff, int stages)
    {
        for (var stage = 0; stage < stages; stage++)
        {
            var w0 = 2.0 * Math.PI * cutoff / sampleRate;
            var cosine = Math.Cos(w0);
            var sine = Math.Sin(w0);
            const double q = 0.7071067811865476;
            var alpha = sine / (2.0 * q);
            var a0 = 1.0 + alpha;

            var b0 = (1.0 - cosine) / 2.0 / a0;
            var b1 = (1.0 - cosine) / a0;
            var b2 = (1.0 - cosine) / 2.0 / a0;
            var a1 = -2.0 * cosine / a0;
            var a2 = (1.0 - alpha) / a0;

            double x1 = 0, x2 = 0, y1 = 0, y2 = 0;

            for (var i = 0; i < buffer.Length; i++)
            {
                var x = buffer[i];
                var y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;

                buffer[i] = (float)y;
                x2 = x1;
                x1 = x;
                y2 = y1;
                y1 = y;
            }
        }
    }

    private static void ApplyEdgeFade(float[] buffer, int sampleRate)
    {
        var fade = Math.Max(1, sampleRate / 200);
        var length = buffer.Length;

        for (var i = 0; i < fade && i < length; i++)
        {
            var gain = i / (float)fade;
            buffer[i] *= gain;
            buffer[length - 1 - i] *= gain;
        }
    }

    private static void NormalizePeak(float[] buffer)
    {
        var peak = 0f;
        foreach (var sample in buffer)
        {
            if (float.IsFinite(sample))
            {
                peak = MathF.Max(peak, MathF.Abs(sample));
            }
        }

        if (peak > 0.98f)
        {
            var gain = 0.98f / peak;
            for (var i = 0; i < buffer.Length; i++)
            {
                buffer[i] *= gain;
            }
        }
    }
}
