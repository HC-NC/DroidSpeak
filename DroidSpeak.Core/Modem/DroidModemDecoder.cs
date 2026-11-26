namespace DroidSpeak.Modem;

/// <summary>
/// Receives droid acoustic-modem packets. The decoder high-pass filters the signal to remove the
/// low-frequency droid voice, locates the FSK preamble, demodulates the bits with an in-house FFT
/// and validates the payload with CRC8.
/// </summary>
public static class DroidModemDecoder
{
    private const float TwoPi = MathF.PI * 2f;
    private const int FftSize = 256;
    private const int AnalysisWindow = 256;
    private const int FrameSize = 256;
    private const int FrameHop = 128;
    private const float BandLow = 3200f;
    private const float BandHigh = 4800f;

    /// <summary>
    /// Attempts to recover a text packet from <paramref name="samples"/>. The v0.9 modes are
    /// tried first, then the legacy v0.8 binary-FSK format for older recordings.
    /// </summary>
    /// <param name="samples">Mono audio, either clean modem tones or droid speech with a hidden packet.</param>
    /// <param name="sampleRate">Sample rate in Hz (at least 10000).</param>
    /// <returns>The decoded text, or <see langword="null"/> when no valid packet was found.</returns>
    public static string? Decode(ReadOnlySpan<float> samples, int sampleRate = 44100) =>
        TryDecode(samples, sampleRate, out var result) ? result.Text : null;

    /// <summary>
    /// Attempts to recover a v0.9 packet and returns FEC, CRC, and clock diagnostics.
    /// </summary>
    public static bool TryDecode(
        ReadOnlySpan<float> samples,
        int sampleRate,
        out ModemDecodeResult result)
    {
        if (AcousticDemodulator.TryDecode(samples, sampleRate, out result))
        {
            return true;
        }

        var legacy = DecodeLegacy(samples, sampleRate);
        if (legacy is null)
        {
            result = default;
            return false;
        }

        result = new ModemDecodeResult(
            legacy,
            ModemMode.HighBandOverlay,
            CorrectedBits: 0,
            CrcValid: true,
            ClockDriftPpm: 0,
            SamplesConsumed: samples.Length,
            Legacy: true);
        return true;
    }

    /// <summary>
    /// Attempts to recover a packet from one specific v0.9 mode.
    /// </summary>
    public static string? Decode(ReadOnlySpan<float> samples, int sampleRate, ModemMode mode) =>
        AcousticDemodulator.TryDecode(samples, sampleRate, mode, out var result) ? result.Text : null;

    private static string? DecodeLegacy(ReadOnlySpan<float> samples, int sampleRate = 44100)
    {
        if (sampleRate < DroidModemProtocol.MinimumSampleRate || sampleRate > 192000)
        {
            return null;
        }

        var bitSamples = DroidModemProtocol.BitDuration * sampleRate;
        var preambleBits = BuildPreambleBits();
        var minimumSamples = (int)Math.Ceiling(preambleBits.Length * bitSamples);
        if (samples.Length < minimumSamples)
        {
            return null;
        }

        var filtered = HighPass(samples, sampleRate);
        var fftReal = new float[FftSize];
        var fftImag = new float[FftSize];
        var onset = FindOnset(filtered, sampleRate, fftReal, fftImag);

        var searchStart = Math.Max(0, onset - FrameSize);
        var searchEnd = Math.Min(filtered.Length - 1, onset + FrameSize);
        var step = Math.Max(1, (int)(bitSamples / 32));

        for (var start = searchStart; start <= searchEnd; start += step)
        {
            if (TryDecodeAt(
                    filtered,
                    start,
                    bitSamples,
                    sampleRate,
                    preambleBits,
                    fftReal,
                    fftImag,
                    out var text))
            {
                return text;
            }
        }

        // Coarse fallback for signals whose onset detector was confused by strong voice energy.
        var coarseStep = Math.Max(1, (int)(bitSamples / 2));
        var limit = Math.Min(filtered.Length, sampleRate * 15);
        for (var start = 0; start + minimumSamples <= limit; start += coarseStep)
        {
            if (start >= searchStart && start <= searchEnd)
            {
                continue;
            }

            if (TryDecodeAt(
                    filtered,
                    start,
                    bitSamples,
                    sampleRate,
                    preambleBits,
                    fftReal,
                    fftImag,
                    out var text))
            {
                return text;
            }
        }

        return null;
    }

    private static int[] BuildPreambleBits()
    {
        var bits = new List<int>(DroidModemProtocol.Preamble.Length * 8);
        foreach (var value in DroidModemProtocol.Preamble)
        {
            DroidModemProtocol.AppendByteBits(bits, value);
        }

        return bits.ToArray();
    }

    private static bool TryDecodeAt(
        ReadOnlySpan<float> samples,
        int start,
        double bitSamples,
        int sampleRate,
        int[] preambleBits,
        float[] fftReal,
        float[] fftImag,
        out string? text)
    {
        text = null;

        if (start < 0 ||
            start + (preambleBits.Length + 24) * bitSamples > samples.Length)
        {
            return false;
        }

        var leaderOnes = 0;
        for (var bitIndex = 0; bitIndex < 16; bitIndex++)
        {
            if (!TryReadBit(
                    samples,
                    start,
                    bitSamples,
                    bitIndex,
                    sampleRate,
                    fftReal,
                    fftImag,
                    out var bit,
                    out _))
            {
                return false;
            }

            leaderOnes += bit;
        }

        // The very first symbols may sit in the speech attack and be weak; the full
        // preamble below is the real synchronization check.
        if (leaderOnes < 12)
        {
            return false;
        }

        var matches = 0;
        for (var bitIndex = 0; bitIndex < preambleBits.Length; bitIndex++)
        {
            if (!TryReadBit(
                    samples,
                    start,
                    bitSamples,
                    bitIndex,
                    sampleRate,
                    fftReal,
                    fftImag,
                    out var bit,
                    out _))
            {
                return false;
            }

            if (bit == preambleBits[bitIndex])
            {
                matches++;
            }
        }

        if (matches < preambleBits.Length - 6)
        {
            return false;
        }

        const int codewordBits = 12;

        if (!TryReadByte(
                samples,
                start,
                bitSamples,
                preambleBits.Length,
                sampleRate,
                fftReal,
                fftImag,
                out var length))
        {
            return false;
        }

        var frameBits = codewordBits + length * codewordBits + codewordBits;
        if (start + (preambleBits.Length + frameBits) * bitSamples > samples.Length)
        {
            return false;
        }

        var dataCodewords = length + 1; // payload bytes plus CRC
        var streamBits = dataCodewords * codewordBits;
        var streamOffset = preambleBits.Length + codewordBits;
        var stream = new int[streamBits];

        for (var i = 0; i < streamBits; i++)
        {
            if (!TryReadBit(
                    samples,
                    start,
                    bitSamples,
                    streamOffset + i,
                    sampleRate,
                    fftReal,
                    fftImag,
                    out stream[i],
                    out _))
            {
                return false;
            }
        }

        var payload = new byte[length];
        byte crc = 0;
        Span<int> codeword = stackalloc int[codewordBits];

        for (var index = 0; index < dataCodewords; index++)
        {
            for (var bitPosition = 0; bitPosition < codewordBits; bitPosition++)
            {
                codeword[bitPosition] = stream[bitPosition * dataCodewords + index];
            }

            if (!DroidModemProtocol.TryDecodeHammingByte(codeword, out var value))
            {
                return false;
            }

            if (index < length)
            {
                payload[index] = value;
            }
            else
            {
                crc = value;
            }
        }

        var crcInput = new byte[length + 1];
        crcInput[0] = length;
        payload.CopyTo(crcInput, 1);

        if (DroidModemProtocol.ComputeCrc8(crcInput) != crc)
        {
            return false;
        }

        return DroidModemProtocol.TryDecodeUtf8(payload, out text);
    }

    private static bool TryReadByte(
        ReadOnlySpan<float> samples,
        int start,
        double bitSamples,
        int bitOffset,
        int sampleRate,
        float[] fftReal,
        float[] fftImag,
        out byte value)
    {
        Span<int> bits = stackalloc int[12];
        for (var i = 0; i < 12; i++)
        {
            if (!TryReadBit(
                    samples,
                    start,
                    bitSamples,
                    bitOffset + i,
                    sampleRate,
                    fftReal,
                    fftImag,
                    out bits[i],
                    out _))
            {
                value = 0;
                return false;
            }
        }

        return DroidModemProtocol.TryDecodeHammingByte(bits, out value);
    }

    private static bool TryReadBit(
        ReadOnlySpan<float> samples,
        int start,
        double bitSamples,
        int bitIndex,
        int sampleRate,
        float[] fftReal,
        float[] fftImag,
        out int bit,
        out float confidence)
    {
        bit = 0;
        confidence = 0f;

        var bitStart = start + (int)Math.Round(bitIndex * bitSamples);
        var bitLength = (int)bitSamples;
        var analysisLength = Math.Min(AnalysisWindow, bitLength);
        if (analysisLength < 16)
        {
            return false;
        }

        var windowStart = bitStart + (bitLength - analysisLength) / 2;
        if (windowStart < 0 || windowStart + analysisLength > samples.Length)
        {
            return false;
        }

        Array.Clear(fftReal);
        Array.Clear(fftImag);

        for (var i = 0; i < analysisLength; i++)
        {
            var hann = 0.5f - 0.5f * MathF.Cos(TwoPi * i / (analysisLength - 1));
            fftReal[i] = samples[windowStart + i] * hann;
        }

        FastFourierTransform.Forward(fftReal, fftImag);

        var zeroMagnitude = BandMagnitude(
            fftReal,
            fftImag,
            sampleRate,
            DroidModemProtocol.Bit0Frequency);
        var oneMagnitude = BandMagnitude(
            fftReal,
            fftImag,
            sampleRate,
            DroidModemProtocol.Bit1Frequency);

        bit = oneMagnitude > zeroMagnitude ? 1 : 0;
        confidence = (oneMagnitude - zeroMagnitude) / (oneMagnitude + zeroMagnitude + 1e-9f);
        return true;
    }

    private static float BandMagnitude(float[] real, float[] imaginary, int sampleRate, float frequency)
    {
        var center = (int)MathF.Round(frequency * FftSize / sampleRate);
        var power = 0f;

        for (var bin = center - 1; bin <= center + 1; bin++)
        {
            if (bin > 0 && bin < FftSize / 2)
            {
                power += real[bin] * real[bin] + imaginary[bin] * imaginary[bin];
            }
        }

        return MathF.Sqrt(power);
    }

    private static int FindOnset(
        ReadOnlySpan<float> samples,
        int sampleRate,
        float[] fftReal,
        float[] fftImag)
    {
        if (samples.Length < FrameSize)
        {
            return 0;
        }

        var frameCount = (samples.Length - FrameSize) / FrameHop + 1;
        var energies = new float[frameCount];
        var hann = new float[FrameSize];
        FastFourierTransform.FillHannWindow(hann, FrameSize);

        var lowBin = Math.Max(1, (int)MathF.Ceiling(BandLow * FrameSize / sampleRate));
        var highBin = Math.Min(
            FrameSize / 2 - 1,
            (int)MathF.Floor(BandHigh * FrameSize / sampleRate));
        var maximum = 0f;

        for (var frame = 0; frame < frameCount; frame++)
        {
            Array.Clear(fftReal);
            Array.Clear(fftImag);

            var offset = frame * FrameHop;
            for (var i = 0; i < FrameSize; i++)
            {
                fftReal[i] = samples[offset + i] * hann[i];
            }

            FastFourierTransform.Forward(fftReal, fftImag);

            var energy = 0f;
            for (var bin = lowBin; bin <= highBin; bin++)
            {
                energy += fftReal[bin] * fftReal[bin] + fftImag[bin] * fftImag[bin];
            }

            energies[frame] = energy;
            maximum = MathF.Max(maximum, energy);
        }

        if (maximum <= 1e-12f)
        {
            return 0;
        }

        var threshold = maximum * 0.06f;
        var requiredRun = Math.Max(4, (int)(0.12 * sampleRate / FrameHop));

        for (var frame = 0; frame + requiredRun <= frameCount; frame++)
        {
            var sustained = true;
            for (var k = 0; k < requiredRun; k++)
            {
                if (energies[frame + k] < threshold)
                {
                    sustained = false;
                    break;
                }
            }

            if (sustained)
            {
                return frame * FrameHop;
            }
        }

        for (var frame = 0; frame < frameCount; frame++)
        {
            if (energies[frame] >= threshold)
            {
                return frame * FrameHop;
            }
        }

        return 0;
    }

    private static float[] HighPass(ReadOnlySpan<float> input, int sampleRate)
    {
        var output = input.ToArray();
        ApplyHighPassStage(output, sampleRate, 2800f);
        ApplyHighPassStage(output, sampleRate, 2800f);
        return output;
    }

    private static void ApplyHighPassStage(float[] buffer, int sampleRate, float cutoff)
    {
        var w0 = 2.0 * Math.PI * cutoff / sampleRate;
        var cosine = Math.Cos(w0);
        var sine = Math.Sin(w0);
        const double q = 0.7071067811865476;
        var alpha = sine / (2.0 * q);
        var a0 = 1.0 + alpha;

        var b0 = (1.0 + cosine) / 2.0 / a0;
        var b1 = -(1.0 + cosine) / a0;
        var b2 = (1.0 + cosine) / 2.0 / a0;
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
