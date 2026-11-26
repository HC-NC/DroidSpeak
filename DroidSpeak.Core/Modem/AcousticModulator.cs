namespace DroidSpeak.Modem;

/// <summary>
/// Physical-layer modulator shared by the clean-packet and speech-embedding APIs.
/// </summary>
internal static class AcousticModulator
{
    private const float TwoPi = MathF.PI * 2f;
    private const float PacketAmplitude = 0.9f;
    private const float DefaultRelativeLevel = 0.15f;
    private const float GapSeconds = 0.02f;

    public static float[] Encode(string text, ModemMode mode, int sampleRate)
    {
        ArgumentNullException.ThrowIfNull(text);
        ValidateSampleRate(mode, sampleRate);

        var frame = AcousticFrameProtocol.BuildFrame(text, mode);
        var symbols = BuildSymbols(frame, mode);
        return Render(symbols, mode, sampleRate, PacketAmplitude);
    }

    public static float[] MixIntoSpeech(
        string text,
        ReadOnlySpan<float> speech,
        ModemMode mode,
        int sampleRate,
        float maskLevel)
    {
        ArgumentNullException.ThrowIfNull(text);
        ValidateSampleRate(mode, sampleRate);

        var modem = Encode(text, mode, sampleRate);
        if (speech.IsEmpty)
        {
            return modem;
        }

        var source = speech.ToArray();
        var peak = Peak(source);
        if (peak < 1e-5f)
        {
            return modem;
        }

        var level = float.IsFinite(maskLevel)
            ? Math.Clamp(maskLevel, 0.05f, 0.30f)
            : DefaultRelativeLevel;
        var relative = MathF.Max(level, DefaultRelativeLevel);

        if (mode == ModemMode.HighBandOverlay)
        {
            return OverlayHighBand(source, modem, peak, relative, sampleRate);
        }

        return PlaceAudible(source, modem, peak, relative, sampleRate);
    }

    private static float[] OverlayHighBand(
        float[] speech,
        float[] modem,
        float peak,
        float relative,
        int sampleRate)
    {
        var start = FindFirstActive(speech, peak);
        var available = speech.Length - start;
        var headroom = 0.995f - peak;

        if (available >= modem.Length && headroom >= 0.025f)
        {
            var gain = MathF.Min(relative * peak, headroom);
            gain = MathF.Max(gain, 0.025f);
            var result = (float[])speech.Clone();
            MixAt(result, modem, start, gain);
            return result;
        }

        return AppendAfterSpeech(speech, modem, sampleRate);
    }

    private static float[] PlaceAudible(
        float[] speech,
        float[] modem,
        float peak,
        float relative,
        int sampleRate)
    {
        var slot = FindQuietSlot(speech, modem.Length, peak, sampleRate);
        var headroom = 0.995f - peak;

        if (slot >= 0 && headroom >= 0.03f)
        {
            var gain = MathF.Min(relative * peak, headroom);
            gain = MathF.Max(gain, 0.03f);
            var result = (float[])speech.Clone();
            MixAt(result, modem, slot, gain);
            return result;
        }

        return AppendAfterSpeech(speech, modem, sampleRate);
    }

    private static float[] AppendAfterSpeech(float[] speech, float[] modem, int sampleRate)
    {
        var gap = Math.Max(1, (int)(GapSeconds * sampleRate));
        var result = new float[speech.Length + gap + modem.Length];
        speech.CopyTo(result, 0);
        modem.CopyTo(result, speech.Length + gap);
        return result;
    }

    private static void MixAt(float[] destination, float[] modem, int start, float gain)
    {
        var count = Math.Min(modem.Length, destination.Length - start);
        for (var i = 0; i < count; i++)
        {
            destination[start + i] += modem[i] * gain;
        }
    }

    private static int[] BuildSymbols(byte[] frame, ModemMode mode)
    {
        var preambleBits = AcousticFrameProtocol.BuildPreambleBits();
        var symbols = new List<int>(preambleBits.Length / 2 + frame.Length * 8);

        for (var i = 0; i < preambleBits.Length; i += 2)
        {
            AcousticFrameProtocol.AppendPackedSymbol(symbols, preambleBits, i);
        }

        var headerBits = new List<int>(AcousticFrameProtocol.HeaderLength * 2 * AcousticFrameProtocol.HammingCodewordBits);
        AcousticFrameProtocol.AppendEncodedByte(headerBits, frame[0]);
        AcousticFrameProtocol.AppendEncodedByte(headerBits, frame[1]);
        var header = headerBits.ToArray();
        for (var i = 0; i < header.Length; i += 2)
        {
            AcousticFrameProtocol.AppendPackedSymbol(symbols, header, i);
        }

        var bodyCodewords = BuildBodyCodewords(frame);
        var interleaved = Interleave(bodyCodewords).ToArray();
        var bodySymbolCount = AcousticFrameProtocol.SymbolsForBits(interleaved.Length);

        for (var bodySymbol = 0; bodySymbol < bodySymbolCount; bodySymbol++)
        {
            if (bodySymbol > 0 && bodySymbol % AcousticFrameProtocol.PilotInterval == 0)
            {
                foreach (var pilot in AcousticFrameProtocol.PilotSymbols)
                {
                    symbols.Add(pilot);
                }
            }

            AcousticFrameProtocol.AppendPackedSymbol(
                symbols,
                interleaved,
                bodySymbol * 2);
        }

        return symbols.ToArray();
    }

    private static List<int[]> BuildBodyCodewords(byte[] frame)
    {
        var bodyLength = frame.Length - AcousticFrameProtocol.HeaderLength;
        var codewords = new List<int[]>(bodyLength * 2);
        Span<int> codeword = stackalloc int[AcousticFrameProtocol.HammingCodewordBits];

        for (var i = AcousticFrameProtocol.HeaderLength; i < frame.Length; i++)
        {
            var value = frame[i];
            AcousticFrameProtocol.EncodeNibble(value >> 4, codeword);
            codewords.Add(codeword.ToArray());
            AcousticFrameProtocol.EncodeNibble(value & 0x0F, codeword);
            codewords.Add(codeword.ToArray());
        }

        return codewords;
    }

    /// <summary>
    /// Interleaves codewords by bit position, so a short acoustic burst damages at most one bit
    /// in each of the affected codewords.
    /// </summary>
    private static List<int> Interleave(List<int[]> codewords)
    {
        var bits = new List<int>(codewords.Count * AcousticFrameProtocol.HammingCodewordBits);
        for (var bitPosition = 0; bitPosition < AcousticFrameProtocol.HammingCodewordBits; bitPosition++)
        {
            foreach (var codeword in codewords)
            {
                bits.Add(codeword[bitPosition]);
            }
        }

        return bits;
    }

    private static float[] Render(int[] symbols, ModemMode mode, int sampleRate, float amplitude)
    {
        var frequencies = AcousticFrameProtocol.Frequencies(mode);
        var period = AcousticFrameProtocol.SymbolDuration(mode) * sampleRate;
        var totalSamples = (int)Math.Ceiling(symbols.Length * period) + Math.Max(1, sampleRate / 100);
        var buffer = new float[totalSamples];
        var phase = 0d;

        for (var symbolIndex = 0; symbolIndex < symbols.Length; symbolIndex++)
        {
            var start = (int)Math.Round(symbolIndex * period);
            var end = Math.Min(totalSamples, (int)Math.Round((symbolIndex + 1) * period));
            var length = end - start;
            if (length <= 0)
            {
                continue;
            }

            var frequency = frequencies[symbols[symbolIndex]];
            var ramp = Math.Max(1, (int)(length * 0.18f));

            for (var n = start; n < end; n++)
            {
                var offset = n - start;
                var envelope = 1f;
                if (offset < ramp)
                {
                    envelope = 0.5f - 0.5f * MathF.Cos(MathF.PI * offset / ramp);
                }
                else if (offset >= length - ramp)
                {
                    envelope = 0.5f - 0.5f * MathF.Cos(MathF.PI * (length - 1 - offset) / ramp);
                }

                phase += TwoPi * frequency / sampleRate;
                if (phase > TwoPi)
                {
                    phase -= TwoPi;
                }

                buffer[n] = MathF.Sin((float)phase) * envelope * amplitude;
            }
        }

        return buffer;
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

    private static int FindQuietSlot(float[] samples, int length, float peak, int sampleRate)
    {
        if (length >= samples.Length)
        {
            return -1;
        }

        var frame = Math.Max(1, sampleRate / 50);
        var threshold = peak * 0.02f;
        var runStart = -1;

        for (var start = 0; start + frame <= samples.Length; start += frame)
        {
            var energy = 0f;
            for (var i = start; i < start + frame; i++)
            {
                energy += samples[i] * samples[i];
            }

            var rms = MathF.Sqrt(energy / frame);
            if (rms <= threshold)
            {
                if (runStart < 0)
                {
                    runStart = start;
                }

                if (start + frame - runStart >= length)
                {
                    return runStart;
                }
            }
            else
            {
                runStart = -1;
            }
        }

        return -1;
    }

    private static float Peak(float[] samples)
    {
        var peak = 0f;
        foreach (var sample in samples)
        {
            if (float.IsFinite(sample))
            {
                peak = MathF.Max(peak, MathF.Abs(sample));
            }
        }

        return peak;
    }

    private static void ValidateSampleRate(ModemMode mode, int sampleRate)
    {
        var minimum = AcousticFrameProtocol.MinimumSampleRate(mode);
        if (sampleRate < minimum || sampleRate > 192000)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sampleRate),
                sampleRate,
                mode == ModemMode.HighBandOverlay
                    ? "HighBandOverlay needs a sample rate of at least 38000 Hz."
                    : "The acoustic modem needs a sample rate between 10000 and 192000 Hz.");
        }
    }
}
