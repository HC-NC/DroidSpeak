namespace DroidSpeak.Modem;

/// <summary>
/// v0.9 non-coherent 4-FSK receiver. It locates the mode preamble, tracks symbol timing with
/// periodic pilots, applies Hamming(7,4) correction, and accepts a packet only when CRC-16 passes.
/// </summary>
internal static class AcousticDemodulator
{
    private const int PreambleTolerance = 4;
    private const int CandidateLimit = 24;
    private const int PilotTolerance = 3;

    public static bool TryDecode(
        ReadOnlySpan<float> samples,
        int sampleRate,
        out ModemDecodeResult result)
    {
        result = default;
        if (sampleRate < DroidModemProtocol.MinimumSampleRate || sampleRate > 192000)
        {
            return false;
        }

        if (sampleRate >= AcousticFrameProtocol.MinimumSampleRate(ModemMode.HighBandOverlay) &&
            TryDecode(samples, sampleRate, ModemMode.HighBandOverlay, out result))
        {
            return true;
        }

        if (sampleRate >= AcousticFrameProtocol.MinimumSampleRate(ModemMode.AudibleChirpStream) &&
            TryDecode(samples, sampleRate, ModemMode.AudibleChirpStream, out result))
        {
            return true;
        }

        return false;
    }

    public static bool TryDecode(
        ReadOnlySpan<float> samples,
        int sampleRate,
        ModemMode mode,
        out ModemDecodeResult result)
    {
        result = default;
        if (sampleRate < AcousticFrameProtocol.MinimumSampleRate(mode) || sampleRate > 192000)
        {
            return false;
        }

        var spec = new ModeSpec(mode, sampleRate);
        if (samples.Length < spec.PreambleLengthSamples)
        {
            return false;
        }

        var candidates = FindCandidates(samples, spec);
        foreach (var candidate in candidates)
        {
            var refined = RefineStart(samples, candidate.Start, spec, out var score);
            if (score < spec.PreambleSymbols.Length - PreambleTolerance)
            {
                continue;
            }

            if (TryDecodeAt(samples, refined, spec, out result))
            {
                return true;
            }

            if (candidate.Start != refined &&
                TryDecodeAt(samples, candidate.Start, spec, out result))
            {
                return true;
            }
        }

        return false;
    }

    private static List<Candidate> FindCandidates(ReadOnlySpan<float> samples, ModeSpec spec)
    {
        var hop = Math.Max(1, spec.PeriodSamples / 4);
        var lastStart = samples.Length - spec.PeriodSamples;
        if (lastStart <= 0)
        {
            return [];
        }

        var positionCount = lastStart / hop + 1;
        var magnitudes = new float[positionCount * 4];
        var windowOffset = (spec.PeriodSamples - spec.AnalysisLength) / 2;

        for (var position = 0; position < positionCount; position++)
        {
            var start = position * hop + windowOffset;
            if (start < 0 || start + spec.AnalysisLength > samples.Length)
            {
                continue;
            }

            ComputeMagnitudes(samples, start, spec, magnitudes.AsSpan(position * 4, 4));
        }

        var preamble = spec.PreambleSymbols;
        var stride = spec.PeriodSamples / hop;
        var candidates = new List<Candidate>();
        var maximumPosition = positionCount - (preamble.Length - 1) * stride;

        for (var phase = 0; phase < stride; phase++)
        {
            for (var position = phase; position < maximumPosition; position += stride)
            {
                var score = 0;
                for (var symbol = 0; symbol < preamble.Length; symbol++)
                {
                    var index = position + symbol * stride;
                    var observed = ArgMax(magnitudes.AsSpan(index * 4, 4));
                    if (observed == preamble[symbol])
                    {
                        score++;
                    }
                }

                if (score >= preamble.Length - PreambleTolerance)
                {
                    candidates.Add(new Candidate(score, position * hop));
                }
            }
        }

        if (candidates.Count == 0)
        {
            return candidates;
        }

        candidates.Sort(static (left, right) => right.Score.CompareTo(left.Score));

        var filtered = new List<Candidate>(Math.Min(candidates.Count, CandidateLimit));
        foreach (var candidate in candidates)
        {
            var duplicate = false;
            foreach (var existing in filtered)
            {
                if (Math.Abs(existing.Start - candidate.Start) < spec.PeriodSamples / 2)
                {
                    duplicate = true;
                    break;
                }
            }

            if (!duplicate)
            {
                filtered.Add(candidate);
                if (filtered.Count == CandidateLimit)
                {
                    break;
                }
            }
        }

        return filtered;
    }

    private static int RefineStart(
        ReadOnlySpan<float> samples,
        int start,
        ModeSpec spec,
        out int bestScore)
    {
        var step = Math.Max(1, spec.PeriodSamples / 32);
        var radius = Math.Max(1, spec.PeriodSamples / 4);
        var bestStart = start;
        bestScore = -1;

        for (var offset = -radius; offset <= radius; offset += step)
        {
            var candidate = start + offset;
            if (candidate < 0)
            {
                continue;
            }

            var score = ScorePreamble(samples, candidate, spec);
            if (score > bestScore)
            {
                bestScore = score;
                bestStart = candidate;
            }
        }

        return bestStart;
    }

    private static int ScorePreamble(ReadOnlySpan<float> samples, int start, ModeSpec spec)
    {
        Span<float> magnitudes = stackalloc float[4];
        var score = 0;
        var position = (double)start;

        foreach (var expected in spec.PreambleSymbols)
        {
            if (!TryReadSymbol(samples, position, spec, magnitudes, out var observed))
            {
                return score;
            }

            if (observed == expected)
            {
                score++;
            }

            position += spec.Period;
        }

        return score;
    }

    private static bool TryDecodeAt(
        ReadOnlySpan<float> samples,
        int start,
        ModeSpec spec,
        out ModemDecodeResult result)
    {
        result = default;
        Span<float> magnitudes = stackalloc float[4];
        var position = (double)start;
        var matches = 0;

        foreach (var expected in spec.PreambleSymbols)
        {
            if (!TryReadSymbol(samples, position, spec, magnitudes, out var observed))
            {
                return false;
            }

            if (observed == expected)
            {
                matches++;
            }

            position += spec.Period;
        }

        if (matches < spec.PreambleSymbols.Length - PreambleTolerance)
        {
            return false;
        }

        var headerBits = new int[AcousticFrameProtocol.HeaderSymbolCount() * 2];
        for (var symbol = 0; symbol < AcousticFrameProtocol.HeaderSymbolCount(); symbol++)
        {
            if (!TryReadSymbol(samples, position, spec, magnitudes, out var observed))
            {
                return false;
            }

            AcousticFrameProtocol.UnpackSymbol(observed, headerBits, symbol * 2);
            position += spec.Period;
        }

        var corrected = 0;
        var header = new byte[AcousticFrameProtocol.HeaderLength];
        for (var codeword = 0; codeword < 4; codeword++)
        {
            if (!TryDecodeCodeword(headerBits, codeword * AcousticFrameProtocol.HammingCodewordBits,
                    out var nibble, out var fixedBit))
            {
                return false;
            }

            corrected += fixedBit ? 1 : 0;
            var byteIndex = codeword / 2;
            header[byteIndex] = codeword % 2 == 0
                ? (byte)(nibble << 4)
                : (byte)(header[byteIndex] | nibble);
        }

        if (!AcousticFrameProtocol.TryReadHeader(header, out var mode, out var payloadLength) || mode != spec.Mode)
        {
            return false;
        }

        var bodySymbolCount = AcousticFrameProtocol.BodySymbolCount(payloadLength);
        var bodyBitCount = bodySymbolCount * 2;
        var bodyBits = new int[bodyBitCount];
        var pilotBlocks = Math.Max(0, (bodySymbolCount - 1) / AcousticFrameProtocol.PilotInterval);

        var required = position +
            (bodySymbolCount + pilotBlocks * AcousticFrameProtocol.PilotSymbols.Length) * spec.Period +
            spec.Period * 0.5;
        if (required > samples.Length)
        {
            return false;
        }

        var initialPeriod = spec.Period;
        var period = spec.Period;

        for (var bodySymbol = 0; bodySymbol < bodySymbolCount; bodySymbol++)
        {
            if (bodySymbol > 0 && bodySymbol % AcousticFrameProtocol.PilotInterval == 0)
            {
                position = TrackPilots(
                    samples,
                    position,
                    ref period,
                    spec,
                    magnitudes,
                    out var pilotError);
                if (pilotError > spec.Period * 0.45)
                {
                    return false;
                }
            }

            if (!TryReadSymbol(samples, position, spec, magnitudes, out var observed))
            {
                return false;
            }

            AcousticFrameProtocol.UnpackSymbol(observed, bodyBits, bodySymbol * 2);
            position += period;
        }

        var codewordCount = (payloadLength + AcousticFrameProtocol.CrcLength) * 2;
        var bytes = new byte[payloadLength + AcousticFrameProtocol.CrcLength];
        Span<int> codewordBits = stackalloc int[AcousticFrameProtocol.HammingCodewordBits];

        for (var index = 0; index < codewordCount; index++)
        {
            for (var bitPosition = 0; bitPosition < AcousticFrameProtocol.HammingCodewordBits; bitPosition++)
            {
                codewordBits[bitPosition] = bodyBits[bitPosition * codewordCount + index];
            }

            if (!AcousticFrameProtocol.TryDecodeNibble(codewordBits, out var nibble, out var fixedBit))
            {
                return false;
            }

            corrected += fixedBit ? 1 : 0;
            var byteIndex = index / 2;
            bytes[byteIndex] = index % 2 == 0
                ? (byte)(nibble << 4)
                : (byte)(bytes[byteIndex] | nibble);
        }

        var receivedCrc = (ushort)((bytes[payloadLength] << 8) | bytes[payloadLength + 1]);
        var crcInput = new byte[AcousticFrameProtocol.HeaderLength + payloadLength];
        header.CopyTo(crcInput, 0);
        bytes.AsSpan(0, payloadLength).CopyTo(crcInput.AsSpan(AcousticFrameProtocol.HeaderLength));

        if (AcousticFrameProtocol.ComputeCrc16(crcInput) != receivedCrc)
        {
            return false;
        }

        if (!DroidModemProtocol.TryDecodeUtf8(bytes.AsSpan(0, payloadLength), out var text) || text is null)
        {
            return false;
        }

        var drift = (period / initialPeriod - 1.0) * 1_000_000.0;
        result = new ModemDecodeResult(
            text,
            mode,
            corrected,
            CrcValid: true,
            drift,
            (int)Math.Ceiling(position),
            Legacy: false);
        return true;
    }

    private static double TrackPilots(
        ReadOnlySpan<float> samples,
        double position,
        ref double period,
        ModeSpec spec,
        Span<float> magnitudes,
        out double error)
    {
        error = 0;
        var search = Math.Max(1, (int)(period * 0.18));
        var firstError = 0.0;

        for (var pilot = 0; pilot < AcousticFrameProtocol.PilotSymbols.Length; pilot++)
        {
            var expected = AcousticFrameProtocol.PilotSymbols[pilot];
            var bestOffset = 0;
            var bestMagnitude = -1f;

            for (var offset = -search; offset <= search; offset++)
            {
                var start = position + offset;
                if (!TryReadMagnitude(samples, start, spec, magnitudes, expected, out var magnitude))
                {
                    continue;
                }

                if (magnitude > bestMagnitude)
                {
                    bestMagnitude = magnitude;
                    bestOffset = offset;
                }
            }

            if (bestMagnitude < 0f)
            {
                return position;
            }

            if (pilot == 0)
            {
                firstError = bestOffset;
                position += bestOffset;
            }
            else
            {
                error = Math.Abs(bestOffset) + Math.Abs(firstError);
                position += bestOffset;
            }

            position += period;
        }

        // A pilot arrives every 24 body symbols. Spread the measured phase error over that
        // interval so a slightly fast or slow sound card is corrected before the next block.
        if (Math.Abs(firstError) > 0.5)
        {
            period += firstError / AcousticFrameProtocol.PilotInterval * 0.6;
        }

        return position;
    }

    private static bool TryReadSymbol(
        ReadOnlySpan<float> samples,
        double symbolStart,
        ModeSpec spec,
        Span<float> magnitudes,
        out int symbol)
    {
        symbol = 0;
        var windowStart = symbolStart + (spec.PeriodSamples - spec.AnalysisLength) / 2.0;
        if (!TryComputeMagnitudes(samples, windowStart, spec, magnitudes))
        {
            return false;
        }

        symbol = ArgMax(magnitudes);
        return true;
    }

    private static bool TryReadMagnitude(
        ReadOnlySpan<float> samples,
        double symbolStart,
        ModeSpec spec,
        Span<float> magnitudes,
        int symbol,
        out float magnitude)
    {
        magnitude = 0f;
        var windowStart = symbolStart + (spec.PeriodSamples - spec.AnalysisLength) / 2.0;
        if (!TryComputeMagnitudes(samples, windowStart, spec, magnitudes))
        {
            return false;
        }

        magnitude = magnitudes[symbol];
        return true;
    }

    private static bool TryComputeMagnitudes(
        ReadOnlySpan<float> samples,
        double windowStart,
        ModeSpec spec,
        Span<float> magnitudes)
    {
        var start = (int)Math.Round(windowStart);
        if (start < 0 || start + spec.AnalysisLength > samples.Length)
        {
            return false;
        }

        ComputeMagnitudes(samples, start, spec, magnitudes);
        return true;
    }

    private static void ComputeMagnitudes(
        ReadOnlySpan<float> samples,
        int start,
        ModeSpec spec,
        Span<float> magnitudes)
    {
        for (var tone = 0; tone < 4; tone++)
        {
            magnitudes[tone] = Goertzel(samples, start, spec, tone);
        }
    }

    private static float Goertzel(ReadOnlySpan<float> samples, int start, ModeSpec spec, int tone)
    {
        var coefficient = spec.Coefficients[tone];
        double s1 = 0;
        double s2 = 0;

        for (var i = 0; i < spec.AnalysisLength; i++)
        {
            var sample = samples[start + i] * spec.Window[i];
            var s0 = sample + coefficient * s1 - s2;
            s2 = s1;
            s1 = s0;
        }

        var power = s1 * s1 + s2 * s2 - coefficient * s1 * s2;
        return power > 0 ? (float)power : 0f;
    }

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        var best = 0;
        for (var i = 1; i < values.Length; i++)
        {
            if (values[i] > values[best])
            {
                best = i;
            }
        }

        return best;
    }

    private static bool TryDecodeCodeword(
        ReadOnlySpan<int> bits,
        int offset,
        out int nibble,
        out bool corrected)
    {
        Span<int> codeword = stackalloc int[AcousticFrameProtocol.HammingCodewordBits];
        bits.Slice(offset, AcousticFrameProtocol.HammingCodewordBits).CopyTo(codeword);
        return AcousticFrameProtocol.TryDecodeNibble(codeword, out nibble, out corrected);
    }

    private readonly record struct Candidate(int Score, int Start);

    private sealed class ModeSpec
    {
        public ModeSpec(ModemMode mode, int sampleRate)
        {
            Mode = mode;
            Frequencies = AcousticFrameProtocol.Frequencies(mode).ToArray();
            Period = AcousticFrameProtocol.SymbolDuration(mode) * sampleRate;
            PeriodSamples = (int)Math.Round(Period);
            AnalysisLength = Math.Max(16, (int)(PeriodSamples * 0.80));
            Window = new float[AnalysisLength];
            FastFourierTransform.FillHannWindow(Window, AnalysisLength);
            Coefficients = new double[4];
            for (var i = 0; i < 4; i++)
            {
                Coefficients[i] = 2.0 * Math.Cos(2.0 * Math.PI * Frequencies[i] / sampleRate);
            }

            PreambleSymbols = AcousticFrameProtocol.BuildPreambleSymbols();
            PreambleLengthSamples = PreambleSymbols.Length * PeriodSamples;
        }

        public ModemMode Mode { get; }
        public float[] Frequencies { get; }
        public double Period { get; }
        public int PeriodSamples { get; }
        public int AnalysisLength { get; }
        public float[] Window { get; }
        public double[] Coefficients { get; }
        public int[] PreambleSymbols { get; }
        public int PreambleLengthSamples { get; }
    }
}
