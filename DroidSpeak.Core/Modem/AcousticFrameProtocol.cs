namespace DroidSpeak.Modem;

/// <summary>
/// Wire format for the v0.9 rich-spectrum modem: a versioned header, Hamming(7,4) protected
/// payload, CRC-16 packet check, and the mode-specific physical-layer constants.
/// </summary>
internal static class AcousticFrameProtocol
{
    public const byte Version = 0x09;
    public const int HeaderLength = 2;
    public const int CrcLength = 2;
    public const int MaxPayloadLength = 255;
    public const int HammingCodewordBits = 7;
    public const int PilotInterval = 24;

    public static readonly int[] PilotSymbols = [0, 3];

    private static readonly float[] HighBandFrequencies = [14600f, 15400f, 16200f, 17000f];
    private static readonly float[] AudibleFrequencies = [2300f, 2900f, 3500f, 4300f];

    public static ReadOnlySpan<float> Frequencies(ModemMode mode) =>
        mode == ModemMode.HighBandOverlay ? HighBandFrequencies : AudibleFrequencies;

    public static double SymbolDuration(ModemMode mode) =>
        mode == ModemMode.HighBandOverlay ? 0.004 : 0.006;

    public static int MinimumSampleRate(ModemMode mode) =>
        mode == ModemMode.HighBandOverlay ? 38000 : 10000;

    public static int MaximumFrequency(ModemMode mode) =>
        mode == ModemMode.HighBandOverlay ? 17000 : 4300;

    /// <summary>
    /// The raw (unprotected) synchronization bits. The leading one-runs give the receiver a
    /// clean carrier onset; the alternating bytes train the symbol clock; the final byte marks
    /// the exact end of the preamble.
    /// </summary>
    public static int[] BuildPreambleBits()
    {
        var bits = new List<int>(DroidModemProtocol.Preamble.Length * 8);
        foreach (var value in DroidModemProtocol.Preamble)
        {
            DroidModemProtocol.AppendByteBits(bits, value);
        }

        return bits.ToArray();
    }

    public static byte[] BuildFrame(string text, ModemMode mode)
    {
        ArgumentNullException.ThrowIfNull(text);

        var payload = DroidModemProtocol.EncodeUtf8(text);
        var frame = new byte[HeaderLength + payload.Length + CrcLength];
        frame[0] = (byte)((Version << 4) | (byte)mode);
        frame[1] = (byte)payload.Length;
        payload.CopyTo(frame, HeaderLength);

        var crc = ComputeCrc16(frame.AsSpan(0, HeaderLength + payload.Length));
        frame[^2] = (byte)(crc >> 8);
        frame[^1] = (byte)crc;
        return frame;
    }

    public static bool TryReadHeader(ReadOnlySpan<byte> header, out ModemMode mode, out int payloadLength)
    {
        mode = default;
        payloadLength = 0;

        if (header.Length < HeaderLength)
        {
            return false;
        }

        var version = header[0] >> 4;
        var modeValue = header[0] & 0x0F;
        if (version != Version || modeValue > (byte)ModemMode.AudibleChirpStream)
        {
            return false;
        }

        mode = (ModemMode)modeValue;
        payloadLength = header[1];
        return true;
    }

    /// <summary>CRC-16/CCITT-FALSE: polynomial 0x1021, initial value 0xFFFF, no reflection.</summary>
    public static ushort ComputeCrc16(ReadOnlySpan<byte> data)
    {
        ushort crc = 0xFFFF;
        foreach (var value in data)
        {
            crc ^= (ushort)(value << 8);
            for (var bit = 0; bit < 8; bit++)
            {
                crc = (crc & 0x8000) != 0
                    ? (ushort)((crc << 1) ^ 0x1021)
                    : (ushort)(crc << 1);
            }
        }

        return crc;
    }

    /// <summary>
    /// Encodes one nibble as Hamming(7,4) in positions 1..7. The data nibble is stored MSB
    /// first, which keeps the mapping easy to inspect in tests.
    /// </summary>
    public static void EncodeNibble(int nibble, Span<int> codeword)
    {
        if (codeword.Length < HammingCodewordBits)
        {
            throw new ArgumentException("A Hamming(7,4) codeword needs seven slots.", nameof(codeword));
        }

        var d1 = (nibble >> 3) & 1;
        var d2 = (nibble >> 2) & 1;
        var d3 = (nibble >> 1) & 1;
        var d4 = nibble & 1;

        codeword[0] = d1 ^ d2 ^ d4;
        codeword[1] = d1 ^ d3 ^ d4;
        codeword[2] = d1;
        codeword[3] = d2 ^ d3 ^ d4;
        codeword[4] = d2;
        codeword[5] = d3;
        codeword[6] = d4;
    }

    /// <summary>
    /// Decodes one Hamming(7,4) codeword, correcting a single flipped bit. Multi-bit patterns
    /// that map onto a valid syndrome are indistinguishable from a single error, so the caller
    /// still relies on CRC-16 for final packet acceptance.
    /// </summary>
    public static bool TryDecodeNibble(ReadOnlySpan<int> codeword, out int nibble, out bool corrected)
    {
        nibble = 0;
        corrected = false;

        if (codeword.Length < HammingCodewordBits)
        {
            return false;
        }

        Span<int> bits = stackalloc int[HammingCodewordBits];
        codeword[..HammingCodewordBits].CopyTo(bits);

        var syndrome =
            (bits[0] ^ bits[2] ^ bits[4] ^ bits[6]) |
            ((bits[1] ^ bits[2] ^ bits[5] ^ bits[6]) << 1) |
            ((bits[3] ^ bits[4] ^ bits[5] ^ bits[6]) << 2);

        if (syndrome is > 0 and <= HammingCodewordBits)
        {
            bits[syndrome - 1] ^= 1;
            corrected = true;
        }
        else if (syndrome != 0)
        {
            return false;
        }

        nibble = (bits[2] << 3) | (bits[4] << 2) | (bits[5] << 1) | bits[6];
        return true;
    }

    public static void AppendEncodedByte(List<int> bits, byte value)
    {
        Span<int> codeword = stackalloc int[HammingCodewordBits];
        EncodeNibble(value >> 4, codeword);
        Append(codeword);
        EncodeNibble(value & 0x0F, codeword);
        Append(codeword);

        void Append(ReadOnlySpan<int> code)
        {
            foreach (var bit in code)
            {
                bits.Add(bit);
            }
        }
    }

    public static void AppendPackedSymbol(List<int> symbols, ReadOnlySpan<int> bits, int offset)
    {
        var first = offset < bits.Length ? bits[offset] : 0;
        var second = offset + 1 < bits.Length ? bits[offset + 1] : 0;
        symbols.Add((first << 1) | second);
    }

    public static void UnpackSymbol(int symbol, Span<int> bits, int offset)
    {
        bits[offset] = (symbol >> 1) & 1;
        bits[offset + 1] = symbol & 1;
    }

    public static int SymbolsForBits(int bitCount) => (bitCount + 1) / 2;

    public static int HeaderSymbolCount() =>
        SymbolsForBits(HeaderLength * 2 * HammingCodewordBits);

    public static int BodySymbolCount(int payloadLength) =>
        SymbolsForBits((payloadLength + CrcLength) * 2 * HammingCodewordBits);

    public static int PhysicalBodyIndex(int bodySymbolIndex) =>
        bodySymbolIndex + (bodySymbolIndex / PilotInterval) * PilotSymbols.Length;
}
