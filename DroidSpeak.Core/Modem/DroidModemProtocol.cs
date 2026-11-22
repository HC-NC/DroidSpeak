using System.Text;

namespace DroidSpeak.Modem;

/// <summary>
/// Shared constants and small helpers for the droid acoustic modem.
/// </summary>
internal static class DroidModemProtocol
{
    /// <summary>FSK carrier used for a zero bit (Hz).</summary>
    public const float Bit0Frequency = 3400f;

    /// <summary>FSK carrier used for a one bit (Hz).</summary>
    public const float Bit1Frequency = 4400f;

    /// <summary>Duration of one FSK symbol in seconds.</summary>
    public const double BitDuration = 0.003;

    /// <summary>Lowest sample rate that can represent both FSK carriers.</summary>
    public const int MinimumSampleRate = 10000;

    /// <summary>Largest payload that fits into the one-byte length field.</summary>
    public const int MaxPayloadLength = 255;

    /// <summary>
    /// Sync preamble: two bytes of 1-bits for a clean energy onset, four alternating
    /// <c>0xAA</c> bytes for bit-phase locking and one <c>0xD5</c> sync marker.
    /// </summary>
    public static ReadOnlySpan<byte> Preamble =>
        [0xFF, 0xFF, 0xAA, 0xAA, 0xAA, 0xAA, 0xD5];

    private static readonly UTF8Encoding StrictUtf8 = new(
        encoderShouldEmitUTF8Identifier: false,
        throwOnInvalidBytes: true);

    public static byte[] EncodeUtf8(string text)
    {
        var bytes = StrictUtf8.GetBytes(text);
        if (bytes.Length > MaxPayloadLength)
        {
            throw new ArgumentException(
                $"The text is too long for one modem packet ({bytes.Length} bytes, maximum {MaxPayloadLength}).",
                nameof(text));
        }

        return bytes;
    }

    public static bool TryDecodeUtf8(ReadOnlySpan<byte> bytes, out string? text)
    {
        try
        {
            text = StrictUtf8.GetString(bytes);
            return true;
        }
        catch (DecoderFallbackException)
        {
            text = null;
            return false;
        }
    }

    /// <summary>CRC-8/ATM (polynomial <c>0x07</c>, initial value <c>0</c>).</summary>
    public static byte ComputeCrc8(ReadOnlySpan<byte> data)
    {
        byte crc = 0;
        foreach (var value in data)
        {
            crc ^= value;
            for (var bit = 0; bit < 8; bit++)
            {
                crc = (byte)((crc & 0x80) != 0 ? (crc << 1) ^ 0x07 : crc << 1);
            }
        }

        return crc;
    }

    public static void AppendByteBits(List<int> bits, byte value)
    {
        for (var bit = 7; bit >= 0; bit--)
        {
            bits.Add((value >> bit) & 1);
        }
    }

    private static ReadOnlySpan<int> HammingDataPositions => [3, 5, 6, 7, 9, 10, 11, 12];

    /// <summary>
    /// Appends a Hamming(12,8) codeword. The physical layer corrects any single-bit error per
    /// byte, while the logical packet still consists of length + UTF-8 + CRC8.
    /// </summary>
    public static void AppendHammingByte(List<int> bits, byte value)
    {
        Span<int> code = stackalloc int[13]; // 1-based positions

        for (var i = 0; i < 8; i++)
        {
            code[HammingDataPositions[i]] = (value >> (7 - i)) & 1;
        }

        code[1] = code[3] ^ code[5] ^ code[7] ^ code[9] ^ code[11];
        code[2] = code[3] ^ code[6] ^ code[7] ^ code[10] ^ code[11];
        code[4] = code[5] ^ code[6] ^ code[7] ^ code[12];
        code[8] = code[9] ^ code[10] ^ code[11] ^ code[12];

        for (var position = 1; position <= 12; position++)
        {
            bits.Add(code[position]);
        }
    }

    /// <summary>
    /// Decodes a Hamming(12,8) codeword and corrects one flipped bit. Returns <see langword="false"/>
    /// when the syndrome points outside the codeword (an uncorrectable pattern).
    /// </summary>
    public static bool TryDecodeHammingByte(ReadOnlySpan<int> bits, out byte value)
    {
        value = 0;
        if (bits.Length < 12)
        {
            return false;
        }

        Span<int> code = stackalloc int[13]; // 1-based positions
        for (var i = 0; i < 12; i++)
        {
            code[i + 1] = bits[i];
        }

        var syndrome =
            (code[1] ^ code[3] ^ code[5] ^ code[7] ^ code[9] ^ code[11]) |
            ((code[2] ^ code[3] ^ code[6] ^ code[7] ^ code[10] ^ code[11]) << 1) |
            ((code[4] ^ code[5] ^ code[6] ^ code[7] ^ code[12]) << 2) |
            ((code[8] ^ code[9] ^ code[10] ^ code[11] ^ code[12]) << 3);

        if (syndrome is > 0 and <= 12)
        {
            code[syndrome] ^= 1;
        }
        else if (syndrome != 0)
        {
            return false;
        }

        for (var i = 0; i < 8; i++)
        {
            value = (byte)((value << 1) | code[HammingDataPositions[i]]);
        }

        return true;
    }
}
