namespace DroidSpeak.Modem;

/// <summary>
/// Compact in-place radix-2 Cooley–Tukey FFT. The decoder uses it to inspect the
/// 3200–4800 Hz modem band without any external DSP dependency.
/// </summary>
internal static class FastFourierTransform
{
    /// <summary>
    /// Transforms <paramref name="real"/> / <paramref name="imaginary"/> in place.
    /// The length must be a power of two.
    /// </summary>
    public static void Forward(Span<float> real, Span<float> imaginary)
    {
        var length = real.Length;
        if (length != imaginary.Length)
        {
            throw new ArgumentException("Real and imaginary buffers must have the same length.");
        }

        if (length == 0 || (length & (length - 1)) != 0)
        {
            throw new ArgumentException("FFT length must be a non-zero power of two.", nameof(real));
        }

        // Bit-reversal permutation.
        for (int i = 1, j = 0; i < length; i++)
        {
            var bit = length >> 1;
            for (; (j & bit) != 0; bit >>= 1)
            {
                j ^= bit;
            }

            j ^= bit;
            if (i < j)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imaginary[i], imaginary[j]) = (imaginary[j], imaginary[i]);
            }
        }

        for (var blockLength = 2; blockLength <= length; blockLength <<= 1)
        {
            var angle = -2.0 * Math.PI / blockLength;
            var twiddleReal = (float)Math.Cos(angle);
            var twiddleImaginary = (float)Math.Sin(angle);
            var half = blockLength >> 1;

            for (var block = 0; block < length; block += blockLength)
            {
                float wReal = 1f;
                float wImaginary = 0f;

                for (var k = 0; k < half; k++)
                {
                    var evenIndex = block + k;
                    var oddIndex = evenIndex + half;

                    var oddReal = real[oddIndex] * wReal - imaginary[oddIndex] * wImaginary;
                    var oddImaginary = real[oddIndex] * wImaginary + imaginary[oddIndex] * wReal;

                    var evenReal = real[evenIndex];
                    var evenImaginary = imaginary[evenIndex];

                    real[evenIndex] = evenReal + oddReal;
                    imaginary[evenIndex] = evenImaginary + oddImaginary;
                    real[oddIndex] = evenReal - oddReal;
                    imaginary[oddIndex] = evenImaginary - oddImaginary;

                    var nextReal = wReal * twiddleReal - wImaginary * twiddleImaginary;
                    wImaginary = wReal * twiddleImaginary + wImaginary * twiddleReal;
                    wReal = nextReal;
                }
            }
        }
    }

    /// <summary>Fills <paramref name="buffer"/> with a Hann window of the given length.</summary>
    public static void FillHannWindow(Span<float> buffer, int length)
    {
        if (length <= 1)
        {
            buffer[..Math.Max(0, length)].Fill(1f);
            return;
        }

        for (var i = 0; i < length; i++)
        {
            buffer[i] = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / (length - 1));
        }
    }
}
