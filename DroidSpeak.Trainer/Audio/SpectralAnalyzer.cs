namespace DroidSpeak.Trainer.Audio;

/// <summary>A pair of vowel formant frequencies recovered from a magnitude spectrum.</summary>
/// <param name="FirstHz">First formant (F1) in Hz.</param>
/// <param name="SecondHz">Second formant (F2) in Hz, always at least 100 Hz above F1.</param>
internal readonly record struct FormantPair(float FirstHz, float SecondHz);

/// <summary>
/// Spectral helpers shared by acoustic extraction: a small radix-2 FFT, a Hann-windowed magnitude
/// spectrum, a zero-crossing rate, a brightness ratio and a two-formant peak picker.
/// <para>
/// Everything here is deterministic and allocation-light. The FFT is the textbook in-place
/// Cooley-Tukey butterfly, chosen over a library transform because the trainer must build with
/// nothing but the SDK and the sizes involved (a pulse is at most a few thousand samples) make the
/// simple implementation fast enough.
/// </para>
/// </summary>
internal static class SpectralAnalyzer
{
    /// <summary>Default transform size used for one pulse, in samples.</summary>
    public const int DefaultFftSize = 1024;

    /// <summary>Default frequency above which energy counts as "bright" for the timbre estimate.</summary>
    public const float DefaultBrightnessCutoffHz = 4000f;

    /// <summary>Lowest first-formant frequency the extractor reports, in Hz.</summary>
    public const float FirstFormantLowHz = 200f;

    /// <summary>Highest first-formant frequency the extractor reports, in Hz.</summary>
    public const float FirstFormantHighHz = 1000f;

    /// <summary>Lowest second-formant frequency the extractor reports, in Hz.</summary>
    public const float SecondFormantLowHz = 700f;

    /// <summary>Highest second-formant frequency the extractor reports, in Hz.</summary>
    public const float SecondFormantHighHz = 3200f;

    /// <summary>Default formant pair used when a band contains no usable energy.</summary>
    public static readonly FormantPair FallbackFormants = new(500f, 1500f);

    /// <summary>
    /// Runs an in-place forward complex FFT.
    /// </summary>
    /// <param name="real">Real parts; overwritten with the transform output.</param>
    /// <param name="imag">Imaginary parts; overwritten with the transform output.</param>
    /// <exception cref="ArgumentException">
    /// The spans differ in length, are empty, or the length is not a power of two.
    /// </exception>
    public static void Transform(Span<float> real, Span<float> imag)
    {
        if (real.Length != imag.Length)
        {
            throw new ArgumentException("Real and imaginary spans must have the same length.", nameof(imag));
        }

        var n = real.Length;
        if (n == 0 || (n & (n - 1)) != 0)
        {
            throw new ArgumentException("The transform length must be a power of two.", nameof(real));
        }

        // Bit-reversal permutation: index i swaps with the reverse of its log2(n) bits.
        for (int i = 1, j = 0; i < n; i++)
        {
            var bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1)
            {
                j ^= bit;
            }

            j ^= bit;

            if (i < j)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imag[i], imag[j]) = (imag[j], imag[i]);
            }
        }

        for (var length = 2; length <= n; length <<= 1)
        {
            var angle = -2f * MathF.PI / length;
            var stepReal = MathF.Cos(angle);
            var stepImag = MathF.Sin(angle);

            for (var start = 0; start < n; start += length)
            {
                var twiddleReal = 1f;
                var twiddleImag = 0f;
                var half = length >> 1;

                for (var k = 0; k < half; k++)
                {
                    var even = start + k;
                    var odd = even + half;

                    var oddReal = (twiddleReal * real[odd]) - (twiddleImag * imag[odd]);
                    var oddImag = (twiddleReal * imag[odd]) + (twiddleImag * real[odd]);

                    real[odd] = real[even] - oddReal;
                    imag[odd] = imag[even] - oddImag;
                    real[even] += oddReal;
                    imag[even] += oddImag;

                    var nextReal = (twiddleReal * stepReal) - (twiddleImag * stepImag);
                    twiddleImag = (twiddleReal * stepImag) + (twiddleImag * stepReal);
                    twiddleReal = nextReal;
                }
            }
        }
    }

    /// <summary>
    /// Computes the one-sided magnitude spectrum of the head of <paramref name="samples"/>.
    /// </summary>
    /// <param name="samples">Mono samples; only the first <paramref name="fftSize"/> are windowed.</param>
    /// <param name="fftSize">Transform size in samples; must be a positive power of two.</param>
    /// <returns>
    /// <c>fftSize / 2 + 1</c> magnitudes, bin <c>k</c> at <c>k * sampleRate / fftSize</c>. The signal
    /// is Hann-windowed over its filled part and zero-padded, so a short pulse gets a smooth
    /// spectrum instead of a rectangular-splatter one.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="fftSize"/> is invalid.</exception>
    public static float[] Magnitudes(ReadOnlySpan<float> samples, int fftSize = DefaultFftSize)
    {
        if (fftSize <= 0 || (fftSize & (fftSize - 1)) != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(fftSize),
                fftSize,
                "The transform size must be a positive power of two.");
        }

        var real = new float[fftSize];
        var imaginary = new float[fftSize];
        var filled = Math.Min(samples.Length, fftSize);

        if (filled > 0)
        {
            if (filled == 1)
            {
                real[0] = samples[0];
            }
            else
            {
                var scale = 2f * MathF.PI / (filled - 1);
                for (var i = 0; i < filled; i++)
                {
                    var value = samples[i];
                    if (!float.IsFinite(value))
                    {
                        value = 0f;
                    }

                    var window = 0.5f - (0.5f * MathF.Cos(scale * i));
                    real[i] = value * window;
                }
            }
        }

        Transform(real, imaginary);

        var bins = (fftSize / 2) + 1;
        var magnitudes = new float[bins];
        for (var k = 0; k < bins; k++)
        {
            magnitudes[k] = MathF.Sqrt((real[k] * real[k]) + (imaginary[k] * imaginary[k]));
        }

        return magnitudes;
    }

    /// <summary>
    /// Computes the zero-crossing rate of <paramref name="samples"/>.
    /// </summary>
    /// <param name="samples">Mono samples.</param>
    /// <param name="sampleRate">Rate of <paramref name="samples"/> in Hz.</param>
    /// <returns>
    /// Crossings per second. A pure sine at <c>f</c> reads roughly <c>2f</c> here, because each cycle
    /// crosses zero twice; the extractor compares the value against twice the detected pitch for
    /// that reason. Returns <c>0</c> for fewer than two samples.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="sampleRate"/> is not positive.</exception>
    public static float ZeroCrossingRate(ReadOnlySpan<float> samples, int sampleRate)
    {
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Rate must be positive.");
        }

        if (samples.Length < 2)
        {
            return 0f;
        }

        var crossings = 0;
        var previous = 0f;

        for (var i = 0; i < samples.Length; i++)
        {
            var value = samples[i];
            if (!float.IsFinite(value))
            {
                value = 0f;
            }

            if ((value > 0f && previous < 0f) || (value < 0f && previous > 0f))
            {
                crossings++;
            }

            if (value != 0f)
            {
                previous = value;
            }
        }

        return crossings * sampleRate / (float)(samples.Length - 1);
    }

    /// <summary>
    /// Computes the share of spectral energy above <paramref name="cutoffHz"/>.
    /// </summary>
    /// <param name="magnitudes">One-sided magnitudes, as produced by <see cref="Magnitudes"/>.</param>
    /// <param name="sampleRate">Rate the spectrum was computed at, in Hz.</param>
    /// <param name="cutoffHz">Split frequency in Hz.</param>
    /// <returns>A ratio in <c>[0, 1]</c>; <c>0</c> when the spectrum has no energy at all.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="sampleRate"/> is not positive or the spectrum is too short.
    /// </exception>
    public static float HighFrequencyRatio(
        ReadOnlySpan<float> magnitudes,
        int sampleRate,
        float cutoffHz = DefaultBrightnessCutoffHz)
    {
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Rate must be positive.");
        }

        if (magnitudes.Length < 2)
        {
            return 0f;
        }

        var binHz = sampleRate / (2f * (magnitudes.Length - 1));
        var total = 0f;
        var high = 0f;

        for (var k = 0; k < magnitudes.Length; k++)
        {
            var magnitude = float.IsFinite(magnitudes[k]) ? magnitudes[k] : 0f;
            total += magnitude;
            if (k * binHz >= cutoffHz)
            {
                high += magnitude;
            }
        }

        return total > 0f ? high / total : 0f;
    }

    /// <summary>
    /// Picks the strongest F1 and F2 peaks from a magnitude spectrum.
    /// </summary>
    /// <param name="magnitudes">One-sided magnitudes, as produced by <see cref="Magnitudes"/>.</param>
    /// <param name="sampleRate">Rate the spectrum was computed at, in Hz.</param>
    /// <returns>
    /// The two peaks, clamped into <c>[200, 1000]</c> Hz and <c>[700, 3200]</c> Hz, then capped at
    /// 95 % of <see cref="Neural.SyllableTargetDecoder.FormantCeiling"/> so the pair always survives
    /// the formant decoder's clamps.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="sampleRate"/> is not positive.
    /// </exception>
    public static FormantPair Formants(ReadOnlySpan<float> magnitudes, int sampleRate)
    {
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Rate must be positive.");
        }

        if (magnitudes.Length < 4)
        {
            return FallbackFormants;
        }

        var binHz = sampleRate / (2f * (magnitudes.Length - 1));
        var ceiling = Neural.SyllableTargetDecoder.FormantCeiling(sampleRate) * 0.95f;

        var peak1 = PeakInBand(magnitudes, binHz, FirstFormantLowHz, MathF.Min(FirstFormantHighHz, ceiling));
        var peak2 = PeakInBand(magnitudes, binHz, SecondFormantLowHz, MathF.Min(SecondFormantHighHz, ceiling));

        if (peak1 <= 0f || peak2 <= 0f)
        {
            peak1 = peak1 > 0f ? peak1 : FallbackFormants.FirstHz;
            peak2 = peak2 > 0f ? peak2 : FallbackFormants.SecondHz;
        }

        var first = Math.Clamp(peak1, FirstFormantLowHz, MathF.Min(FirstFormantHighHz, ceiling));
        var second = Math.Clamp(peak2, SecondFormantLowHz, MathF.Min(SecondFormantHighHz, ceiling));

        if (second < first + 100f)
        {
            second = first + 100f;
        }

        return new FormantPair(first, second);
    }

    /// <summary>Returns the frequency of the strongest bin in <c>[lowHz, highHz]</c>, or <c>0</c>.</summary>
    private static float PeakInBand(ReadOnlySpan<float> magnitudes, float binHz, float lowHz, float highHz)
    {
        if (highHz < lowHz || binHz <= 0f)
        {
            return 0f;
        }

        var best = 0f;
        var bestMagnitude = 0f;

        for (var k = 0; k < magnitudes.Length; k++)
        {
            var frequency = k * binHz;
            if (frequency < lowHz || frequency > highHz)
            {
                continue;
            }

            var magnitude = float.IsFinite(magnitudes[k]) ? magnitudes[k] : 0f;
            if (magnitude > bestMagnitude)
            {
                bestMagnitude = magnitude;
                best = frequency;
            }
        }

        return bestMagnitude > 0f ? best : 0f;
    }
}
