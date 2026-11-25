namespace DroidSpeak.DSP;

/// <summary>
/// Single-precision, allocation-free RBJ band-pass biquad (constant 0 dB peak gain).
/// <para>
/// The filter is a value type with five coefficients and four state words, so a whole formant
/// bank can live on the stack. Coefficients are recalculated only when the target frequency
/// moves, and every intermediate value is clamped to a safe range, which keeps the filter stable
/// from 8 kHz up to 192 kHz.
/// </para>
/// </summary>
internal struct BiquadBandPass
{
    private const float MinFrequency = 20f;
    private const float MaxQ = 12f;
    private const float MinQ = 0.35f;

    // Normalized direct-form-I coefficients (b1 is always zero for a band-pass).
    private float _b0;
    private float _b2;
    private float _a1;
    private float _a2;

    // Direct-form-I history.
    private float _x1;
    private float _x2;
    private float _y1;
    private float _y2;

    /// <summary>Center frequency last passed to <see cref="SetBandPass"/>.</summary>
    public float Frequency { get; private set; }

    /// <summary>Recomputes the coefficients for a band-pass centered on <paramref name="frequency"/>.</summary>
    public void SetBandPass(float frequency, float q, int sampleRate)
    {
        Frequency = frequency;

        // Keep the center comfortably below Nyquist so the bilinear transform never folds.
        var maxFrequency = sampleRate * 0.45f;
        var clamped = Math.Clamp(frequency, MinFrequency, maxFrequency);
        var clampedQ = Math.Clamp(q, MinQ, MaxQ);

        var w0 = MathF.Tau * clamped / sampleRate;
        var cosW0 = MathF.Cos(w0);
        var alpha = MathF.Sin(w0) / (2f * clampedQ);
        var a0 = 1f + alpha;
        var inverseA0 = 1f / a0;

        _b0 = alpha * inverseA0;
        _b2 = -alpha * inverseA0;
        _a1 = -2f * cosW0 * inverseA0;
        _a2 = (1f - alpha) * inverseA0;
    }

    /// <summary>Runs one sample through the filter.</summary>
    public float Process(float input)
    {
        var output = _b0 * input + _b2 * _x2 - _a1 * _y1 - _a2 * _y2;

        _x2 = _x1;
        _x1 = input;
        _y2 = _y1;
        _y1 = output;

        if (!float.IsFinite(output))
        {
            // A denormal storm or a pathological coefficient must never poison the whole buffer.
            Reset();
            return 0f;
        }

        return output;
    }

    /// <summary>Clears the filter history without touching the coefficients.</summary>
    public void Reset()
    {
        _x1 = 0f;
        _x2 = 0f;
        _y1 = 0f;
        _y2 = 0f;
    }
}
