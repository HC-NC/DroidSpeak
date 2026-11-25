namespace DroidSpeak.Neural;

/// <summary>
/// Scalar helpers shared by every neural layer. Everything here is allocation free, inlines to a
/// handful of instructions and uses only <see cref="MathF"/>, so the same weights always produce
/// the same output on every supported runtime.
/// </summary>
public static class NeuralMathF
{
    /// <summary>Slope applied to negative inputs by <see cref="LeakyRelu"/>.</summary>
    public const float LeakyReluAlpha = 0.01f;

    /// <summary>Slope applied outside the pass band by <see cref="ModRelu"/>.</summary>
    public const float ModReluLeak = 0.1f;

    /// <summary>Upper edge of the identity pass band of <see cref="ModRelu"/>.</summary>
    public const float ModReluKnee = 1f;

    /// <summary>
    /// Largest magnitude a non-finite value is folded back to for activations that are not
    /// naturally bounded. Chosen well inside <see cref="float.MaxValue"/> so a following layer
    /// cannot overflow.
    /// </summary>
    public const float UnboundedLimit = 10f;

    /// <summary>Logistic sigmoid, saturating smoothly at <c>0</c> and <c>1</c>.</summary>
    public static float Sigmoid(float value) => 1f / (1f + MathF.Exp(-value));

    /// <summary>Hyperbolic tangent, saturating smoothly at <c>-1</c> and <c>1</c>.</summary>
    public static float Tanh(float value) => MathF.Tanh(value);

    /// <summary>Rectified linear unit.</summary>
    public static float Relu(float value) => value > 0f ? value : 0f;

    /// <summary>ReLU with a small negative slope so no unit can die completely.</summary>
    public static float LeakyRelu(float value) => value > 0f ? value : LeakyReluAlpha * value;

    /// <summary>
    /// The modulo ReLU from the legacy engine: identity on <c>[0, 1]</c>, and a shallow leak of
    /// <c>0.1 * x</c> (shifted by one) above the knee and <c>0.1 * x</c> below zero.
    /// </summary>
    public static float ModRelu(float value)
    {
        if (value < 0f)
        {
            return ModReluLeak * value;
        }

        return value > ModReluKnee ? ModReluLeak * value + ModReluKnee : value;
    }

    /// <summary>Identity.</summary>
    public static float Linear(float value) => value;

    /// <summary>Applies <paramref name="activation"/> to a single pre-activation value.</summary>
    public static float Apply(float value, ActivationType activation) => activation switch
    {
        ActivationType.Sigmoid => Sigmoid(value),
        ActivationType.Tanh => Tanh(value),
        ActivationType.Relu => Relu(value),
        ActivationType.LeakyRelu => LeakyRelu(value),
        ActivationType.ModRelu => ModRelu(value),
        _ => Linear(value),
    };

    /// <summary>Applies <paramref name="activation"/> to every element of <paramref name="values"/>.</summary>
    public static void ApplyInPlace(Span<float> values, ActivationType activation)
    {
        for (var i = 0; i < values.Length; i++)
        {
            values[i] = Apply(values[i], activation);
        }
    }

    /// <summary>
    /// Folds a non-finite activation result back into the range the activation can legitimately
    /// produce. Finite values are returned untouched, so the hot path costs one predictable branch.
    /// </summary>
    public static float Sanitize(float value, ActivationType activation)
    {
        if (float.IsFinite(value))
        {
            return value;
        }

        // NaN carries no direction, so it falls back to the neutral point of the activation.
        if (float.IsNaN(value))
        {
            return activation == ActivationType.Sigmoid ? 0.5f : 0f;
        }

        return activation switch
        {
            ActivationType.Sigmoid => value > 0f ? 1f : 0f,
            ActivationType.Tanh => value > 0f ? 1f : -1f,
            _ => value > 0f ? UnboundedLimit : -UnboundedLimit,
        };
    }

    /// <summary>Clamps into <c>[0, 1]</c>; <see cref="float.NaN"/> maps to <c>0</c>.</summary>
    public static float Clamp01(float value) =>
        float.IsNaN(value) ? 0f : Math.Clamp(value, 0f, 1f);

    /// <summary>
    /// Clamps into <c>[min, max]</c>. Infinite values move to the nearest bound and
    /// <see cref="float.NaN"/> falls back to <paramref name="fallback"/>, mirroring
    /// <see cref="DroidVoiceProfile.Normalized"/>.
    /// </summary>
    public static float ClampFinite(float value, float min, float max, float fallback) =>
        float.IsNaN(value) ? fallback : Math.Clamp(value, min, max);

    /// <summary>Linear interpolation between <paramref name="from"/> and <paramref name="to"/>.</summary>
    public static float Lerp(float from, float to, float amount) =>
        from + (to - from) * Clamp01(amount);
}
