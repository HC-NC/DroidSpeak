using DroidSpeak.Neural;

namespace DroidSpeak.Trainer;

/// <summary>
/// Forward activations and exact analytical derivatives for every <see cref="ActivationType"/> the
/// inference engine supports.
/// <para>
/// The forward half deliberately calls straight into <see cref="NeuralMathF"/>: a trainer that used
/// its own sigmoid would minimize the error of a network that does not exist at inference time. The
/// derivative half mirrors the branch conditions of those same functions, including the edge cases
/// the runtime picks at a tie (<see cref="NeuralMathF.Relu"/> is zero at the origin, so its slope
/// there is taken as zero; <see cref="NeuralMathF.ModRelu"/> leaks outside <c>[0, 1]</c>, so it is
/// flat there rather than a pass-through).
/// </para>
/// </summary>
internal static class ActivationMath
{
    /// <summary>
    /// Applies the activation to one pre-activation value, exactly as
    /// <see cref="MlpModel"/> does during inference, non-finite results included.
    /// </summary>
    public static float Forward(float preActivation, ActivationType activation) =>
        NeuralMathF.Sanitize(NeuralMathF.Apply(preActivation, activation), activation);

    /// <summary>
    /// Slope of the activation at <paramref name="preActivation"/>. Non-finite inputs return zero so
    /// a blown-up unit stops contributing gradient instead of poisoning the whole tensor.
    /// </summary>
    public static float Derivative(float preActivation, ActivationType activation)
    {
        if (!float.IsFinite(preActivation))
        {
            return 0f;
        }

        switch (activation)
        {
            case ActivationType.Sigmoid:
                // d/dx sigmoid(x) = s * (1 - s). Recomputing the logistic keeps this consistent with
                // the sanitized forward value at both tails, where s has rounded to 0 or 1.
                var s = NeuralMathF.Sigmoid(preActivation);
                return s * (1f - s);

            case ActivationType.Tanh:
                var t = NeuralMathF.Tanh(preActivation);
                return 1f - (t * t);

            case ActivationType.Relu:
                return preActivation > 0f ? 1f : 0f;

            case ActivationType.LeakyRelu:
                return preActivation > 0f ? 1f : NeuralMathF.LeakyReluAlpha;

            case ActivationType.ModRelu:
                // Identity inside the pass band, constant leak outside it.
                return preActivation < 0f || preActivation > NeuralMathF.ModReluKnee
                    ? NeuralMathF.ModReluLeak
                    : 1f;

            default:
                return 1f;
        }
    }
}
