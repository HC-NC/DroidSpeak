namespace DroidSpeak.Neural;

/// <summary>
/// The non-linearity applied to the output of a dense layer. The numeric values are persisted in
/// saved models, so existing members must never be renumbered.
/// </summary>
public enum ActivationType : byte
{
    /// <summary><c>1 / (1 + e^-x)</c>, compresses any input into <c>(0, 1)</c>.</summary>
    Sigmoid = 0,

    /// <summary>Centred sigmoid, compresses any input into <c>(-1, 1)</c>.</summary>
    Tanh = 1,

    /// <summary><c>max(0, x)</c>.</summary>
    Relu = 2,

    /// <summary>ReLU with a <see cref="NeuralMathF.LeakyReluAlpha"/> slope below zero.</summary>
    LeakyRelu = 3,

    /// <summary>
    /// The modulo ReLU of the legacy engine: identity on <c>[0, 1]</c> and a
    /// <see cref="NeuralMathF.ModReluLeak"/>-slope leak outside it, which stops units from dying
    /// while keeping the useful range unsquashed.
    /// </summary>
    ModRelu = 4,

    /// <summary>Identity, used for the output layer and for purely linear projections.</summary>
    Linear = 5,
}
