namespace DroidSpeak.Neural;

/// <summary>
/// A single dense (fully connected) layer with flat, row-major weights. Row <c>n</c> of the weight
/// tensor holds the incoming connections of neuron <c>n</c>, so a forward pass walks memory in
/// order and never has to stride.
/// <para>
/// The tensors are owned by the layer and are copied on <see cref="Create"/>, which means a model
/// loaded from disk or built by a trainer can hand its scratch buffers back to the pool. Weights
/// are always <see cref="float"/>, never <see cref="double"/>: inference is single precision on
/// every platform, which keeps neural prosody bit-reproducible.
/// </para>
/// </summary>
public sealed class DenseLayer
{
    private readonly float[] _weights;
    private readonly float[] _biases;

    internal DenseLayer(
        float[] weights,
        float[] biases,
        int inputSize,
        int outputSize,
        ActivationType activation)
    {
        _weights = weights;
        _biases = biases;
        InputSize = inputSize;
        OutputSize = outputSize;
        Activation = activation;
    }

    /// <summary>Creates a layer and copies the supplied tensors into it.</summary>
    /// <param name="weights">
    /// <paramref name="inputSize"/> * <paramref name="outputSize"/> weights in row-major order, one
    /// row per output neuron.
    /// </param>
    /// <param name="biases">One bias per output neuron.</param>
    /// <param name="inputSize">Number of incoming activations.</param>
    /// <param name="outputSize">Number of neurons.</param>
    /// <param name="activation">Non-linearity applied to each neuron.</param>
    /// <exception cref="ArgumentOutOfRangeException">
    /// A size is outside <c>1..</c><see cref="MlpModel.MaxSupportedLayerWidth"/>, or the tensors do
    /// not match the shape.
    /// </exception>
    /// <exception cref="ArgumentException">A weight or bias is not finite.</exception>
    public static DenseLayer Create(
        ReadOnlySpan<float> weights,
        ReadOnlySpan<float> biases,
        int inputSize,
        int outputSize,
        ActivationType activation)
    {
        ValidateShape(weights, biases, inputSize, outputSize, activation);

        var storage = new float[weights.Length];
        weights.CopyTo(storage);

        var biasStorage = new float[biases.Length];
        biases.CopyTo(biasStorage);

        return new DenseLayer(storage, biasStorage, inputSize, outputSize, activation);
    }

    /// <summary>Number of incoming activations.</summary>
    public int InputSize { get; }

    /// <summary>Number of neurons in the layer.</summary>
    public int OutputSize { get; }

    /// <summary>Non-linearity applied to each neuron.</summary>
    public ActivationType Activation { get; }

    /// <summary>Row-major weights, one row of <see cref="InputSize"/> values per neuron.</summary>
    public ReadOnlySpan<float> Weights => _weights;

    /// <summary>One bias per neuron.</summary>
    public ReadOnlySpan<float> Biases => _biases;

    /// <summary>Total number of trainable values in the layer (weights plus biases).</summary>
    public int ParameterCount => _weights.Length + _biases.Length;

    internal float[] WeightStorage => _weights;

    internal float[] BiasStorage => _biases;

    internal static void ValidateShape(
        ReadOnlySpan<float> weights,
        ReadOnlySpan<float> biases,
        int inputSize,
        int outputSize,
        ActivationType activation)
    {
        if (inputSize is < 1 or > MlpModel.MaxSupportedLayerWidth)
        {
            throw new ArgumentOutOfRangeException(
                nameof(inputSize),
                inputSize,
                $"Layer input size must be between 1 and {MlpModel.MaxSupportedLayerWidth}.");
        }

        if (outputSize is < 1 or > MlpModel.MaxSupportedLayerWidth)
        {
            throw new ArgumentOutOfRangeException(
                nameof(outputSize),
                outputSize,
                $"Layer output size must be between 1 and {MlpModel.MaxSupportedLayerWidth}.");
        }

        if (activation is < ActivationType.Sigmoid or > ActivationType.Linear)
        {
            throw new ArgumentOutOfRangeException(
                nameof(activation),
                activation,
                "Unknown activation type.");
        }

        var expectedWeights = checked(inputSize * outputSize);
        if (weights.Length != expectedWeights)
        {
            throw new ArgumentException(
                $"Expected {expectedWeights} weights for a {inputSize}x{outputSize} layer, " +
                $"but received {weights.Length}.",
                nameof(weights));
        }

        if (biases.Length != outputSize)
        {
            throw new ArgumentException(
                $"Expected {outputSize} biases for a layer with {outputSize} neurons, " +
                $"but received {biases.Length}.",
                nameof(biases));
        }

        for (var i = 0; i < weights.Length; i++)
        {
            if (!float.IsFinite(weights[i]))
            {
                throw new ArgumentException(
                    $"Weight at index {i} is not finite. Non-finite weights would poison every " +
                    "downstream sample.",
                    nameof(weights));
            }
        }

        for (var i = 0; i < biases.Length; i++)
        {
            if (!float.IsFinite(biases[i]))
            {
                throw new ArgumentException(
                    $"Bias at index {i} is not finite.",
                    nameof(biases));
            }
        }
    }
}
