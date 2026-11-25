namespace DroidSpeak.Neural;

/// <summary>
/// The shape and non-linearity of one dense layer, used to describe a network before its tensors
/// exist. Sizing the layer is the caller's job: <see cref="MlpModel.CreateInitialized"/> turns a
/// list of specs into a Xavier/He initialized model, and <see cref="DenseLayer.Create"/> turns it
/// into a layer with supplied weights.
/// </summary>
public readonly struct LayerSpec
{
    /// <summary>Describes a dense layer.</summary>
    /// <param name="inputSize">Number of incoming activations.</param>
    /// <param name="outputSize">Number of neurons in the layer.</param>
    /// <param name="activation">Non-linearity applied to each neuron.</param>
    public LayerSpec(int inputSize, int outputSize, ActivationType activation)
    {
        InputSize = inputSize;
        OutputSize = outputSize;
        Activation = activation;
    }

    /// <summary>Number of incoming activations.</summary>
    public int InputSize { get; }

    /// <summary>Number of neurons in the layer.</summary>
    public int OutputSize { get; }

    /// <summary>Non-linearity applied to each neuron.</summary>
    public ActivationType Activation { get; }

    /// <summary>Number of weights the layer needs (<see cref="InputSize"/> * <see cref="OutputSize"/>).</summary>
    public int WeightCount => InputSize * OutputSize;

    /// <summary>
    /// Scale factor applied to a deterministic uniform or gaussian source when filling the layer,
    /// following Glorot for bounded activations and He for the ReLU family.
    /// </summary>
    internal float InitScale => Activation switch
    {
        ActivationType.Relu => MathF.Sqrt(2f / InputSize),
        ActivationType.LeakyRelu => MathF.Sqrt(
            2f / ((1f + LeakyReluGain) * InputSize)),
        ActivationType.ModRelu => MathF.Sqrt(2f / InputSize),
        _ => MathF.Sqrt(6f / (InputSize + OutputSize)),
    };

    /// <summary>
    /// Whether initialization should draw from a gaussian rather than a uniform source. Only the
    /// ReLU family uses the gaussian He variant; everything else uses Glorot uniform.
    /// </summary>
    internal bool UseGaussianInit =>
        Activation is ActivationType.Relu or ActivationType.LeakyRelu or ActivationType.ModRelu;

    private const float LeakyReluGain = NeuralMathF.LeakyReluAlpha * NeuralMathF.LeakyReluAlpha;
}
