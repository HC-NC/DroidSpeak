namespace DroidSpeak.Neural;

/// <summary>
/// A flat, fully connected multi-layer perceptron used for inference.
/// <para>
/// <see cref="Predict"/> is the hot path and is written to be completely allocation free: the
/// activations of every layer live in two stack buffers that ping-pong between layers, so a
/// prediction never touches the managed heap and never boxes a value. The layer tensors themselves
/// are allocated once, when the model is built or loaded.
/// </para>
/// <para>
/// Every layer is bounded by <see cref="MaxSupportedLayerWidth"/>, which is what makes the stack
/// footprint of a prediction a compile-time constant
/// (<c>2 * MaxSupportedLayerWidth * sizeof(float)</c> = 2 KB) instead of a heap tensor. That bound
/// is generous for prosody: a network wider than 256 neurons would be far too slow for real-time
/// speech on a microcontroller-class target anyway.
/// </para>
/// </summary>
public sealed partial class MlpModel
{
    /// <summary>
    /// Largest input, output or hidden width the zero-allocation forward pass supports. The value is
    /// part of the model contract: loading a file with a wider layer fails instead of falling back to
    /// a heap buffer.
    /// </summary>
    public const int MaxSupportedLayerWidth = 256;

    /// <summary>Deepest network the inference path supports.</summary>
    public const int MaxLayerCount = 32;

    /// <summary>Seed used by <see cref="CreateInitialized(ReadOnlySpan{LayerSpec}, uint)"/> when none is given.</summary>
    public const uint DefaultInitSeed = 0x5EED0001u;

    private readonly DenseLayer[] _layers;

    /// <summary>Builds a model from a fully specified chain of layers.</summary>
    /// <param name="layers">
    /// Layers in evaluation order. Each layer's input size must equal the previous layer's output
    /// size, and every width must stay within <see cref="MaxSupportedLayerWidth"/>.
    /// </param>
    /// <exception cref="ArgumentException">The chain is empty, discontinuous, or too deep.</exception>
    public MlpModel(ReadOnlySpan<DenseLayer> layers)
    {
        if (layers.IsEmpty)
        {
            throw new ArgumentException("A model needs at least one layer.", nameof(layers));
        }

        if (layers.Length > MaxLayerCount)
        {
            throw new ArgumentException(
                $"A model supports at most {MaxLayerCount} layers, but {layers.Length} were given.",
                nameof(layers));
        }

        _layers = new DenseLayer[layers.Length];
        layers.CopyTo(_layers);

        for (var i = 0; i < _layers.Length; i++)
        {
            var layer = _layers[i];
            ArgumentNullException.ThrowIfNull(layer);

            if (layer.InputSize is < 1 or > MaxSupportedLayerWidth)
            {
                throw new ArgumentException(
                    $"Layer {i} has an input size of {layer.InputSize}, which is outside " +
                    $"1..{MaxSupportedLayerWidth}.",
                    nameof(layers));
            }

            if (layer.OutputSize is < 1 or > MaxSupportedLayerWidth)
            {
                throw new ArgumentException(
                    $"Layer {i} has an output size of {layer.OutputSize}, which is outside " +
                    $"1..{MaxSupportedLayerWidth}.",
                    nameof(layers));
            }

            if (i > 0 && _layers[i - 1].OutputSize != layer.InputSize)
            {
                throw new ArgumentException(
                    $"Layer {i - 1} produces {_layers[i - 1].OutputSize} values but layer {i} " +
                    $"expects {layer.InputSize}.",
                    nameof(layers));
            }
        }

        InputSize = _layers[0].InputSize;
        OutputSize = _layers[^1].OutputSize;
    }

    /// <summary>Width of the input vector the model expects.</summary>
    public int InputSize { get; }

    /// <summary>Width of the output vector the model produces.</summary>
    public int OutputSize { get; }

    /// <summary>Number of dense layers.</summary>
    public int LayerCount => _layers.Length;

    /// <summary>The layers in evaluation order.</summary>
    public ReadOnlySpan<DenseLayer> Layers => _layers;

    /// <summary>Reads a single layer.</summary>
    /// <param name="index">Zero-based layer index.</param>
    public DenseLayer this[int index] => _layers[index];

    /// <summary>Total number of weights and biases in the model.</summary>
    public int ParameterCount
    {
        get
        {
            var count = 0;
            for (var i = 0; i < _layers.Length; i++)
            {
                count += _layers[i].ParameterCount;
            }

            return count;
        }
    }

    /// <summary>
    /// Runs the forward pass.
    /// </summary>
    /// <param name="input">Exactly <see cref="InputSize"/> activations.</param>
    /// <param name="output">
    /// At least <see cref="OutputSize"/> floats; the first <see cref="OutputSize"/> entries are
    /// overwritten.
    /// </param>
    /// <remarks>
    /// This method allocates zero bytes: layer activations are computed into two
    /// <c>stackalloc</c> buffers. Nothing is boxed, no delegate is invoked and no array is rented.
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// <paramref name="input"/> does not have <see cref="InputSize"/> entries, or
    /// <paramref name="output"/> is shorter than <see cref="OutputSize"/>.
    /// </exception>
    public void Predict(ReadOnlySpan<float> input, Span<float> output)
    {
        if (input.Length != InputSize)
        {
            throw new ArgumentException(
                $"Expected {InputSize} input values, but received {input.Length}.",
                nameof(input));
        }

        if (output.Length < OutputSize)
        {
            throw new ArgumentException(
                $"The output buffer must hold at least {OutputSize} values, but has {output.Length}.",
                nameof(output));
        }

        Span<float> first = stackalloc float[MaxSupportedLayerWidth];
        Span<float> second = stackalloc float[MaxSupportedLayerWidth];

        input.CopyTo(first);

        var source = first;
        var destination = second;
        var sourceLength = InputSize;

        for (var i = 0; i < _layers.Length; i++)
        {
            var layer = _layers[i];
            Forward(layer, source[..sourceLength], destination[..layer.OutputSize]);

            var swap = source;
            source = destination;
            destination = swap;
            sourceLength = layer.OutputSize;
        }

        source[..OutputSize].CopyTo(output[..OutputSize]);
    }

    /// <summary>
    /// Builds a deterministically initialized network: Glorot/Xavier uniform for sigmoid, tanh and
    /// linear layers, and He normal for the ReLU family. Biases start at zero.
    /// </summary>
    /// <param name="layers">
    /// Layer shapes and activations in evaluation order. The model's input and output widths are the
    /// first and last entries.
    /// </param>
    /// <param name="seed">
    /// Initialization seed. The same seed and shape always produce byte-identical weights, which is
    /// what lets tests pin a network's output without shipping a weight file.
    /// </param>
    /// <exception cref="ArgumentException">The chain is empty, discontinuous, or too wide.</exception>
    public static MlpModel CreateInitialized(ReadOnlySpan<LayerSpec> layers, uint seed = DefaultInitSeed)
    {
        if (layers.IsEmpty)
        {
            throw new ArgumentException("A model needs at least one layer.", nameof(layers));
        }

        if (layers.Length > MaxLayerCount)
        {
            throw new ArgumentException(
                $"A model supports at most {MaxLayerCount} layers, but {layers.Length} were given.",
                nameof(layers));
        }

        var random = new NeuralRandom(seed);
        var dense = new DenseLayer[layers.Length];

        for (var i = 0; i < layers.Length; i++)
        {
            var spec = layers[i];

            if (spec.InputSize is < 1 or > MaxSupportedLayerWidth ||
                spec.OutputSize is < 1 or > MaxSupportedLayerWidth)
            {
                throw new ArgumentException(
                    $"Layer {i} is {spec.InputSize}x{spec.OutputSize}, which exceeds the " +
                    $"{MaxSupportedLayerWidth} neuron limit of the zero-allocation forward pass.",
                    nameof(layers));
            }

            if (i > 0 && layers[i - 1].OutputSize != spec.InputSize)
            {
                throw new ArgumentException(
                    $"Layer {i - 1} produces {layers[i - 1].OutputSize} values but layer {i} " +
                    $"expects {spec.InputSize}.",
                    nameof(layers));
            }

            var weightCount = checked(spec.InputSize * spec.OutputSize);
            var weights = new float[weightCount];
            var biases = new float[spec.OutputSize];
            var scale = spec.InitScale;

            if (spec.UseGaussianInit)
            {
                for (var w = 0; w < weights.Length; w++)
                {
                    weights[w] = random.NextGaussian() * scale;
                }
            }
            else
            {
                for (var w = 0; w < weights.Length; w++)
                {
                    weights[w] = random.NextSigned(scale);
                }
            }

            dense[i] = new DenseLayer(weights, biases, spec.InputSize, spec.OutputSize, spec.Activation);
        }

        return new MlpModel(dense);
    }

    /// <summary>
    /// Builds a network from pre-allocated tensors. Intended for the trainer in v0.7, which owns
    /// pooled arrays and hands ownership to the resulting model.
    /// </summary>
    /// <param name="specs">Layer shapes and activations in evaluation order.</param>
    /// <param name="weights">One weight tensor per layer, in row-major order.</param>
    /// <param name="biases">One bias tensor per layer.</param>
    /// <exception cref="ArgumentException">A tensor does not match its layer shape.</exception>
    public static MlpModel Create(
        ReadOnlySpan<LayerSpec> specs,
        ReadOnlySpan<float[]> weights,
        ReadOnlySpan<float[]> biases)
    {
        if (specs.Length != weights.Length || specs.Length != biases.Length)
        {
            throw new ArgumentException(
                "Every layer needs exactly one weight tensor and one bias tensor.",
                nameof(specs));
        }

        var dense = new DenseLayer[specs.Length];

        for (var i = 0; i < specs.Length; i++)
        {
            var spec = specs[i];
            var layerWeights = weights[i] ?? throw new ArgumentException(
                $"Weight tensor {i} is null.",
                nameof(weights));
            var layerBiases = biases[i] ?? throw new ArgumentException(
                $"Bias tensor {i} is null.",
                nameof(biases));

            DenseLayer.ValidateShape(
                layerWeights,
                layerBiases,
                spec.InputSize,
                spec.OutputSize,
                spec.Activation);

            dense[i] = new DenseLayer(
                layerWeights,
                layerBiases,
                spec.InputSize,
                spec.OutputSize,
                spec.Activation);
        }

        return new MlpModel(dense);
    }

    /// <summary>
    /// Multiplies <paramref name="input"/> by the layer, adds the biases and applies the activation.
    /// Accumulation order is fixed (bias first, then inputs ascending) so results are reproducible
    /// across runtimes and are not reordered by the vectorizing JIT into a different rounding path.
    /// </summary>
    private static void Forward(
        DenseLayer layer,
        ReadOnlySpan<float> input,
        Span<float> output)
    {
        var weights = layer.Weights;
        var biases = layer.Biases;
        var width = layer.InputSize;
        var activation = layer.Activation;

        for (var neuron = 0; neuron < layer.OutputSize; neuron++)
        {
            var row = neuron * width;
            var sum = biases[neuron];

            for (var i = 0; i < width; i++)
            {
                sum += weights[row + i] * input[i];
            }

            output[neuron] = NeuralMathF.Sanitize(NeuralMathF.Apply(sum, activation), activation);
        }
    }
}
