using DroidSpeak.Neural;

namespace DroidSpeak.Tests;

public class MlpModelTests
{
    [Fact]
    public void Predict_AppliesASingleLinearLayerExactly()
    {
        // 3 -> 2 sigmoid layer with hand-computed pre-activations:
        //   n0 = 0.1 + 0.5*1   + (-1)*(-2) + 2*0.5   = 3.6
        //   n1 = -0.3 + (-2)*1 + 0.25*(-2) + 1*0.5   = -2.3
        var model = new MlpModel(
        [
            DenseLayer.Create(
                [0.5f, -1f, 2f, -2f, 0.25f, 1f],
                [0.1f, -0.3f],
                inputSize: 3,
                outputSize: 2,
                activation: ActivationType.Sigmoid),
        ]);

        Span<float> input = [1f, -2f, 0.5f];
        Span<float> output = stackalloc float[2];
        model.Predict(input, output);

        Assert.Equal(NeuralMathF.Sigmoid(3.6f), output[0], 6);
        Assert.Equal(NeuralMathF.Sigmoid(-2.3f), output[1], 6);
    }

    [Fact]
    public void Predict_PingsPongsThroughTwoLayers()
    {
        // Layer 1 (2 -> 3, ReLU) on input [2, -1]:
        //   n0 = 1*2 + 1*(-1)  =  1 ->  1
        //   n1 = 0.5*2 - 1*(-1)=  2 ->  2
        //   n2 = -1*2 - 1*(-1) = -1 ->  0
        // Layer 2 (3 -> 1, Tanh):
        //   z = 0.5 + 1*1 + 2*2 + (-3)*0 = 5.5
        var model = new MlpModel(
        [
            DenseLayer.Create(
                [1f, 1f, 0.5f, -1f, -1f, -1f],
                [0f, 0f, 0f],
                inputSize: 2,
                outputSize: 3,
                activation: ActivationType.Relu),
            DenseLayer.Create(
                [1f, 2f, -3f],
                [0.5f],
                inputSize: 3,
                outputSize: 1,
                activation: ActivationType.Tanh),
        ]);

        Span<float> input = [2f, -1f];
        Span<float> output = stackalloc float[1];
        model.Predict(input, output);

        Assert.Equal(MathF.Tanh(5.5f), output[0], 6);
    }

    [Fact]
    public void Predict_BiasesDriveTheOutputWhenTheInputIsZero()
    {
        var model = new MlpModel(
        [
            DenseLayer.Create(
                [0f, 0f, 0f, 0f],
                [2f, -2f],
                inputSize: 2,
                outputSize: 2,
                activation: ActivationType.Tanh),
        ]);

        Span<float> input = stackalloc float[2];
        Span<float> output = stackalloc float[2];
        model.Predict(input, output);

        Assert.Equal(MathF.Tanh(2f), output[0], 6);
        Assert.Equal(MathF.Tanh(-2f), output[1], 6);
    }

    [Fact]
    public void Predict_SupportsEveryActivation()
    {
        Span<float> input = [-0.7f];
        Span<float> output = stackalloc float[1];

        foreach (ActivationType activation in Enum.GetValues<ActivationType>())
        {
            var model = new MlpModel(
            [
                DenseLayer.Create(
                    [1f],
                    [0.4f],
                    inputSize: 1,
                    outputSize: 1,
                    activation),
            ]);

            model.Predict(input, output);

            Assert.Equal(NeuralMathF.Apply(0.4f - 0.7f, activation), output[0], 6);
        }
    }

    [Fact]
    public void Predict_WritesOnlyOutputSizeEntries()
    {
        var model = CreateNetwork();

        Span<float> input = stackalloc float[model.InputSize];
        Span<float> output = stackalloc float[model.OutputSize + 4];
        output[^4..].Fill(-123f);
        model.Predict(input, output);

        Assert.Equal(-123f, output[^4]);
        Assert.Equal(-123f, output[^1]);
    }

    [Fact]
    public void Predict_IsDeterministicAcrossCalls()
    {
        var model = CreateNetwork();
        Span<float> input = stackalloc float[model.InputSize];
        Span<float> first = stackalloc float[model.OutputSize];
        Span<float> second = stackalloc float[model.OutputSize];

        for (var i = 0; i < input.Length; i++)
        {
            input[i] = (i % 7) * 0.13f;
        }

        model.Predict(input, first);
        model.Predict(input, second);

        Assert.True(first.SequenceEqual(second));
    }

    [Fact]
    public void Predict_AllocatesZeroBytesOnTheWarmPath()
    {
        var model = CreateNetwork();
        Span<float> input = stackalloc float[model.InputSize];
        Span<float> output = stackalloc float[model.OutputSize];

        // Warm up tier-0 JIT and any first-call fixups.
        for (var i = 0; i < 16; i++)
        {
            model.Predict(input, output);
        }

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 1000; i++)
        {
            model.Predict(input, output);
        }

        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(0, allocated);
    }

    [Fact]
    public void Predict_RejectsWrongInputLength()
    {
        var model = CreateNetwork();
        var input = new float[model.InputSize - 1];
        var output = new float[model.OutputSize];

        Assert.Throws<ArgumentException>(() => model.Predict(input, output));
    }

    [Fact]
    public void Predict_RejectsShortOutputBuffer()
    {
        var model = CreateNetwork();
        var input = new float[model.InputSize];
        var output = new float[model.OutputSize - 1];

        Assert.Throws<ArgumentException>(() => model.Predict(input, output));
    }

    [Fact]
    public void Constructor_RejectsAnEmptyLayerList()
    {
        Assert.Throws<ArgumentException>(() => new MlpModel([]));
    }

    [Fact]
    public void Constructor_RejectsDiscontinuousLayers()
    {
        var first = DenseLayer.Create(
            [1f, 2f, 3f, 4f],
            [0f, 0f],
            inputSize: 2,
            outputSize: 2,
            ActivationType.Linear);
        var second = DenseLayer.Create(
            [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f],
            [0f, 0f, 0f],
            inputSize: 3,
            outputSize: 3,
            ActivationType.Linear);

        Assert.Throws<ArgumentException>(() => new MlpModel([first, second]));
    }

    [Fact]
    public void Constructor_RejectsTooManyLayers()
    {
        var layer = DenseLayer.Create([1f], [0f], 1, 1, ActivationType.Linear);
        var layers = new DenseLayer[MlpModel.MaxLayerCount + 1];
        Array.Fill(layers, layer);

        Assert.Throws<ArgumentException>(() => new MlpModel(layers));
    }

    [Fact]
    public void DenseLayer_RejectsWrongTensorShapes()
    {
        Assert.Throws<ArgumentException>(() =>
            DenseLayer.Create([1f, 2f, 3f], [0f, 0f], 2, 2, ActivationType.Relu));

        Assert.Throws<ArgumentException>(() =>
            DenseLayer.Create([1f, 2f, 3f, 4f], [0f], 2, 2, ActivationType.Relu));
    }

    [Fact]
    public void DenseLayer_RejectsWidthsAboveTheStackBufferLimit()
    {
        var width = MlpModel.MaxSupportedLayerWidth + 1;
        var weights = new float[width];

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DenseLayer.Create(weights, [0f], width, 1, ActivationType.Relu));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            DenseLayer.Create(weights, [0f], 1, width, ActivationType.Relu));
    }

    [Fact]
    public void DenseLayer_RejectsNonFiniteTensors()
    {
        Assert.Throws<ArgumentException>(() =>
            DenseLayer.Create([1f, float.NaN, 3f, 4f], [0f, 0f], 2, 2, ActivationType.Relu));

        Assert.Throws<ArgumentException>(() =>
            DenseLayer.Create(
                [1f, 2f, 3f, 4f],
                [0f, float.PositiveInfinity],
                2,
                2,
                ActivationType.Relu));
    }

    [Fact]
    public void DenseLayer_CopiesTheSuppliedTensors()
    {
        var weights = new float[] { 1f, 2f, 3f, 4f };
        var layer = DenseLayer.Create(weights, [0f, 0f], 2, 2, ActivationType.Linear);

        weights[0] = 99f;

        Assert.Equal(1f, layer.Weights[0]);
        Assert.Equal(2, layer.InputSize);
        Assert.Equal(2, layer.OutputSize);
        Assert.Equal(ActivationType.Linear, layer.Activation);
        Assert.Equal(6, layer.ParameterCount);
    }

    [Fact]
    public void CreateInitialized_IsDeterministicPerSeed()
    {
        var first = CreateNetwork(seed: 1234u);
        var second = CreateNetwork(seed: 1234u);
        var other = CreateNetwork(seed: 4321u);

        Span<float> input = stackalloc float[first.InputSize];
        Span<float> outputA = stackalloc float[first.OutputSize];
        Span<float> outputB = stackalloc float[second.OutputSize];
        Span<float> outputC = stackalloc float[other.OutputSize];

        for (var i = 0; i < input.Length; i++)
        {
            input[i] = i * 0.05f;
        }

        first.Predict(input, outputA);
        second.Predict(input, outputB);
        other.Predict(input, outputC);

        Assert.True(outputA.SequenceEqual(outputB));
        Assert.False(outputA.SequenceEqual(outputC));
    }

    [Fact]
    public void CreateInitialized_UsesBoundedWeightsAndZeroBiases()
    {
        var model = CreateNetwork(seed: 7u);
        var layer = model[0];

        // Glorot uniform for a tanh layer of this shape keeps |w| well below one, so a bounded
        // input vector can never blow up into infinity in the first layer.
        foreach (var weight in layer.Weights)
        {
            Assert.True(float.IsFinite(weight));
            Assert.InRange(weight, -1f, 1f);
        }

        foreach (var bias in layer.Biases)
        {
            Assert.Equal(0f, bias);
        }
    }

    [Fact]
    public void CreateInitialized_ScalesReluLayersWithHeVariance()
    {
        var relu = LayerSpecFor(ActivationType.Relu, inputSize: 64);
        var sigmoid = LayerSpecFor(ActivationType.Sigmoid, inputSize: 64);

        var reluModel = MlpModel.CreateInitialized([relu]);
        var sigmoidModel = MlpModel.CreateInitialized([sigmoid]);

        var reluSpread = StandardDeviation(reluModel[0].Weights);
        var sigmoidSpread = StandardDeviation(sigmoidModel[0].Weights);

        // He: std = sqrt(2 / fanIn) = 0.177. Glorot uniform: std = sqrt(6/(64+64)) / sqrt(3) ~ 0.125.
        Assert.InRange(reluSpread, 0.12f, 0.24f);
        Assert.InRange(sigmoidSpread, 0.08f, 0.16f);
        Assert.True(reluSpread > sigmoidSpread);
    }

    [Fact]
    public void CreateInitialized_RejectsWidthsAboveTheLimit()
    {
        var tooWide = new LayerSpec(1, MlpModel.MaxSupportedLayerWidth + 1, ActivationType.Relu);

        Assert.Throws<ArgumentException>(() => MlpModel.CreateInitialized([tooWide]));
    }

    [Fact]
    public void CreateInitialized_RejectsDiscontinuousSpecs()
    {
        Assert.Throws<ArgumentException>(() => MlpModel.CreateInitialized(
        [
            new LayerSpec(2, 3, ActivationType.Tanh),
            new LayerSpec(4, 1, ActivationType.Linear),
        ]));
    }

    [Fact]
    public void Create_UsesTheSuppliedTensorsWithoutCopying()
    {
        var weights = new float[] { 2f, 3f };
        var biases = new float[] { 1f };
        var model = MlpModel.Create(
            [new LayerSpec(2, 1, ActivationType.Linear)],
            [weights],
            [biases]);

        weights[0] = 100f;

        Assert.Equal(2, model.InputSize);
        Assert.Equal(1, model.OutputSize);
        Assert.Equal(1, model.LayerCount);

        Span<float> input = [1f, 1f];
        Span<float> output = stackalloc float[1];
        model.Predict(input, output);

        // 1 + 100*1 + 3*1 after the in-place mutation, proving the arrays are owned, not copied.
        Assert.Equal(104f, output[0]);
    }

    [Fact]
    public void Create_RejectsMismatchedTensorCounts()
    {
        Assert.Throws<ArgumentException>(() => MlpModel.Create(
            [new LayerSpec(2, 1, ActivationType.Linear)],
            [],
            []));
    }

    [Fact]
    public void Model_ExposesItsShape()
    {
        var model = CreateNetwork();

        Assert.Equal(4, model.InputSize);
        Assert.Equal(2, model.OutputSize);
        Assert.Equal(3, model.LayerCount);
        Assert.Equal(2, model[1].OutputSize);
        Assert.Equal(model.LayerCount, model.Layers.Length);
        Assert.Equal(model[0].ParameterCount + model[1].ParameterCount + model[2].ParameterCount,
            model.ParameterCount);
    }

    private static MlpModel CreateNetwork(uint seed = 20261125u) => MlpModel.CreateInitialized(
    [
        new LayerSpec(4, 8, ActivationType.Tanh),
        new LayerSpec(8, 2, ActivationType.Relu),
        new LayerSpec(2, 2, ActivationType.Sigmoid),
    ], seed);

    private static LayerSpec LayerSpecFor(ActivationType activation, int inputSize) =>
        new(inputSize, inputSize, activation);

    private static float StandardDeviation(ReadOnlySpan<float> values)
    {
        var sum = 0f;
        for (var i = 0; i < values.Length; i++)
        {
            sum += values[i];
        }

        var mean = sum / values.Length;
        var squared = 0f;
        for (var i = 0; i < values.Length; i++)
        {
            var delta = values[i] - mean;
            squared += delta * delta;
        }

        return MathF.Sqrt(squared / values.Length);
    }
}
