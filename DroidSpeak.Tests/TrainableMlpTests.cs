using DroidSpeak.Neural;
using DroidSpeak.Trainer;

namespace DroidSpeak.Tests;

public class TrainableMlpTests
{
    [Theory]
    [InlineData(ActivationType.Linear)]
    [InlineData(ActivationType.Sigmoid)]
    [InlineData(ActivationType.Tanh)]
    [InlineData(ActivationType.Relu)]
    [InlineData(ActivationType.LeakyRelu)]
    [InlineData(ActivationType.ModRelu)]
    public void BackwardMatchesCentralDifferences(ActivationType activation)
    {
        var network = new TrainableMlp([new(2, 3, activation), new(3, 2, ActivationType.Linear)],
            42, 0.9f, 0.999f, 1e-8f);
        float[] input = [0.5f, -0.25f];
        float[] target = [0.1f, -0.4f];
        // Hidden preactivations span the negative, pass-through and above-knee branches,
        // far enough from 0 and 1 that the perturbation never crosses a kink.
        float[] parameters = [0.1f, 0.2f, 0.2f, -0.2f, 0.3f, 0.2f, -0.6f, 0.25f, 1.4f,
            0.2f, -0.4f, 0.35f, -0.3f, 0.2f, 0.6f, 0.1f, -0.15f];
        network.LoadParameters(parameters);
        network.TrainSample(input, target);
        var gradients = new float[parameters.Length];
        network.CopyGradients(gradients);

        // A near-zero gradient vector would satisfy any tolerance, so prove there is signal first.
        var largest = 0f;
        foreach (var gradient in gradients)
        {
            largest = MathF.Max(largest, MathF.Abs(gradient));
        }

        Assert.True(largest > 1e-2f, $"{activation}: gradients are essentially zero.");

        const float h = 0.002f;
        for (int i = 0; i < parameters.Length; i++)
        {
            var original = parameters[i];
            parameters[i] = original + h;
            network.LoadParameters(parameters);
            var plus = network.Evaluate(input, target);
            parameters[i] = original - h;
            network.LoadParameters(parameters);
            var minus = network.Evaluate(input, target);
            parameters[i] = original;
            var expected = (plus - minus) / (2 * h);
            Assert.True(Math.Abs(expected - gradients[i]) < 5e-5f + 0.001f * Math.Abs(expected),
                $"{activation}, parameter {i}: numerical {expected}, analytical {gradients[i]}");
        }
    }

    [Fact]
    public void AdamFirstStepUsesBiasCorrectedMoments()
    {
        var network = new TrainableMlp([new(1, 1, ActivationType.Linear)], 42, 0.9f, 0.999f, 1e-8f);
        network.LoadParameters([0f, 0f]);
        network.TrainSample([-2f], [1f]);
        network.ApplyAdam(0.03f, 1f);
        var parameters = new float[2];
        network.CopyParameters(parameters);
        // At t=1, mHat=g and vHat=g². Both updates have magnitude lr despite unequal gradients.
        Assert.Equal(-0.03f, parameters[0], 6);
        Assert.Equal(0.03f, parameters[1], 6);
    }

    [Fact]
    public void AdamConvergesOnExclusiveOr()
    {
        // XOR is not linearly separable, so a low loss here means the hidden layer really learned.
        var options = TrainingOptions.Default with
        {
            HiddenWidths = [8],
            HiddenActivation = ActivationType.Tanh,
            Epochs = 400,
            BatchSize = 4,
            LearningRate = 0.05f,
            Schedule = LearningRateSchedule.Constant,
            ValidationFraction = 0.5f,
            LogEvery = 400,
        };

        var samples = new TrainingSet(
            [0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f],
            [0f, 1f, 1f, 0f],
            inputSize: 2,
            outputSize: 1);

        var network = new TrainableMlp(
            options.BuildSpecs(2, 1), options.Seed, options.Beta1, options.Beta2, options.Epsilon);

        var result = new TrainingRun(options, _ => { }).Run(network, samples, samples);

        Assert.True(result.BestValidationLoss < 1e-3f,
            $"XOR stayed at {result.BestValidationLoss:E3}.");

        Span<float> output = stackalloc float[1];
        for (var index = 0; index < samples.Count; index++)
        {
            network.Forward(samples.Features(index));
            network.CopyOutput(output);
            Assert.True(
                MathF.Abs(output[0] - samples.Targets(index)[0]) < 0.1f,
                $"Case {index} produced {output[0]:F3}.");
        }
    }
}
