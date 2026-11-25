using System.Globalization;
using DroidSpeak.Neural;

namespace DroidSpeak.Trainer;

/// <summary>
/// A two-bit XOR probe: the smallest problem that proves backpropagation and Adam actually work,
/// runnable from the CLI before any distillation corpus exists.
/// </summary>
/// <remarks>
/// XOR is not linearly separable, so a network that solves it has really learned through its hidden
/// layer. The probe doubles as the trainer's self-check: if this fails, nothing the
/// <c>distill</c> command reports can be trusted.
/// </remarks>
internal static class XorProbe
{
    private static readonly float[] Features =
    [
        0f, 0f,
        0f, 1f,
        1f, 0f,
        1f, 1f,
    ];

    private static readonly float[] Targets = [0f, 1f, 1f, 0f];

    /// <summary>Runs the probe and reports whether the network converged.</summary>
    public static bool Run()
    {
        var options = TrainingOptions.Default with
        {
            HiddenWidths = [8],
            HiddenActivation = ActivationType.Tanh,
            Epochs = 400,
            BatchSize = 4,
            LearningRate = 0.05f,
            Schedule = LearningRateSchedule.Constant,
            ValidationFraction = 0.5f,
            LogEvery = 100,
        };

        var samples = new TrainingSet(
            (float[])Features.Clone(),
            (float[])Targets.Clone(),
            inputSize: 2,
            outputSize: 1);

        var network = new TrainableMlp(
            options.BuildSpecs(2, 1),
            options.Seed,
            options.Beta1,
            options.Beta2,
            options.Epsilon);

        Console.WriteLine("training a 2-8-1 tanh network on XOR");
        var result = new TrainingRun(options, log: null).Run(network, samples, samples);

        var worstError = 0f;
        Span<float> output = stackalloc float[1];

        for (var index = 0; index < samples.Count; index++)
        {
            var input = samples.Features(index);
            network.Forward(input);
            network.CopyOutput(output);
            worstError = MathF.Max(worstError, MathF.Abs(output[0] - Targets[index]));

            Console.WriteLine(string.Format(
                CultureInfo.InvariantCulture,
                "  {0} xor {1} = {2:F3} (expect {3:F3})",
                input[0] > 0.5f ? 1 : 0,
                input[1] > 0.5f ? 1 : 0,
                output[0],
                Targets[index]));
        }

        Console.WriteLine(string.Format(
            CultureInfo.InvariantCulture,
            "best validation loss {0:E3} at epoch {1}, worst output error {2:F3}",
            result.BestValidationLoss,
            result.BestEpoch,
            worstError));

        var converged = worstError < 0.1f;
        Console.WriteLine(converged ? "XOR probe passed." : "XOR probe FAILED.");
        return converged;
    }
}
