using System.Globalization;
using DroidSpeak.Neural;

namespace DroidSpeak.Trainer;

/// <summary>
/// The epoch loop: mini-batch Adam over a shuffled training split, validation scoring after every
/// epoch, and a best-checkpoint policy that exports a loadable <see cref="MlpModel"/>.
/// </summary>
internal sealed class TrainingRun
{
    private readonly TrainingOptions _options;
    private readonly Action<string> _log;

    public TrainingRun(TrainingOptions options, Action<string>? log = null)
    {
        _options = options;
        _log = log ?? DefaultLog;
    }

    /// <summary>Outcome of a run, enough to reconstruct the checkpoint's meaning.</summary>
    /// <param name="Model">The best model seen, ready to save or evaluate.</param>
    /// <param name="BestEpoch">One-based epoch that produced <see cref="Model"/>.</param>
    /// <param name="BestValidationLoss">Lowest validation mean squared error seen.</param>
    /// <param name="FinalTrainingLoss">Training error of the last epoch, before that rollback.</param>
    internal readonly record struct Result(
        MlpModel Model,
        int BestEpoch,
        float BestValidationLoss,
        float FinalTrainingLoss);

    /// <summary>
    /// Trains until the schedule is exhausted and returns the best model, not the last one.
    /// </summary>
    /// <param name="network">The freshly initialized network.</param>
    /// <param name="training">Training split; reshuffled every epoch.</param>
    /// <param name="validation">Held-out split, scored but never trained on.</param>
    /// <remarks>
    /// Keeping the best validation checkpoint matters here because a distilled prosody network
    /// over-fits the teacher's phrasing after a few dozen epochs: the last epoch is usually a worse
    /// voice than the best one, and the file shipped with the library is the best one.
    /// </remarks>
    public Result Run(TrainableMlp network, TrainingSet training, TrainingSet validation)
    {
        var random = new RandomSource(_options.Seed ^ 0x517B15u);
        var bestLoss = float.PositiveInfinity;
        var bestEpoch = 0;
        MlpModel? best = null;
        var finalTrainLoss = 0f;

        _log(string.Format(
            CultureInfo.InvariantCulture,
            "network {0} -> {1} -> {2} ({3} parameters), {4} samples, {5} validation, " +
            "{6} epochs, batch {7}, lr {8:G5}",
            network.InputSize,
            string.Join(",", _options.HiddenWidths),
            network.OutputSize,
            network.ParameterCount,
            training.Count,
            validation.Count,
            _options.Epochs,
            _options.BatchSize,
            _options.LearningRate));

        for (var epoch = 0; epoch < _options.Epochs; epoch++)
        {
            var learningRate = _options.LearningRateAt(epoch);
            training.Shuffle(ref random);
            var trainingLoss = TrainEpoch(network, training, learningRate);
            var validationLoss = Validate(network, validation);

            if (validationLoss < bestLoss)
            {
                bestLoss = validationLoss;
                bestEpoch = epoch + 1;
                best = network.Export();
            }

            finalTrainLoss = trainingLoss;

            if ((epoch + 1) % _options.LogEvery == 0 || epoch + 1 == _options.Epochs)
            {
                _log(string.Format(
                    CultureInfo.InvariantCulture,
                    "  epoch {0,3}/{1}  lr {2:G5}  train {3:E4}  val {4:E4}  best {5:E4} (e{6})",
                    epoch + 1,
                    _options.Epochs,
                    learningRate,
                    trainingLoss,
                    validationLoss,
                    bestLoss,
                    bestEpoch));
            }
        }

        // An empty checkpoint cannot happen once at least one epoch ran, but the compiler does not
        // know the loop body executed, and a run with zero validation improvement still trained.
        return new Result(best ?? network.Export(), bestEpoch, bestLoss, finalTrainLoss);
    }

    private float TrainEpoch(TrainableMlp network, TrainingSet training, float learningRate)
    {
        var total = 0f;
        var batch = 0;

        network.ZeroGradients();

        for (var position = 0; position < training.Count; position++)
        {
            total += network.TrainSample(
                training.FeaturesAt(position),
                training.TargetsAt(position));
            batch++;

            // The tail batch is applied with its own size, so a corpus that does not divide evenly
            // still gets a correctly averaged final update instead of an over-scaled one.
            if (batch == _options.BatchSize || position + 1 == training.Count)
            {
                network.ApplyAdam(learningRate, 1f / batch);
                network.ZeroGradients();
                batch = 0;
            }
        }

        return total / training.Count;
    }

    private static float Validate(TrainableMlp network, TrainingSet validation)
    {
        var total = 0f;

        for (var index = 0; index < validation.Count; index++)
        {
            total += network.Evaluate(validation.Features(index), validation.Targets(index));
        }

        return total / validation.Count;
    }

    private static void DefaultLog(string message) => Console.WriteLine(message);
}
