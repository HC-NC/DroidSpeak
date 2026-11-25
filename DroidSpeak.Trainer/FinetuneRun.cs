using System.Globalization;
using DroidSpeak.Neural;

namespace DroidSpeak.Trainer;

/// <summary>
/// Fine-tunes an existing prosody network on an extracted acoustic dataset while replaying the
/// procedural distillation corpus, so the network adapts to real reference audio without forgetting
/// the rule-based teacher.
/// <para>
/// The two losses are tracked separately and reported separately. That matters because the mixed
/// objective hides exactly the failure this run exists to detect: a network can drive the acoustic
/// validation loss down while its replay loss climbs, which is catastrophic forgetting dressed up as
/// a successful fine-tune. Keeping the numbers apart makes the trade-off visible in the checkpoint
/// report instead of buried in one averaged figure.
/// </para>
/// <para>
/// Replay is interleaved, not appended. After each acoustic sample the loop advances a fractional
/// replay cursor by <c>replayRatio</c> and trains on the replay corpus whenever it crosses an
/// integer. The replay set is cycled, so a ratio above one sees it several times per epoch and a
/// ratio below one sees a subset, and the point is that the optimizer never goes many batches with
/// only one distribution in view.
/// </para>
/// </summary>
internal sealed class FinetuneRun
{
    private readonly TrainingOptions _options;
    private readonly float _replayRatio;
    private readonly Action<string>? _log;

    /// <summary>Creates a run.</summary>
    /// <param name="options">Epoch, batch, optimizer and schedule settings.</param>
    /// <param name="replayRatio">Replay samples trained per acoustic sample; <c>1</c> is balanced.</param>
    /// <param name="log">Progress sink; <c>null</c> silences the run.</param>
    public FinetuneRun(TrainingOptions options, float replayRatio, Action<string>? log = null)
    {
        if (replayRatio < 0f || !float.IsFinite(replayRatio))
        {
            throw new ArgumentOutOfRangeException(
                nameof(replayRatio),
                replayRatio,
                "The replay ratio must be a finite, non-negative number.");
        }

        _options = options;
        _replayRatio = replayRatio;
        _log = log;
    }

    /// <summary>Outcome of a fine-tune, with the acoustic and replay losses kept apart.</summary>
    /// <param name="Model">Best checkpoint by combined validation loss.</param>
    /// <param name="BestEpoch">One-based epoch that produced <paramref name="Model"/>; 0 means the seed model.</param>
    /// <param name="InitialAcousticValidation">Acoustic validation loss before any update.</param>
    /// <param name="InitialReplayValidation">Replay validation loss before any update.</param>
    /// <param name="BestAcousticValidation">Acoustic validation loss of the chosen checkpoint.</param>
    /// <param name="BestReplayValidation">Replay validation loss of the chosen checkpoint.</param>
    /// <param name="FinalAcousticTraining">Acoustic training loss of the last epoch.</param>
    /// <param name="FinalReplayTraining">Replay training loss of the last epoch.</param>
    /// <param name="Completed">Whether the run reached its final epoch without diverging.</param>
    /// <param name="FailureReason">Why the run stopped early, when it did.</param>
    internal readonly record struct Result(
        MlpModel Model,
        int BestEpoch,
        float InitialAcousticValidation,
        float InitialReplayValidation,
        float BestAcousticValidation,
        float BestReplayValidation,
        float FinalAcousticTraining,
        float FinalReplayTraining,
        bool Completed,
        string? FailureReason);

    /// <summary>
    /// Trains the network and returns the best checkpoint, not the last one.
    /// </summary>
    /// <param name="network">Network pre-loaded with the seed weights.</param>
    /// <param name="acousticTraining">Extracted training split.</param>
    /// <param name="acousticValidation">Extracted held-out split, scored but never trained on.</param>
    /// <param name="replayTraining">Procedural replay split.</param>
    /// <param name="replayValidation">Procedural replay held-out split.</param>
    /// <remarks>
    /// The checkpoint is chosen by the sum of the two validation losses, so a model that improves one
    /// side by wrecking the other cannot win. The two numbers are still reported apart so the caller
    /// can apply a stricter retention policy than the checkpoint selector does.
    /// </remarks>
    public Result Run(
        TrainableMlp network,
        TrainingSet acousticTraining,
        TrainingSet acousticValidation,
        TrainingSet replayTraining,
        TrainingSet replayValidation)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(acousticTraining);
        ArgumentNullException.ThrowIfNull(acousticValidation);
        ArgumentNullException.ThrowIfNull(replayTraining);
        ArgumentNullException.ThrowIfNull(replayValidation);

        var random = new RandomSource(_options.Seed ^ 0x4FB1A7E5u);
        var finalAcousticTraining = 0f;
        var finalReplayTraining = 0f;

        var initialAcoustic = Validate(network, acousticValidation);
        var initialReplay = Validate(network, replayValidation);

        if (!float.IsFinite(initialAcoustic) || !float.IsFinite(initialReplay))
        {
            return new Result(
                network.Export(),
                0,
                initialAcoustic,
                initialReplay,
                initialAcoustic,
                initialReplay,
                0f,
                0f,
                false,
                "The seed network already produces a non-finite validation loss.");
        }

        // The seed model is the first candidate checkpoint. A fine-tune that never beats it (or that
        // diverges early) then hands the caller the network it was given rather than a worse one, and
        // Epoch 0 in the report unambiguously means "no update won".
        var bestLoss = initialAcoustic + initialReplay;
        var bestEpoch = 0;
        var bestAcoustic = initialAcoustic;
        var bestReplay = initialReplay;
        MlpModel? bestModel = network.Export();

        Log(string.Format(
            CultureInfo.InvariantCulture,
            "fine-tune {0} -> {1}, {2} acoustic / {3} replay training, {4} / {5} validation, " +
            "{6} epochs, batch {7}, replay ratio {8:G4}, lr {9:G5}",
            network.InputSize,
            network.OutputSize,
            acousticTraining.Count,
            replayTraining.Count,
            acousticValidation.Count,
            replayValidation.Count,
            _options.Epochs,
            _options.BatchSize,
            _replayRatio,
            _options.LearningRate));

        var completed = true;
        string? failure = null;

        for (var epoch = 0; epoch < _options.Epochs; epoch++)
        {
            var learningRate = _options.LearningRateAt(epoch);
            acousticTraining.Shuffle(ref random);
            replayTraining.Shuffle(ref random);

            var (acousticTrain, replayTrain) = TrainEpoch(
                network,
                acousticTraining,
                replayTraining,
                learningRate,
                _options.BatchSize,
                _replayRatio);

            var acousticValidate = Validate(network, acousticValidation);
            var replayValidate = Validate(network, replayValidation);

            if (!float.IsFinite(acousticTrain) ||
                !float.IsFinite(replayTrain) ||
                !float.IsFinite(acousticValidate) ||
                !float.IsFinite(replayValidate))
            {
                completed = false;
                failure = $"Divergence at epoch {epoch + 1}: a loss became non-finite.";
                break;
            }

            finalAcousticTraining = acousticTrain;
            finalReplayTraining = replayTrain;

            var combined = acousticValidate + replayValidate;
            if (combined < bestLoss)
            {
                bestLoss = combined;
                bestEpoch = epoch + 1;
                bestAcoustic = acousticValidate;
                bestReplay = replayValidate;
                bestModel = network.Export();
            }

            if ((epoch + 1) % _options.LogEvery == 0 || epoch + 1 == _options.Epochs)
            {
                Log(string.Format(
                    CultureInfo.InvariantCulture,
                    "  epoch {0,3}/{1}  lr {2:G5}  acoustic {3:E4}/{4:E4}  replay {5:E4}/{6:E4}  " +
                    "best {7:E4} (e{8})",
                    epoch + 1,
                    _options.Epochs,
                    learningRate,
                    acousticTrain,
                    acousticValidate,
                    replayTrain,
                    replayValidate,
                    bestLoss,
                    bestEpoch));
            }
        }

        if (bestModel is null)
        {
            // No epoch improved on the seed, or the run diverged before the first checkpoint. The
            // seed weights are still a valid model, so hand them back rather than nothing.
            bestModel = network.Export();
            bestAcoustic = initialAcoustic;
            bestReplay = initialReplay;
        }

        return new Result(
            bestModel,
            bestEpoch,
            initialAcoustic,
            initialReplay,
            bestAcoustic,
            bestReplay,
            finalAcousticTraining,
            finalReplayTraining,
            completed,
            failure);
    }

    /// <summary>
    /// Runs one epoch of mixed acoustic and replayed training, keeping the two loss sums apart.
    /// </summary>
    /// <param name="network">Network to update.</param>
    /// <param name="acoustic">Shuffled acoustic split.</param>
    /// <param name="replay">Shuffled replay split; may be empty.</param>
    /// <param name="learningRate">Step size for this epoch.</param>
    /// <param name="batchSize">Acoustic samples per Adam update.</param>
    /// <param name="replayRatio">Replay samples per acoustic sample.</param>
    /// <returns>The mean acoustic and replay training errors over the epoch.</returns>
    /// <remarks>
    /// Allocates nothing once the JIT has warmed the method: the shuffle is in place, the per-sample
    /// work reuses the network's buffers, and the only arithmetic that allocates anywhere is the
    /// Adam bias correction's <see cref="MathF.Pow(float, float)"/>, which does not.
    /// </remarks>
    internal static (float Acoustic, float Replay) TrainEpoch(
        TrainableMlp network,
        TrainingSet acoustic,
        TrainingSet replay,
        float learningRate,
        int batchSize,
        float replayRatio)
    {
        var acousticLoss = 0f;
        var replayLoss = 0f;
        var acousticBatch = 0;
        var replayBatch = 0;
        var replayPosition = 0;
        var replayCursor = 0f;

        network.ZeroGradients();

        for (var position = 0; position < acoustic.Count; position++)
        {
            acousticLoss += network.TrainSample(
                acoustic.FeaturesAt(position),
                acoustic.TargetsAt(position));
            acousticBatch++;

            replayCursor += replayRatio;
            while (replayCursor >= 1f && replay.Count > 0)
            {
                replayCursor -= 1f;
                var replayIndex = replayPosition % replay.Count;
                replayLoss += network.TrainSample(
                    replay.FeaturesAt(replayIndex),
                    replay.TargetsAt(replayIndex));
                replayPosition++;
                replayBatch++;
            }

            if (acousticBatch == batchSize || position + 1 == acoustic.Count)
            {
                // The gradient is the mean over every sample summed into it, acoustic and replay, so
                // the update does not depend on the ratio of the two sources inside the batch.
                var total = acousticBatch + replayBatch;
                network.ApplyAdam(learningRate, 1f / total);
                network.ZeroGradients();
                acousticBatch = 0;
                replayBatch = 0;
            }
        }

        var acousticMean = acoustic.Count > 0 ? acousticLoss / acoustic.Count : 0f;
        var replayMean = replayPosition > 0 ? replayLoss / replayPosition : 0f;
        return (acousticMean, replayMean);
    }

    /// <summary>Mean squared error of a network over a split, allocation free.</summary>
    internal static float Validate(TrainableMlp network, TrainingSet set)
    {
        if (set.Count == 0)
        {
            return 0f;
        }

        var total = 0f;
        for (var index = 0; index < set.Count; index++)
        {
            total += network.Evaluate(set.Features(index), set.Targets(index));
        }

        return total / set.Count;
    }

    private void Log(string message)
    {
        if (_log is not null)
        {
            _log(message);
        }
    }
}