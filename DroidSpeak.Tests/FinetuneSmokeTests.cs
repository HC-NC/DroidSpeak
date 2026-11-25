using DroidSpeak;
using DroidSpeak.DSP;
using DroidSpeak.Neural;
using DroidSpeak.NLP;
using DroidSpeak.Trainer;
using DroidSpeak.Trainer.Datasets;

namespace DroidSpeak.Tests;

/// <summary>
/// The stage-6 fine-tune path, deliberately small enough to run in a unit test: mix an extracted
/// acoustic split with a procedural replay split, train, and check convergence, retention and the
/// allocation-free hot path. The numbers are not a real training run; the assertions are about the
/// machinery and the direction of the losses.
/// </summary>
public class FinetuneSmokeTests
{
    private const string Utterance = "Привет, мир! R-2 is... ready?";
    private const int Rate = AcousticFixtures.Rate;

    private static TrainingOptions TestOptions => TrainingOptions.Default with
    {
        HiddenWidths = [32, 24],
        Epochs = 60,
        BatchSize = 16,
        LearningRate = 0.01f,
        MinLearningRate = 0.0005f,
        ValidationFraction = 0.25f,
        LogEvery = 60,
    };

    [Fact]
    public void FineTuningConvergesOnAcousticsAndRetainsReplay()
    {
        var profile = DroidVoiceProfile.R2;
        var options = TestOptions;
        var dataset = AcousticFixtures.BuildDataset(8, profile, Rate, options.Seed);

        var splitRandom = new RandomSource(options.Seed ^ 0x5A17C0DEu);
        dataset.SplitByClip(
            profile,
            options.ValidationFraction,
            ref splitRandom,
            out var acousticTraining,
            out var acousticValidation);

        var replayTraining = DistillationCorpus.Build(
            200, profile, options.Seed ^ 0x4EB1E5u, Rate);
        var replayValidation = DistillationCorpus.Build(
            64, profile, options.Seed ^ 0x9E3779B9u, Rate);

        var network = CreateSeedNetwork(options, seedSeed: options.Seed);
        var result = new FinetuneRun(options, replayRatio: 1f).Run(
            network,
            acousticTraining,
            acousticValidation,
            replayTraining,
            replayValidation);

        Assert.True(result.Completed, result.FailureReason);
        Assert.True(result.BestEpoch >= 1);
        Assert.True(
            result.BestAcousticValidation < result.InitialAcousticValidation,
            $"Acoustic validation went from {result.InitialAcousticValidation:E4} to " +
            $"{result.BestAcousticValidation:E4}; fine-tuning did not learn the clips.");

        // The replay side is allowed to move, but not to collapse: a network that halves its acoustic
        // loss by forgetting the teacher is exactly the failure this stage is meant to prevent.
        Assert.True(float.IsFinite(result.BestReplayValidation));
        Assert.True(
            result.BestReplayValidation < result.InitialReplayValidation * 2f,
            $"Replay validation went from {result.InitialReplayValidation:E4} to " +
            $"{result.BestReplayValidation:E4}; that is catastrophic forgetting.");
    }

    [Fact]
    public void FineTunedCheckpointLoadsAndRendersWithoutAllocating()
    {
        var profile = DroidVoiceProfile.R2;
        var options = TestOptions with { Epochs = 30 };
        var dataset = AcousticFixtures.BuildDataset(6, profile, Rate, options.Seed);

        var splitRandom = new RandomSource(options.Seed ^ 0x5A17C0DEu);
        dataset.SplitByClip(
            profile,
            options.ValidationFraction,
            ref splitRandom,
            out var acousticTraining,
            out var acousticValidation);

        var replayTraining = DistillationCorpus.Build(160, profile, 11u, Rate);
        var replayValidation = DistillationCorpus.Build(64, profile, 12u, Rate);

        var network = CreateSeedNetwork(options, options.Seed);
        var result = new FinetuneRun(options, replayRatio: 1f).Run(
            network,
            acousticTraining,
            acousticValidation,
            replayTraining,
            replayValidation);

        using var stream = new MemoryStream();
        result.Model.Save(stream);
        stream.Position = 0;
        var restored = MlpModel.Load(stream);

        Assert.Equal(AcousticDataset.InputSize, restored.InputSize);
        Assert.Equal(AcousticDataset.TargetSize, restored.OutputSize);

        var synthesizer = new DroidSynthesizer(
            sampleRate: DroidSynthesizer.DefaultSampleRate,
            seed: 3,
            profile: profile,
            predictor: new NeuralProsodyPredictor(restored));

        var phrases = TextParser.Parse(Utterance);
        var audio = new float[synthesizer.GetSampleCount(phrases)];

        synthesizer.Synthesize(phrases, audio);
        Array.Fill(audio, float.NaN);

        var before = GC.GetAllocatedBytesForCurrentThread();
        var written = synthesizer.Synthesize(phrases, audio);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(0, allocated);
        Assert.Equal(audio.Length, written);

        var peak = 0f;
        foreach (var sample in audio)
        {
            Assert.InRange(sample, -1f, 1f);
            peak = MathF.Max(peak, MathF.Abs(sample));
        }

        Assert.True(peak > 0.05f, "The fine-tuned voice must produce audible signal.");
    }

    [Fact]
    public void TrainEpochAndValidationAllocateNothingAfterWarmup()
    {
        var profile = DroidVoiceProfile.R2;
        var options = TestOptions with { Epochs = 1 };
        var dataset = AcousticFixtures.BuildDataset(4, profile, Rate, options.Seed);
        var acoustic = dataset.Encode(profile);
        var replay = DistillationCorpus.Build(64, profile, 5u, Rate);
        var network = CreateSeedNetwork(options, options.Seed);

        // Warm the JIT and the network's buffers with one full epoch and one validation pass.
        FinetuneRun.TrainEpoch(network, acoustic, replay, 0.001f, options.BatchSize, 1f);
        var warm = FinetuneRun.Validate(network, acoustic);
        Assert.True(float.IsFinite(warm));

        var beforeEpoch = GC.GetAllocatedBytesForCurrentThread();
        var (acousticLoss, replayLoss) = FinetuneRun.TrainEpoch(
            network, acoustic, replay, 0.001f, options.BatchSize, 1f);
        var epochBytes = GC.GetAllocatedBytesForCurrentThread() - beforeEpoch;

        Assert.Equal(0, epochBytes);
        Assert.True(float.IsFinite(acousticLoss));
        Assert.True(float.IsFinite(replayLoss));

        var beforeValidation = GC.GetAllocatedBytesForCurrentThread();
        var validation = FinetuneRun.Validate(network, acoustic);
        var validationBytes = GC.GetAllocatedBytesForCurrentThread() - beforeValidation;

        Assert.Equal(0, validationBytes);
        Assert.True(float.IsFinite(validation));
    }

    [Fact]
    public void EmptyReplayRatioOnlyTrainsAcoustics()
    {
        var profile = DroidVoiceProfile.R2;
        var options = TestOptions with { Epochs = 1 };
        var dataset = AcousticFixtures.BuildDataset(4, profile, Rate, options.Seed);
        var acoustic = dataset.Encode(profile);
        var replay = DistillationCorpus.Build(32, profile, 5u, Rate);
        var network = CreateSeedNetwork(options, options.Seed);

        var (acousticLoss, replayLoss) = FinetuneRun.TrainEpoch(
            network, acoustic, replay, 0.001f, options.BatchSize, replayRatio: 0f);

        Assert.True(acousticLoss > 0f);
        Assert.Equal(0f, replayLoss);
    }

    private static TrainableMlp CreateSeedNetwork(TrainingOptions options, uint seedSeed)
    {
        var specs = options.BuildSpecs(AcousticDataset.InputSize, AcousticDataset.TargetSize);
        var template = new TrainableMlp(
            specs, seedSeed, options.Beta1, options.Beta2, options.Epsilon);
        var seedModel = template.Export();

        var network = new TrainableMlp(
            specs, options.Seed, options.Beta1, options.Beta2, options.Epsilon);
        network.Import(seedModel);
        return network;
    }
}