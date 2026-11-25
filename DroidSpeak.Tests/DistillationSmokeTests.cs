using DroidSpeak.DSP;
using DroidSpeak.Neural;
using DroidSpeak.NLP;
using DroidSpeak.Trainer;
using DroidSpeak.Trainer.Datasets;

namespace DroidSpeak.Tests;

/// <summary>
/// The end-to-end path a release depends on, deliberately too small to be a real training run: build
/// a corpus, distil it, save the checkpoint, load it back, and render through it. The point is that
/// every glue step works and that the loss the file carries is the loss the file produces.
/// </summary>
public class DistillationSmokeTests
{
    private const string Utterance = "Привет, мир! R2-D2 is... ready?";

    [Fact]
    public void BestCheckpointCutsHeldOutLossAndRendersWithoutAllocating()
    {
        var options = TrainingOptions.Default with { HiddenWidths = [24, 16], Epochs = 15 };
        var corpus = DistillationCorpus.Build(
            500, DroidVoiceProfile.R2, options.Seed, DroidSynthesizer.DefaultSampleRate);

        var splitRandom = new RandomSource(options.Seed ^ 0x5A17C0DEu);
        var validationRows = (int)MathF.Round(corpus.Count * options.ValidationFraction);
        corpus.Split(validationRows, ref splitRandom, out var training, out var validation);

        var network = new TrainableMlp(
            options.BuildSpecs(corpus.InputSize, corpus.OutputSize),
            options.Seed,
            options.Beta1,
            options.Beta2,
            options.Epsilon);

        // A sigmoid head starts near its 0.5 neutral point, so this is the do-nothing baseline the
        // trained checkpoint has to beat by a wide margin.
        var before = ValidationLoss(network.Export(), validation);
        var result = new TrainingRun(options, _ => { }).Run(network, training, validation);

        using var stream = new MemoryStream();
        result.Model.Save(stream);
        stream.Position = 0;
        var restored = MlpModel.Load(stream);

        // The saved file must score what the run reported (up to float summation order), and clearly
        // better than doing nothing.
        var after = ValidationLoss(restored, validation);
        Assert.True(
            MathF.Abs(after - result.BestValidationLoss) < 1e-4f,
            $"The loaded checkpoint scores {after:E4} but the run reported " +
            $"{result.BestValidationLoss:E4}.");
        Assert.True(after < before * 0.5f,
            $"Held-out MSE went from {before:E3} to {after:E3}; that is not a distillation.");

        AssertRendersAudibleAudioWithoutAllocating(restored);
    }

    private static void AssertRendersAudibleAudioWithoutAllocating(MlpModel model)
    {
        var synthesizer = new DroidSynthesizer(
            sampleRate: DroidSynthesizer.DefaultSampleRate,
            seed: 3,
            profile: DroidVoiceProfile.R2,
            predictor: new NeuralProsodyPredictor(model));

        var phrases = TextParser.Parse(Utterance);
        var audio = new float[synthesizer.GetSampleCount(phrases)];

        // Warm the renderer, array pool and JIT before measuring.
        synthesizer.Synthesize(phrases, audio);
        Array.Fill(audio, float.NaN);

        var before = GC.GetAllocatedBytesForCurrentThread();
        var written = synthesizer.Synthesize(phrases, audio);

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.Equal(audio.Length, written);

        var peak = 0f;
        foreach (var sample in audio)
        {
            // NaN from a partial write would fail here, so this also proves the whole buffer filled.
            Assert.InRange(sample, -1f, 1f);
            peak = MathF.Max(peak, MathF.Abs(sample));
        }

        Assert.True(peak > 0.05f, "The restored voice must produce audible signal.");
    }

    private static float ValidationLoss(MlpModel model, TrainingSet validation)
    {
        Span<float> output = stackalloc float[model.OutputSize];
        var sum = 0d;

        for (var row = 0; row < validation.Count; row++)
        {
            model.Predict(validation.Features(row), output);
            var targets = validation.Targets(row);

            for (var i = 0; i < output.Length; i++)
            {
                var error = output[i] - targets[i];
                sum += (double)error * error;
            }
        }

        return (float)(sum / ((double)validation.Count * model.OutputSize));
    }
}