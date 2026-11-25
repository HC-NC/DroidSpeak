using DroidSpeak.Demo;
using DroidSpeak.Neural;

namespace DroidSpeak.Tests;

// These tests cover the demo's internal helpers, which are visible to this assembly through
// [assembly: InternalsVisibleTo("DroidSpeak.Tests")] in DroidSpeak.Demo/Properties/AssemblyInfo.cs,
// without widening the demo's public surface.
public class DemoBrainTests
{
    private static string CreateTempDirectory()
    {
        var root = Path.Combine(
            Path.GetTempPath(),
            "droidspeak-demo-tests",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        return root;
    }

    private static string SaveBrain(string fileName, MlpModel model)
    {
        var path = Path.Combine(CreateTempDirectory(), fileName);
        using (var stream = File.Create(path))
        {
            model.Save(stream);
        }

        return path;
    }

    private static MlpModel CreateValidBrain() => MlpModel.CreateInitialized(
    [
        new LayerSpec(SyllableFeatureEncoder.InputSize, 8, ActivationType.Tanh),
        new LayerSpec(8, NeuralProsodyPredictor.OutputSize, ActivationType.Sigmoid),
    ]);

    [Fact]
    public void ParseArguments_AcceptsTheBrainFlag()
    {
        var options = DemoCommandLine.ParseArguments(["--brain", "x.dsbrain"]);

        Assert.Null(options.Error);
        Assert.Equal("x.dsbrain", options.Brain);
        Assert.Null(options.Text);
        Assert.Null(options.Music);
    }

    [Fact]
    public void ParseArguments_RejectsATrailingBrainFlag()
    {
        var options = DemoCommandLine.ParseArguments(["--brain"]);

        Assert.NotNull(options.Error);
        Assert.Contains("missing", options.Error, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("--brain", options.Error, StringComparison.Ordinal);
        Assert.Null(options.Brain);
    }

    [Fact]
    public void ParseArguments_KeepsVoiceAndBrainTogether()
    {
        var options = DemoCommandLine.ParseArguments(["--voice", "grumpy", "--brain", "x.dsbrain"]);

        Assert.Null(options.Error);
        Assert.Equal("grumpy", options.Voice);
        Assert.Equal("x.dsbrain", options.Brain);
    }

    [Fact]
    public void ParseArguments_LeavesBrainNullWithoutTheFlag()
    {
        var options = DemoCommandLine.ParseArguments(["hello", "world"]);

        Assert.Null(options.Error);
        Assert.Null(options.Brain);
        Assert.Equal("hello world", options.Text);
    }

    [Fact]
    public void ParseArguments_RejectsUnknownOptions()
    {
        var options = DemoCommandLine.ParseArguments(["--bogus"]);

        Assert.NotNull(options.Error);
        Assert.Contains("unknown option", options.Error, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("--bogus", options.Error, StringComparison.Ordinal);
    }

    [Fact]
    public void ParseArguments_RejectsTextCombinedWithMusic()
    {
        var options = DemoCommandLine.ParseArguments(["hello", "-m", "song"]);

        Assert.NotNull(options.Error);
        Assert.Contains("--music", options.Error, StringComparison.Ordinal);
    }

    [Fact]
    public void ParseArguments_PreservesTheExistingOptionBehaviour()
    {
        var options = DemoCommandLine.ParseArguments(
            ["-v", "tiny", "-o", "out.wav", "--solo", "--stream", "--no-play", "hi"]);

        Assert.Null(options.Error);
        Assert.Equal("tiny", options.Voice);
        Assert.Equal("out.wav", options.OutputPath);
        Assert.True(options.Solo);
        Assert.True(options.Stream);
        Assert.False(options.Play);
        Assert.Equal("hi", options.Text);
    }

    [Fact]
    public void TryLoadBrain_ReportsAMissingPath()
    {
        var missing = Path.Combine(CreateTempDirectory(), "nope.dsbrain");

        var (predictor, error) = DemoCommandLine.TryLoadBrain(missing);

        Assert.Null(predictor);
        Assert.NotNull(error);
        Assert.Contains("not found", error, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TryLoadBrain_ReportsACorruptFile()
    {
        var path = Path.Combine(CreateTempDirectory(), "garbage.dsbrain");
        File.WriteAllBytes(path, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);

        var (predictor, error) = DemoCommandLine.TryLoadBrain(path);

        Assert.Null(predictor);
        Assert.NotNull(error);
        Assert.Contains("cannot load", error, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TryLoadBrain_LoadsAValidModel()
    {
        var path = SaveBrain("valid.dsbrain", CreateValidBrain());

        var (predictor, error) = DemoCommandLine.TryLoadBrain(path);

        Assert.Null(error);
        Assert.NotNull(predictor);
        Assert.Equal(SyllableFeatureEncoder.InputSize, predictor!.Model.InputSize);
        Assert.Equal(NeuralProsodyPredictor.OutputSize, predictor.Model.OutputSize);

        // The predictor constructor is the 60->17 shape gate; a successful load means it accepted
        // the model.
        Assert.IsType<NeuralProsodyPredictor>(predictor);
    }

    [Fact]
    public void TryLoadBrain_ReturnsAClearErrorForAWrongShapedModel()
    {
        var model = MlpModel.CreateInitialized(
        [
            new LayerSpec(SyllableFeatureEncoder.InputSize, 8, ActivationType.Tanh),
            new LayerSpec(8, 16, ActivationType.Sigmoid),
        ]);
        var path = SaveBrain("wrong-shape.dsbrain", model);

        var (predictor, error) = DemoCommandLine.TryLoadBrain(path);

        Assert.Null(predictor);
        Assert.NotNull(error);
        Assert.Contains("outputs", error, StringComparison.Ordinal);
    }

    [Fact]
    public void TryLoadBrain_RejectsAnEmptyPath()
    {
        var (predictor, error) = DemoCommandLine.TryLoadBrain("   ");

        Assert.Null(predictor);
        Assert.NotNull(error);
        Assert.Contains("no .dsbrain model path", error, StringComparison.OrdinalIgnoreCase);
    }
}
