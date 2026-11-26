using DroidSpeak.DSP;
using DroidSpeak.Neural;
using DroidSpeak.Prosody;

namespace DroidSpeak.Tests;

// Coverage for the v1.0.0 convenience brain-loading APIs: MlpModel.Load(path) and the
// DroidSynthesizer.FromBrainFile / FromBrainBytes factories. The classic rule-based predictor stays
// the default; a neural model is only ever opt-in, so these tests also pin that a factory-built
// synthesizer renders exactly what an explicitly injected NeuralProsodyPredictor renders.
public class BrainLoadingTests
{
    private const string SampleText = "Beep boop! Systems online.";

    private static string CreateTempDirectory()
    {
        var root = Path.Combine(
            Path.GetTempPath(),
            "droidspeak-brain-tests",
            Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        return root;
    }

    private static MlpModel CreateValidBrain(uint seed = MlpModel.DefaultInitSeed) => MlpModel.CreateInitialized(
    [
        new LayerSpec(SyllableFeatureEncoder.InputSize, 8, ActivationType.Tanh),
        new LayerSpec(8, NeuralProsodyPredictor.OutputSize, ActivationType.Sigmoid),
    ], seed);

    private static byte[] Serialize(MlpModel model)
    {
        using var stream = new MemoryStream();
        model.Save(stream);
        return stream.ToArray();
    }

    private static string SaveBrain(MlpModel model)
    {
        var path = Path.Combine(CreateTempDirectory(), "voice.dsbrain");
        using (var stream = File.Create(path))
        {
            model.Save(stream);
        }

        return path;
    }

    private static void AssertPredictionsMatch(MlpModel expected, MlpModel actual)
    {
        Assert.Equal(expected.InputSize, actual.InputSize);
        Assert.Equal(expected.OutputSize, actual.OutputSize);

        var input = new float[expected.InputSize];
        for (var i = 0; i < input.Length; i++)
        {
            input[i] = (i % 7) / 7f;
        }

        var a = new float[expected.OutputSize];
        var b = new float[actual.OutputSize];
        expected.Predict(input, a);
        actual.Predict(input, b);
        Assert.Equal(a, b);
    }

    [Fact]
    public void LoadFromPath_MatchesSpanLoad()
    {
        var model = CreateValidBrain();
        var path = SaveBrain(model);

        var fromFile = MlpModel.Load(path);
        var fromBytes = MlpModel.Load(Serialize(model));

        AssertPredictionsMatch(model, fromFile);
        AssertPredictionsMatch(model, fromBytes);
    }

    [Fact]
    public void LoadFromPath_MissingFile_Throws()
    {
        var path = Path.Combine(CreateTempDirectory(), "absent.dsbrain");

        Assert.Throws<FileNotFoundException>(() => MlpModel.Load(path));
    }

    [Fact]
    public void LoadFromPath_MalformedFile_Throws()
    {
        var path = Path.Combine(CreateTempDirectory(), "broken.dsbrain");
        File.WriteAllBytes(path, [0x44, 0x53, 0x42, 0x52, 0x09]);

        Assert.Throws<InvalidDataException>(() => MlpModel.Load(path));
    }

    [Fact]
    public void LoadFromPath_NullPath_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => MlpModel.Load((string)null!));
    }

    [Fact]
    public void FromBrainFile_MatchesFromBrainBytes()
    {
        var model = CreateValidBrain();
        var path = SaveBrain(model);

        var fromFile = DroidSynthesizer.FromBrainFile(path, sampleRate: 44100, seed: 7);
        var fromBytes = DroidSynthesizer.FromBrainBytes(Serialize(model), sampleRate: 44100, seed: 7);

        Assert.Equal(fromFile.Synthesize(SampleText), fromBytes.Synthesize(SampleText));
    }

    [Fact]
    public void FromBrain_MatchesExplicitNeuralPredictor()
    {
        var model = CreateValidBrain();
        var bytes = Serialize(model);

        var expected = new DroidSynthesizer(
            sampleRate: DroidSynthesizer.DefaultSampleRate,
            seed: 3,
            profile: DroidVoiceProfile.Grumpy,
            predictor: new NeuralProsodyPredictor(model));

        var factory = DroidSynthesizer.FromBrainBytes(
            bytes,
            sampleRate: DroidSynthesizer.DefaultSampleRate,
            seed: 3,
            profile: DroidVoiceProfile.Grumpy);

        Assert.Equal(expected.Synthesize(SampleText), factory.Synthesize(SampleText));
    }

    [Fact]
    public void FromBrain_ProducesAudioDifferentFromTheRuleEngine()
    {
        var neural = DroidSynthesizer.FromBrainBytes(
            Serialize(CreateValidBrain()));

        var classic = new DroidSynthesizer();

        // A deterministically initialized network is not the rule engine, so the two must diverge;
        // this also guards against the factory silently falling back to RuleBasedProsodyPredictor.
        Assert.NotEqual(classic.Synthesize(SampleText), neural.Synthesize(SampleText));
    }

    [Fact]
    public void FromBrainFile_MissingFile_Throws()
    {
        var path = Path.Combine(CreateTempDirectory(), "absent.dsbrain");

        Assert.Throws<FileNotFoundException>(() => DroidSynthesizer.FromBrainFile(path));
    }

    [Fact]
    public void FromBrainBytes_InvalidData_Throws()
    {
        Assert.Throws<InvalidDataException>(
            () => DroidSynthesizer.FromBrainBytes(new byte[] { 1, 2, 3, 4 }));
    }

    [Fact]
    public void FromBrainBytes_WrongShape_Throws()
    {
        var wrongShape = MlpModel.CreateInitialized(
        [
            new LayerSpec(4, 5, ActivationType.Tanh),
            new LayerSpec(5, 2, ActivationType.Sigmoid),
        ]);

        Assert.Throws<ArgumentException>(
            () => DroidSynthesizer.FromBrainBytes(Serialize(wrongShape)));
    }

    [Fact]
    public void Constructor_DefaultsToRuleBasedProsody()
    {
        var synthesizer = new DroidSynthesizer(seed: 11);
        var classic = new DroidSynthesizer(
            predictor: RuleBasedProsodyPredictor.Shared,
            seed: 11);

        Assert.Equal(classic.Synthesize(SampleText), synthesizer.Synthesize(SampleText));
    }
}
