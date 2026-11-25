using DroidSpeak;
using DroidSpeak.DSP;
using DroidSpeak.Neural;
using DroidSpeak.Trainer;
using DroidSpeak.Trainer.Datasets;

// Offline trainer for DroidSpeak prosody networks. It exists so a voice can be distilled from the
// rule-based engine into a small .dsbrain file on a developer machine, without shipping any training
// code or any dependency to the synthesizer.
if (args.Length == 0)
{
    PrintUsage();
    return 0;
}

try
{
    switch (args[0].ToLowerInvariant())
    {
        case "selftest":
            // Proves the backpropagation and Adam path works before any corpus is involved.
            return XorProbe.Run() ? 0 : 1;

        case "distill":
            return Distill(args[1..]);

        case "eval":
            return Eval(args[1..]);

        case "-h":
        case "--help":
        case "help":
            PrintUsage();
            return 0;

        default:
            Console.Error.WriteLine($"error: unknown command '{args[0]}'.");
            Console.Error.WriteLine();
            PrintUsage();
            return 1;
    }
}
catch (Exception exception)
    when (exception is ArgumentException or IOException or InvalidDataException
        or InvalidOperationException)
{
    // Bad options and unreadable checkpoints are user mistakes, not crashes: report and exit 2 so a
    // script can tell them apart from a failed model quality check (exit 1).
    Console.Error.WriteLine($"error: {exception.Message}");
    return 2;
}

static int Distill(string[] arguments)
{
    var defaults = TrainingOptions.Default;
    var options = CliOptions.Parse(
        arguments,
        "samples",
        "epochs",
        "batch-size",
        "learning-rate",
        "lr-min",
        "arch",
        "activation",
        "output-activation",
        "seed",
        "validation",
        "schedule",
        "step-size",
        "decay",
        "beta1",
        "beta2",
        "epsilon",
        "profile",
        "sample-rate",
        "log-every",
        "out");

    var samples = options.Int("samples", 4000);
    var sampleRate = options.Int("sample-rate", DroidSynthesizer.DefaultSampleRate);
    var profileName = options.Value("profile") ?? "r2";
    var profile = options.Profile(profileName);

    var training = defaults with
    {
        HiddenWidths = options.IntList("arch", defaults.HiddenWidths),
        HiddenActivation = ParseActivation(
            options.Value("activation"), defaults.HiddenActivation, "activation"),
        OutputActivation = ParseActivation(
            options.Value("output-activation"), defaults.OutputActivation, "output-activation"),
        Epochs = options.Int("epochs", defaults.Epochs),
        BatchSize = options.Int("batch-size", defaults.BatchSize),
        LearningRate = options.Single("learning-rate", defaults.LearningRate),
        MinLearningRate = options.Single("lr-min", defaults.MinLearningRate),
        Schedule = ParseSchedule(options.Value("schedule"), defaults.Schedule),
        StepEpochs = options.Int("step-size", defaults.StepEpochs),
        StepFactor = options.Single("decay", defaults.StepFactor),
        Beta1 = options.Single("beta1", defaults.Beta1),
        Beta2 = options.Single("beta2", defaults.Beta2),
        Epsilon = options.Single("epsilon", defaults.Epsilon),
        Seed = options.UInt("seed", defaults.Seed),
        ValidationFraction = options.Single("validation", defaults.ValidationFraction),
        LogEvery = options.Int("log-every", defaults.LogEvery),
    };

    if (samples < 2)
    {
        throw new ArgumentException("--samples must be at least 2.");
    }

    if (sampleRate is < 8000 or > 192000)
    {
        throw new ArgumentException("--sample-rate must be between 8000 and 192000 Hz.");
    }

    if (training.Validate() is { } problem)
    {
        throw new ArgumentException(problem);
    }

    var output = options.Value("out") ?? "prosody.dsbrain";

    Console.WriteLine(
        $"building {samples} distillation samples from profile '{profileName}' " +
        $"(seed {training.Seed}, {sampleRate} Hz)...");

    var corpus = DistillationCorpus.BuildDetailed(samples, profile, training.Seed, sampleRate);

    var validationCount = Math.Clamp(
        (int)MathF.Round(corpus.Set.Count * training.ValidationFraction),
        1,
        corpus.Set.Count - 1);
    var splitRandom = new RandomSource(training.Seed ^ 0x5A17C0DEu);
    corpus.Set.Split(
        validationCount,
        ref splitRandom,
        out var trainingSet,
        out var validationSet);

    var specs = training.BuildSpecs(
        SyllableFeatureEncoder.InputSize, SyllableTargetEncoder.TargetSize);
    var network = new TrainableMlp(
        specs, training.Seed, training.Beta1, training.Beta2, training.Epsilon);

    var result = new TrainingRun(training).Run(network, trainingSet, validationSet);

    using (var stream = File.Create(output))
    {
        result.Model.Save(stream);
    }

    Console.WriteLine(
        $"saved best checkpoint (epoch {result.BestEpoch}, validation MSE " +
        $"{result.BestValidationLoss:E4}) to {output}");
    return 0;
}

static int Eval(string[] arguments)
{
    var options = CliOptions.Parse(
        arguments, "brain", "samples", "seed", "profile", "sample-rate");

    var brain = options.Value("brain")
        ?? throw new ArgumentException("eval requires --brain <path.dsbrain>.");
    var samples = options.Int("samples", 1500);
    if (samples < 1)
    {
        throw new ArgumentException("--samples must be at least 1.");
    }

    var sampleRate = options.Int("sample-rate", DroidSynthesizer.DefaultSampleRate);
    if (sampleRate is < 8000 or > 192000)
    {
        throw new ArgumentException("--sample-rate must be between 8000 and 192000 Hz.");
    }

    // A validator should pass a seed the training run never used, so the report measures held-out
    // syllables rather than remembered ones.
    var seed = options.UInt("seed", 0xE7A1_2026u);
    var profile = options.Profile(options.Value("profile") ?? "r2");

    MlpModel model;
    using (var stream = File.OpenRead(brain))
    {
        model = MlpModel.Load(stream);
    }

    Console.WriteLine(
        $"loaded {brain}: {model.InputSize} -> {model.OutputSize}, " +
        $"{model.ParameterCount} parameters");

    return Evaluation.Report(model, profile, sampleRate, seed, samples) ? 0 : 1;
}

static ActivationType ParseActivation(
    string? name,
    ActivationType fallback,
    string option)
{
    if (string.IsNullOrWhiteSpace(name))
    {
        return fallback;
    }

    return name.Trim().ToLowerInvariant() switch
    {
        "sigmoid" => ActivationType.Sigmoid,
        "tanh" => ActivationType.Tanh,
        "relu" => ActivationType.Relu,
        "leaky-relu" or "leakyrelu" or "leaky_relu" => ActivationType.LeakyRelu,
        "mod-relu" or "modrelu" or "mod_relu" => ActivationType.ModRelu,
        "linear" => ActivationType.Linear,
        _ => throw new ArgumentException(
            $"--{option} must be sigmoid, tanh, relu, leaky-relu, mod-relu or linear."),
    };
}

static LearningRateSchedule ParseSchedule(
    string? name,
    LearningRateSchedule fallback)
{
    if (string.IsNullOrWhiteSpace(name))
    {
        return fallback;
    }

    return name.Trim().ToLowerInvariant() switch
    {
        "constant" or "none" => LearningRateSchedule.Constant,
        "cosine" => LearningRateSchedule.Cosine,
        "step" => LearningRateSchedule.Step,
        _ => throw new ArgumentException("--schedule must be constant, cosine or step."),
    };
}

static void PrintUsage()
{
    Console.WriteLine("""
        DroidSpeak.Trainer - offline prosody network training

        usage: dotnet run --project DroidSpeak.Trainer -- <command> [options]

        commands:
          selftest   train a tiny XOR network to prove backprop and Adam work

          distill    build a procedural corpus, distil the rule-based teacher into an MLP with
                     Adam, and save the best validation checkpoint as a .dsbrain file
                     options:
                       --samples N             corpus syllables, default 4000
                       --epochs N              full passes over the training split, default 60
                       --batch-size N          samples per Adam update, default 32
                       --learning-rate F       base step size, default 0.008
                       --lr-min F              cosine floor, default 0.0002
                       --arch A,B,...          hidden widths, default 64,48
                       --activation NAME       hidden non-linearity: sigmoid, tanh, relu,
                                               leaky-relu, mod-relu, linear; default tanh
                       --output-activation N   output head non-linearity, default sigmoid
                       --seed N                corpus, init and shuffle seed, default 20260107
                       --validation F          held-out fraction, default 0.1
                       --schedule NAME         constant, cosine or step; default cosine
                       --step-size N           epochs per step decay, default 10
                       --decay F               step-decay multiplier, default 0.5
                       --beta1 F               Adam first-moment decay, default 0.9
                       --beta2 F               Adam second-moment decay, default 0.999
                       --epsilon F             Adam denominator floor, default 1e-8
                       --profile NAME          r2, grumpy, bass, tiny or a profile JSON path;
                                               default r2
                       --sample-rate HZ        corpus and render rate, default 44100
                       --log-every N           epochs between progress lines, default 1
                       --out PATH              checkpoint path, default prosody.dsbrain

          eval       score a .dsbrain file against the rule-based teacher
                     options:
                       --brain PATH            checkpoint to load (required)
                       --samples N             held-out syllables to score, default 1500
                       --seed N                corpus seed; use one training never saw
                       --profile NAME          voice the model was distilled for, default r2
                       --sample-rate HZ        render rate, default 44100
        """);
}
