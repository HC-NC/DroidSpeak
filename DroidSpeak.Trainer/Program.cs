using DroidSpeak;
using DroidSpeak.DSP;
using DroidSpeak.Neural;
using DroidSpeak.Trainer;
using DroidSpeak.Trainer.Audio;
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

        case "extract":
            return Extract(args[1..]);

        case "finetune":
            return Finetune(args[1..]);

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

static int Extract(string[] arguments)
{
    var options = CliOptions.Parse(
        arguments,
        "input",
        "out",
        "profile",
        "render-rate",
        "analysis-rate",
        "seed",
        "resampler",
        "frame-ms",
        "hop-ratio",
        "open-factor",
        "close-factor",
        "min-pulse-ms",
        "min-gap-ms",
        "word-gap-ms");

    var input = options.Value("input")
        ?? throw new ArgumentException(
            "extract requires --input <file-or-directory[;...]> (semicolon-separated).");
    var output = options.Value("out") ?? "prosody.dsdata";
    var profileName = options.Value("profile") ?? "r2";
    var profile = options.Profile(profileName);

    var renderRate = options.Int("render-rate", DroidSynthesizer.DefaultSampleRate);
    if (renderRate is < 8000 or > 192000)
    {
        throw new ArgumentException("--render-rate must be between 8000 and 192000 Hz.");
    }

    var analysisRate = options.Int("analysis-rate", WavReader.DefaultAnalysisRate);
    if (analysisRate is < 8000 or > 192000)
    {
        throw new ArgumentException("--analysis-rate must be between 8000 and 192000 Hz.");
    }

    var seed = options.UInt("seed", 20260107u);
    var resampler = ParseResampler(options.Value("resampler"));

    // The CLI exposes the segmenter in milliseconds and ratios, the units a human tunes in; the
    // segmenter itself works in seconds because that is what makes it rate independent.
    var segmenter = PulseSegmenter.Options.Default with
    {
        FrameSeconds = options.Single("frame-ms", PulseSegmenter.DefaultFrameSeconds * 1000f) / 1000f,
        HopFraction = options.Single("hop-ratio", PulseSegmenter.DefaultHopFraction),
        OpenFactor = options.Single("open-factor", PulseSegmenter.DefaultOpenFactor),
        CloseFactor = options.Single("close-factor", PulseSegmenter.DefaultCloseFactor),
        MinPulseSeconds =
            options.Single("min-pulse-ms", PulseSegmenter.DefaultMinPulseSeconds * 1000f) / 1000f,
        MinGapSeconds =
            options.Single("min-gap-ms", PulseSegmenter.DefaultMinGapSeconds * 1000f) / 1000f,
        WordGapSeconds =
            options.Single("word-gap-ms", PulseSegmenter.DefaultWordGapSeconds * 1000f) / 1000f,
    };

    if (segmenter.Validate() is { } problem)
    {
        throw new ArgumentException(problem);
    }

    var files = ResolveFiles(input, "*.wav");
    if (files.Count == 0)
    {
        throw new ArgumentException($"No .wav files found under '{input}'.");
    }

    Console.WriteLine(
        $"extracting {files.Count} clip(s) with profile '{profileName}' " +
        $"(analysis {analysisRate} Hz, render {renderRate} Hz, seed {seed})...");

    var clips = new List<AcousticClip>();
    var totalSyllables = 0;

    foreach (var file in files)
    {
        var audio = WavReader.Read(file, analysisRate, resampler);
        var segmentation = PulseSegmenter.Segment(audio.Samples, audio.SampleRate, segmenter);
        var (contexts, frames) = AcousticContextMapper.Extract(
            audio, segmentation, profile, file, renderRate, seed);

        if (contexts.Length == 0)
        {
            Console.WriteLine($"  skip {Path.GetFileName(file)}: no usable pulses");
            continue;
        }

        clips.Add(new AcousticClip(
            AcousticContextMapper.ClipId(file), file, contexts, frames));
        totalSyllables += contexts.Length;

        Console.WriteLine(
            $"  {Path.GetFileName(file)}: {contexts.Length} syllables in " +
            $"{segmentation.Words.Length} words");
    }

    if (clips.Count == 0)
    {
        throw new InvalidOperationException(
            "No clip produced any syllables; check --frame-ms and the threshold factors.");
    }

    var dataset = new AcousticDataset(clips, renderRate, seed);
    DsDataFile.Save(dataset, output);

    Console.WriteLine(
        $"saved {dataset.ClipCount} clips / {totalSyllables} syllables at {renderRate} Hz to " +
        $"{output}");
    return 0;
}

static int Finetune(string[] arguments)
{
    var defaults = TrainingOptions.Default;
    var options = CliOptions.Parse(
        arguments,
        "brain",
        "data",
        "out",
        "profile",
        "epochs",
        "batch-size",
        "learning-rate",
        "lr-min",
        "schedule",
        "step-size",
        "decay",
        "beta1",
        "beta2",
        "epsilon",
        "seed",
        "validation",
        "replay",
        "replay-ratio",
        "log-every");

    var brain = options.Value("brain")
        ?? throw new ArgumentException("finetune requires --brain <path.dsbrain>.");
    var data = options.Value("data")
        ?? throw new ArgumentException(
            "finetune requires --data <path.dsdata[;...]> (semicolon-separated).");
    var output = options.Value("out") ?? "prosody-finetuned.dsbrain";
    var profileName = options.Value("profile") ?? "r2";
    var profile = options.Profile(profileName);

    var replaySamples = options.Int("replay", 3000);
    if (replaySamples < 2)
    {
        throw new ArgumentException("--replay must be at least 2.");
    }

    var replayRatio = options.Single("replay-ratio", 1f);
    if (replayRatio < 0f || !float.IsFinite(replayRatio))
    {
        throw new ArgumentException("--replay-ratio must be a finite, non-negative number.");
    }

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

    // The hidden architecture comes from the seed brain, not from --arch: the checkpoint is being
    // refined, and Import refuses a shape mismatch, so an --arch that disagreed would only be a trap.
    _ = training.Validate();

    MlpModel seedModel;
    using (var stream = File.OpenRead(brain))
    {
        seedModel = MlpModel.Load(stream);
    }

    if (seedModel.InputSize != SyllableFeatureEncoder.InputSize ||
        seedModel.OutputSize != SyllableTargetEncoder.TargetSize)
    {
        throw new InvalidDataException(
            $"{brain} is a {seedModel.InputSize} -> {seedModel.OutputSize} model; fine-tuning " +
            $"expects {SyllableFeatureEncoder.InputSize} -> {SyllableTargetEncoder.TargetSize}.");
    }

    var dataFiles = ResolveFiles(data, "*.dsdata");
    if (dataFiles.Count == 0)
    {
        throw new ArgumentException($"No .dsdata files found under '{data}'.");
    }

    var datasets = new List<AcousticDataset>();
    foreach (var file in dataFiles)
    {
        datasets.Add(DsDataFile.Load(file));
    }

    var dataset = AcousticDataset.Combine(datasets);

    Console.WriteLine(
        $"loaded {brain}: {seedModel.InputSize} -> {seedModel.OutputSize}, " +
        $"{seedModel.ParameterCount} parameters");
    Console.WriteLine(
        $"fine-tuning on {dataset.ClipCount} clips / {dataset.SyllableCount} syllables at " +
        $"{dataset.RenderRate} Hz, {replaySamples} replay samples per epoch");

    var splitRandom = new RandomSource(training.Seed ^ 0x5A17C0DEu);
    dataset.SplitByClip(
        profile,
        training.ValidationFraction,
        ref splitRandom,
        out var acousticTraining,
        out var acousticValidation);

    // Replay is generated at the dataset's own render rate so duration and pause targets invert
    // through the same rate the acoustic rows were encoded with.
    var replayTraining = DistillationCorpus.Build(
        replaySamples, profile, training.Seed ^ 0x4EB1E5u, dataset.RenderRate);
    var replayValidation = DistillationCorpus.Build(
        Math.Max(64, replaySamples / 4),
        profile,
        training.Seed ^ 0x9E3779B9u,
        dataset.RenderRate);

    var specs = new LayerSpec[seedModel.LayerCount];
    for (var layer = 0; layer < specs.Length; layer++)
    {
        var existing = seedModel[layer];
        specs[layer] = new LayerSpec(existing.InputSize, existing.OutputSize, existing.Activation);
    }

    var network = new TrainableMlp(
        specs, training.Seed, training.Beta1, training.Beta2, training.Epsilon);
    network.Import(seedModel);

    var result = new FinetuneRun(training, replayRatio, Console.WriteLine).Run(
        network,
        acousticTraining,
        acousticValidation,
        replayTraining,
        replayValidation);

    using (var stream = File.Create(output))
    {
        result.Model.Save(stream);
    }

    var acousticRetention = Retention(result.InitialAcousticValidation, result.BestAcousticValidation);
    var replayRetention = Retention(result.InitialReplayValidation, result.BestReplayValidation);

    Console.WriteLine(
        $"acoustic  initial {result.InitialAcousticValidation:E4}  " +
        $"best {result.BestAcousticValidation:E4}  ({acousticRetention:P1} of initial)");
    Console.WriteLine(
        $"replay    initial {result.InitialReplayValidation:E4}  " +
        $"best {result.BestReplayValidation:E4}  ({replayRetention:P1} of initial)");
    Console.WriteLine(
        $"saved best checkpoint (epoch {result.BestEpoch}) to {output}");

    if (!result.Completed)
    {
        Console.Error.WriteLine($"warning: {result.FailureReason}");
        return 1;
    }

    // The checkpoint must actually learn the new audio and must not forget the teacher. The replay
    // budget allows a quarter of regression before the run is called a failure.
    var improved = result.BestAcousticValidation < result.InitialAcousticValidation;
    var retained = replayRetention <= 1.25f;

    if (!improved)
    {
        Console.Error.WriteLine(
            "warning: fine-tuning did not reduce the held-out acoustic loss; the checkpoint is " +
            "no better on the reference clips than the seed brain.");
    }

    if (!retained)
    {
        Console.Error.WriteLine(
            $"warning: replay loss grew to {replayRetention:P1} of its initial value; raise " +
            "--replay or --replay-ratio to limit forgetting.");
    }

    return improved && retained ? 0 : 1;
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

static WavResampler ParseResampler(string? name)
{
    if (string.IsNullOrWhiteSpace(name))
    {
        return WavResampler.Linear;
    }

    return name.Trim().ToLowerInvariant() switch
    {
        "linear" => WavResampler.Linear,
        "cubic" => WavResampler.Cubic,
        _ => throw new ArgumentException("--resampler must be linear or cubic."),
    };
}

static List<string> ResolveFiles(string input, string searchPattern)
{
    var files = new List<string>();

    foreach (var entry in input.Split(
        ';',
        StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
    {
        if (File.Exists(entry))
        {
            files.Add(entry);
            continue;
        }

        if (Directory.Exists(entry))
        {
            files.AddRange(Directory.EnumerateFiles(entry, searchPattern, SearchOption.AllDirectories));
            continue;
        }

        throw new ArgumentException($"'{entry}' is neither an existing file nor a directory.");
    }

    // Directory enumeration order is not defined, and extraction must be reproducible, so the final
    // list is sorted by ordinal path.
    files.Sort(StringComparer.Ordinal);
    return files;
}

static float Retention(float initial, float best) =>
    initial > 0f ? best / initial : 0f;

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

          extract    segment reference WAVs into syllables, map each pulse to a context and a frame,
                     and save the result as a .dsdata dataset
                     options:
                       --input PATH[;PATH]     WAV files or directories to scan (required)
                       --out PATH              dataset path, default prosody.dsdata
                       --profile NAME          voice the targets encode against, default r2
                       --render-rate HZ        rate baked into contexts and frames, default 44100
                       --analysis-rate HZ      rate WAVs are resampled to, default 22050
                       --resampler NAME        linear or cubic; default linear
                       --seed N                extraction seed, default 20260107
                       --frame-ms F            RMS frame length, default 11
                       --hop-ratio F           analysis hop as a fraction of the frame, default 0.5
                       --open-factor F         floor multiplier that opens a pulse, default 3.0
                       --close-factor F        floor multiplier that closes a pulse, default 1.6
                       --min-pulse-ms F        shortest accepted pulse, default 25
                       --min-gap-ms F          shortest silence that ends a pulse, default 8
                       --word-gap-ms F         gap that starts a new word, default 100

          finetune   refine a .dsbrain network on a .dsdata dataset while replaying the procedural
                     corpus, so the voice adapts without forgetting the teacher
                     options:
                       --brain PATH            seed checkpoint to refine (required)
                       --data PATH[;PATH]      .dsdata datasets to train on (required)
                       --out PATH              checkpoint path, default prosody-finetuned.dsbrain
                       --profile NAME          voice the targets encode against, default r2
                       --epochs N              full passes over the acoustic split, default 60
                       --batch-size N          acoustic samples per Adam update, default 32
                       --learning-rate F       base step size, default 0.008
                       --lr-min F              cosine floor, default 0.0002
                       --schedule NAME         constant, cosine or step; default cosine
                       --step-size N           epochs per step decay, default 10
                       --decay F               step-decay multiplier, default 0.5
                       --beta1 F               Adam first-moment decay, default 0.9
                       --beta2 F               Adam second-moment decay, default 0.999
                       --epsilon F             Adam denominator floor, default 1e-8
                       --seed N                init and shuffle seed, default 20260107
                       --validation F          held-out fraction, by clip; default 0.1
                       --replay N              replay samples per epoch, default 3000
                       --replay-ratio F        replay samples per acoustic sample, default 1
                       --log-every N           epochs between progress lines, default 1

          eval       score a .dsbrain file against the rule-based teacher
                     options:
                       --brain PATH            checkpoint to load (required)
                       --samples N             held-out syllables to score, default 1500
                       --seed N                corpus seed; use one training never saw
                       --profile NAME          voice the model was distilled for, default r2
                       --sample-rate HZ        render rate, default 44100
        """);
}
