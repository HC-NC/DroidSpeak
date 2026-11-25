using DroidSpeak.Neural;

namespace DroidSpeak.Trainer;

/// <summary>
/// How the learning rate moves during a run.
/// </summary>
internal enum LearningRateSchedule
{
    /// <summary>Constant rate.</summary>
    Constant = 0,

    /// <summary>
    /// Half-cosine from the base rate down to a floor. Default for distillation: most of the loss is
    /// won early, and a large step near the end only jitters the rhythm outputs.
    /// </summary>
    Cosine = 1,

    /// <summary>Multiplies the rate by a factor every fixed number of epochs.</summary>
    Step = 2,
}

/// <summary>
/// Everything a training run needs, parsed from the CLI once and passed down unchanged.
/// </summary>
/// <param name="HiddenWidths">Hidden layer sizes, between the 60 features and the 17 outputs.</param>
/// <param name="HiddenActivation">Non-linearity used by every hidden layer.</param>
/// <param name="Epochs">Full passes over the training split.</param>
/// <param name="BatchSize">Samples summed before one Adam update.</param>
/// <param name="LearningRate">Base step size.</param>
/// <param name="Schedule">Learning-rate decay shape.</param>
/// <param name="MinLearningRate">Floor the cosine schedule decays to.</param>
/// <param name="StepEpochs">Epochs per decay for the step schedule.</param>
/// <param name="StepFactor">Multiplier per decay for the step schedule.</param>
/// <param name="Beta1">Adam first-moment decay.</param>
/// <param name="Beta2">Adam second-moment decay.</param>
/// <param name="Epsilon">Adam denominator floor.</param>
/// <param name="Seed">Initialization and shuffling seed.</param>
/// <param name="ValidationFraction">Share of the corpus held out for scoring.</param>
/// <param name="LogEvery">Epochs between progress lines.</param>
internal readonly record struct TrainingOptions(
    int[] HiddenWidths,
    ActivationType HiddenActivation,
    int Epochs,
    int BatchSize,
    float LearningRate,
    LearningRateSchedule Schedule,
    float MinLearningRate,
    int StepEpochs,
    float StepFactor,
    float Beta1,
    float Beta2,
    float Epsilon,
    uint Seed,
    float ValidationFraction,
    int LogEvery)
{
    /// <summary>The documented v0.7 defaults: 64, 48, tanh, 60 epochs, batch 32, cosine decay.</summary>
    public static TrainingOptions Default => new(
        HiddenWidths: [64, 48],
        HiddenActivation: ActivationType.Tanh,
        Epochs: 60,
        BatchSize: 32,
        LearningRate: 0.008f,
        Schedule: LearningRateSchedule.Cosine,
        MinLearningRate: 0.0002f,
        StepEpochs: 10,
        StepFactor: 0.5f,
        Beta1: 0.9f,
        Beta2: 0.999f,
        Epsilon: 1e-8f,
        Seed: 20260107u,
        ValidationFraction: 0.1f,
        LogEvery: 1);

    /// <summary>Builds the layer chain: features, hidden layers, 17 normalized outputs.</summary>
    /// <param name="inputSize">Width of the encoded syllable features.</param>
    /// <param name="outputSize">Width of the encoded targets.</param>
    public LayerSpec[] BuildSpecs(int inputSize, int outputSize)
    {
        var specs = new LayerSpec[HiddenWidths.Length + 1];
        var from = inputSize;

        for (var i = 0; i < HiddenWidths.Length; i++)
        {
            specs[i] = new LayerSpec(from, HiddenWidths[i], HiddenActivation);
            from = HiddenWidths[i];
        }

        // The output layer must land inside [0, 1] because the decoder treats every value as a
        // normalized amount; a linear or ReLU output would train targets the decoder then clamps.
        specs[^1] = new LayerSpec(from, outputSize, ActivationType.Sigmoid);
        return specs;
    }

    /// <summary>
    /// Resolves the learning rate for one epoch. Cosine is measured against the whole run so the
    /// floor is reached on the last epoch, which is what makes a longer run finer rather than wider.
    /// </summary>
    public float LearningRateAt(int epoch) => Schedule switch
    {
        LearningRateSchedule.Cosine => MinLearningRate +
            (0.5f * (LearningRate - MinLearningRate) *
                (1f + MathF.Cos(MathF.PI * epoch / MathF.Max(1, Epochs - 1)))),
        LearningRateSchedule.Step => LearningRate *
            MathF.Pow(StepFactor, epoch / MathF.Max(1, StepEpochs)),
        _ => LearningRate,
    };

    /// <summary>Rejects values that would produce a network the inference engine cannot load.</summary>
    public string? Validate()
    {
        if (HiddenWidths.Length == 0)
        {
            return "At least one hidden layer is required (--arch).";
        }

        foreach (var width in HiddenWidths)
        {
            if (width is < 1 or > MlpModel.MaxSupportedLayerWidth)
            {
                return $"Hidden width {width} is outside 1..{MlpModel.MaxSupportedLayerWidth}.";
            }
        }

        if (HiddenWidths.Length + 1 > MlpModel.MaxLayerCount)
        {
            return $"A network of {HiddenWidths.Length + 1} layers exceeds " +
                $"{MlpModel.MaxLayerCount}.";
        }

        if (Epochs < 1)
        {
            return "--epochs must be at least 1.";
        }

        if (BatchSize < 1)
        {
            return "--batch must be at least 1.";
        }

        if (LearningRate <= 0f || MinLearningRate < 0f || MinLearningRate > LearningRate)
        {
            return "--lr must be positive and --lr-min must sit between 0 and --lr.";
        }

        if (StepEpochs < 1 || StepFactor is <= 0f or > 1f)
        {
            return "--step-epochs must be positive and --step-factor must sit in (0, 1].";
        }

        if (Beta1 is <= 0f or >= 1f || Beta2 is <= 0f or >= 1f)
        {
            return "--beta1 and --beta2 must sit strictly between 0 and 1.";
        }

        if (Epsilon <= 0f)
        {
            return "--epsilon must be positive.";
        }

        if (ValidationFraction is <= 0f or >= 0.9f)
        {
            return "--validation must sit inside (0, 0.9).";
        }

        if (LogEvery < 1)
        {
            return "--log-every must be at least 1.";
        }

        return null;
    }
}
