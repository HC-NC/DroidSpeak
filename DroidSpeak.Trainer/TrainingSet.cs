namespace DroidSpeak.Trainer;

/// <summary>
/// A block of feature/target pairs held as flat row-major tensors, plus the shuffling and splitting
/// the training loop needs.
/// <para>
/// Rows are contiguous on purpose. A trainer that walked an array of vectors would chase references
/// through the heap on every sample; walking one <see cref="float"/> array reads sequentially and
/// keeps the row in cache for the whole forward and backward pass.
/// </para>
/// </summary>
internal sealed class TrainingSet
{
    private readonly float[] _features;
    private readonly float[] _targets;
    private int[] _order;

    public TrainingSet(float[] features, float[] targets, int inputSize, int outputSize)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(targets);

        if (inputSize is < 1 || outputSize is < 1)
        {
            throw new ArgumentException("A sample needs at least one feature and one target.");
        }

        if (features.Length % inputSize != 0)
        {
            throw new ArgumentException(
                $"{features.Length} features do not divide into rows of {inputSize}.",
                nameof(features));
        }

        if (targets.Length % outputSize != 0)
        {
            throw new ArgumentException(
                $"{targets.Length} targets do not divide into rows of {outputSize}.",
                nameof(targets));
        }

        Count = features.Length / inputSize;
        if (Count * outputSize != targets.Length)
        {
            throw new ArgumentException(
                $"The feature tensor holds {Count} rows but the target tensor holds " +
                $"{targets.Length / outputSize}.",
                nameof(targets));
        }

        _features = features;
        _targets = targets;
        InputSize = inputSize;
        OutputSize = outputSize;
        _order = Indices(0, Count);
    }

    /// <summary>Number of samples.</summary>
    public int Count { get; }

    /// <summary>Width of one feature row.</summary>
    public int InputSize { get; }

    /// <summary>Width of one target row.</summary>
    public int OutputSize { get; }

    /// <summary>Current visit order, so an epoch can be resumed sample by sample.</summary>
    public ReadOnlySpan<int> Order => _order;

    /// <summary>Features of the sample at <paramref name="position"/> in the current order.</summary>
    public ReadOnlySpan<float> FeaturesAt(int position)
    {
        var row = _order[position] * InputSize;
        return _features.AsSpan(row, InputSize);
    }

    /// <summary>Targets of the sample at <paramref name="position"/> in the current order.</summary>
    public ReadOnlySpan<float> TargetsAt(int position)
    {
        var row = _order[position] * OutputSize;
        return _targets.AsSpan(row, OutputSize);
    }

    /// <summary>Features of sample <paramref name="index"/> in storage order.</summary>
    public ReadOnlySpan<float> Features(int index) => _features.AsSpan(index * InputSize, InputSize);

    /// <summary>Targets of sample <paramref name="index"/> in storage order.</summary>
    public ReadOnlySpan<float> Targets(int index) => _targets.AsSpan(index * OutputSize, OutputSize);

    /// <summary>Feature storage, for tests and for exporting a corpus alongside a model.</summary>
    public float[] FeatureStorage => _features;

    /// <summary>Target storage.</summary>
    public float[] TargetStorage => _targets;

    /// <summary>Reorders the visit sequence with a Fisher-Yates shuffle.</summary>
    /// <param name="random">Deterministic source; the same seed always produces the same epoch.</param>
    public void Shuffle(ref RandomSource random)
    {
        for (var i = _order.Length - 1; i > 0; i--)
        {
            var j = random.NextInt(i + 1);
            var swap = _order[i];
            _order[i] = _order[j];
            _order[j] = swap;
        }
    }

    /// <summary>Restores storage order after a shuffle.</summary>
    public void ResetOrder() => _order = Indices(0, Count);

    /// <summary>
    /// Partitions this set into training and validation halves by index, without copying rows.
    /// </summary>
    /// <param name="validationCount">Number of samples to reserve for validation.</param>
    /// <param name="random">Source used to pick which samples become validation rows.</param>
    /// <param name="training">Receives the remaining samples.</param>
    /// <param name="validation">Receives the reserved samples.</param>
    /// <remarks>
    /// The split is drawn before any training happens and never reshuffled, so the validation numbers
    /// of two runs measure the same held-out samples and are directly comparable.
    /// </remarks>
    public void Split(
        int validationCount,
        ref RandomSource random,
        out TrainingSet training,
        out TrainingSet validation)
    {
        if (validationCount < 0 || validationCount >= Count)
        {
            throw new ArgumentException(
                $"A split must reserve between 0 and {Count - 1} of {Count} samples.",
                nameof(validationCount));
        }

        // Partial Fisher-Yates: the first validationCount entries of the permutation are the held-out
        // rows, everything after them is training. One pass, no sorting.
        var permutation = Indices(0, Count);
        for (var i = 0; i < validationCount; i++)
        {
            var j = i + random.NextInt(Count - i);
            var swap = permutation[i];
            permutation[i] = permutation[j];
            permutation[j] = swap;
        }

        var trainingRows = Count - validationCount;
        var trainFeatures = new float[trainingRows * InputSize];
        var trainTargets = new float[trainingRows * OutputSize];
        var valFeatures = new float[validationCount * InputSize];
        var valTargets = new float[validationCount * OutputSize];

        for (var i = 0; i < validationCount; i++)
        {
            CopyRow(permutation[i], valFeatures, valTargets, i);
        }

        for (var i = validationCount; i < Count; i++)
        {
            CopyRow(permutation[i], trainFeatures, trainTargets, i - validationCount);
        }

        training = new TrainingSet(trainFeatures, trainTargets, InputSize, OutputSize);
        validation = new TrainingSet(valFeatures, valTargets, InputSize, OutputSize);
    }

    private void CopyRow(int source, float[] features, float[] targets, int destination)
    {
        Features(source).CopyTo(features.AsSpan(destination * InputSize, InputSize));
        Targets(source).CopyTo(targets.AsSpan(destination * OutputSize, OutputSize));
    }

    private static int[] Indices(int from, int length)
    {
        var values = new int[length];
        for (var i = 0; i < length; i++)
        {
            values[i] = from + i;
        }

        return values;
    }
}
