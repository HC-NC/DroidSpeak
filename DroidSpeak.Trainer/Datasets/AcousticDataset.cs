using DroidSpeak;
using DroidSpeak.Neural;
using DroidSpeak.Prosody;

namespace DroidSpeak.Trainer.Datasets;

/// <summary>
/// One extracted reference clip: the path it came from, its stable id, and the paired contexts and
/// frames produced by <see cref="Audio.AcousticContextMapper"/>.
/// </summary>
/// <param name="ClipId">Stable hash of the path, used for reporting and holdout bookkeeping.</param>
/// <param name="Path">Path the clip was read from.</param>
/// <param name="Contexts">One context per acoustic pulse, in speaking order.</param>
/// <param name="Frames">One frame per context, index-aligned with <paramref name="Contexts"/>.</param>
internal readonly record struct AcousticClip(
    uint ClipId,
    string Path,
    SyllableContext[] Contexts,
    SyllableFrame[] Frames)
{
    /// <summary>Number of syllables in the clip.</summary>
    public int SyllableCount => Contexts.Length;
}

/// <summary>
/// A set of extracted clips that can be encoded into feature/target tensors and split for training.
/// <para>
/// The split is by clip, not by syllable. Syllables from one utterance share a phrasing and a noise
/// seed, so a row-level split would leak the held-out clip's character into training and flatter the
/// validation loss. Grouping every clip into one side of the split keeps the validation set an honest
/// measurement of a never-seen recording.
/// </para>
/// </summary>
internal sealed class AcousticDataset
{
    private readonly AcousticClip[] _clips;
    private readonly int[] _clipRowCounts;

    /// <summary>Wraps a validated list of clips.</summary>
    /// <param name="clips">Clips in a deterministic order; at least one is required.</param>
    /// <param name="renderRate">Render rate baked into every context.</param>
    /// <param name="seed">Global seed recorded for reporting; contexts carry their own copy.</param>
    /// <exception cref="ArgumentException">The list is empty or a clip is malformed.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="renderRate"/> is out of range.</exception>
    public AcousticDataset(IReadOnlyList<AcousticClip> clips, int renderRate, uint seed)
    {
        ArgumentNullException.ThrowIfNull(clips);

        if (clips.Count == 0)
        {
            throw new ArgumentException("A dataset needs at least one clip.", nameof(clips));
        }

        if (renderRate is < 8000 or > 192000)
        {
            throw new ArgumentOutOfRangeException(
                nameof(renderRate),
                renderRate,
                "The render rate must be between 8000 and 192000 Hz.");
        }

        _clips = new AcousticClip[clips.Count];
        _clipRowCounts = new int[clips.Count];
        var syllables = 0;

        for (var i = 0; i < clips.Count; i++)
        {
            var clip = clips[i];
            ArgumentNullException.ThrowIfNull(clip.Contexts);
            ArgumentNullException.ThrowIfNull(clip.Frames);
            ArgumentNullException.ThrowIfNull(clip.Path);

            if (clip.Contexts.Length != clip.Frames.Length)
            {
                throw new ArgumentException(
                    $"Clip '{clip.Path}' has {clip.Contexts.Length} contexts but " +
                    $"{clip.Frames.Length} frames.",
                    nameof(clips));
            }

            for (var j = 0; j < clip.Contexts.Length; j++)
            {
                if (clip.Contexts[j].SampleRate != renderRate)
                {
                    throw new ArgumentException(
                        $"Clip '{clip.Path}' carries a {clip.Contexts[j].SampleRate} Hz context, " +
                        $"but the dataset renders at {renderRate} Hz.",
                        nameof(clips));
                }
            }

            _clips[i] = clip;
            _clipRowCounts[i] = clip.Contexts.Length;
            syllables += clip.Contexts.Length;
        }

        RenderRate = renderRate;
        Seed = seed;
        SyllableCount = syllables;
    }

    /// <summary>Clips in dataset order.</summary>
    public IReadOnlyList<AcousticClip> Clips => _clips;

    /// <summary>Number of clips.</summary>
    public int ClipCount => _clips.Length;

    /// <summary>Total encoded rows (one per syllable).</summary>
    public int SyllableCount { get; }

    /// <summary>Render rate every context in the dataset shares.</summary>
    public int RenderRate { get; }

    /// <summary>Seed recorded at extraction time.</summary>
    public uint Seed { get; }

    /// <summary>Rows contributed by each clip, in dataset order.</summary>
    public ReadOnlySpan<int> ClipRowCounts => _clipRowCounts;

    /// <summary>Width of one encoded feature row.</summary>
    public static int InputSize => SyllableFeatureEncoder.InputSize;

    /// <summary>Width of one encoded target row.</summary>
    public static int TargetSize => SyllableTargetEncoder.TargetSize;

    /// <summary>
    /// Concatenates datasets that share a render rate. Contexts keep their own global seed, so mixing
    /// datasets extracted with different seeds is allowed; only the rate must agree because it is
    /// baked into durations and pauses.
    /// </summary>
    /// <exception cref="InvalidDataException">The datasets do not render at the same rate.</exception>
    public static AcousticDataset Combine(IReadOnlyList<AcousticDataset> datasets)
    {
        ArgumentNullException.ThrowIfNull(datasets);

        if (datasets.Count == 0)
        {
            throw new ArgumentException("At least one dataset is required.", nameof(datasets));
        }

        var renderRate = datasets[0].RenderRate;
        var clips = new List<AcousticClip>();

        foreach (var dataset in datasets)
        {
            if (dataset.RenderRate != renderRate)
            {
                throw new InvalidDataException(
                    $"Cannot combine datasets at {renderRate} Hz and {dataset.RenderRate} Hz.");
            }

            clips.AddRange(dataset.Clips);
        }

        return new AcousticDataset(clips, renderRate, datasets[0].Seed);
    }

    /// <summary>
    /// Encodes every syllable into the 60-wide feature / 17-wide target tensors the trainer consumes,
    /// in clip order and reading order. Every value is checked for finiteness: a corrupt clip must
    /// fail here rather than poison the weights with a NaN after a few epochs.
    /// </summary>
    /// <exception cref="InvalidDataException">An encoded feature or target is not finite.</exception>
    public TrainingSet Encode(DroidVoiceProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);

        var features = new float[SyllableCount * InputSize];
        var targets = new float[SyllableCount * TargetSize];
        Span<float> featureRow = stackalloc float[InputSize];
        Span<float> targetRow = stackalloc float[TargetSize];
        var row = 0;

        foreach (var clip in _clips)
        {
            for (var i = 0; i < clip.Contexts.Length; i++)
            {
                SyllableFeatureEncoder.Encode(clip.Contexts, i, featureRow);
                SyllableTargetEncoder.EncodeTargets(
                    clip.Frames[i],
                    clip.Contexts[i],
                    profile,
                    targetRow);

                EnsureFinite(featureRow, clip.Path, "feature");
                EnsureFinite(targetRow, clip.Path, "target");

                featureRow.CopyTo(features.AsSpan(row * InputSize, InputSize));
                targetRow.CopyTo(targets.AsSpan(row * TargetSize, TargetSize));
                row++;
            }
        }

        return new TrainingSet(features, targets, InputSize, TargetSize);
    }

    /// <summary>
    /// Partitions the dataset into a training and a validation split without splitting a clip across
    /// the two. If the dataset has a single clip the split cannot be honest, so the call fails loudly
    /// instead of silently leaking rows.
    /// </summary>
    /// <param name="profile">Voice profile targets are encoded against.</param>
    /// <param name="validationFraction">Share of clips held out.</param>
    /// <param name="random">Deterministic source used to choose which clips are held out.</param>
    /// <param name="training">Receives the training rows.</param>
    /// <param name="validation">Receives the held-out rows.</param>
    /// <exception cref="InvalidOperationException">The dataset has fewer than two clips.</exception>
    public void SplitByClip(
        DroidVoiceProfile profile,
        float validationFraction,
        ref RandomSource random,
        out TrainingSet training,
        out TrainingSet validation)
    {
        ArgumentNullException.ThrowIfNull(profile);

        if (ClipCount < 2)
        {
            throw new InvalidOperationException(
                "Clip-level holdout needs at least two clips; extract more reference files or " +
                "lower the validation fraction into a row-level split explicitly.");
        }

        if (validationFraction is <= 0f or >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(validationFraction),
                validationFraction,
                "The validation fraction must sit strictly between 0 and 1.");
        }

        var encoded = Encode(profile);
        var validationClips = Math.Clamp(
            (int)MathF.Round(ClipCount * validationFraction),
            1,
            ClipCount - 1);

        // Partial Fisher-Yates over the clip indices: the first validationClips entries are held out.
        var permutation = new int[ClipCount];
        for (var i = 0; i < permutation.Length; i++)
        {
            permutation[i] = i;
        }

        for (var i = 0; i < validationClips; i++)
        {
            var j = i + random.NextInt(ClipCount - i);
            (permutation[i], permutation[j]) = (permutation[j], permutation[i]);
        }

        var heldOut = new bool[ClipCount];
        for (var i = 0; i < validationClips; i++)
        {
            heldOut[permutation[i]] = true;
        }

        var validationRows = 0;
        for (var clip = 0; clip < ClipCount; clip++)
        {
            if (heldOut[clip])
            {
                validationRows += _clipRowCounts[clip];
            }
        }

        var trainingRows = SyllableCount - validationRows;
        if (trainingRows == 0)
        {
            throw new InvalidOperationException(
                "Every clip was held out for validation; the split produced no training rows.");
        }

        var trainFeatures = new float[trainingRows * InputSize];
        var trainTargets = new float[trainingRows * TargetSize];
        var valFeatures = new float[validationRows * InputSize];
        var valTargets = new float[validationRows * TargetSize];

        var trainRow = 0;
        var valRow = 0;
        var sourceRow = 0;

        for (var clip = 0; clip < ClipCount; clip++)
        {
            var count = _clipRowCounts[clip];

            if (heldOut[clip])
            {
                CopyRows(encoded, sourceRow, valFeatures, valRow, count, valTargets);
                valRow += count;
            }
            else
            {
                CopyRows(encoded, sourceRow, trainFeatures, trainRow, count, trainTargets);
                trainRow += count;
            }

            sourceRow += count;
        }

        training = new TrainingSet(trainFeatures, trainTargets, InputSize, TargetSize);
        validation = new TrainingSet(valFeatures, valTargets, InputSize, TargetSize);
    }

    private static void CopyRows(
        TrainingSet source,
        int sourceRow,
        float[] destinationFeatures,
        int destinationRow,
        int rows,
        float[] destinationTargets)
    {
        Array.Copy(
            source.FeatureStorage,
            sourceRow * InputSize,
            destinationFeatures,
            destinationRow * InputSize,
            rows * InputSize);
        Array.Copy(
            source.TargetStorage,
            sourceRow * TargetSize,
            destinationTargets,
            destinationRow * TargetSize,
            rows * TargetSize);
    }

    private static void EnsureFinite(ReadOnlySpan<float> values, string path, string kind)
    {
        foreach (var value in values)
        {
            if (!float.IsFinite(value))
            {
                throw new InvalidDataException(
                    $"Clip '{path}' produced a non-finite {kind}; the extraction is corrupt.");
            }
        }
    }
}