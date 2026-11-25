using DroidSpeak;
using DroidSpeak.Neural;
using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.Trainer.Datasets;

/// <summary>
/// Builds the knowledge-distillation corpus: procedural text, run through the real parser and the
/// rule-based prosody engine, then converted into the exact feature/target pairs the synthesizer's
/// network consumes.
/// <para>
/// Nothing here approximates the teacher. Each row is produced by the same code path the shipped
/// <see cref="RuleBasedProsodyPredictor"/> uses at runtime, so the network is trained on the truth
/// rather than on a description of it. That is also why the corpus is generated instead of stored:
/// a corpus committed to the repository would silently disagree with the engine the next time the
/// rule model changed.
/// </para>
/// <para>
/// A row is one syllable: a 60-float window from <see cref="SyllableFeatureEncoder"/> paired with
/// the 17-float normalized target from <see cref="SyllableTargetEncoder"/>. The canonical coverage
/// utterances are encoded first, then randomly composed sentences fill the requested count, so a
/// small corpus still spans the vowel inventory, the punctuation set and the emotion grid.
/// </para>
/// </summary>
internal static class DistillationCorpus
{
    /// <summary>Width of one encoded feature row, re-exported for readability in callers.</summary>
    public const int InputSize = SyllableFeatureEncoder.InputSize;

    /// <summary>Width of one encoded target row, re-exported for readability in callers.</summary>
    public const int TargetSize = SyllableTargetEncoder.TargetSize;

    /// <summary>
    /// Builds a corpus of exactly <paramref name="samples"/> syllables for one voice profile.
    /// </summary>
    /// <param name="samples">Rows requested; the count is reached exactly by truncating the last utterance.</param>
    /// <param name="profile">
    /// The voice being distilled. Rows must all share it: the network input has no profile channel,
    /// so mixing two voices would train two different targets from identical features.
    /// </param>
    /// <param name="seed">Corpus seed; the same seed always produces the same dataset.</param>
    /// <param name="sampleRate">
    /// Sample rate baked into the contexts. It must match the rate the eval render uses, because the
    /// duration and pause targets invert through it.
    /// </param>
    /// <returns>A set of 60-wide features paired with 17-wide normalized targets.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="samples"/> is below one, or <paramref name="sampleRate"/> is outside
    /// 8000..192000 Hz.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// The vocabulary or seed produced no usable syllables.
    /// </exception>
    public static TrainingSet Build(
        int samples,
        DroidVoiceProfile profile,
        uint seed,
        int sampleRate) =>
        BuildDetailed(samples, profile, seed, sampleRate).Set;

    /// <summary>
    /// Builds a corpus and keeps the per-row <see cref="SyllableContext"/> alongside it, which is
    /// what lets a test decode a target row back through <see cref="SyllableTargetDecoder"/> and
    /// confirm the encoder is a true inverse outside the trainer.
    /// </summary>
    /// <param name="samples">Rows requested.</param>
    /// <param name="profile">The voice being distilled.</param>
    /// <param name="seed">Corpus seed.</param>
    /// <param name="sampleRate">Sample rate baked into the contexts.</param>
    /// <returns>The tensors plus one context per row, in the same order.</returns>
    public static DistillationData BuildDetailed(
        int samples,
        DroidVoiceProfile profile,
        uint seed,
        int sampleRate)
    {
        ArgumentNullException.ThrowIfNull(profile);

        if (samples < 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(samples),
                samples,
                "A corpus needs at least one sample.");
        }

        if (sampleRate is < 8000 or > 192000)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sampleRate),
                sampleRate,
                "The sample rate must be between 8000 and 192000 Hz.");
        }

        profile = profile.Normalized();

        var random = new RandomSource(seed);
        var features = new List<float>(samples * InputSize);
        var targets = new List<float>(samples * TargetSize);
        var contexts = new List<SyllableContext>(samples);
        var rows = 0;

        // Coverage first: even a small corpus sees every vowel class, hyphen, apostrophe, shout,
        // punctuation mark and emotion cell. Any row budget left over is filled at random below.
        foreach (var canonical in CorpusVocabulary.CoverageUtterances)
        {
            if (rows >= samples)
            {
                break;
            }

            rows += Encode(
                canonical,
                profile,
                sampleRate,
                random.NextUInt(),
                samples - rows,
                features,
                targets,
                contexts);
        }

        // The attempt cap exists only so a pathological seed cannot spin forever; the composer always
        // yields at least one syllable per call, so the loop ends on the row budget in practice.
        for (var attempt = 0; rows < samples && attempt < 200_000; attempt++)
        {
            var utterance = CorpusVocabulary.ComposeRandom(ref random);
            rows += Encode(
                utterance,
                profile,
                sampleRate,
                random.NextUInt(),
                samples - rows,
                features,
                targets,
                contexts);
        }

        if (rows == 0)
        {
            throw new InvalidOperationException(
                "The corpus generator produced no syllables; the vocabulary or seed is unusable.");
        }

        if (rows < samples)
        {
            throw new InvalidOperationException(
                $"The corpus generator produced only {rows} of {samples} requested samples.");
        }

        return new DistillationData(
            new TrainingSet([.. features], [.. targets], InputSize, TargetSize),
            [.. contexts]);
    }

    /// <summary>
    /// Parses and encodes one utterance, appending at most <paramref name="remaining"/> rows.
    /// </summary>
    /// <returns>The number of rows actually appended.</returns>
    private static int Encode(
        string utterance,
        DroidVoiceProfile profile,
        int sampleRate,
        uint globalSeed,
        int remaining,
        List<float> features,
        List<float> targets,
        List<SyllableContext> contexts)
    {
        if (remaining <= 0)
        {
            return 0;
        }

        var parsed = TextParser.Parse(utterance);
        if (parsed.Count == 0)
        {
            return 0;
        }

        var count = SyllableContextBuilder.Count(parsed);
        if (count == 0)
        {
            return 0;
        }

        // Dataset generation is an offline job, so plain arrays are fine here; the inference hot
        // path is the one that must stay allocation free.
        var contextArray = new SyllableContext[count];
        SyllableContextBuilder.Build(parsed, sampleRate, globalSeed, contextArray);

        var frames = new SyllableFrame[count];
        RuleBasedProsodyPredictor.Shared.Predict(contextArray, frames, profile);

        Span<float> featureRow = stackalloc float[InputSize];
        Span<float> targetRow = stackalloc float[TargetSize];

        var written = Math.Min(count, remaining);
        for (var i = 0; i < written; i++)
        {
            SyllableFeatureEncoder.Encode(contextArray, i, featureRow);
            SyllableTargetEncoder.EncodeTargets(frames[i], contextArray[i], profile, targetRow);

            for (var f = 0; f < featureRow.Length; f++)
            {
                features.Add(featureRow[f]);
            }

            for (var t = 0; t < targetRow.Length; t++)
            {
                targets.Add(targetRow[t]);
            }

            contexts.Add(contextArray[i]);
        }

        return written;
    }
}

/// <summary>
/// A generated corpus together with the context of each row, so callers that need to interpret a
/// target can decode it exactly the way the predictor would.
/// </summary>
/// <param name="Set">The flat feature/target tensors.</param>
/// <param name="Contexts">One context per row, in storage order.</param>
internal readonly record struct DistillationData(TrainingSet Set, SyllableContext[] Contexts);
