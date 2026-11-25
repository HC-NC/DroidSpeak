using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.Neural;

/// <summary>
/// Turns a sliding window of <see cref="SyllableContext"/> values into the fixed feature vector a
/// prosody network consumes.
/// <para>
/// The layout is part of the model contract: a network trained against this order must always be
/// fed this order, so every index is named, every value is normalised into <c>[0, 1]</c>, and the
/// whole function is pure. Nothing here reads text or hashes, which keeps the encoder stable across
/// library versions and free of any allocation.
/// </para>
/// <para>
/// A window is <c>[previous, current, next]</c>. Missing neighbours (the first and last syllable of
/// the utterance) leave their slot at zero rather than duplicating the edge syllable, so the network
/// can learn "utterance boundary" from an all-zero slot instead of mixing two different pronunciations
/// into the same features.
/// </para>
/// </summary>
public static class SyllableFeatureEncoder
{
    /// <summary>Number of features describing one syllable.</summary>
    public const int FeaturesPerSyllable = 20;

    /// <summary>Syllables in the window: previous, current, next.</summary>
    public const int WindowSize = 3;

    /// <summary>Total width of the encoded window, and of any compatible model input layer.</summary>
    public const int InputSize = FeaturesPerSyllable * WindowSize;

    /// <summary>Start of the vowel one-hot block (<c>a</c>, <c>e</c>, <c>i</c>, <c>o</c>, <c>u</c>, other).</summary>
    public const int NucleusOffset = 0;

    /// <summary>Index carrying <see cref="SyllableContext.IsStressed"/>.</summary>
    public const int StressIndex = 6;

    /// <summary>Index carrying <see cref="SyllableContext.HasVoicedOnset"/>.</summary>
    public const int VoicedOnsetIndex = 7;

    /// <summary>Index carrying <see cref="SyllableContext.HasSibilantOnset"/>.</summary>
    public const int SibilantOnsetIndex = 8;

    /// <summary>Index carrying <see cref="SyllableContext.IsShout"/>.</summary>
    public const int ShoutIndex = 9;

    /// <summary>Index carrying <see cref="SyllableContext.IsHyphenBoundary"/>.</summary>
    public const int HyphenBoundaryIndex = 10;

    /// <summary>Index carrying <see cref="SyllableContext.PreviousWordHadComma"/>.</summary>
    public const int PreviousCommaIndex = 11;

    /// <summary>Index carrying the syllable's relative position inside its word.</summary>
    public const int SyllablePositionIndex = 12;

    /// <summary>Index carrying the word's relative position inside its phrase.</summary>
    public const int WordPositionIndex = 13;

    /// <summary>Index carrying the owning word's syllable count, normalised.</summary>
    public const int WordLengthIndex = 14;

    /// <summary>Index carrying phrase valence mapped from <c>[-1, 1]</c> to <c>[0, 1]</c>.</summary>
    public const int ValenceIndex = 15;

    /// <summary>Index carrying phrase arousal mapped from <c>[-1, 1]</c> to <c>[0, 1]</c>.</summary>
    public const int ArousalIndex = 16;

    /// <summary>Index carrying phrase tempo mapped from its legal range to <c>[0, 1]</c>.</summary>
    public const int TempoIndex = 17;

    /// <summary>Index that is <c>1</c> only for a question phrase.</summary>
    public const int QuestionIndex = 18;

    /// <summary>Index that is <c>1</c> for terminal emphasis: an exclamation or an ellipsis.</summary>
    public const int EmphasisIndex = 19;

    /// <summary>Offset of the current syllable inside the encoded window.</summary>
    public const int CurrentSlotOffset = FeaturesPerSyllable;

    /// <summary>Offset of the next syllable inside the encoded window.</summary>
    public const int NextSlotOffset = FeaturesPerSyllable * 2;

    /// <summary>Lowest tempo the parser can produce, matching <see cref="Phrase"/> normalization.</summary>
    public const float MinTempoScale = 0.55f;

    /// <summary>Highest tempo the parser can produce, matching <see cref="Phrase"/> normalization.</summary>
    public const float MaxTempoScale = 2.00f;

    /// <summary>Word syllable count that saturates <see cref="WordLengthIndex"/>.</summary>
    public const int WordLengthSaturation = 8;

    /// <summary>
    /// Encodes the window centred on <paramref name="index"/>.
    /// </summary>
    /// <param name="contexts">The whole utterance, in speaking order.</param>
    /// <param name="index">Index of the syllable being described.</param>
    /// <param name="features">Destination; at least <see cref="InputSize"/> floats.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="index"/> is outside the span.</exception>
    /// <exception cref="ArgumentException"><paramref name="features"/> is too short.</exception>
    public static void Encode(
        ReadOnlySpan<SyllableContext> contexts,
        int index,
        Span<float> features)
    {
        if (features.Length < InputSize)
        {
            throw new ArgumentException(
                $"The feature buffer must hold at least {InputSize} values, but has " +
                $"{features.Length}.",
                nameof(features));
        }

        if ((uint)index >= (uint)contexts.Length)
        {
            throw new ArgumentOutOfRangeException(
                nameof(index),
                index,
                $"The utterance only has {contexts.Length} syllables.");
        }

        // A missing neighbour keeps an all-zero slot: the network reads that as "utterance boundary"
        // instead of having the edge syllable pronounced twice.
        if (index > 0)
        {
            EncodeSyllable(contexts[index - 1], features);
        }
        else
        {
            features[..FeaturesPerSyllable].Clear();
        }

        EncodeSyllable(contexts[index], features[CurrentSlotOffset..]);

        if (index + 1 < contexts.Length)
        {
            EncodeSyllable(contexts[index + 1], features[NextSlotOffset..]);
        }
        else
        {
            features[NextSlotOffset..InputSize].Clear();
        }
    }

    /// <summary>
    /// Encodes a single syllable into <paramref name="destination"/>, which must hold at least
    /// <see cref="FeaturesPerSyllable"/> values. The slot is cleared first, so calling this directly
    /// on a reused buffer cannot leave stale vowel-class bits behind. Values past
    /// <see cref="FeaturesPerSyllable"/> are untouched.
    /// </summary>
    /// <param name="context">The syllable to describe.</param>
    /// <param name="destination">Destination slot.</param>
    /// <exception cref="ArgumentException"><paramref name="destination"/> is too short.</exception>
    public static void EncodeSyllable(in SyllableContext context, Span<float> destination)
    {
        if (destination.Length < FeaturesPerSyllable)
        {
            throw new ArgumentException(
                $"A syllable slot needs at least {FeaturesPerSyllable} values, but has " +
                $"{destination.Length}.",
                nameof(destination));
        }

        destination[..FeaturesPerSyllable].Clear();

        destination[NucleusOffset + VowelClass(context.Nucleus)] = 1f;
        destination[StressIndex] = context.IsStressed ? 1f : 0f;
        destination[VoicedOnsetIndex] = context.HasVoicedOnset ? 1f : 0f;
        destination[SibilantOnsetIndex] = context.HasSibilantOnset ? 1f : 0f;
        destination[ShoutIndex] = context.IsShout ? 1f : 0f;
        destination[HyphenBoundaryIndex] = context.IsHyphenBoundary ? 1f : 0f;
        destination[PreviousCommaIndex] = context.PreviousWordHadComma ? 1f : 0f;
        destination[SyllablePositionIndex] = SyllablePosition(in context);
        destination[WordPositionIndex] = WordPosition(in context);
        destination[WordLengthIndex] = NeuralMathF.Clamp01(context.WordSyllableCount / (float)WordLengthSaturation);
        destination[ValenceIndex] = NeuralMathF.Clamp01((context.EmotionValence + 1f) * 0.5f);
        destination[ArousalIndex] = NeuralMathF.Clamp01((context.EmotionArousal + 1f) * 0.5f);
        destination[TempoIndex] = NeuralMathF.Clamp01(
            (context.TempoScale - MinTempoScale) / (MaxTempoScale - MinTempoScale));
        destination[QuestionIndex] = context.Punctuation == PhrasePunctuation.Question ? 1f : 0f;
        destination[EmphasisIndex] = context.Punctuation is PhrasePunctuation.Exclamation
            or PhrasePunctuation.Ellipsis ? 1f : 0f;
    }

    /// <summary>
    /// Maps a nucleus letter onto one of the six vowel classes, treating the palatalised Cyrillic
    /// vowels as the sound they actually carry. <c>'\0'</c> (a consonant-only syllable) lands in
    /// <see cref="OtherVowelClass"/>.
    /// </summary>
    /// <param name="nucleus">The nucleus character, in any case.</param>
    /// <returns>An index in <c>0..5</c> to add to <see cref="NucleusOffset"/>.</returns>
    public static int VowelClass(char nucleus) => char.ToLowerInvariant(nucleus) switch
    {
        'a' or 'а' or 'я' => 0,
        'e' or 'э' or 'е' or 'ё' => 1,
        'i' or 'и' or 'y' or 'ы' => 2,
        'o' or 'о' => 3,
        'u' or 'у' or 'ю' => 4,
        _ => OtherVowelClass,
    };

    /// <summary>The vowel class used for consonant-only syllables and unfamiliar letters.</summary>
    public const int OtherVowelClass = 5;

    private static float SyllablePosition(in SyllableContext context)
    {
        // A single-syllable word sits at the start of its word, which is also the end: 0 keeps the
        // feature monotonic instead of dividing by zero.
        if (context.WordSyllableCount <= 1)
        {
            return 0f;
        }

        return NeuralMathF.Clamp01(
            context.SyllableIndex / (float)(context.WordSyllableCount - 1));
    }

    private static float WordPosition(in SyllableContext context)
    {
        // Phrases do not expose a word count, so the syllable count bounds the position instead: a
        // word deep into a long phrase reads as a late position.
        var span = context.PhraseSyllableCount - 1;
        if (span <= 0)
        {
            return 0f;
        }

        return NeuralMathF.Clamp01(context.WordIndexInPhrase / (float)span);
    }
}
