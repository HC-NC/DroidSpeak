using DroidSpeak.NLP;

namespace DroidSpeak.Prosody;

/// <summary>
/// Flattens parsed phrases into the value-type <see cref="SyllableContext"/> stream consumed by
/// <see cref="IProsodyPredictor"/>.
/// <para>
/// Everything here uses indexed loops over <see cref="IReadOnlyList{T}"/>. A <c>foreach</c> over an
/// interface-typed list boxes its enumerator, which would break the allocation-free hot path.
/// </para>
/// </summary>
internal static class SyllableContextBuilder
{
    /// <summary>Counts how many syllables the parsed phrases contain.</summary>
    public static int Count(IReadOnlyList<Phrase> phrases)
    {
        var count = 0;
        for (var phraseIndex = 0; phraseIndex < phrases.Count; phraseIndex++)
        {
            var words = phrases[phraseIndex].Words;
            for (var wordIndex = 0; wordIndex < words.Count; wordIndex++)
            {
                count += words[wordIndex].Syllables.Count;
            }
        }

        return count;
    }

    /// <summary>
    /// Fills <paramref name="contexts"/> in speaking order. The destination must be at least as
    /// long as <see cref="Count"/>.
    /// </summary>
    public static void Build(
        IReadOnlyList<Phrase> phrases,
        int sampleRate,
        uint globalSeed,
        Span<SyllableContext> contexts)
    {
        var index = 0;
        var phraseIndex = 0;
        var firstSpokenPhrase = true;

        for (var phrasePosition = 0; phrasePosition < phrases.Count; phrasePosition++)
        {
            var phrase = phrases[phrasePosition];
            var words = phrase.Words;
            if (words.Count == 0)
            {
                continue;
            }

            var phraseSyllableCount = CountSyllables(words);
            if (phraseSyllableCount == 0)
            {
                continue;
            }

            var previousWordHadComma = false;

            for (var wordIndex = 0; wordIndex < words.Count; wordIndex++)
            {
                var word = words[wordIndex];
                var syllables = word.Syllables;

                for (var syllableIndex = 0; syllableIndex < syllables.Count; syllableIndex++)
                {
                    var syllable = syllables[syllableIndex];

                    contexts[index++] = new SyllableContext
                    {
                        SampleRate = sampleRate,
                        GlobalSeed = globalSeed,
                        WordHash = word.Hash,
                        PhraseIndex = phraseIndex,
                        WordIndexInPhrase = wordIndex,
                        SyllableIndex = syllableIndex,
                        WordSyllableCount = syllables.Count,
                        PhraseSyllableCount = phraseSyllableCount,
                        WordTextLength = word.Text.Length,
                        TempoScale = phrase.TempoScale,
                        EmotionValence = phrase.Emotion.Valence,
                        EmotionArousal = phrase.Emotion.Arousal,
                        Punctuation = phrase.Punctuation,
                        IsFirstSpokenPhrase = firstSpokenPhrase,
                        PreviousWordHadComma = previousWordHadComma,
                        IsShout = word.IsShout,
                        IsStressed = syllable.IsStressed,
                        HasVoicedOnset = syllable.HasVoicedOnset,
                        HasSibilantOnset = syllable.HasSibilantOnset,
                        IsHyphenBoundary = syllableIndex > 0 && syllable.Text.Contains('-'),
                    };
                }

                previousWordHadComma = word.IsFollowedByComma;
            }

            phraseIndex++;
            firstSpokenPhrase = false;
        }
    }

    private static int CountSyllables(IReadOnlyList<Word> words)
    {
        var count = 0;
        for (var i = 0; i < words.Count; i++)
        {
            count += words[i].Syllables.Count;
        }

        return count;
    }
}
