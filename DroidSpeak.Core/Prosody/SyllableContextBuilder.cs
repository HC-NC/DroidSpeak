using DroidSpeak.NLP;

namespace DroidSpeak.Prosody;

/// <summary>
/// Flattens parsed phrases into the value-type <see cref="SyllableContext"/> stream consumed by
/// <see cref="IProsodyPredictor"/>.
/// </summary>
internal static class SyllableContextBuilder
{
    /// <summary>Counts how many syllables the parsed phrases contain.</summary>
    public static int Count(IReadOnlyList<Phrase> phrases)
    {
        var count = 0;
        foreach (var phrase in phrases)
        {
            foreach (var word in phrase.Words)
            {
                count += word.Syllables.Count;
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

        foreach (var phrase in phrases)
        {
            if (phrase.Words.Count == 0 || phrase.SyllableCount == 0)
            {
                continue;
            }

            var phraseSyllableCount = phrase.SyllableCount;
            var previousWordHadComma = false;

            for (var wordIndex = 0; wordIndex < phrase.Words.Count; wordIndex++)
            {
                var word = phrase.Words[wordIndex];

                for (var syllableIndex = 0; syllableIndex < word.Syllables.Count; syllableIndex++)
                {
                    var syllable = word.Syllables[syllableIndex];

                    contexts[index++] = new SyllableContext
                    {
                        SampleRate = sampleRate,
                        GlobalSeed = globalSeed,
                        WordHash = word.Hash,
                        PhraseIndex = phraseIndex,
                        WordIndexInPhrase = wordIndex,
                        SyllableIndex = syllableIndex,
                        WordSyllableCount = word.Syllables.Count,
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
}
