namespace DroidSpeak.NLP;

/// <summary>
/// Splits words into syllables by vowel groups for Russian (Cyrillic) and English (Latin) text.
/// </summary>
public static class SyllableSplitter
{
    /// <summary>Splits a word using a stress position derived from its stable hash.</summary>
    public static IReadOnlyList<Syllable> Split(string word)
    {
        ArgumentNullException.ThrowIfNull(word);
        return Split(word, Phonetics.StableHash(word));
    }

    /// <summary>
    /// Splits a word into syllables. Consonants between two vowels go to the following syllable,
    /// except for larger clusters where up to two consonants are kept as the previous coda
    /// (for example <c>ex-tra</c>, <c>hel-lo</c>).
    /// </summary>
    /// <param name="word">A single word without surrounding punctuation.</param>
    /// <param name="seed">A stable seed used to pick the stressed syllable.</param>
    public static IReadOnlyList<Syllable> Split(string word, uint seed)
    {
        ArgumentNullException.ThrowIfNull(word);

        if (word.Length == 0)
        {
            return Array.Empty<Syllable>();
        }

        var vowelPositions = new List<int>(4);
        for (var i = 0; i < word.Length; i++)
        {
            if (Phonetics.IsVowel(word[i]))
            {
                vowelPositions.Add(i);
            }
        }

        if (vowelPositions.Count == 0)
        {
            return new[]
            {
                new Syllable(
                    word,
                    word,
                    '\0',
                    string.Empty,
                    IsStressed: true,
                    HasSibilantOnset: Phonetics.ContainsSibilant(word),
                    HasVoicedOnset: Phonetics.ContainsVoicedConsonant(word)),
            };
        }

        var stressedIndex = (int)((seed >> 8) % (uint)vowelPositions.Count);
        var syllables = new List<Syllable>(vowelPositions.Count);
        var start = 0;

        for (var i = 0; i < vowelPositions.Count; i++)
        {
            var vowelIndex = vowelPositions[i];
            int boundary;

            if (i == vowelPositions.Count - 1)
            {
                boundary = word.Length;
            }
            else
            {
                var gap = vowelPositions[i + 1] - vowelIndex - 1;
                var codaLength = gap >= 3 ? Math.Min(gap - 2, 2) : 0;
                boundary = vowelIndex + 1 + codaLength;
            }

            var onset = word[start..vowelIndex];
            var coda = word[(vowelIndex + 1)..boundary];

            syllables.Add(new Syllable(
                word[start..boundary],
                onset,
                word[vowelIndex],
                coda,
                IsStressed: i == stressedIndex,
                HasSibilantOnset: Phonetics.ContainsSibilant(onset),
                HasVoicedOnset: Phonetics.ContainsVoicedConsonant(onset)));

            start = boundary;
        }

        return syllables;
    }
}
