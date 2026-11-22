namespace DroidSpeak.NLP;

/// <summary>
/// Locale-aware (Cyrillic + Latin) helpers for vowels, consonant articulation classes
/// and deterministic word hashing.
/// </summary>
public static class Phonetics
{
    private const uint FnvOffsetBasis = 2166136261u;
    private const uint FnvPrime = 16777619u;

    /// <summary>Returns <see langword="true"/> for Cyrillic or Latin vowel letters.</summary>
    public static bool IsVowel(char value)
    {
        return char.ToLowerInvariant(value) switch
        {
            'а' or 'е' or 'ё' or 'и' or 'о' or 'у' or 'ы' or 'э' or 'ю' or 'я' => true,
            'a' or 'e' or 'i' or 'o' or 'u' or 'y' => true,
            _ => false,
        };
    }

    /// <summary>
    /// Returns <see langword="true"/> for hissing/fricative consonants that add a short
    /// white-noise transient to a syllable attack.
    /// </summary>
    public static bool IsSibilant(char value)
    {
        return char.ToLowerInvariant(value) switch
        {
            'с' or 'ш' or 'щ' or 'ч' or 'ф' => true,
            's' or 'f' => true,
            _ => false,
        };
    }

    /// <summary>
    /// Returns <see langword="true"/> for voiced consonants that add a low-frequency
    /// amplitude/frequency modulation ("growl") to a syllable.
    /// </summary>
    public static bool IsVoicedConsonant(char value)
    {
        return char.ToLowerInvariant(value) switch
        {
            'б' or 'д' or 'г' or 'р' or 'ж' or 'з' => true,
            'b' or 'd' or 'g' or 'r' or 'z' => true,
            _ => false,
        };
    }

    /// <summary>Checks an onset cluster for a sibilant, including the Latin <c>sh</c>/<c>ch</c> digraphs.</summary>
    public static bool ContainsSibilant(ReadOnlySpan<char> onset)
    {
        for (var i = 0; i < onset.Length; i++)
        {
            var current = char.ToLowerInvariant(onset[i]);
            if (IsSibilant(current))
            {
                return true;
            }

            if (i + 1 < onset.Length)
            {
                var next = char.ToLowerInvariant(onset[i + 1]);
                if ((current == 'c' && next == 'h') ||
                    (current == 's' && next == 'h'))
                {
                    return true;
                }
            }
        }

        return false;
    }

    /// <summary>Checks an onset cluster for a voiced consonant.</summary>
    public static bool ContainsVoicedConsonant(ReadOnlySpan<char> onset)
    {
        foreach (var character in onset)
        {
            if (IsVoicedConsonant(character))
            {
                return true;
            }
        }

        return false;
    }

    /// <summary>
    /// Stable FNV-1a (32-bit) hash of a word. The hash is case-insensitive and does not depend on
    /// the process-wide randomized <see cref="string.GetHashCode()"/> implementation, so the same
    /// word always receives the same synthesis seed.
    /// </summary>
    public static uint StableHash(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return StableHash(text.AsSpan());
    }

    /// <inheritdoc cref="StableHash(string)"/>
    public static uint StableHash(ReadOnlySpan<char> text)
    {
        var hash = FnvOffsetBasis;
        foreach (var character in text)
        {
            hash ^= char.ToLowerInvariant(character);
            hash *= FnvPrime;
        }

        return hash;
    }

    /// <summary>Trims surrounding punctuation and whitespace from a word.</summary>
    public static string NormalizeWord(string word)
    {
        ArgumentNullException.ThrowIfNull(word);

        var start = 0;
        var end = word.Length;
        while (start < end && !char.IsLetterOrDigit(word[start]))
        {
            start++;
        }

        while (end > start && !char.IsLetterOrDigit(word[end - 1]))
        {
            end--;
        }

        return word[start..end];
    }
}
