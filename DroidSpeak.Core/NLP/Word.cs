namespace DroidSpeak.NLP;

/// <summary>A parsed word together with its stable hash and syllable breakdown.</summary>
public sealed class Word
{
    /// <summary>Creates a parsed word.</summary>
    /// <param name="text">The original word text.</param>
    /// <param name="isShout">Whether the word is written in uppercase (a droid "scream").</param>
    /// <param name="isFollowedByComma">Whether the word is followed by a comma-like pause.</param>
    /// <param name="syllables">The pre-computed syllable breakdown.</param>
    public Word(string text, bool isShout, bool isFollowedByComma, IReadOnlyList<Syllable> syllables)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(syllables);

        Text = text;
        IsShout = isShout;
        IsFollowedByComma = isFollowedByComma;
        Syllables = syllables;
        Hash = Phonetics.StableHash(text);
    }

    /// <summary>The original word text.</summary>
    public string Text { get; }

    /// <summary>A deterministic FNV-1a hash of the word.</summary>
    public uint Hash { get; }

    /// <summary>Whether the word is written in uppercase (at least two uppercase letters).</summary>
    public bool IsShout { get; }

    /// <summary>Whether the word is followed by a comma, semicolon or colon.</summary>
    public bool IsFollowedByComma { get; }

    /// <summary>The syllable breakdown of the word.</summary>
    public IReadOnlyList<Syllable> Syllables { get; }
}
