namespace DroidSpeak.NLP;

/// <summary>
/// One syllable of a word, split into onset / nucleus / coda.
/// </summary>
/// <param name="Text">The original syllable text (for example, <c>"при"</c>).</param>
/// <param name="Onset">Consonants preceding the nucleus.</param>
/// <param name="Nucleus">The vowel letter, or <c>'\0'</c> when the word has no vowels.</param>
/// <param name="Coda">Consonants following the nucleus.</param>
/// <param name="IsStressed">Whether this syllable receives the (deterministic) word stress.</param>
/// <param name="HasSibilantOnset">Whether the onset contains a hissing consonant.</param>
/// <param name="HasVoicedOnset">Whether the onset contains a voiced consonant.</param>
public readonly record struct Syllable(
    string Text,
    string Onset,
    char Nucleus,
    string Coda,
    bool IsStressed,
    bool HasSibilantOnset,
    bool HasVoicedOnset)
{
    /// <summary>Whether the syllable contains a vowel.</summary>
    public bool HasNucleus => Nucleus != '\0';
}
