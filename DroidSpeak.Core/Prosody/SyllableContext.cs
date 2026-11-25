using DroidSpeak.NLP;

namespace DroidSpeak.Prosody;

/// <summary>
/// Everything the prosody layer needs to know about one syllable: what the text says, how it is
/// stressed, and which phrase / word / syllable it belongs to.
/// <para>
/// A context contains no audio data and no references, so a whole utterance can be described with
/// a flat array of value types that is safe for NativeAOT and WebAssembly.
/// </para>
/// </summary>
public struct SyllableContext
{
    /// <summary>Output sample rate in Hz.</summary>
    public int SampleRate;

    /// <summary>Global seed mixed into every syllable seed.</summary>
    public uint GlobalSeed;

    /// <summary>Stable FNV-1a hash of the owning word.</summary>
    public uint WordHash;

    /// <summary>Zero-based index of the owning phrase among the non-empty phrases.</summary>
    public int PhraseIndex;

    /// <summary>Zero-based index of the owning word inside its phrase.</summary>
    public int WordIndexInPhrase;

    /// <summary>Zero-based index of this syllable inside its word.</summary>
    public int SyllableIndex;

    /// <summary>Number of syllables in the owning word.</summary>
    public int WordSyllableCount;

    /// <summary>Number of syllables in the owning phrase.</summary>
    public int PhraseSyllableCount;

    /// <summary>Length of the owning word's text, in characters.</summary>
    public int WordTextLength;

    /// <summary>Phrase tempo multiplier: <c>&lt; 1</c> is faster, <c>&gt; 1</c> is slower.</summary>
    public float TempoScale;

    /// <summary>Raw phrase valence in <c>[-1, 1]</c>.</summary>
    public float EmotionValence;

    /// <summary>Raw phrase arousal in <c>[-1, 1]</c>.</summary>
    public float EmotionArousal;

    /// <summary>Terminal punctuation of the owning phrase.</summary>
    public PhrasePunctuation Punctuation;

    /// <summary>Whether this is the first phrase that actually contains syllables.</summary>
    public bool IsFirstSpokenPhrase;

    /// <summary>Whether the preceding word in this phrase ended with a comma-like pause.</summary>
    public bool PreviousWordHadComma;

    /// <summary>Whether the owning word is written in uppercase.</summary>
    public bool IsShout;

    /// <summary>Whether this syllable carries the word stress.</summary>
    public bool IsStressed;

    /// <summary>Whether the onset contains a voiced consonant.</summary>
    public bool HasVoicedOnset;

    /// <summary>Whether the onset contains a hissing consonant.</summary>
    public bool HasSibilantOnset;

    /// <summary>Whether this syllable starts a new part of a hyphenated word.</summary>
    public bool IsHyphenBoundary;
}
