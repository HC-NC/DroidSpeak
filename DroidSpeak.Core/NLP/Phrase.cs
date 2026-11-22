namespace DroidSpeak.NLP;

/// <summary>
/// A parsed phrase: its words, emotion and intonation contour.
/// </summary>
public sealed class Phrase
{
    /// <summary>Creates a parsed phrase.</summary>
    public Phrase(
        string text,
        EmotionVector emotion,
        PhrasePunctuation punctuation,
        IReadOnlyList<Word> words,
        float shoutRatio)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(words);

        Text = text;
        Emotion = emotion;
        Punctuation = punctuation;
        Words = words;
        ShoutRatio = Math.Clamp(shoutRatio, 0f, 1f);

        var tempo = punctuation switch
        {
            PhrasePunctuation.Exclamation => 0.82f,
            PhrasePunctuation.Ellipsis => 1.40f,
            PhrasePunctuation.Question => 0.95f,
            _ => 1.00f,
        };

        // Higher arousal makes the droid speak faster; low arousal (sadness) slows it down.
        tempo *= 1f - 0.16f * Emotion.Arousal;
        TempoScale = Math.Clamp(tempo, 0.55f, 2.00f);
    }

    /// <summary>The phrase text without its terminal punctuation.</summary>
    public string Text { get; }

    /// <summary>The emotion derived from typography and the emotion lexicon.</summary>
    public EmotionVector Emotion { get; }

    /// <summary>The terminal punctuation that produced the intonation contour.</summary>
    public PhrasePunctuation Punctuation { get; }

    /// <summary>The parsed words in order.</summary>
    public IReadOnlyList<Word> Words { get; }

    /// <summary>The fraction of words written in uppercase, in <c>[0, 1]</c>.</summary>
    public float ShoutRatio { get; }

    /// <summary>A tempo multiplier: <c>&lt; 1</c> is faster, <c>&gt; 1</c> is slower.</summary>
    public float TempoScale { get; }

    /// <summary>Whether the phrase ends with a rising question glide.</summary>
    public bool RisingGlide => Punctuation == PhrasePunctuation.Question;

    /// <summary>Whether the phrase ends with a falling ellipsis contour.</summary>
    public bool FallingGlide => Punctuation == PhrasePunctuation.Ellipsis;

    /// <summary>Whether any word is shouted in uppercase.</summary>
    public bool IsShout => ShoutRatio > 0f;

    /// <summary>The total number of syllables in the phrase.</summary>
    public int SyllableCount
    {
        get
        {
            var count = 0;
            foreach (var word in Words)
            {
                count += word.Syllables.Count;
            }

            return count;
        }
    }
}
