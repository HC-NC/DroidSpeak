namespace DroidSpeak.NLP;

/// <summary>The terminal punctuation of a phrase, which drives its intonation contour.</summary>
public enum PhrasePunctuation
{
    /// <summary>A plain statement with a gently falling final tone.</summary>
    Statement,

    /// <summary>A question with a rising glide over the final two syllables.</summary>
    Question,

    /// <summary>An exclamation with raised arousal, loudness and FM overload.</summary>
    Exclamation,

    /// <summary>An ellipsis with slower tempo and a falling, subdued contour.</summary>
    Ellipsis,
}
