namespace DroidSpeak.NLP;

/// <summary>
/// A compact two-dimensional affect model used to drive the synthesizer.
/// <para>
/// <see cref="Valence"/> describes pleasantness (<c>-1</c> = negative, <c>+1</c> = positive) and
/// <see cref="Arousal"/> describes activation (<c>-1</c> = calm, <c>+1</c> = excited).
/// Both components are always clamped to <c>[-1, 1]</c>.
/// </para>
/// </summary>
public readonly record struct EmotionVector(float Valence, float Arousal)
{
    /// <summary>A neutral emotion: neither positive nor excited.</summary>
    public static readonly EmotionVector Neutral = new(0f, 0f);

    /// <summary>Pleasantness in the range <c>[-1, 1]</c>.</summary>
    public float Valence { get; } = Math.Clamp(Valence, -1f, 1f);

    /// <summary>Activation in the range <c>[-1, 1]</c>.</summary>
    public float Arousal { get; } = Math.Clamp(Arousal, -1f, 1f);

    /// <summary>Returns a copy with a new valence, clamped to the valid range.</summary>
    public EmotionVector WithValence(float valence) => new(valence, Arousal);

    /// <summary>Returns a copy with a new arousal, clamped to the valid range.</summary>
    public EmotionVector WithArousal(float arousal) => new(Valence, arousal);

    /// <summary>Adds deltas to both components. The result is clamped.</summary>
    public EmotionVector Add(float valenceDelta, float arousalDelta) =>
        new(Valence + valenceDelta, Arousal + arousalDelta);

    /// <summary>Scales both components. The result is clamped.</summary>
    public EmotionVector Scale(float factor) => new(Valence * factor, Arousal * factor);

    /// <summary>Linear interpolation towards <paramref name="other"/>.</summary>
    public EmotionVector Lerp(EmotionVector other, float amount) =>
        new(
            Valence + (other.Valence - Valence) * amount,
            Arousal + (other.Arousal - Arousal) * amount);

    public static EmotionVector operator +(EmotionVector left, EmotionVector right) =>
        new(left.Valence + right.Valence, left.Arousal + right.Arousal);

    public static EmotionVector operator *(EmotionVector value, float factor) =>
        new(value.Valence * factor, value.Arousal * factor);

    public override string ToString() => $"Emotion(V={Valence:0.00}, A={Arousal:0.00})";
}
