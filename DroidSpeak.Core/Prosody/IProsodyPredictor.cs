namespace DroidSpeak.Prosody;

/// <summary>
/// Turns the linguistic description of an utterance into per-syllable acoustic frames.
/// <para>
/// Implementations must be deterministic: the same contexts and profile must always produce the
/// same frames. The default implementation is <c>RuleBasedProsodyPredictor</c>, which reproduces
/// the v0.3.0 procedural voice exactly.
/// </para>
/// </summary>
public interface IProsodyPredictor
{
    /// <summary>
    /// Predicts one frame per context.
    /// </summary>
    /// <param name="contexts">Syllables in speaking order.</param>
    /// <param name="frames">
    /// Destination for the predicted frames. Must be at least as long as <paramref name="contexts"/>;
    /// only the first <c>contexts.Length</c> entries are written.
    /// </param>
    /// <param name="profile">Normalized voice profile.</param>
    void Predict(
        ReadOnlySpan<SyllableContext> contexts,
        Span<SyllableFrame> frames,
        DroidVoiceProfile profile);
}
