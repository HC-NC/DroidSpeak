namespace DroidSpeak.Modem;

/// <summary>
/// Encodes UTF-8 text as a v0.9 rich-spectrum acoustic packet and places it in droid speech
/// without filtering or normalizing the voice.
/// <para>
/// <see cref="ModemMode.HighBandOverlay"/> is the default: the complete audible voice band is
/// preserved and the packet occupies 14.6–17 kHz. <see cref="ModemMode.AudibleChirpStream"/>
/// uses 2.3–4.3 kHz chirps in a natural pause or after the utterance.
/// </para>
/// </summary>
public static class DroidModemEncoder
{
    /// <summary>Default modem amplitude relative to the speech peak.</summary>
    public const float DefaultMaskLevel = 0.15f;

    /// <summary>
    /// Encodes <paramref name="text"/> with a v0.9 rich-spectrum mode.
    /// </summary>
    /// <param name="text">Text to transmit (UTF-8, at most 255 bytes).</param>
    /// <param name="mode">Physical-layer mode.</param>
    /// <param name="sampleRate">Output sample rate in Hz.</param>
    public static float[] Encode(string text, ModemMode mode, int sampleRate = 44100) =>
        AcousticModulator.Encode(text, mode, sampleRate);

    /// <summary>
    /// Encodes <paramref name="text"/> as a clean <see cref="ModemMode.HighBandOverlay"/> packet.
    /// </summary>
    /// <param name="text">Text to transmit (UTF-8, at most 255 bytes).</param>
    /// <param name="sampleRate">Output sample rate in Hz (at least 38000).</param>
    public static float[] Encode(string text, int sampleRate = 44100) =>
        Encode(text, ModemMode.HighBandOverlay, sampleRate);

    /// <summary>
    /// Encodes <paramref name="text"/> with a v0.9 rich-spectrum mode and places it in
    /// <paramref name="speech"/> without filtering or normalizing the voice.
    /// </summary>
    /// <param name="text">Text to transmit (UTF-8, at most 255 bytes).</param>
    /// <param name="speech">Mono speech samples in <c>[-1, 1]</c>.</param>
    /// <param name="mode">Physical-layer mode.</param>
    /// <param name="sampleRate">Sample rate in Hz.</param>
    /// <param name="maskLevel">Relative modem level (0.05–0.30).</param>
    public static float[] EncodeIntoSpeech(
        string text,
        ReadOnlySpan<float> speech,
        ModemMode mode,
        int sampleRate = 44100,
        float maskLevel = DefaultMaskLevel) =>
        AcousticModulator.MixIntoSpeech(text, speech, mode, sampleRate, maskLevel);

    /// <summary>
    /// Encodes <paramref name="text"/> and places it in <paramref name="speech"/> using the
    /// default <see cref="ModemMode.HighBandOverlay"/> mode.
    /// </summary>
    public static float[] EncodeIntoSpeech(
        string text,
        ReadOnlySpan<float> speech,
        int sampleRate = 44100,
        float maskLevel = DefaultMaskLevel) =>
        EncodeIntoSpeech(text, speech, ModemMode.HighBandOverlay, sampleRate, maskLevel);
}
