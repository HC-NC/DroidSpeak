namespace DroidSpeak.Modem;

/// <summary>
/// Selects how a v0.9 acoustic packet shares the channel with droid speech.
/// </summary>
public enum ModemMode
{
    /// <summary>
    /// Keeps the complete voice band untouched and overlays a narrow 4-FSK packet in the
    /// 14.6–17 kHz band. This is the default for rich, full-spectrum audio.
    /// </summary>
    HighBandOverlay = 0,

    /// <summary>
    /// Places shaped 4-FSK chirps in the 2.3–4.3 kHz band, using a natural pause or the
    /// decorative tail after an utterance instead of filtering the voice.
    /// </summary>
    AudibleChirpStream = 1,
}
