namespace DroidSpeak.Modem;

/// <summary>
/// A successfully received acoustic packet plus the physical-layer diagnostics that help tune
/// a real speaker/microphone link.
/// </summary>
/// <param name="Text">Decoded UTF-8 payload.</param>
/// <param name="Mode">Physical-layer mode that produced the packet.</param>
/// <param name="CorrectedBits">Number of Hamming syndromes that corrected a bit.</param>
/// <param name="CrcValid">Whether CRC-16 accepted the packet. Successful results are always true.</param>
/// <param name="ClockDriftPpm">Estimated sample-clock error in parts per million.</param>
/// <param name="SamplesConsumed">Samples consumed from the beginning of the fed buffer.</param>
/// <param name="Legacy">True for a v0.8 binary-FSK packet.</param>
public readonly record struct ModemDecodeResult(
    string Text,
    ModemMode Mode,
    int CorrectedBits,
    bool CrcValid,
    double ClockDriftPpm,
    int SamplesConsumed,
    bool Legacy);
