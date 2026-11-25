namespace DroidSpeak.Prosody;

/// <summary>
/// A short mechanical chatter burst that fills the silence before a word or hyphen part.
/// <para>
/// The spec is embedded in the first <see cref="SyllableFrame"/> of the segment that follows the
/// gap. <see cref="StartOffsetSamples"/> is therefore negative: the burst begins before the frame.
/// </para>
/// </summary>
public struct ChatterSpec
{
    /// <summary>Whether this frame carries a chatter burst at all.</summary>
    public bool Present;

    /// <summary>Offset from the frame start, in samples. Negative when the burst fills a gap.</summary>
    public int StartOffsetSamples;

    /// <summary>Chatter length in samples.</summary>
    public int LengthSamples;

    /// <summary>Which chatter flavour to render.</summary>
    public ChatterKind Kind;

    /// <summary>First arpeggio step frequency in Hz.</summary>
    public float Frequency1;

    /// <summary>Second arpeggio step frequency in Hz.</summary>
    public float Frequency2;

    /// <summary>Third arpeggio step frequency in Hz.</summary>
    public float Frequency3;

    /// <summary>Number of arpeggio steps (2 or 3).</summary>
    public int StepCount;

    /// <summary>Relay-click glide start frequency in Hz.</summary>
    public float StartFrequency;

    /// <summary>Relay-click glide end frequency in Hz.</summary>
    public float EndFrequency;

    /// <summary>Peak amplitude.</summary>
    public float Amplitude;

    /// <summary>Deterministic noise seed.</summary>
    public uint Seed;
}
