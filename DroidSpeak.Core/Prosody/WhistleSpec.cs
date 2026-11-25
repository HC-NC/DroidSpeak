namespace DroidSpeak.Prosody;

/// <summary>
/// A short high whistle or sob layered on top of a syllable.
/// <para>
/// The spec is embedded in the <see cref="SyllableFrame"/> that owns the syllable, so the renderer
/// can schedule it without a second list. <see cref="StartOffsetSamples"/> is relative to the
/// frame's <see cref="SyllableFrame.StartSample"/> and is always positive for whistles.
/// </para>
/// </summary>
public struct WhistleSpec
{
    /// <summary>Whether this frame carries a whistle at all.</summary>
    public bool Present;

    /// <summary>Offset from the frame start, in samples.</summary>
    public int StartOffsetSamples;

    /// <summary>Whistle length in samples.</summary>
    public int LengthSamples;

    /// <summary>Glide start frequency in Hz.</summary>
    public float StartFrequency;

    /// <summary>Glide end frequency in Hz.</summary>
    public float EndFrequency;

    /// <summary>Peak amplitude.</summary>
    public float Amplitude;

    /// <summary>Vibrato depth in Hz.</summary>
    public float VibratoDepth;

    /// <summary>Vibrato rate in Hz.</summary>
    public float VibratoRate;

    /// <summary>Sob (slow tremolo) depth.</summary>
    public float SobDepth;

    /// <summary>Deterministic noise seed.</summary>
    public uint Seed;
}
