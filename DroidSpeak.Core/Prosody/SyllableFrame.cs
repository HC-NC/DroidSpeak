namespace DroidSpeak.Prosody;

/// <summary>
/// The complete acoustic description of one syllable, produced by an <see cref="IProsodyPredictor"/>
/// and consumed by the DSP renderer.
/// <para>
/// Every ramped parameter keeps an explicit start value next to its target value, because the
/// renderer glides between them. This mirrors the v0.3.0 internal note event exactly, which is what
/// lets the rule-based predictor reproduce the old audio bit-for-bit.
/// </para>
/// <para>
/// <see cref="MidFreq"/> curves the pitch contour, <see cref="RingModMix"/>,
/// <see cref="FormantF1"/>, <see cref="FormantF2"/>, <see cref="FormantMix"/> and
/// <see cref="AttackChirp"/> drive the v0.5 expressive layer. All of them default to zero, which
/// keeps the renderer on the bit-identical v0.4 path.
/// </para>
/// </summary>
public struct SyllableFrame
{
    /// <summary>Absolute start of the syllable in the output buffer, in samples.</summary>
    public int StartSample;

    /// <summary>Syllable length in samples.</summary>
    public int LengthSamples;

    /// <summary>Sounded duration in seconds (<see cref="LengthSamples"/> divided by the sample rate).</summary>
    public float DurationSeconds;

    /// <summary>Silence before the syllable, in samples (phrase, word or hyphen gap).</summary>
    public int PauseBeforeSamples;

    /// <summary>Silence after the syllable, in samples (non-zero only at a phrase boundary).</summary>
    public int PauseAfterSamples;

    /// <summary>Silence before the syllable, in seconds.</summary>
    public float PauseBeforeSeconds;

    /// <summary>Silence after the syllable, in seconds.</summary>
    public float PauseAfterSeconds;

    /// <summary>Glide start frequency in Hz.</summary>
    public float StartFreq;

    /// <summary>
    /// Pitch at the midpoint of the contour, in Hz. <c>0</c> keeps the legacy two-point glide;
    /// a value above or below the geometric mean of <see cref="StartFreq"/> and
    /// <see cref="EndFreq"/> bends the glide into an arc that still starts and ends exactly on
    /// the requested frequencies.
    /// </summary>
    public float MidFreq;

    /// <summary>Glide end frequency in Hz.</summary>
    public float EndFreq;

    /// <summary>Gain at the start of the syllable.</summary>
    public float StartGain;

    /// <summary>Gain at the end of the syllable.</summary>
    public float Gain;

    /// <summary>FM depth at the start of the syllable.</summary>
    public float StartFmDepth;

    /// <summary>FM depth at the end of the syllable.</summary>
    public float FmDepth;

    /// <summary>Second (inharmonic) FM depth at the start of the syllable.</summary>
    public float StartFmDepth2;

    /// <summary>Second (inharmonic) FM depth at the end of the syllable.</summary>
    public float FmDepth2;

    /// <summary>First modulator frequency ratio.</summary>
    public float FmRatio;

    /// <summary>Second modulator frequency ratio.</summary>
    public float FmRatio2;

    /// <summary>Growl modulation depth at the start of the syllable.</summary>
    public float StartGrowlIntensity;

    /// <summary>Growl modulation depth at the end of the syllable.</summary>
    public float GrowlIntensity;

    /// <summary>Growl modulation rate in Hz.</summary>
    public float GrowlFrequency;

    /// <summary>White-noise transient amount for sibilant onsets.</summary>
    public float NoiseMix;

    /// <summary>Reserved for v0.5 ring modulation. Ignored by the v0.4 renderer.</summary>
    public float RingModMix;

    /// <summary>
    /// Signed strength of the 14 ms pitch-attack chirp (for example <c>0.12</c> for a 12 %
    /// upward chirp). <c>0</c> keeps the legacy onset scoop.
    /// </summary>
    public float AttackChirp;

    /// <summary>Reserved for v0.5 formant shaping. Ignored by the v0.4 renderer.</summary>
    public float FormantF1;

    /// <summary>Reserved for v0.5 formant shaping. Ignored by the v0.4 renderer.</summary>
    public float FormantF2;

    /// <summary>Reserved for v0.5 formant shaping. Ignored by the v0.4 renderer.</summary>
    public float FormantMix;

    /// <summary>Word-level attack time in seconds.</summary>
    public float AttackTime;

    /// <summary>Word-level decay time in seconds.</summary>
    public float DecayTime;

    /// <summary>Word-level release time in seconds.</summary>
    public float ReleaseTime;

    /// <summary>Word-level sustain level.</summary>
    public float SustainLevel;

    /// <summary>Trill depth in Hz (<c>0</c> when the syllable does not trill).</summary>
    public float TrillDepth;

    /// <summary>Trill rate in Hz.</summary>
    public float TrillRate;

    /// <summary>Soft-clipping drive.</summary>
    public float Drive;

    /// <summary>Soft-clipping amount in <c>[0, 0.9]</c>.</summary>
    public float Saturation;

    /// <summary>Deterministic seed for the syllable's noise generator.</summary>
    public uint Seed;

    /// <summary>Whether the syllable starts with a voiced consonant.</summary>
    public bool HasVoicedOnset;

    /// <summary>Absolute start of the owning word in the output buffer, in samples.</summary>
    public int WordStartSample;

    /// <summary>Absolute end of the owning word in the output buffer, in samples.</summary>
    public int WordEndSample;

    /// <summary>Whether this frame starts a legato segment (word or hyphen part).</summary>
    public bool StartsLegatoSegment;

    /// <summary>Optional whistle layered on this syllable.</summary>
    public WhistleSpec Whistle;

    /// <summary>Optional chatter burst that fills the gap before this syllable's segment.</summary>
    public ChatterSpec Chatter;
}
