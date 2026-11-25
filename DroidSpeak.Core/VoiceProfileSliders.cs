namespace DroidSpeak;

/// <summary>
/// A UI-friendly, quantized view of a <see cref="DroidVoiceProfile"/>: every parameter is stored as
/// an integer (percent, milliseconds or hertz) so it can bind directly to range inputs without
/// decimal-separator issues. The web playground keeps its slider state in this value type, which
/// lets the round trip <see cref="FromProfile"/> → <see cref="ToProfile"/> be tested against the
/// shipped presets, including the v0.5 expressive parameters.
/// </summary>
public struct VoiceProfileSliders
{
    /// <summary>Minimum glide frequency in Hz.</summary>
    public int MinFreq;

    /// <summary>Maximum glide frequency in Hz.</summary>
    public int MaxFreq;

    /// <summary>Global speed multiplier, in percent (100 = 1.00x).</summary>
    public int SpeedPct;

    /// <summary>FM intensity, in percent.</summary>
    public int FmPct;

    /// <summary>Chatter / trill probability, in percent.</summary>
    public int ChatterPct;

    /// <summary>Triangle-wave blend, in percent.</summary>
    public int TrianglePct;

    /// <summary>Word-level attack time in milliseconds.</summary>
    public int AttackMs;

    /// <summary>Word-level decay time in milliseconds.</summary>
    public int DecayMs;

    /// <summary>Word-level sustain level, in percent.</summary>
    public int SustainPct;

    /// <summary>Word-level release time in milliseconds.</summary>
    public int ReleaseMs;

    /// <summary>Filter cutoff multiplier, in percent.</summary>
    public int FilterCutoffPct;

    /// <summary>Filter resonance Q, in percent.</summary>
    public int FilterResonancePct;

    /// <summary>Whistle probability multiplier, in percent.</summary>
    public int WhistlePct;

    /// <summary>Vowel formant bank mix, in percent (v0.5).</summary>
    public int FormantPct;

    /// <summary>Ring modulation mix, in percent (v0.5).</summary>
    public int RingModPct;

    /// <summary>Saturation drive in centiunits, so <c>400</c> means 4.00 (v0.5).</summary>
    public int SaturationCenti;

    /// <summary>Early-reflection room send, in percent (v0.5).</summary>
    public int RoomPct;

    /// <summary>Three-point pitch curvature amount, in percent (v0.5).</summary>
    public int PitchCurvePct;

    /// <summary>Quantizes a profile into slider state, rounding to the supported grid.</summary>
    public static VoiceProfileSliders FromProfile(DroidVoiceProfile profile)
    {
        profile = profile.Normalized();
        return new VoiceProfileSliders
        {
            MinFreq = (int)MathF.Round(profile.MinFreq),
            MaxFreq = (int)MathF.Round(profile.MaxFreq),
            SpeedPct = (int)MathF.Round(profile.SpeedMultiplier * 100f),
            FmPct = (int)MathF.Round(profile.FmIntensity * 100f),
            ChatterPct = (int)MathF.Round(profile.ChatterProbability * 100f),
            TrianglePct = (int)MathF.Round(profile.TriangleMix * 100f),
            AttackMs = (int)MathF.Round(profile.AttackSeconds * 1000f),
            DecayMs = (int)MathF.Round(profile.DecaySeconds * 1000f),
            SustainPct = (int)MathF.Round(profile.SustainLevel * 100f),
            ReleaseMs = (int)MathF.Round(profile.ReleaseSeconds * 1000f),
            FilterCutoffPct = (int)MathF.Round(profile.FilterCutoffMultiplier * 100f),
            FilterResonancePct = (int)MathF.Round(profile.FilterResonance * 100f),
            WhistlePct = (int)MathF.Round(profile.WhistleProbability * 100f),
            FormantPct = (int)MathF.Round(profile.FormantMix * 100f),
            RingModPct = (int)MathF.Round(profile.RingModMix * 100f),
            SaturationCenti = (int)MathF.Round(profile.SaturationDrive * 100f),
            RoomPct = (int)MathF.Round(profile.RoomMix * 100f),
            PitchCurvePct = (int)MathF.Round(profile.PitchCurve * 100f),
        };
    }

    /// <summary>
    /// Rebuilds a normalized profile from slider state. A default-initialized instance carries
    /// zeroed v0.5 parameters, reproducing the legacy v0.4 voice until a preset or imported
    /// JSON enables the expressive layer.
    /// </summary>
    public readonly DroidVoiceProfile ToProfile() => new DroidVoiceProfile
    {
        MinFreq = MinFreq,
        MaxFreq = Math.Max(MinFreq + 50, MaxFreq),
        SpeedMultiplier = SpeedPct / 100f,
        FmIntensity = FmPct / 100f,
        ChatterProbability = ChatterPct / 100f,
        TriangleMix = TrianglePct / 100f,
        AttackSeconds = AttackMs / 1000f,
        DecaySeconds = DecayMs / 1000f,
        SustainLevel = SustainPct / 100f,
        ReleaseSeconds = ReleaseMs / 1000f,
        FilterCutoffMultiplier = FilterCutoffPct / 100f,
        FilterResonance = FilterResonancePct / 100f,
        WhistleProbability = WhistlePct / 100f,
        FormantMix = FormantPct / 100f,
        RingModMix = RingModPct / 100f,
        SaturationDrive = SaturationCenti / 100f,
        RoomMix = RoomPct / 100f,
        PitchCurve = PitchCurvePct / 100f,
    }.Normalized();
}
