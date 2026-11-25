using System.Text.Json;
using System.Text.Json.Serialization;

namespace DroidSpeak;

/// <summary>
/// Describes the timbre and behaviour of a droid voice. Profiles can be created in code or
/// round-tripped through JSON without reflection, which keeps them NativeAOT-friendly.
/// </summary>
public sealed record DroidVoiceProfile
{
    /// <summary>Lowest carrier frequency in Hz.</summary>
    public float MinFreq { get; init; } = 350f;

    /// <summary>Highest carrier frequency in Hz.</summary>
    public float MaxFreq { get; init; } = 1600f;

    /// <summary>
    /// Speech speed multiplier. Values above <c>1</c> speak faster, values below <c>1</c> slower.
    /// </summary>
    public float SpeedMultiplier { get; init; } = 1f;

    /// <summary>Metallic / raspy FM intensity multiplier. <c>1</c> is the classic R2 timbre.</summary>
    public float FmIntensity { get; init; } = 1f;

    /// <summary>
    /// Probability scale for trills, micro-arpeggios and relay clicks (0–1). Higher values make
    /// the droid chatter more.
    /// </summary>
    public float ChatterProbability { get; init; } = 0.50f;

    /// <summary>Share of the triangle wave in the carrier mix (0–1).</summary>
    public float TriangleMix { get; init; } = 0.23f;

    /// <summary>Word-level ADSR attack duration in seconds (0.002–0.100).</summary>
    public float AttackSeconds { get; init; } = 0.010f;

    /// <summary>Word-level ADSR decay duration in seconds (0.005–0.150).</summary>
    public float DecaySeconds { get; init; } = 0.022f;

    /// <summary>Word-level ADSR sustain gain (0.10–1.00).</summary>
    public float SustainLevel { get; init; } = 0.74f;

    /// <summary>Word-level ADSR release duration in seconds (0.005–0.200).</summary>
    public float ReleaseSeconds { get; init; } = 0.016f;

    /// <summary>
    /// Base cutoff multiplier for the dynamic resonant low-pass filter, relative to
    /// <see cref="MaxFreq"/> (0.80–4.00). Higher values keep the voice brighter.
    /// </summary>
    public float FilterCutoffMultiplier { get; init; } = 1.65f;

    /// <summary>Q factor of the resonant low-pass filter (0.50–4.00).</summary>
    public float FilterResonance { get; init; } = 1.45f;

    /// <summary>
    /// Multiplier for the chance of stressed / emotional whistles (0.00–1.50).
    /// <c>0</c> disables whistles completely.
    /// </summary>
    public float WhistleProbability { get; init; } = 1.00f;

    /// <summary>
    /// Strength of the parallel vowel formant bank (0.00–1.00). <c>0</c> keeps the classic
    /// FM-only timbre.
    /// </summary>
    public float FormantMix { get; init; } = 0f;

    /// <summary>
    /// Strength of the inharmonic ring modulator (0.00–1.00). Adds metallic sidebands.
    /// </summary>
    public float RingModMix { get; init; } = 0f;

    /// <summary>
    /// Analog soft-clipping drive applied before the resonant low-pass (0.00–4.00).
    /// <c>0</c> leaves the signal untouched.
    /// </summary>
    public float SaturationDrive { get; init; } = 0f;

    /// <summary>
    /// Early-reflection / micro-reverb send (0.00–1.00). <c>0</c> keeps the voice completely dry.
    /// </summary>
    public float RoomMix { get; init; } = 0f;

    /// <summary>
    /// Amount of three-point pitch curvature the prosody model may add (0.00–1.00).
    /// <c>0</c> keeps straight start-to-end glides.
    /// </summary>
    public float PitchCurve { get; init; } = 0f;

    /// <summary>Classic bright astromech, the default voice.</summary>
    public static DroidVoiceProfile R2 { get; } = new()
    {
        // Expressive vowel formants, a light metallic shimmer and a touch of room ambience.
        FormantMix = 0.55f,
        RingModMix = 0.14f,
        SaturationDrive = 0.55f,
        RoomMix = 0.18f,
        PitchCurve = 0.45f,
    };

    /// <summary>Lower, raspier and slightly slower — a grumpy Chopper-like voice.</summary>
    public static DroidVoiceProfile Grumpy { get; } = new()
    {
        MinFreq = 220f,
        MaxFreq = 1100f,
        SpeedMultiplier = 0.92f,
        FmIntensity = 1.45f,
        ChatterProbability = 0.34f,
        TriangleMix = 0.36f,
        // Darker and more nasal: a lower cutoff with a stronger resonant peak.
        FilterCutoffMultiplier = 1.35f,
        FilterResonance = 1.85f,
        // Heavy ring modulation and drive: a gritty, nasal Chopper-like growl.
        FormantMix = 0.50f,
        RingModMix = 0.42f,
        SaturationDrive = 1.60f,
        RoomMix = 0.26f,
        PitchCurve = 0.30f,
    };

    /// <summary>Heavy, bass-heavy droid with a slow, deliberate cadence.</summary>
    public static DroidVoiceProfile Bass { get; } = new()
    {
        MinFreq = 120f,
        MaxFreq = 650f,
        SpeedMultiplier = 0.82f,
        FmIntensity = 1.20f,
        ChatterProbability = 0.26f,
        TriangleMix = 0.30f,
        // Slow, deliberate swells with a long tail.
        AttackSeconds = 0.020f,
        ReleaseSeconds = 0.030f,
        // Deep, warm saturation with a long room tail and restrained sidebands.
        FormantMix = 0.42f,
        RingModMix = 0.10f,
        SaturationDrive = 2.20f,
        RoomMix = 0.32f,
        PitchCurve = 0.18f,
    };

    /// <summary>Very fast, high-pitched chatterbox.</summary>
    public static DroidVoiceProfile Tiny { get; } = new()
    {
        MinFreq = 600f,
        MaxFreq = 2400f,
        SpeedMultiplier = 1.38f,
        FmIntensity = 1.10f,
        ChatterProbability = 0.78f,
        TriangleMix = 0.16f,
        // Snappy attacks, short releases and extra whistles.
        AttackSeconds = 0.006f,
        ReleaseSeconds = 0.009f,
        WhistleProbability = 1.30f,
        // Bright formants and fast pitch arcs make it sound permanently excited.
        FormantMix = 0.62f,
        RingModMix = 0.26f,
        SaturationDrive = 0.35f,
        RoomMix = 0.10f,
        PitchCurve = 0.85f,
    };

    /// <summary>Serializes this profile to JSON using a source-generated contract.</summary>
    public string ToJson() =>
        JsonSerializer.Serialize(this, DroidVoiceProfileJsonContext.Default.DroidVoiceProfile);

    /// <summary>
    /// Deserializes a profile from JSON using a source-generated contract and validates it.
    /// Properties missing from the document keep their default values, so profiles written by
    /// older versions of the library continue to load unchanged.
    /// </summary>
    /// <exception cref="JsonException">The JSON is malformed or does not describe a profile.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A numeric value is outside its allowed range.</exception>
    public static DroidVoiceProfile FromJson(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);

        var model = JsonSerializer.Deserialize(
                json,
                DroidVoiceProfileJsonContext.Default.DroidVoiceProfileJsonModel)
            ?? throw new JsonException("The JSON document does not contain a voice profile.");

        var defaults = new DroidVoiceProfile();
        var profile = new DroidVoiceProfile
        {
            MinFreq = model.MinFreq ?? defaults.MinFreq,
            MaxFreq = model.MaxFreq ?? defaults.MaxFreq,
            SpeedMultiplier = model.SpeedMultiplier ?? defaults.SpeedMultiplier,
            FmIntensity = model.FmIntensity ?? defaults.FmIntensity,
            ChatterProbability = model.ChatterProbability ?? defaults.ChatterProbability,
            TriangleMix = model.TriangleMix ?? defaults.TriangleMix,
            AttackSeconds = model.AttackSeconds ?? defaults.AttackSeconds,
            DecaySeconds = model.DecaySeconds ?? defaults.DecaySeconds,
            SustainLevel = model.SustainLevel ?? defaults.SustainLevel,
            ReleaseSeconds = model.ReleaseSeconds ?? defaults.ReleaseSeconds,
            FilterCutoffMultiplier = model.FilterCutoffMultiplier ?? defaults.FilterCutoffMultiplier,
            FilterResonance = model.FilterResonance ?? defaults.FilterResonance,
            WhistleProbability = model.WhistleProbability ?? defaults.WhistleProbability,
            FormantMix = model.FormantMix ?? defaults.FormantMix,
            RingModMix = model.RingModMix ?? defaults.RingModMix,
            SaturationDrive = model.SaturationDrive ?? defaults.SaturationDrive,
            RoomMix = model.RoomMix ?? defaults.RoomMix,
            PitchCurve = model.PitchCurve ?? defaults.PitchCurve,
        };

        profile.Validate();
        return profile;
    }

    /// <summary>Returns a copy clamped into safe synthesis ranges.</summary>
    public DroidVoiceProfile Normalized()
    {
        var min = Math.Clamp(MinFreq, 40f, 7000f);
        var max = Math.Clamp(MaxFreq, min + 1f, 9000f);
        var speed = Math.Clamp(SpeedMultiplier, 0.25f, 4f);
        var fm = Math.Clamp(FmIntensity, 0f, 4f);
        var chatter = Math.Clamp(ChatterProbability, 0f, 1f);
        var triangle = Math.Clamp(TriangleMix, 0f, 1f);
        var attack = ClampFinite(AttackSeconds, 0.002f, 0.100f, 0.010f);
        var decay = ClampFinite(DecaySeconds, 0.005f, 0.150f, 0.022f);
        var sustain = ClampFinite(SustainLevel, 0.10f, 1.00f, 0.74f);
        var release = ClampFinite(ReleaseSeconds, 0.005f, 0.200f, 0.016f);
        var cutoff = ClampFinite(FilterCutoffMultiplier, 0.80f, 4.00f, 1.65f);
        var resonance = ClampFinite(FilterResonance, 0.50f, 4.00f, 1.45f);
        var whistle = ClampFinite(WhistleProbability, 0.00f, 1.50f, 1.00f);
        var formant = ClampFinite(FormantMix, 0.00f, 1.00f, 0f);
        var ring = ClampFinite(RingModMix, 0.00f, 1.00f, 0f);
        var saturationDrive = ClampFinite(SaturationDrive, 0.00f, 4.00f, 0f);
        var room = ClampFinite(RoomMix, 0.00f, 1.00f, 0f);
        var pitchCurve = ClampFinite(PitchCurve, 0.00f, 1.00f, 0f);

        // Hot paths call this on every render; avoid a record copy when nothing needs clamping.
        if (min == MinFreq &&
            max == MaxFreq &&
            speed == SpeedMultiplier &&
            fm == FmIntensity &&
            chatter == ChatterProbability &&
            triangle == TriangleMix &&
            attack == AttackSeconds &&
            decay == DecaySeconds &&
            sustain == SustainLevel &&
            release == ReleaseSeconds &&
            cutoff == FilterCutoffMultiplier &&
            resonance == FilterResonance &&
            whistle == WhistleProbability &&
            formant == FormantMix &&
            ring == RingModMix &&
            saturationDrive == SaturationDrive &&
            room == RoomMix &&
            pitchCurve == PitchCurve)
        {
            return this;
        }

        return this with
        {
            MinFreq = min,
            MaxFreq = max,
            SpeedMultiplier = speed,
            FmIntensity = fm,
            ChatterProbability = chatter,
            TriangleMix = triangle,
            AttackSeconds = attack,
            DecaySeconds = decay,
            SustainLevel = sustain,
            ReleaseSeconds = release,
            FilterCutoffMultiplier = cutoff,
            FilterResonance = resonance,
            WhistleProbability = whistle,
            FormantMix = formant,
            RingModMix = ring,
            SaturationDrive = saturationDrive,
            RoomMix = room,
            PitchCurve = pitchCurve,
        };
    }

    private static float ClampFinite(float value, float min, float max, float fallback) =>
        // ±Infinity is clamped to the nearest bound; NaN falls back to the property default.
        float.IsNaN(value) ? fallback : Math.Clamp(value, min, max);

    private void Validate()
    {
        if (!(MinFreq > 0f) || !float.IsFinite(MinFreq))
        {
            throw new ArgumentOutOfRangeException(nameof(MinFreq), MinFreq, "MinFreq must be positive.");
        }

        if (!(MaxFreq > MinFreq) || !float.IsFinite(MaxFreq))
        {
            throw new ArgumentOutOfRangeException(nameof(MaxFreq), MaxFreq, "MaxFreq must exceed MinFreq.");
        }

        if (!(SpeedMultiplier > 0f) || !float.IsFinite(SpeedMultiplier))
        {
            throw new ArgumentOutOfRangeException(
                nameof(SpeedMultiplier),
                SpeedMultiplier,
                "SpeedMultiplier must be positive.");
        }

        if (!(FmIntensity >= 0f) || !float.IsFinite(FmIntensity))
        {
            throw new ArgumentOutOfRangeException(
                nameof(FmIntensity),
                FmIntensity,
                "FmIntensity must be zero or positive.");
        }

        if (!(ChatterProbability >= 0f && ChatterProbability <= 1f))
        {
            throw new ArgumentOutOfRangeException(
                nameof(ChatterProbability),
                ChatterProbability,
                "ChatterProbability must be between 0 and 1.");
        }

        if (!(TriangleMix >= 0f && TriangleMix <= 1f))
        {
            throw new ArgumentOutOfRangeException(
                nameof(TriangleMix),
                TriangleMix,
                "TriangleMix must be between 0 and 1.");
        }

        ValidateRange(AttackSeconds, 0.002f, 0.100f, nameof(AttackSeconds));
        ValidateRange(DecaySeconds, 0.005f, 0.150f, nameof(DecaySeconds));
        ValidateRange(SustainLevel, 0.10f, 1.00f, nameof(SustainLevel));
        ValidateRange(ReleaseSeconds, 0.005f, 0.200f, nameof(ReleaseSeconds));
        ValidateRange(FilterCutoffMultiplier, 0.80f, 4.00f, nameof(FilterCutoffMultiplier));
        ValidateRange(FilterResonance, 0.50f, 4.00f, nameof(FilterResonance));
        ValidateRange(WhistleProbability, 0.00f, 1.50f, nameof(WhistleProbability));
        ValidateRange(FormantMix, 0.00f, 1.00f, nameof(FormantMix));
        ValidateRange(RingModMix, 0.00f, 1.00f, nameof(RingModMix));
        ValidateRange(SaturationDrive, 0.00f, 4.00f, nameof(SaturationDrive));
        ValidateRange(RoomMix, 0.00f, 1.00f, nameof(RoomMix));
        ValidateRange(PitchCurve, 0.00f, 1.00f, nameof(PitchCurve));
    }

    private static void ValidateRange(float value, float min, float max, string name)
    {
        // NaN fails both comparisons and is rejected too.
        if (!(value >= min && value <= max))
        {
            throw new ArgumentOutOfRangeException(
                name,
                value,
                $"{name} must be between {min} and {max}.");
        }
    }
}

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(DroidVoiceProfile))]
[JsonSerializable(typeof(DroidVoiceProfileJsonModel))]
internal sealed partial class DroidVoiceProfileJsonContext : JsonSerializerContext
{
}

/// <summary>
/// Nullable mirror of <see cref="DroidVoiceProfile"/> used for deserialization. Source-generated
/// deserializers pass <c>default</c> for missing members, which would silently zero out
/// properties that were added after the JSON was written; nullable members let the caller apply
/// the real defaults instead.
/// </summary>
internal sealed class DroidVoiceProfileJsonModel
{
    public float? MinFreq { get; init; }

    public float? MaxFreq { get; init; }

    public float? SpeedMultiplier { get; init; }

    public float? FmIntensity { get; init; }

    public float? ChatterProbability { get; init; }

    public float? TriangleMix { get; init; }

    public float? AttackSeconds { get; init; }

    public float? DecaySeconds { get; init; }

    public float? SustainLevel { get; init; }

    public float? ReleaseSeconds { get; init; }

    public float? FilterCutoffMultiplier { get; init; }

    public float? FilterResonance { get; init; }

    public float? WhistleProbability { get; init; }

    public float? FormantMix { get; init; }

    public float? RingModMix { get; init; }

    public float? SaturationDrive { get; init; }

    public float? RoomMix { get; init; }

    public float? PitchCurve { get; init; }
}
