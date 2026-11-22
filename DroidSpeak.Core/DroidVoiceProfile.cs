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

    /// <summary>Classic bright astromech, the default voice.</summary>
    public static DroidVoiceProfile R2 { get; } = new();

    /// <summary>Lower, raspier and slightly slower — a grumpy Chopper-like voice.</summary>
    public static DroidVoiceProfile Grumpy { get; } = new()
    {
        MinFreq = 220f,
        MaxFreq = 1100f,
        SpeedMultiplier = 0.92f,
        FmIntensity = 1.45f,
        ChatterProbability = 0.34f,
        TriangleMix = 0.36f,
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
    };

    /// <summary>Serializes this profile to JSON using a source-generated contract.</summary>
    public string ToJson() =>
        JsonSerializer.Serialize(this, DroidVoiceProfileJsonContext.Default.DroidVoiceProfile);

    /// <summary>
    /// Deserializes a profile from JSON using a source-generated contract and validates it.
    /// </summary>
    /// <exception cref="JsonException">The JSON is malformed or does not describe a profile.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A numeric value is outside its allowed range.</exception>
    public static DroidVoiceProfile FromJson(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);

        var profile = JsonSerializer.Deserialize(
                json,
                DroidVoiceProfileJsonContext.Default.DroidVoiceProfile)
            ?? throw new JsonException("The JSON document does not contain a voice profile.");

        profile.Validate();
        return profile;
    }

    /// <summary>Returns a copy clamped into safe synthesis ranges.</summary>
    public DroidVoiceProfile Normalized()
    {
        var min = Math.Clamp(MinFreq, 40f, 7000f);
        var max = Math.Clamp(MaxFreq, min + 1f, 9000f);

        return this with
        {
            MinFreq = min,
            MaxFreq = max,
            SpeedMultiplier = Math.Clamp(SpeedMultiplier, 0.25f, 4f),
            FmIntensity = Math.Clamp(FmIntensity, 0f, 4f),
            ChatterProbability = Math.Clamp(ChatterProbability, 0f, 1f),
            TriangleMix = Math.Clamp(TriangleMix, 0f, 1f),
        };
    }

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
    }
}

[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(DroidVoiceProfile))]
internal sealed partial class DroidVoiceProfileJsonContext : JsonSerializerContext
{
}
