using System.Text.Json;
using DroidSpeak.DSP;

namespace DroidSpeak.Tests;

public class DroidVoiceProfileTests
{
    [Fact]
    public void Presets_AreValidAndWithinSupportedRanges()
    {
        var presets = new[]
        {
            DroidVoiceProfile.R2,
            DroidVoiceProfile.Grumpy,
            DroidVoiceProfile.Bass,
            DroidVoiceProfile.Tiny,
        };

        foreach (var profile in presets)
        {
            Assert.True(profile.MinFreq > 0f);
            Assert.True(profile.MaxFreq > profile.MinFreq);
            Assert.True(profile.SpeedMultiplier > 0f);
            Assert.True(profile.FmIntensity >= 0f);
            Assert.InRange(profile.ChatterProbability, 0f, 1f);
            Assert.InRange(profile.TriangleMix, 0f, 1f);
            Assert.InRange(profile.AttackSeconds, 0.002f, 0.100f);
            Assert.InRange(profile.DecaySeconds, 0.005f, 0.150f);
            Assert.InRange(profile.SustainLevel, 0.10f, 1.00f);
            Assert.InRange(profile.ReleaseSeconds, 0.005f, 0.200f);
            Assert.InRange(profile.FilterCutoffMultiplier, 0.80f, 4.00f);
            Assert.InRange(profile.FilterResonance, 0.50f, 4.00f);
            Assert.InRange(profile.WhistleProbability, 0.00f, 1.50f);
            Assert.InRange(profile.FormantMix, 0.00f, 1.00f);
            Assert.InRange(profile.RingModMix, 0.00f, 1.00f);
            Assert.InRange(profile.SaturationDrive, 0.00f, 4.00f);
            Assert.InRange(profile.RoomMix, 0.00f, 1.00f);
            Assert.InRange(profile.PitchCurve, 0.00f, 1.00f);
        }
    }

    [Fact]
    public void Presets_ShowcaseTheNewEnvelopeFilterAndWhistleParameters()
    {
        Assert.True(
            DroidVoiceProfile.Grumpy.FilterCutoffMultiplier < DroidVoiceProfile.R2.FilterCutoffMultiplier,
            "Grumpy should be darker than R2.");
        Assert.True(
            DroidVoiceProfile.Grumpy.FilterResonance > DroidVoiceProfile.R2.FilterResonance,
            "Grumpy should have a stronger resonant peak than R2.");
        Assert.True(
            DroidVoiceProfile.Bass.AttackSeconds > DroidVoiceProfile.R2.AttackSeconds,
            "Bass should have a slower attack than R2.");
        Assert.True(
            DroidVoiceProfile.Bass.ReleaseSeconds > DroidVoiceProfile.R2.ReleaseSeconds,
            "Bass should have a longer release than R2.");
        Assert.True(
            DroidVoiceProfile.Tiny.AttackSeconds < DroidVoiceProfile.R2.AttackSeconds,
            "Tiny should have a snappier attack than R2.");
        Assert.True(
            DroidVoiceProfile.Tiny.ReleaseSeconds < DroidVoiceProfile.R2.ReleaseSeconds,
            "Tiny should have a shorter release than R2.");
        Assert.True(
            DroidVoiceProfile.Tiny.WhistleProbability > DroidVoiceProfile.R2.WhistleProbability,
            "Tiny should whistle more often than R2.");
    }

    [Fact]
    public void JsonRoundTrip_PreservesEveryField()
    {
        var expected = new DroidVoiceProfile
        {
            MinFreq = 210f,
            MaxFreq = 1234f,
            SpeedMultiplier = 1.17f,
            FmIntensity = 2.35f,
            ChatterProbability = 0.63f,
            TriangleMix = 0.41f,
            AttackSeconds = 0.031f,
            DecaySeconds = 0.044f,
            SustainLevel = 0.55f,
            ReleaseSeconds = 0.052f,
            FilterCutoffMultiplier = 2.20f,
            FilterResonance = 1.90f,
            WhistleProbability = 1.25f,
            FormantMix = 0.62f,
            RingModMix = 0.31f,
            SaturationDrive = 1.70f,
            RoomMix = 0.24f,
            PitchCurve = 0.55f,
        };

        var json = expected.ToJson();
        var actual = DroidVoiceProfile.FromJson(json);

        Assert.Equal(expected, actual);
    }

    [Fact]
    public void ToJson_WritesReadableProperties()
    {
        var json = DroidVoiceProfile.Grumpy.ToJson();

        Assert.Contains("\"MinFreq\"", json);
        Assert.Contains("\"MaxFreq\"", json);
        Assert.Contains("\"SpeedMultiplier\"", json);
        Assert.Contains("\"ChatterProbability\"", json);
        Assert.Contains("\"TriangleMix\"", json);
        Assert.Contains("\"AttackSeconds\"", json);
        Assert.Contains("\"DecaySeconds\"", json);
        Assert.Contains("\"SustainLevel\"", json);
        Assert.Contains("\"ReleaseSeconds\"", json);
        Assert.Contains("\"FilterCutoffMultiplier\"", json);
        Assert.Contains("\"FilterResonance\"", json);
        Assert.Contains("\"WhistleProbability\"", json);
        Assert.Contains("\"FormantMix\"", json);
        Assert.Contains("\"RingModMix\"", json);
        Assert.Contains("\"SaturationDrive\"", json);
        Assert.Contains("\"RoomMix\"", json);
        Assert.Contains("\"PitchCurve\"", json);

        using var document = JsonDocument.Parse(json);
        Assert.Equal(JsonValueKind.Object, document.RootElement.ValueKind);
    }

    [Fact]
    public void FromJson_RejectsMalformedJson()
    {
        Assert.Throws<JsonException>(() => DroidVoiceProfile.FromJson("not json"));
    }

    [Fact]
    public void FromJson_RejectsEmptyInput()
    {
        Assert.Throws<ArgumentException>(() => DroidVoiceProfile.FromJson("   "));
    }

    [Theory]
    [InlineData("{\"MinFreq\":800,\"MaxFreq\":400}")]
    [InlineData("{\"MinFreq\":-10,\"MaxFreq\":400}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"SpeedMultiplier\":0}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"FmIntensity\":-1}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"ChatterProbability\":1.5}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"TriangleMix\":-0.1}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"AttackSeconds\":0.001}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"DecaySeconds\":0.2}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"SustainLevel\":0.05}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"ReleaseSeconds\":0.25}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"FilterCutoffMultiplier\":0.5}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"FilterResonance\":5}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"WhistleProbability\":1.6}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"WhistleProbability\":-0.1}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"FormantMix\":1.1}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"FormantMix\":-0.1}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"RingModMix\":2}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"SaturationDrive\":-1}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"SaturationDrive\":4.5}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"RoomMix\":1.5}")]
    [InlineData("{\"MinFreq\":100,\"MaxFreq\":400,\"PitchCurve\":-0.2}")]
    public void FromJson_RejectsInvalidValues(string json)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => DroidVoiceProfile.FromJson(json));
    }

    [Fact]
    public void FromJson_LegacyDocumentWithoutNewProperties_UsesDefaults()
    {
        // Profiles written before the ADSR/filter/whistle parameters existed must keep working.
        const string legacy = """
            {
              "MinFreq": 220,
              "MaxFreq": 1100,
              "SpeedMultiplier": 0.92,
              "FmIntensity": 1.45,
              "ChatterProbability": 0.34,
              "TriangleMix": 0.36
            }
            """;

        var profile = DroidVoiceProfile.FromJson(legacy);

        Assert.Equal(0.010f, profile.AttackSeconds);
        Assert.Equal(0.022f, profile.DecaySeconds);
        Assert.Equal(0.74f, profile.SustainLevel);
        Assert.Equal(0.016f, profile.ReleaseSeconds);
        Assert.Equal(1.65f, profile.FilterCutoffMultiplier);
        Assert.Equal(1.45f, profile.FilterResonance);
        Assert.Equal(1.00f, profile.WhistleProbability);
        Assert.Equal(0f, profile.FormantMix);
        Assert.Equal(0f, profile.RingModMix);
        Assert.Equal(0f, profile.SaturationDrive);
        Assert.Equal(0f, profile.RoomMix);
        Assert.Equal(0f, profile.PitchCurve);
    }

    [Fact]
    public void Normalized_ClampsOutOfRangeValues()
    {
        var wild = new DroidVoiceProfile
        {
            MinFreq = -100f,
            MaxFreq = 50000f,
            SpeedMultiplier = 20f,
            FmIntensity = -3f,
            ChatterProbability = 2f,
            TriangleMix = -1f,
            AttackSeconds = 0.0001f,
            DecaySeconds = 9f,
            SustainLevel = -1f,
            ReleaseSeconds = 5f,
            FilterCutoffMultiplier = 0.1f,
            FilterResonance = 100f,
            WhistleProbability = 9f,
            FormantMix = 3f,
            RingModMix = -2f,
            SaturationDrive = 9f,
            RoomMix = 4f,
            PitchCurve = -1f,
        };

        var normalized = wild.Normalized();

        Assert.InRange(normalized.MinFreq, 40f, 7000f);
        Assert.True(normalized.MaxFreq > normalized.MinFreq);
        Assert.InRange(normalized.MaxFreq, normalized.MinFreq + 1f, 9000f);
        Assert.InRange(normalized.SpeedMultiplier, 0.25f, 4f);
        Assert.InRange(normalized.FmIntensity, 0f, 4f);
        Assert.InRange(normalized.ChatterProbability, 0f, 1f);
        Assert.InRange(normalized.TriangleMix, 0f, 1f);
        Assert.Equal(0.002f, normalized.AttackSeconds);
        Assert.Equal(0.150f, normalized.DecaySeconds);
        Assert.Equal(0.10f, normalized.SustainLevel);
        Assert.Equal(0.200f, normalized.ReleaseSeconds);
        Assert.Equal(0.80f, normalized.FilterCutoffMultiplier);
        Assert.Equal(4.00f, normalized.FilterResonance);
        Assert.Equal(1.50f, normalized.WhistleProbability);
        Assert.Equal(1.00f, normalized.FormantMix);
        Assert.Equal(0.00f, normalized.RingModMix);
        Assert.Equal(4.00f, normalized.SaturationDrive);
        Assert.Equal(1.00f, normalized.RoomMix);
        Assert.Equal(0.00f, normalized.PitchCurve);
    }

    [Fact]
    public void Normalized_ReplacesNaNWithDefaultsAndClampsInfinity()
    {
        var wild = new DroidVoiceProfile
        {
            AttackSeconds = float.NaN,
            DecaySeconds = float.PositiveInfinity,
            SustainLevel = float.NegativeInfinity,
            ReleaseSeconds = float.NaN,
            FilterCutoffMultiplier = float.PositiveInfinity,
            FilterResonance = float.NaN,
            WhistleProbability = float.NegativeInfinity,
            FormantMix = float.NaN,
            RingModMix = float.PositiveInfinity,
            SaturationDrive = float.NaN,
            RoomMix = float.NegativeInfinity,
            PitchCurve = float.NaN,
        };

        var normalized = wild.Normalized();

        Assert.Equal(0.010f, normalized.AttackSeconds);
        Assert.Equal(0.150f, normalized.DecaySeconds);
        Assert.Equal(0.10f, normalized.SustainLevel);
        Assert.Equal(0.016f, normalized.ReleaseSeconds);
        Assert.Equal(4.00f, normalized.FilterCutoffMultiplier);
        Assert.Equal(1.45f, normalized.FilterResonance);
        Assert.Equal(0.00f, normalized.WhistleProbability);
        Assert.Equal(0.00f, normalized.FormantMix);
        Assert.Equal(1.00f, normalized.RingModMix);
        Assert.Equal(0.00f, normalized.SaturationDrive);
        Assert.Equal(0.00f, normalized.RoomMix);
        Assert.Equal(0.00f, normalized.PitchCurve);
    }

    [Fact]
    public void Profiles_ProduceDistinctAudio()
    {
        const string text = "Бип-буп. Привет!";
        var r2 = new DroidSynthesizer(seed: 5, profile: DroidVoiceProfile.R2).Synthesize(text);
        var tiny = new DroidSynthesizer(seed: 5, profile: DroidVoiceProfile.Tiny).Synthesize(text);

        Assert.False(r2.AsSpan().SequenceEqual(tiny));
    }

    [Fact]
    public void NeutralV05Parameters_LeaveAudioUntouched()
    {
        const string text = "Бип-буп, R2!";
        var classic = new DroidSynthesizer(seed: 9, profile: DroidVoiceProfile.R2);
        var neutral = new DroidSynthesizer(seed: 9, profile: DroidVoiceProfile.R2 with
        {
            FormantMix = 0f,
            RingModMix = 0f,
            SaturationDrive = 0f,
            RoomMix = 0f,
            PitchCurve = 0f,
        });

        Assert.Equal(classic.Synthesize(text), neutral.Synthesize(text));
    }

    [Fact]
    public void RoomMixAndSaturation_ChangeRenderedAudioWithoutChangingLength()
    {
        const string text = "Тревога!";
        var dry = new DroidSynthesizer(seed: 5, profile: DroidVoiceProfile.R2);
        var wet = new DroidSynthesizer(seed: 5, profile: DroidVoiceProfile.R2 with
        {
            RoomMix = 0.65f,
            SaturationDrive = 2.20f,
        });

        var dryAudio = dry.Synthesize(text);
        var wetAudio = wet.Synthesize(text);

        Assert.Equal(dryAudio.Length, wetAudio.Length);
        Assert.False(dryAudio.AsSpan().SequenceEqual(wetAudio));

        foreach (var sample in wetAudio)
        {
            Assert.True(float.IsFinite(sample));
            Assert.InRange(sample, -1f, 1f);
        }
    }

    [Fact]
    public void FasterProfile_ProducesShorterAudio()
    {
        const string text = "Проверка скорости речи";
        var slow = new DroidSynthesizer(seed: 5, profile: DroidVoiceProfile.Grumpy);
        var fast = new DroidSynthesizer(seed: 5, profile: DroidVoiceProfile.Tiny);

        Assert.True(
            fast.GetSampleCount(text) < slow.GetSampleCount(text),
            "A faster profile should produce a shorter utterance.");
    }

    [Fact]
    public void FilterSettings_ChangeRenderedAudio()
    {
        const string text = "Тревога!";
        var dark = new DroidSynthesizer(seed: 5, profile: DroidVoiceProfile.R2 with
        {
            FilterCutoffMultiplier = 0.90f,
            FilterResonance = 0.70f,
        });
        var bright = new DroidSynthesizer(seed: 5, profile: DroidVoiceProfile.R2 with
        {
            FilterCutoffMultiplier = 3.00f,
            FilterResonance = 3.00f,
        });

        var darkAudio = dark.Synthesize(text);
        var brightAudio = bright.Synthesize(text);

        Assert.Equal(darkAudio.Length, brightAudio.Length);
        Assert.False(darkAudio.AsSpan().SequenceEqual(brightAudio));
    }
}
