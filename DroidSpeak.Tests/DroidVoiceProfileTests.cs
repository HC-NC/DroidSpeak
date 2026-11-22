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
        }
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
    public void FromJson_RejectsInvalidValues(string json)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => DroidVoiceProfile.FromJson(json));
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
        };

        var normalized = wild.Normalized();

        Assert.InRange(normalized.MinFreq, 40f, 7000f);
        Assert.True(normalized.MaxFreq > normalized.MinFreq);
        Assert.InRange(normalized.MaxFreq, normalized.MinFreq + 1f, 9000f);
        Assert.InRange(normalized.SpeedMultiplier, 0.25f, 4f);
        Assert.InRange(normalized.FmIntensity, 0f, 4f);
        Assert.InRange(normalized.ChatterProbability, 0f, 1f);
        Assert.InRange(normalized.TriangleMix, 0f, 1f);
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
    public void FasterProfile_ProducesShorterAudio()
    {
        const string text = "Проверка скорости речи";
        var slow = new DroidSynthesizer(seed: 5, profile: DroidVoiceProfile.Grumpy);
        var fast = new DroidSynthesizer(seed: 5, profile: DroidVoiceProfile.Tiny);

        Assert.True(
            fast.GetSampleCount(text) < slow.GetSampleCount(text),
            "A faster profile should produce a shorter utterance.");
    }
}