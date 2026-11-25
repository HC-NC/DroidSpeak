using System.Reflection;
using DroidSpeak;
using DroidSpeak.DSP;

namespace DroidSpeak.Tests;

// Guards the Web playground's profile <-> slider conversion. The bug these tests pin is that the
// Blazor UI once rebuilt its active profile from only the pre-v0.5 sliders, silently zeroing the
// expressive layer and making web audio byte-identical to v0.4.
public class VoiceProfileSlidersTests
{
    [Theory]
    [InlineData("R2")]
    [InlineData("Grumpy")]
    [InlineData("Bass")]
    [InlineData("Tiny")]
    public void RoundTrip_PreservesV05Parameters(string name)
    {
        var preset = Preset(name);

        var roundTripped = VoiceProfileSliders.FromProfile(preset).ToProfile();

        Assert.Equal(preset.FormantMix, roundTripped.FormantMix, 3);
        Assert.Equal(preset.RingModMix, roundTripped.RingModMix, 3);
        Assert.Equal(preset.SaturationDrive, roundTripped.SaturationDrive, 3);
        Assert.Equal(preset.RoomMix, roundTripped.RoomMix, 3);
        Assert.Equal(preset.PitchCurve, roundTripped.PitchCurve, 3);
    }

    [Theory]
    [InlineData("R2")]
    [InlineData("Grumpy")]
    [InlineData("Bass")]
    [InlineData("Tiny")]
    public void RoundTrip_PreservesLegacyParameters(string name)
    {
        var preset = Preset(name);

        var roundTripped = VoiceProfileSliders.FromProfile(preset).ToProfile();

        Assert.Equal(preset.MinFreq, roundTripped.MinFreq, 1);
        Assert.Equal(preset.MaxFreq, roundTripped.MaxFreq, 1);
        Assert.Equal(preset.SpeedMultiplier, roundTripped.SpeedMultiplier, 3);
        Assert.Equal(preset.FmIntensity, roundTripped.FmIntensity, 3);
        Assert.Equal(preset.ChatterProbability, roundTripped.ChatterProbability, 3);
        Assert.Equal(preset.TriangleMix, roundTripped.TriangleMix, 3);
        Assert.Equal(preset.FilterCutoffMultiplier, roundTripped.FilterCutoffMultiplier, 3);
        Assert.Equal(preset.FilterResonance, roundTripped.FilterResonance, 3);
        Assert.Equal(preset.WhistleProbability, roundTripped.WhistleProbability, 3);
    }

    [Fact]
    public void DefaultSliders_StayOnTheLegacyPath()
    {
        // A fresh customizer must reproduce the neutral (v0.4) voice: every expressive parameter
        // is zero until a preset or imported JSON fills the sliders.
        var profile = default(VoiceProfileSliders).ToProfile();

        Assert.Equal(0f, profile.FormantMix);
        Assert.Equal(0f, profile.RingModMix);
        Assert.Equal(0f, profile.SaturationDrive);
        Assert.Equal(0f, profile.RoomMix);
        Assert.Equal(0f, profile.PitchCurve);
    }

    [Theory]
    [InlineData("R2")]
    [InlineData("Grumpy")]
    [InlineData("Bass")]
    [InlineData("Tiny")]
    public void PresetSliders_ChangeRenderedAudioVersusNeutral(string name)
    {
        const string text = "Beep-boop! Systems online?";
        var viaSliders = VoiceProfileSliders.FromProfile(Preset(name)).ToProfile();

        var rich = new DroidSynthesizer(seed: 0, profile: viaSliders).Synthesize(text);
        var neutral = new DroidSynthesizer(seed: 0, profile: viaSliders with
        {
            FormantMix = 0f,
            RingModMix = 0f,
            SaturationDrive = 0f,
            RoomMix = 0f,
            PitchCurve = 0f,
        }).Synthesize(text);

        Assert.Equal(rich.Length, neutral.Length);
        Assert.False(rich.AsSpan().SequenceEqual(neutral));
    }

    [Fact]
    public void RoundTrip_CoversEveryProfileProperty()
    {
        // The original defect was structural: the UI rebuilt its profile from an enumerated list
        // of sliders, so new properties silently fell back to their neutral defaults. This test
        // walks the actual property list, so any future DroidVoiceProfile parameter must also be
        // represented in VoiceProfileSliders or the suite fails.
        var scratch = new DroidVoiceProfile
        {
            MinFreq = 300f,
            MaxFreq = 1500f,
            SpeedMultiplier = 1.25f,
            FmIntensity = 1.5f,
            ChatterProbability = 0.55f,
            TriangleMix = 0.25f,
            AttackSeconds = 0.012f,
            DecaySeconds = 0.024f,
            SustainLevel = 0.7f,
            ReleaseSeconds = 0.018f,
            FilterCutoffMultiplier = 1.75f,
            FilterResonance = 1.5f,
            WhistleProbability = 1.2f,
            FormantMix = 0.65f,
            RingModMix = 0.35f,
            SaturationDrive = 2.5f,
            RoomMix = 0.45f,
            PitchCurve = 0.75f,
        };

        var roundTripped = VoiceProfileSliders.FromProfile(scratch).ToProfile();

        var properties = typeof(DroidVoiceProfile)
            .GetProperties(BindingFlags.Public | BindingFlags.Instance)
            .Where(p => p.GetIndexParameters().Length == 0)
            .ToArray();

        // Any new profile property (of any type) must be given a slider field; update this count
        // when intentionally adding one.
        Assert.Equal(18, properties.Length);

        foreach (var property in properties)
        {
            Assert.Equal(typeof(float), property.PropertyType);

            var expected = (float)property.GetValue(scratch)!;
            var actual = (float)property.GetValue(roundTripped)!;
            var tolerance = Math.Max(0.005f, Math.Abs(expected) * 0.01f);

            Assert.True(
                MathF.Abs(expected - actual) <= tolerance,
                $"{property.Name} did not survive the slider round trip: " +
                $"expected {expected:0.###}, got {actual:0.###}. " +
                "Add it to VoiceProfileSliders so the Web customizer keeps it.");
        }
    }

    [Theory]
    [InlineData("R2")]
    [InlineData("Grumpy")]
    [InlineData("Bass")]
    [InlineData("Tiny")]
    public void SlidersRenderTheSameAudioAsThePresetObject(string name)
    {
        // The Demo passes the preset object straight to DroidSynthesizer; the Web playground
        // rebuilds it from slider state. Both must render byte-identical audio, which is only
        // true if no v0.5 parameter is dropped along the way.
        const string text = "Beep-boop! Systems online?";
        var preset = Preset(name);
        var viaSliders = VoiceProfileSliders.FromProfile(preset).ToProfile();

        var direct = new DroidSynthesizer(seed: 0, profile: preset).Synthesize(text);
        var rebuilt = new DroidSynthesizer(seed: 0, profile: viaSliders).Synthesize(text);

        Assert.Equal(direct.Length, rebuilt.Length);
        Assert.True(direct.AsSpan().SequenceEqual(rebuilt));
    }

    private static DroidVoiceProfile Preset(string name) => name switch
    {
        "R2" => DroidVoiceProfile.R2,
        "Grumpy" => DroidVoiceProfile.Grumpy,
        "Bass" => DroidVoiceProfile.Bass,
        "Tiny" => DroidVoiceProfile.Tiny,
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown preset."),
    };
}
