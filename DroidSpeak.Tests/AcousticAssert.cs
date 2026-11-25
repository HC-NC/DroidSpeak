using DroidSpeak.Prosody;

namespace DroidSpeak.Tests;

/// <summary>
/// Field-by-field assertions for the two large value types the acoustic pipeline produces. Comparing
/// every field explicitly keeps a broken round-trip from hiding behind a struct's default equality,
/// which would ignore fields the serializer forgot to persist.
/// </summary>
internal static class AcousticAssert
{
    public static void ContextEqual(in SyllableContext expected, in SyllableContext actual)
    {
        Assert.Equal(expected.SampleRate, actual.SampleRate);
        Assert.Equal(expected.GlobalSeed, actual.GlobalSeed);
        Assert.Equal(expected.WordHash, actual.WordHash);
        Assert.Equal(expected.PhraseIndex, actual.PhraseIndex);
        Assert.Equal(expected.WordIndexInPhrase, actual.WordIndexInPhrase);
        Assert.Equal(expected.SyllableIndex, actual.SyllableIndex);
        Assert.Equal(expected.WordSyllableCount, actual.WordSyllableCount);
        Assert.Equal(expected.PhraseSyllableCount, actual.PhraseSyllableCount);
        Assert.Equal(expected.WordTextLength, actual.WordTextLength);
        Assert.Equal(expected.TempoScale, actual.TempoScale);
        Assert.Equal(expected.EmotionValence, actual.EmotionValence);
        Assert.Equal(expected.EmotionArousal, actual.EmotionArousal);
        Assert.Equal(expected.Punctuation, actual.Punctuation);
        Assert.Equal(expected.IsFirstSpokenPhrase, actual.IsFirstSpokenPhrase);
        Assert.Equal(expected.PreviousWordHadComma, actual.PreviousWordHadComma);
        Assert.Equal(expected.IsShout, actual.IsShout);
        Assert.Equal(expected.IsStressed, actual.IsStressed);
        Assert.Equal(expected.HasVoicedOnset, actual.HasVoicedOnset);
        Assert.Equal(expected.HasSibilantOnset, actual.HasSibilantOnset);
        Assert.Equal(expected.IsHyphenBoundary, actual.IsHyphenBoundary);
        Assert.Equal(expected.Nucleus, actual.Nucleus);
    }

    public static void FrameEqual(in SyllableFrame expected, in SyllableFrame actual)
    {
        Assert.Equal(expected.StartSample, actual.StartSample);
        Assert.Equal(expected.LengthSamples, actual.LengthSamples);
        Assert.Equal(expected.DurationSeconds, actual.DurationSeconds);
        Assert.Equal(expected.PauseBeforeSamples, actual.PauseBeforeSamples);
        Assert.Equal(expected.PauseAfterSamples, actual.PauseAfterSamples);
        Assert.Equal(expected.PauseBeforeSeconds, actual.PauseBeforeSeconds);
        Assert.Equal(expected.PauseAfterSeconds, actual.PauseAfterSeconds);
        Assert.Equal(expected.StartFreq, actual.StartFreq);
        Assert.Equal(expected.MidFreq, actual.MidFreq);
        Assert.Equal(expected.EndFreq, actual.EndFreq);
        Assert.Equal(expected.StartGain, actual.StartGain);
        Assert.Equal(expected.Gain, actual.Gain);
        Assert.Equal(expected.StartFmDepth, actual.StartFmDepth);
        Assert.Equal(expected.FmDepth, actual.FmDepth);
        Assert.Equal(expected.StartFmDepth2, actual.StartFmDepth2);
        Assert.Equal(expected.FmDepth2, actual.FmDepth2);
        Assert.Equal(expected.FmRatio, actual.FmRatio);
        Assert.Equal(expected.FmRatio2, actual.FmRatio2);
        Assert.Equal(expected.StartGrowlIntensity, actual.StartGrowlIntensity);
        Assert.Equal(expected.GrowlIntensity, actual.GrowlIntensity);
        Assert.Equal(expected.GrowlFrequency, actual.GrowlFrequency);
        Assert.Equal(expected.NoiseMix, actual.NoiseMix);
        Assert.Equal(expected.RingModMix, actual.RingModMix);
        Assert.Equal(expected.AttackChirp, actual.AttackChirp);
        Assert.Equal(expected.FormantF1, actual.FormantF1);
        Assert.Equal(expected.FormantF2, actual.FormantF2);
        Assert.Equal(expected.FormantMix, actual.FormantMix);
        Assert.Equal(expected.AttackTime, actual.AttackTime);
        Assert.Equal(expected.DecayTime, actual.DecayTime);
        Assert.Equal(expected.ReleaseTime, actual.ReleaseTime);
        Assert.Equal(expected.SustainLevel, actual.SustainLevel);
        Assert.Equal(expected.TrillDepth, actual.TrillDepth);
        Assert.Equal(expected.TrillRate, actual.TrillRate);
        Assert.Equal(expected.Drive, actual.Drive);
        Assert.Equal(expected.Saturation, actual.Saturation);
        Assert.Equal(expected.Seed, actual.Seed);
        Assert.Equal(expected.HasVoicedOnset, actual.HasVoicedOnset);
        Assert.Equal(expected.WordStartSample, actual.WordStartSample);
        Assert.Equal(expected.WordEndSample, actual.WordEndSample);
        Assert.Equal(expected.StartsLegatoSegment, actual.StartsLegatoSegment);
    }
}