using DroidSpeak.DSP;
using DroidSpeak.NLP;

namespace DroidSpeak.Tests;

public class DroidChatterTests
{
    private static readonly DroidVoiceProfile Chatty = new()
    {
        MinFreq = 350f,
        MaxFreq = 1600f,
        SpeedMultiplier = 1.2f,
        FmIntensity = 1f,
        ChatterProbability = 1f,
        TriangleMix = 0.23f,
    };

    [Fact]
    public void EmotionalSpeech_ProducesTrills()
    {
        var synthesizer = new DroidSynthesizer(seed: 7, profile: Chatty);
        var plan = synthesizer.BuildPlan(TextParser.Parse("ТРЕВОГА! Перегрев реактора!"));

        Assert.Contains(plan.Events, note => note.TrillDepth > 0f);
        Assert.All(
            plan.Events.Where(note => note.TrillDepth > 0f),
            note => Assert.InRange(note.TrillRate, 27f, 35f));
    }

    [Fact]
    public void FastSpeech_ProducesMicroChatterInGaps()
    {
        var synthesizer = new DroidSynthesizer(seed: 7, profile: Chatty);
        var plan = synthesizer.BuildPlan(TextParser.Parse("ТРЕВОГА! Перегрев реактора!"));

        Assert.NotEmpty(plan.Chatters);
        foreach (var chatter in plan.Chatters)
        {
            // Micro-arpeggios and relay clicks are 15–25 ms long.
            Assert.InRange(chatter.Length, 600, 1105);
            Assert.InRange(chatter.Amplitude, 0.05f, 0.25f);

            if (chatter.Kind == DroidSynthesizer.ChatterKind.Arpeggio)
            {
                Assert.InRange(chatter.StepCount, 2, 3);
                Assert.True(chatter.Frequency1 > 0f);
                Assert.True(chatter.Frequency2 > chatter.Frequency1);
                Assert.True(chatter.Frequency3 > chatter.Frequency2);
            }
            else
            {
                Assert.True(chatter.StartFrequency > chatter.EndFrequency);
            }
        }
    }

    [Fact]
    public void ZeroChatterProbability_DisablesTrillsAndChatter()
    {
        var quiet = Chatty with { ChatterProbability = 0f };
        var synthesizer = new DroidSynthesizer(seed: 7, profile: quiet);
        var plan = synthesizer.BuildPlan(TextParser.Parse("ТРЕВОГА! Перегрев реактора!"));

        Assert.DoesNotContain(plan.Events, note => note.TrillDepth > 0f);
        Assert.Empty(plan.Chatters);
    }

    [Fact]
    public void Profile_KeepsEveryNoteInsideItsFrequencyRange()
    {
        var bass = new DroidSynthesizer(seed: 11, profile: DroidVoiceProfile.Bass);
        var plan = bass.BuildPlan(TextParser.Parse("ТРЕВОГА! Перегрев реактора! Как дела?"));

        Assert.NotEmpty(plan.Events);
        Assert.All(plan.Events, note =>
        {
            Assert.InRange(note.StartFrequency, bass.Profile.MinFreq, bass.Profile.MaxFreq);
            Assert.InRange(note.EndFrequency, bass.Profile.MinFreq, bass.Profile.MaxFreq);
        });
    }

    [Fact]
    public void ChatterEvents_NeverOverlapNotedAudio()
    {
        var synthesizer = new DroidSynthesizer(seed: 21, profile: Chatty);
        var plan = synthesizer.BuildPlan(TextParser.Parse("ТРЕВОГА! Перегрев реактора!"));

        foreach (var chatter in plan.Chatters)
        {
            var chatterEnd = chatter.StartSample + chatter.Length;
            foreach (var note in plan.Events)
            {
                var noteStart = note.StartSample;
                var noteEnd = note.StartSample + note.Length;
                var overlaps = chatter.StartSample < noteEnd && noteStart < chatterEnd;
                Assert.False(overlaps, "Chatter must be placed in a real pause.");
            }
        }
    }
}