using DroidSpeak.Music;

namespace DroidSpeak.Tests;

public sealed class MusicTests
{
    private const string SampleRtttl =
        "OdeToJoy:d=4,o=5,b=125:e,e,f,g,g,f,e,d,c,c,d,e,e.,8d,2d";

    [Fact]
    public void RtttlParser_ParsesValidString()
    {
        var score = RtttlParser.Shared.Parse(SampleRtttl);

        Assert.Equal("OdeToJoy", score.Title);
        Assert.Equal(15, score.Notes.Count);
        Assert.True(score.TotalDurationSeconds > 0f);
    }

    [Fact]
    public void MidiParser_ParsesValidSmfHeaderAndNotes()
    {
        // Minimal Format 0 MIDI: 1 track, 480 ticks/quarter, 120 BPM, single C5 quarter note
        byte[] midiBytes =
{
            0x4D, 0x54, 0x68, 0x64,  0x00, 0x00, 0x00, 0x06,
            0x00, 0x00,  0x00, 0x01,  0x01, 0xE0,
            0x4D, 0x54, 0x72, 0x6B,  0x00, 0x00, 0x00, 0x0D,
            0x00, 0x90, 0x48, 0x64,
            0x83, 0x60, 0x80, 0x48, 0x00,
            0x00, 0xFF, 0x2F, 0x00
        };

        var score = MidiParser.Shared.Parse(midiBytes);

        var note = Assert.Single(score.Notes);
        Assert.Equal(0.5f, note.DurationSeconds, precision: 3);
        Assert.InRange(note.FrequencyHz, 523.0f, 523.5f);
    }

    [Fact]
    public void DroidMusicSynthesizer_RendersValidAudioAndSupportsSoloMode()
    {
        var score = new DroidScore(
        [
            new ScoreNote(0.0f, 0.5f, 523.25f, 0.9f), // Lead note (C5)
            new ScoreNote(0.0f, 0.5f, 261.63f, 0.8f), // Harmony note (C4)
        ]);

        var synth = new DroidMusicSynthesizer(profile: DroidVoiceProfile.R2);

        int expectedSamples = synth.GetSampleCount(score);
        float[] polyphonic = synth.Render(score, soloMelodyOnly: false);
        float[] solo = synth.Render(score, soloMelodyOnly: true);

        Assert.Equal(expectedSamples, polyphonic.Length);
        Assert.Equal(expectedSamples, solo.Length);
        Assert.NotEqual(polyphonic, solo);

        Assert.All(solo, s =>
        {
            Assert.True(float.IsFinite(s));
            Assert.InRange(s, -1f, 1f);
        });

        Span<float> spanBuffer = new float[expectedSamples];
        int written = synth.Render(score, spanBuffer, soloMelodyOnly: true);
        Assert.Equal(expectedSamples, written);
        Assert.Equal(solo, spanBuffer.ToArray());
    }
}
