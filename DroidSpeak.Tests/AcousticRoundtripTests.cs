using DroidSpeak.DSP;
using DroidSpeak.Neural;
using DroidSpeak.Prosody;
using DroidSpeak.Trainer.Audio;

namespace DroidSpeak.Tests;

/// <summary>
/// End-to-end checks that a rendered, re-read and segmented clip becomes frames the target encoder
/// can invert: every target lands in <c>[0, 1]</c> and decoding those targets reproduces the frame.
/// </summary>
public class AcousticRoundtripTests
{
    [Theory]
    [InlineData(22050)]
    [InlineData(44100)]
    public void ExtractedTargetsRoundTripThroughTheDecoder(int sampleRate)
    {
        var path = Path.Combine(
            Path.GetTempPath(),
            $"droidspeak-roundtrip-{sampleRate}-{Guid.NewGuid():N}.wav");

        try
        {
            var synthesizer = new DroidSynthesizer(sampleRate, seed: 0, DroidVoiceProfile.R2);
            var audio = synthesizer.Synthesize("beep. boop.");
            WavWriter.Write(path, audio, sampleRate);

            // The reader canonicalizes to 22050 Hz, so the render rate and the analysis rate differ
            // whenever the theory runs at 44100; the mapper has to bridge them.
            var clip = WavReader.Read(path);
            var segmentation = PulseSegmenter.Segment(clip.Samples, clip.SampleRate);
            var (contexts, frames) = AcousticContextMapper.Extract(
                clip,
                segmentation,
                DroidVoiceProfile.R2,
                "beep.boop.wav",
                sampleRate,
                seed: 5u);

            Assert.NotEmpty(contexts);
            Assert.Equal(contexts.Length, frames.Length);

            var profile = DroidVoiceProfile.R2;
            for (var i = 0; i < contexts.Length; i++)
            {
                var context = contexts[i];
                var frame = frames[i];
                var targets = new float[SyllableTargetEncoder.TargetSize];
                SyllableTargetEncoder.EncodeTargets(frame, context, profile, targets);

                foreach (var value in targets)
                {
                    Assert.InRange(value, 0f, 1f);
                }

                SyllableTargetDecoder.DecodeTargets(targets, context, profile, out var decoded);
                AssertMatches(frame, decoded);
            }
        }
        finally
        {
            if (File.Exists(path))
            {
                File.Delete(path);
            }
        }
    }

    /// <summary>Asserts the decoder reproduces every model-controlled field of the frame.</summary>
    private static void AssertMatches(in SyllableFrame frame, in PredictedFrame decoded)
    {
        Assert.Equal(frame.LengthSamples, decoded.LengthSamples);

        Close(frame.StartFreq, decoded.StartFrequency, 0.05f);
        Close(frame.EndFreq, decoded.EndFrequency, 0.05f);
        Close(frame.Gain, decoded.Gain, 1e-3f);
        Close(frame.FmDepth, decoded.FmDepth, 1e-3f);
        Close(frame.FmRatio, decoded.FmRatio, 1e-3f);
        Close(frame.GrowlIntensity, decoded.Growl, 1e-3f);
        Close(frame.NoiseMix, decoded.Noise, 1e-3f);
        Close(frame.RingModMix, decoded.RingMod, 1e-3f);
        Close(frame.FormantF1, decoded.Formant1, 0.05f);
        Close(frame.FormantF2, decoded.Formant2, 0.05f);
        Close(frame.FormantMix, decoded.FormantBlend, 1e-3f);
        Close(frame.AttackTime, decoded.AttackSeconds, 1e-5f);
        Close(frame.ReleaseTime, decoded.ReleaseSeconds, 1e-5f);
        Close(frame.Saturation, decoded.Saturation, 1e-3f);

        if (frame.PauseAfterSamples > 0)
        {
            // One sample of slack covers the round of the pause back onto the sample grid.
            Assert.InRange(
                decoded.PauseAfterSamples,
                frame.PauseAfterSamples - 1,
                frame.PauseAfterSamples + 1);
        }
    }

    /// <summary>Asserts a value matches within a relative-or-absolute tolerance.</summary>
    private static void Close(float expected, float actual, float absoluteTolerance)
    {
        var tolerance = MathF.Max(absoluteTolerance, MathF.Abs(expected) * 0.01f);
        Assert.InRange(actual, expected - tolerance, expected + tolerance);
    }
}
