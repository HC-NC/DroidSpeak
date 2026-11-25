using DroidSpeak.Neural;
using DroidSpeak.Prosody;

namespace DroidSpeak.Trainer.Audio;

/// <summary>
/// Turns one segmented pulse into a <see cref="SyllableFrame"/> whose model-controlled fields already
/// sit on the values <see cref="SyllableTargetDecoder"/> can produce.
/// <para>
/// The extractor never writes a raw measurement into a decoder-controlled field. It measures an
/// amount, pushes it through the matching <see cref="SyllableTargetDecoder"/> method, and keeps the
/// decoded result. That is what makes the frame invertible: the distillation trainer can feed it
/// straight to <see cref="SyllableTargetEncoder"/> and get a target vector whose decode reproduces
/// the frame (<see cref="AcousticContextMapper"/> snaps the quantized duration and pause the same
/// way). A frame with a hand-set gain would encode to a target the network could never realize.
/// </para>
/// <para>
/// Every measurement is a timbre proxy, not a transcription. A droid beep carries no vowel to
/// estimate, so brightness stands in for FM depth, the zero-crossing excess stands in for sibilant
/// noise, and the envelope coefficient of variation stands in for growl. These are the physical
/// quantities that vary with the expressive layer, which is what the network needs to learn.
/// </para>
/// </summary>
internal static class AcousticFeatureExtractor
{
    /// <summary>Fraction-of-peak the attack and release measurement uses as its low knee.</summary>
    private const float EnvelopeLowKnee = 0.10f;

    /// <summary>Fraction-of-peak the attack and release measurement uses as its high knee.</summary>
    private const float EnvelopeHighKnee = 0.90f;

    /// <summary>Brightness ratio that maps to a fully open FM depth.</summary>
    private const float BrightnessFullScale = 0.30f;

    /// <summary>Coefficient of variation that maps to a fully open growl.</summary>
    private const float GrowlFullScale = 0.50f;

    /// <summary>Length of one envelope sub-frame used for the growl estimate, in seconds.</summary>
    private const float GrowlFrameSeconds = 0.005f;

    /// <summary>Crest factor of a pure sine, used as the unsaturated reference.</summary>
    private const float SineCrestFactor = 1.41f;

    /// <summary>
    /// Extracts the acoustic frame of one pulse.
    /// </summary>
    /// <param name="samples">The whole clip's mono samples; only the pulse's span is read.</param>
    /// <param name="sampleRate">Rate of <paramref name="samples"/> in Hz.</param>
    /// <param name="pulse">The pulse to describe.</param>
    /// <param name="profile">Voice profile; normalized internally.</param>
    /// <param name="isSegmentStart">
    /// Whether the pulse opens a legato segment (a word or a hyphen part). Only then is the attack
    /// envelope measured from the audio; later pulses inherit the profile attack.
    /// </param>
    /// <returns>
    /// A frame whose model-controlled fields are all representable by the target decoder. Procedural
    /// fields the mapper owns (<c>Seed</c>, <c>HasVoicedOnset</c>, sample positions, pause) are left
    /// at their defaults.
    /// </returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="sampleRate"/> is not positive.</exception>
    public static SyllableFrame ExtractFrame(
        ReadOnlySpan<float> samples,
        int sampleRate,
        in AcousticPulse pulse,
        DroidVoiceProfile profile,
        bool isSegmentStart)
    {
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Rate must be positive.");
        }

        profile = profile.Normalized();

        var start = Math.Clamp(pulse.StartSample, 0, samples.Length);
        var end = Math.Clamp(pulse.EndSample, start, samples.Length);
        var length = end - start;

        var frame = new SyllableFrame
        {
            StartSample = start,
            LengthSamples = length,
            DurationSeconds = length / (float)sampleRate,
            WordStartSample = start,
            WordEndSample = end,
            StartsLegatoSegment = isSegmentStart,
            DecayTime = profile.DecaySeconds,
            SustainLevel = profile.SustainLevel,
        };

        if (length == 0)
        {
            return frame;
        }

        var slice = samples.Slice(start, length);
        var (lowFrequency, highFrequency) = Band(profile);

        // Pitch is measured on the two halves so a glide inside the pulse is captured; a voiced result
        // outside the profile band is impossible after the clamp, and an unvoiced pulse falls back to
        // the geometric centre of the band.
        var half = length / 2;
        var startFrequency = MeasureFrequency(slice[..half], sampleRate, lowFrequency, highFrequency, profile);
        var endFrequency = MeasureFrequency(slice[half..], sampleRate, lowFrequency, highFrequency, profile);
        var pitchReference = 0.5f * (startFrequency + endFrequency);

        frame.StartFreq = startFrequency;
        frame.EndFreq = endFrequency;

        var rms = Rms(slice);
        var peak = PeakAbsolute(slice);

        var gainAmount = Math.Clamp(((rms * MathF.Sqrt(2f)) - 0.14f) / (0.58f - 0.14f), 0f, 1f);
        frame.Gain = SyllableTargetDecoder.Gain(gainAmount);
        frame.StartGain = frame.Gain;

        var fftSize = TransformSize(length);
        var magnitudes = SpectralAnalyzer.Magnitudes(slice, fftSize);
        var cutoff = MathF.Min(SpectralAnalyzer.DefaultBrightnessCutoffHz, sampleRate * 0.45f);
        var brightness = Math.Clamp(
            SpectralAnalyzer.HighFrequencyRatio(magnitudes, sampleRate, cutoff) / BrightnessFullScale,
            0f,
            1f);

        var fmDepth = SyllableTargetDecoder.FmDepth(brightness, profile);
        frame.FmDepth = fmDepth;
        frame.StartFmDepth = fmDepth;
        frame.FmDepth2 = SyllableTargetDecoder.FmDepth2(fmDepth);
        frame.StartFmDepth2 = frame.FmDepth2;
        frame.FmRatio = SyllableTargetDecoder.FmRatio(brightness);
        frame.FmRatio2 = SyllableTargetDecoder.FmRatio2(frame.FmRatio);

        var zeroCrossing = SpectralAnalyzer.ZeroCrossingRate(slice, sampleRate);
        var noiseThreshold = 2.5f * MathF.Max(1f, pitchReference);
        var noiseAmount = Math.Clamp((zeroCrossing - noiseThreshold) / MathF.Max(1f, noiseThreshold), 0f, 1f);
        frame.NoiseMix = SyllableTargetDecoder.Noise(noiseAmount);

        var growlAmount = Math.Clamp(EnvelopeVariation(slice, sampleRate) / GrowlFullScale, 0f, 1f);
        frame.GrowlIntensity = SyllableTargetDecoder.Growl(growlAmount);
        frame.StartGrowlIntensity = frame.GrowlIntensity;
        frame.GrowlFrequency = 30f;

        frame.RingModMix = SyllableTargetDecoder.RingMod(brightness, profile);

        var formants = SpectralAnalyzer.Formants(magnitudes, sampleRate);
        var formant1Amount = InvertLerp(
            SyllableTargetDecoder.MinFormantHz,
            SyllableTargetDecoder.FormantCeiling(sampleRate) * 0.70f,
            formants.FirstHz);
        frame.FormantF1 = SyllableTargetDecoder.Formant1(formant1Amount, sampleRate);

        var secondLow = frame.FormantF1 + SyllableTargetDecoder.FormantGuardHz;
        var secondHigh = MathF.Max(secondLow, SyllableTargetDecoder.FormantCeiling(sampleRate));
        var formant2Amount = InvertLerp(secondLow, secondHigh, formants.SecondHz);
        frame.FormantF2 = SyllableTargetDecoder.Formant2(formant2Amount, frame.FormantF1, sampleRate);
        frame.FormantMix = SyllableTargetDecoder.FormantMix(Math.Clamp(0.5f + (0.5f * brightness), 0f, 1f), profile);

        var crest = rms > 1e-6f ? peak / rms : SineCrestFactor;
        var saturationAmount = Math.Clamp((SineCrestFactor - crest) / SineCrestFactor, 0f, 1f);
        frame.Saturation = SyllableTargetDecoder.Saturation(saturationAmount);
        frame.Drive = SyllableTargetDecoder.Drive(saturationAmount);

        var attack = isSegmentStart ? MeasureAttack(slice, sampleRate) : profile.AttackSeconds;
        frame.AttackTime = SyllableTargetDecoder.AttackSeconds(
            InvertEnvelope(attack, profile.AttackSeconds),
            profile);

        var release = MeasureRelease(slice, sampleRate);
        frame.ReleaseTime = SyllableTargetDecoder.ReleaseSeconds(
            InvertEnvelope(release, profile.ReleaseSeconds),
            profile);

        return frame;
    }

    /// <summary>Clamps the profile band into the range the YIN estimator can search.</summary>
    private static (float Low, float High) Band(DroidVoiceProfile profile)
    {
        var low = Math.Clamp(
            profile.MinFreq,
            YinPitchTracker.DefaultMinFrequency,
            YinPitchTracker.DefaultMaxFrequency - 1f);
        var high = Math.Clamp(
            profile.MaxFreq,
            low + 1f,
            YinPitchTracker.DefaultMaxFrequency);
        return (low, high);
    }

    /// <summary>Detects a clampable fundamental in a slice, or the band centre when unvoiced.</summary>
    private static float MeasureFrequency(
        ReadOnlySpan<float> slice,
        int sampleRate,
        float low,
        float high,
        DroidVoiceProfile profile)
    {
        var estimate = YinPitchTracker.Detect(slice, sampleRate, minFrequency: low, maxFrequency: high);
        var frequency = estimate.IsVoiced
            ? estimate.FrequencyHz
            : MathF.Sqrt(MathF.Max(1f, profile.MinFreq) * MathF.Max(1f, profile.MaxFreq));

        if (!float.IsFinite(frequency))
        {
            frequency = profile.MinFreq;
        }

        return Math.Clamp(frequency, profile.MinFreq, profile.MaxFreq);
    }

    /// <summary>Root-mean-square of a slice, ignoring non-finite samples.</summary>
    private static float Rms(ReadOnlySpan<float> slice)
    {
        if (slice.IsEmpty)
        {
            return 0f;
        }

        var sum = 0d;
        for (var i = 0; i < slice.Length; i++)
        {
            var value = float.IsFinite(slice[i]) ? slice[i] : 0f;
            sum += value * value;
        }

        return (float)Math.Sqrt(sum / slice.Length);
    }

    /// <summary>Largest absolute sample in a slice, ignoring non-finite samples.</summary>
    private static float PeakAbsolute(ReadOnlySpan<float> slice)
    {
        var peak = 0f;
        for (var i = 0; i < slice.Length; i++)
        {
            var value = float.IsFinite(slice[i]) ? slice[i] : 0f;
            peak = MathF.Max(peak, MathF.Abs(value));
        }

        return peak;
    }

    /// <summary>
    /// Measures the seconds the envelope takes to rise from 10 % to 90 % of its peak.
    /// </summary>
    private static float MeasureAttack(ReadOnlySpan<float> slice, int sampleRate)
    {
        var peak = PeakAbsolute(slice);
        if (peak <= 0f)
        {
            return 0f;
        }

        var low = peak * EnvelopeLowKnee;
        var high = peak * EnvelopeHighKnee;
        var lowIndex = -1;
        var highIndex = -1;

        for (var i = 0; i < slice.Length; i++)
        {
            var value = MathF.Abs(float.IsFinite(slice[i]) ? slice[i] : 0f);
            if (lowIndex < 0 && value >= low)
            {
                lowIndex = i;
            }

            if (value >= high)
            {
                highIndex = i;
                break;
            }
        }

        if (lowIndex < 0 || highIndex < lowIndex)
        {
            return 0f;
        }

        return (highIndex - lowIndex) / (float)sampleRate;
    }

    /// <summary>
    /// Measures the seconds the envelope takes to fall from 90 % to 10 % of its peak at the tail.
    /// </summary>
    private static float MeasureRelease(ReadOnlySpan<float> slice, int sampleRate)
    {
        var peak = PeakAbsolute(slice);
        if (peak <= 0f)
        {
            return 0f;
        }

        var low = peak * EnvelopeLowKnee;
        var high = peak * EnvelopeHighKnee;
        var highIndex = -1;
        var lowIndex = -1;

        for (var i = slice.Length - 1; i >= 0; i--)
        {
            var value = MathF.Abs(float.IsFinite(slice[i]) ? slice[i] : 0f);
            if (highIndex < 0 && value >= high)
            {
                highIndex = i;
            }

            if (value <= low)
            {
                lowIndex = i;
                break;
            }
        }

        if (highIndex < 0 || lowIndex < 0 || highIndex < lowIndex)
        {
            return 0f;
        }

        return (highIndex - lowIndex) / (float)sampleRate;
    }

    /// <summary>
    /// Coefficient of variation of the short-time RMS over the central half of the slice. A steady
    /// pulse reads near zero; a growl-modulated one reads well above it.
    /// </summary>
    private static float EnvelopeVariation(ReadOnlySpan<float> slice, int sampleRate)
    {
        var length = slice.Length;
        var from = length / 4;
        var to = length - from;
        var frame = Math.Max(1, (int)MathF.Round(GrowlFrameSeconds * sampleRate));

        if (to - from < frame)
        {
            return 0f;
        }

        var count = 0;
        var sum = 0d;
        var sumSquares = 0d;

        for (var position = from; position + frame <= to; position += frame)
        {
            var value = Rms(slice.Slice(position, frame));
            sum += value;
            sumSquares += value * value;
            count++;
        }

        if (count < 2)
        {
            return 0f;
        }

        var mean = sum / count;
        if (mean <= 1e-6)
        {
            return 0f;
        }

        var variance = Math.Max(0d, (sumSquares / count) - (mean * mean));
        return (float)(Math.Sqrt(variance) / mean);
    }

    /// <summary>
    /// Largest power of two that is no bigger than the pulse, clipped into the transform's sane range.
    /// </summary>
    private static int TransformSize(int length)
    {
        var size = 16;
        while (size < length && size < SpectralAnalyzer.DefaultFftSize)
        {
            size <<= 1;
        }

        return Math.Min(size, SpectralAnalyzer.DefaultFftSize);
    }

    /// <summary>
    /// Inverts an envelope stage: the decoder scales the profile seconds across
    /// <c>[0.6, 1.4]</c>, so the amount that reproduces <paramref name="seconds"/> divides it back out.
    /// </summary>
    private static float InvertEnvelope(float seconds, float profileSeconds)
    {
        if (!(profileSeconds > 0f))
        {
            return SyllableTargetEncoder.NeutralTarget;
        }

        return Math.Clamp(
            ((seconds / profileSeconds) - SyllableTargetDecoder.EnvelopeScaleMin) /
            (SyllableTargetDecoder.EnvelopeScaleMax - SyllableTargetDecoder.EnvelopeScaleMin),
            0f,
            1f);
    }

    /// <summary>Recovers the <c>[0, 1]</c> amount that makes a lerp produce <paramref name="value"/>.</summary>
    private static float InvertLerp(float from, float to, float value) =>
        to > from
            ? Math.Clamp((value - from) / (to - from), 0f, 1f)
            : SyllableTargetEncoder.NeutralTarget;
}
