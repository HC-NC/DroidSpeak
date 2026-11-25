using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.Neural;

/// <summary>
/// Turns an acoustic <see cref="SyllableFrame"/> back into the 17 normalized <c>[0, 1]</c> values a
/// prosody network must produce to reproduce it.
/// <para>
/// This is the exact inverse of <see cref="SyllableTargetDecoder"/> (and therefore of the decoding
/// <see cref="NeuralProsodyPredictor"/> performs on its hot path), so a knowledge-distillation
/// trainer can convert rule-based teacher frames into ground-truth targets without loss inside the
/// representable range. Every formula here mirrors one decoder branch; keep the two files in sync.
/// </para>
/// <para>
/// Not every frame is reachable: a linear map plus clamps is not injective, some fields are
/// quantized to whole samples, and a profile with a disabled stage silences fields the network
/// cannot then control. Where the forward mapping genuinely cannot produce the frame value, the
/// target falls back to the neutral point of its output (documented per index below) so training
/// data stays inside <c>[0, 1]</c> and never fights the clamps.
/// </para>
/// <para>
/// The method allocates nothing. It is intentionally not on any synthesis hot path; it exists for
/// dataset generation and tests.
/// </para>
/// </summary>
public static class SyllableTargetEncoder
{
    /// <summary>Number of normalized targets, identical to <see cref="NeuralProsodyPredictor.OutputSize"/>.</summary>
    public const int TargetSize = NeuralProsodyPredictor.OutputSize;

    /// <summary>The neutral target for outputs the frame value cannot distinguish.</summary>
    public const float NeutralTarget = 0.5f;

    /// <summary>
    /// Encodes one syllable frame into its 17 network targets.
    /// </summary>
    /// <param name="frame">
    /// The frame to imitate. The model-controlled fields (<c>StartFreq</c>, <c>MidFreq</c>,
    /// <c>EndFreq</c>, <c>LengthSamples</c>, <c>Gain</c>, <c>FmDepth</c>, <c>FmRatio</c>,
    /// <c>GrowlIntensity</c>, <c>NoiseMix</c>, <c>RingModMix</c>, <c>FormantF1</c>, <c>FormantF2</c>,
    /// <c>FormantMix</c>, <c>AttackTime</c>, <c>ReleaseTime</c>, <c>Saturation</c> and
    /// <c>PauseAfterSeconds</c>) are inverted; every other field is procedural and ignored.
    /// </param>
    /// <param name="context">
    /// The syllable's own context. Tempo, punctuation and sample rate shape the duration and pause
    /// inverses exactly the way they shape decoding.
    /// </param>
    /// <param name="profile">Voice profile; normalized internally, like the predictor does.</param>
    /// <param name="targets">
    /// Destination for at least <see cref="TargetSize"/> values, written in the
    /// <see cref="NeuralProsodyPredictor"/> output layout. Every value is inside <c>[0, 1]</c>.
    /// </param>
    /// <exception cref="ArgumentException"><paramref name="targets"/> is too short.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="context"/> carries a sample rate outside the synthesizer's legal range.
    /// </exception>
    public static void EncodeTargets(
        in SyllableFrame frame,
        in SyllableContext context,
        DroidVoiceProfile profile,
        Span<float> targets)
    {
        if (targets.Length < TargetSize)
        {
            throw new ArgumentException(
                $"The target buffer must hold at least {TargetSize} values, but has " +
                $"{targets.Length}.",
                nameof(targets));
        }

        if (context.SampleRate is < 8000 or > 192000)
        {
            throw new ArgumentOutOfRangeException(
                nameof(context),
                context.SampleRate,
                "The syllable context must carry a sample rate between 8000 and 192000 Hz.");
        }

        profile = profile.Normalized();

        targets[NeuralProsodyPredictor.StartFrequencyIndex] =
            FrequencyTarget(frame.StartFreq, profile);
        targets[NeuralProsodyPredictor.MidFrequencyIndex] =
            CurveTarget(frame, profile);
        targets[NeuralProsodyPredictor.EndFrequencyIndex] =
            FrequencyTarget(frame.EndFreq, profile);
        targets[NeuralProsodyPredictor.DurationIndex] =
            DurationTarget(frame, context, profile);
        targets[NeuralProsodyPredictor.GainIndex] =
            InvertLerp(SyllableTargetDecoder.MinGain, SyllableTargetDecoder.MaxGain, frame.Gain);
        targets[NeuralProsodyPredictor.FmDepthIndex] =
            FmDepthTarget(frame.FmDepth, profile);
        targets[NeuralProsodyPredictor.FmRatioIndex] =
            InvertLerp(SyllableTargetDecoder.MinFmRatio, SyllableTargetDecoder.MaxFmRatio, frame.FmRatio);
        targets[NeuralProsodyPredictor.GrowlIndex] =
            NeuralMathF.Clamp01(frame.GrowlIntensity / SyllableTargetDecoder.MaxGrowl);
        targets[NeuralProsodyPredictor.NoiseIndex] =
            NeuralMathF.Clamp01(frame.NoiseMix / SyllableTargetDecoder.MaxNoise);
        targets[NeuralProsodyPredictor.RingModIndex] =
            profile.RingModMix > 0f
                ? NeuralMathF.Clamp01(frame.RingModMix / profile.RingModMix)
                : NeutralTarget;
        targets[NeuralProsodyPredictor.Formant1Index] =
            InvertLerp(
                SyllableTargetDecoder.MinFormantHz,
                SyllableTargetDecoder.FormantCeiling(context.SampleRate) * 0.70f,
                frame.FormantF1);
        targets[NeuralProsodyPredictor.Formant2Index] =
            Formant2Target(frame, context.SampleRate);
        targets[NeuralProsodyPredictor.FormantMixIndex] =
            profile.FormantMix > 0f
                ? NeuralMathF.Clamp01(frame.FormantMix / profile.FormantMix)
                : NeutralTarget;
        targets[NeuralProsodyPredictor.AttackIndex] =
            EnvelopeTarget(frame.AttackTime, profile.AttackSeconds);
        targets[NeuralProsodyPredictor.ReleaseIndex] =
            EnvelopeTarget(frame.ReleaseTime, profile.ReleaseSeconds);
        // Saturation is the canonical source: Drive is derived from the same output by the decoder,
        // and only Saturation carries a plain normalized scale inside [0, MaxSaturation].
        targets[NeuralProsodyPredictor.SaturationIndex] =
            NeuralMathF.Clamp01(frame.Saturation / SyllableTargetDecoder.MaxSaturation);
        targets[NeuralProsodyPredictor.PauseAfterIndex] =
            PauseTarget(frame, context, profile);
    }

    /// <summary>
    /// Maps a frequency back onto the profile band. Values outside the band clamp, because the
    /// decoder clamps on the way out; the predictor never emits an out-of-band frame frequency.
    /// </summary>
    private static float FrequencyTarget(float frequency, DroidVoiceProfile profile) =>
        NeuralMathF.Clamp01(
            (frequency - profile.MinFreq) / (profile.MaxFreq - profile.MinFreq));

    /// <summary>
    /// Inverts the pitch-curve mapping: undo the profile band position, take the base-2 logarithm
    /// of the ratio to the geometric midpoint and re-center <c>[-1, 1]</c> onto <c>[0, 1]</c>.
    /// Without curvature the decoder forces <c>MidFreq</c> to <c>0</c> regardless of the output, so
    /// a frame that carries one is unrepresentable and lands on the neutral point.
    /// </summary>
    private static float CurveTarget(in SyllableFrame frame, DroidVoiceProfile profile)
    {
        if (profile.PitchCurve <= 0f || frame.MidFreq <= 0f)
        {
            return NeutralTarget;
        }

        var geometric = MathF.Sqrt(MathF.Max(1f, frame.StartFreq) * MathF.Max(1f, frame.EndFreq));
        var bipolar = MathF.Log2(frame.MidFreq / geometric) /
            (profile.PitchCurve * SyllableTargetDecoder.MaxCurveOctaves);
        return NeuralMathF.Clamp01((bipolar + 1f) * 0.5f);
    }

    /// <summary>
    /// Inverts the duration mapping: re-derive the seconds the network must have asked for from the
    /// whole-sample length, then undo the tempo and speed scaling. The quantization to samples and
    /// the clamp to the legal synthesis range mean the recovered target lands within a sample of the
    /// forward value, not bit-exactly.
    /// </summary>
    private static float DurationTarget(
        in SyllableFrame frame,
        in SyllableContext context,
        DroidVoiceProfile profile)
    {
        var seconds = frame.LengthSamples / (float)context.SampleRate;
        var baseSeconds = seconds * profile.SpeedMultiplier / context.TempoScale;
        return InvertLerp(
            SyllableTargetDecoder.MinDurationSeconds,
            SyllableTargetDecoder.MaxDurationSeconds,
            baseSeconds);
    }

    /// <summary>
    /// Inverts the FM depth mapping. The profile intensity multiplier is applied before the clamp,
    /// so it is divided back out first. A profile with zero intensity decodes to the depth floor no
    /// matter what the network says; the neutral target keeps such samples from pulling weights.
    /// </summary>
    private static float FmDepthTarget(float depth, DroidVoiceProfile profile)
    {
        if (profile.FmIntensity <= 0f)
        {
            return NeutralTarget;
        }

        return InvertLerp(
            SyllableTargetDecoder.MinFmDepth,
            SyllableTargetDecoder.MaxFmDepth,
            depth / profile.FmIntensity);
    }

    /// <summary>
    /// Inverts the second-formant mapping. The band depends on the decoded first formant, so the
    /// inverse uses the frame's own <see cref="SyllableFrame.FormantF1"/> as the band floor, exactly
    /// as the decoder does. A collapsed band (guard above the ceiling) lands on the neutral point.
    /// </summary>
    private static float Formant2Target(in SyllableFrame frame, int sampleRate)
    {
        var low = frame.FormantF1 + SyllableTargetDecoder.FormantGuardHz;
        var high = MathF.Max(low, SyllableTargetDecoder.FormantCeiling(sampleRate));
        if (high <= low)
        {
            return NeutralTarget;
        }

        return InvertLerp(low, high, frame.FormantF2);
    }

    /// <summary>
    /// Inverts an envelope stage: the output scales the profile seconds across
    /// <c>[EnvelopeScaleMin, EnvelopeScaleMax]</c>, so divide the frame seconds back out. Envelope
    /// shrinking (words too short for the requested ADSR) is a decoder-level effect the same output
    /// passes through, so an already-shrunk frame value simply asks for the same scale the predictor
    /// would have applied before its own shrink.
    /// </summary>
    private static float EnvelopeTarget(float frameSeconds, float profileSeconds)
    {
        var scale = frameSeconds / profileSeconds;
        return InvertLerp(
            SyllableTargetDecoder.EnvelopeScaleMin,
            SyllableTargetDecoder.EnvelopeScaleMax,
            scale);
    }

    /// <summary>
    /// Inverts the pause mapping exactly as <see cref="SyllableTargetDecoder.PauseSeconds"/> applies
    /// it: divide the rendered pause back through the profile speed, the punctuation base pause and
    /// the <c>[PauseScaleMin, PauseScaleMax]</c> lerp. Syllables that are not phrase-final carry no
    /// pause, and the decoder reads their output only at a boundary, so they get the neutral target
    /// (<c>0.5</c> decodes to exactly the rule engine's base pause at unity speed).
    /// </summary>
    private static float PauseTarget(
        in SyllableFrame frame,
        in SyllableContext context,
        DroidVoiceProfile profile)
    {
        if (frame.PauseAfterSeconds <= 0f)
        {
            return NeutralTarget;
        }

        var basePause = context.Punctuation == PhrasePunctuation.Ellipsis
            ? SyllableTargetDecoder.EllipsisPauseSeconds
            : SyllableTargetDecoder.PhrasePauseSeconds;

        var scale = frame.PauseAfterSeconds * profile.SpeedMultiplier / basePause;
        return InvertLerp(
            SyllableTargetDecoder.PauseScaleMin,
            SyllableTargetDecoder.PauseScaleMax,
            scale);
    }

    /// <summary>
    /// Recovers the <c>[0, 1]</c> amount that makes <see cref="NeuralMathF.Lerp"/> produce
    /// <paramref name="value"/>. Outside the band the result clamps: a value a lerp between
    /// <paramref name="from"/> and <paramref name="to"/> can never produce has no representable
    /// target.
    /// </summary>
    private static float InvertLerp(float from, float to, float value) =>
        to > from
            ? NeuralMathF.Clamp01((value - from) / (to - from))
            : NeutralTarget;
}
