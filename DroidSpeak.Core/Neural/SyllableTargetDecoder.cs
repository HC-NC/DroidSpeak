using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.Neural;

/// <summary>
/// The forward mapping from the 17 normalized network outputs of <see cref="NeuralProsodyPredictor"/>
/// to concrete syllable acoustics, plus the decoded, post-processing-free snapshot of one target
/// vector.
/// <para>
/// Every scalar mapping the neural predictor uses to turn a network output into a frame field lives
/// here, so <see cref="SyllableTargetEncoder"/> can invert exactly the same formulas instead of
/// shadow-copying constants. A trainer that produced targets against a different mapping would teach
/// the network the wrong voice.
/// </para>
/// <para>
/// The methods are pure and allocation free; the ranges mirror the constants of
/// <see cref="NeuralProsodyPredictor"/>, and the predictor delegates to them on its hot path.
/// </para>
/// </summary>
public static class SyllableTargetDecoder
{
    /// <summary>Shortest syllable the duration output can request, in seconds.</summary>
    public const float MinDurationSeconds = 0.045f;

    /// <summary>Longest syllable the duration output can request, in seconds.</summary>
    public const float MaxDurationSeconds = 0.50f;

    /// <summary>Quietest gain the gain output can request.</summary>
    public const float MinGain = 0.18f;

    /// <summary>Loudest gain the gain output can request.</summary>
    public const float MaxGain = 0.62f;

    /// <summary>Lowest FM depth (before the profile intensity) the depth output can request.</summary>
    public const float MinFmDepth = 0.20f;

    /// <summary>Highest FM depth (before the profile intensity) the depth output can request.</summary>
    public const float MaxFmDepth = 6.50f;

    /// <summary>Second modulator depth as a fraction of the first.</summary>
    public const float FmDepth2Scale = 0.40f;

    /// <summary>Lowest modulator ratio the ratio output can request.</summary>
    public const float MinFmRatio = 1.00f;

    /// <summary>Highest modulator ratio the ratio output can request.</summary>
    public const float MaxFmRatio = 3.50f;

    /// <summary>Second modulator ratio as a multiple of the first, tuned to the classic 3.07 Hz timbre.</summary>
    public const float FmRatio2Scale = 3.07f / 2f;

    /// <summary>Growl depth at a fully open growl output.</summary>
    public const float MaxGrowl = 0.35f;

    /// <summary>Sibilant noise at a fully open noise output.</summary>
    public const float MaxNoise = 0.25f;

    /// <summary>Soft-clip drive at a closed saturation output.</summary>
    public const float MinDrive = 0.80f;

    /// <summary>Soft-clip drive at a fully open saturation output.</summary>
    public const float MaxDrive = 3.40f;

    /// <summary>Soft-clip amount at a fully open saturation output.</summary>
    public const float MaxSaturation = 0.72f;

    /// <summary>Lowest attack time the attack output can request, in seconds.</summary>
    public const float AttackSecondsFloor = 0.002f;

    /// <summary>Highest attack time the attack output can request, in seconds.</summary>
    public const float AttackSecondsCeiling = 0.100f;

    /// <summary>Lowest release time the release output can request, in seconds.</summary>
    public const float ReleaseSecondsFloor = 0.005f;

    /// <summary>Highest release time the release output can request, in seconds.</summary>
    public const float ReleaseSecondsCeiling = 0.200f;

    /// <summary>Envelope scale applied to the profile value at a closed envelope output.</summary>
    public const float EnvelopeScaleMin = 0.60f;

    /// <summary>Envelope scale applied to the profile value at a fully open envelope output.</summary>
    public const float EnvelopeScaleMax = 1.40f;

    /// <summary>Statement or question pause the pause output scales, in seconds.</summary>
    public const float PhrasePauseSeconds = 0.12f;

    /// <summary>Ellipsis pause the pause output scales, in seconds.</summary>
    public const float EllipsisPauseSeconds = 0.18f;

    /// <summary>Pause scale at a closed pause output.</summary>
    public const float PauseScaleMin = 0.50f;

    /// <summary>Pause scale at a fully open pause output.</summary>
    public const float PauseScaleMax = 1.50f;

    /// <summary>Lowest formant frequency the formant outputs can request, in Hz.</summary>
    public const float MinFormantHz = 150f;

    /// <summary>Minimum separation kept between the two formants, in Hz.</summary>
    public const float FormantGuardHz = 100f;

    /// <summary>Largest pitch bend, in octaves, the curve output can request at full curvature.</summary>
    public const float MaxCurveOctaves = 0.16f;

    /// <summary>
    /// Maps a normalized frequency amount into the profile band, clamped so the carrier can never
    /// leave the voice's legal range.
    /// </summary>
    public static float Frequency(float amount, DroidVoiceProfile profile) =>
        Math.Clamp(
            profile.MinFreq + NeuralMathF.Clamp01(amount) * (profile.MaxFreq - profile.MinFreq),
            profile.MinFreq,
            profile.MaxFreq);

    /// <summary>
    /// Places the contour midpoint between the two endpoints of the glide. The amount selects how
    /// far the arc bulges around the geometric mean; a profile without curvature yields <c>0</c>,
    /// which keeps the two-point legacy glide.
    /// </summary>
    public static float CurveCenter(
        float amount,
        float startFrequency,
        float endFrequency,
        DroidVoiceProfile profile)
    {
        if (profile.PitchCurve <= 0f)
        {
            return 0f;
        }

        var geometric = MathF.Sqrt(MathF.Max(1f, startFrequency) * MathF.Max(1f, endFrequency));
        var bipolar = (NeuralMathF.Clamp01(amount) * 2f) - 1f;
        return Math.Clamp(
            geometric * MathF.Pow(2f, bipolar * profile.PitchCurve * MaxCurveOctaves),
            profile.MinFreq,
            profile.MaxFreq);
    }

    /// <summary>
    /// Decodes the duration output into seconds, scaled by the phrase tempo and the profile speed,
    /// then clamped into the range the synthesizer can render.
    /// </summary>
    public static float DurationSeconds(float amount, float tempoScale, DroidVoiceProfile profile)
    {
        var seconds = NeuralMathF.Lerp(MinDurationSeconds, MaxDurationSeconds, amount);
        return Math.Clamp(
            seconds * tempoScale / profile.SpeedMultiplier,
            MinDurationSeconds,
            MaxDurationSeconds);
    }

    /// <summary>Decodes the gain output into the audible gain band.</summary>
    public static float Gain(float amount) => NeuralMathF.Lerp(MinGain, MaxGain, amount);

    /// <summary>
    /// Decodes the FM depth output, folds in the profile intensity and keeps the result inside the
    /// stable modulation range.
    /// </summary>
    public static float FmDepth(float amount, DroidVoiceProfile profile) =>
        Math.Clamp(
            NeuralMathF.Lerp(MinFmDepth, MaxFmDepth, amount) * profile.FmIntensity,
            MinFmDepth,
            MaxFmDepth);

    /// <summary>Decodes the modulator ratio output.</summary>
    public static float FmRatio(float amount) => NeuralMathF.Lerp(MinFmRatio, MaxFmRatio, amount);

    /// <summary>Second (inharmonic) modulator ratio derived from the decoded first one.</summary>
    public static float FmRatio2(float ratio) => ratio * FmRatio2Scale;

    /// <summary>Second modulator depth derived from the decoded first one.</summary>
    public static float FmDepth2(float depth) => depth * FmDepth2Scale;

    /// <summary>Decodes the growl output into a modulation depth.</summary>
    public static float Growl(float amount) => NeuralMathF.Clamp01(amount) * MaxGrowl;

    /// <summary>Decodes the noise output into a white-noise transient amount.</summary>
    public static float Noise(float amount) => NeuralMathF.Clamp01(amount) * MaxNoise;

    /// <summary>Decodes the ring-modulation output; silent when the profile disables the stage.</summary>
    public static float RingMod(float amount, DroidVoiceProfile profile) =>
        profile.RingModMix > 0f
            ? Math.Clamp(profile.RingModMix * NeuralMathF.Clamp01(amount), 0f, 1f)
            : 0f;

    /// <summary>Upper edge of the whole formant band for a sample rate.</summary>
    public static float FormantCeiling(int sampleRate) => MathF.Max(400f, sampleRate * 0.42f);

    /// <summary>Decodes the first-formant output; the band tops out at 70% of the ceiling.</summary>
    public static float Formant1(float amount, int sampleRate)
    {
        var ceiling = FormantCeiling(sampleRate) * 0.70f;
        return Math.Clamp(
            NeuralMathF.Lerp(MinFormantHz, ceiling, amount),
            MinFormantHz,
            ceiling);
    }

    /// <summary>
    /// Decodes the second-formant output. The band starts one guard distance above the decoded first
    /// formant, so the network cannot ask the two formants to collide.
    /// </summary>
    public static float Formant2(float amount, float firstFormant, int sampleRate)
    {
        var low = firstFormant + FormantGuardHz;
        var high = MathF.Max(low, FormantCeiling(sampleRate));
        return Math.Clamp(NeuralMathF.Lerp(low, high, amount), low, high);
    }

    /// <summary>Decodes the formant-blend output; silent when the profile disables the stage.</summary>
    public static float FormantMix(float amount, DroidVoiceProfile profile) =>
        profile.FormantMix > 0f
            ? Math.Clamp(profile.FormantMix * NeuralMathF.Clamp01(amount), 0f, 1f)
            : 0f;

    /// <summary>
    /// Scales a profile envelope stage by a model output and keeps it inside the ranges
    /// <see cref="DroidVoiceProfile.Normalized"/> accepts.
    /// </summary>
    public static float EnvelopeSeconds(
        float amount,
        float profileSeconds,
        float minimum,
        float maximum) =>
        Math.Clamp(
            profileSeconds * NeuralMathF.Lerp(EnvelopeScaleMin, EnvelopeScaleMax, amount),
            minimum,
            maximum);

    /// <summary>Decodes the attack output against the profile attack time.</summary>
    public static float AttackSeconds(float amount, DroidVoiceProfile profile) =>
        EnvelopeSeconds(amount, profile.AttackSeconds, AttackSecondsFloor, AttackSecondsCeiling);

    /// <summary>Decodes the release output against the profile release time.</summary>
    public static float ReleaseSeconds(float amount, DroidVoiceProfile profile) =>
        EnvelopeSeconds(amount, profile.ReleaseSeconds, ReleaseSecondsFloor, ReleaseSecondsCeiling);

    /// <summary>Decodes the saturation output into the soft-clip drive.</summary>
    public static float Drive(float amount) => NeuralMathF.Lerp(MinDrive, MaxDrive, amount);

    /// <summary>Decodes the saturation output into the soft-clip amount.</summary>
    public static float Saturation(float amount) => NeuralMathF.Clamp01(amount) * MaxSaturation;

    /// <summary>
    /// Decodes the pause output into a phrase-boundary pause in seconds. The punctuation picks the
    /// base length, the model scales it, and the profile speed shortens or stretches it.
    /// </summary>
    public static float PauseSeconds(
        float amount,
        PhrasePunctuation punctuation,
        DroidVoiceProfile profile)
    {
        var basePause = punctuation == PhrasePunctuation.Ellipsis
            ? EllipsisPauseSeconds
            : PhrasePauseSeconds;

        var seconds = NeuralMathF.Lerp(PauseScaleMin, PauseScaleMax, amount) * basePause
            / profile.SpeedMultiplier;

        return Math.Clamp(seconds, 0.010f, 1.00f);
    }

    /// <summary>
    /// Decodes a full 17-value target vector into the model-controlled fields of one syllable,
    /// without the procedural post-processing (legato joins, envelope shrinking) the predictor adds
    /// on top. This is the exact forward half of the <see cref="SyllableTargetEncoder"/> round trip.
    /// </summary>
    /// <param name="targets">At least <see cref="SyllableTargetEncoder.TargetSize"/> normalized values.</param>
    /// <param name="context">The syllable the targets describe.</param>
    /// <param name="profile">Voice profile; normalized internally.</param>
    /// <param name="frame">Receives the decoded acoustic snapshot before post-processing.</param>
    /// <exception cref="ArgumentException"><paramref name="targets"/> is too short.</exception>
    public static void DecodeTargets(
        ReadOnlySpan<float> targets,
        in SyllableContext context,
        DroidVoiceProfile profile,
        out PredictedFrame frame)
    {
        if (targets.Length < SyllableTargetEncoder.TargetSize)
        {
            throw new ArgumentException(
                $"The target buffer must hold at least {SyllableTargetEncoder.TargetSize} values, " +
                $"but has {targets.Length}.",
                nameof(targets));
        }

        profile = profile.Normalized();
        var sampleRate = context.SampleRate;

        var startFrequency = Frequency(
            targets[NeuralProsodyPredictor.StartFrequencyIndex], profile);
        var endFrequency = Frequency(
            targets[NeuralProsodyPredictor.EndFrequencyIndex], profile);
        var fmDepth = FmDepth(targets[NeuralProsodyPredictor.FmDepthIndex], profile);
        var fmRatio = FmRatio(targets[NeuralProsodyPredictor.FmRatioIndex]);
        var formant1 = Formant1(targets[NeuralProsodyPredictor.Formant1Index], sampleRate);
        var saturation = targets[NeuralProsodyPredictor.SaturationIndex];
        var pauseSeconds = PauseSeconds(
            targets[NeuralProsodyPredictor.PauseAfterIndex],
            context.Punctuation,
            profile);
        var durationSeconds = DurationSeconds(
            targets[NeuralProsodyPredictor.DurationIndex],
            context.TempoScale,
            profile);

        frame = new PredictedFrame(
            startFrequency,
            CurveCenter(
                targets[NeuralProsodyPredictor.MidFrequencyIndex],
                startFrequency,
                endFrequency,
                profile),
            endFrequency,
            Math.Max(1, (int)MathF.Round(durationSeconds * sampleRate)),
            Gain(targets[NeuralProsodyPredictor.GainIndex]),
            fmDepth,
            FmDepth2(fmDepth),
            fmRatio,
            Growl(targets[NeuralProsodyPredictor.GrowlIndex]),
            Noise(targets[NeuralProsodyPredictor.NoiseIndex]),
            RingMod(targets[NeuralProsodyPredictor.RingModIndex], profile),
            formant1,
            Formant2(targets[NeuralProsodyPredictor.Formant2Index], formant1, sampleRate),
            FormantMix(targets[NeuralProsodyPredictor.FormantMixIndex], profile),
            AttackSeconds(targets[NeuralProsodyPredictor.AttackIndex], profile),
            ReleaseSeconds(targets[NeuralProsodyPredictor.ReleaseIndex], profile),
            Drive(saturation),
            Saturation(saturation),
            Math.Max(0, (int)MathF.Round(pauseSeconds * sampleRate)));
    }
}

/// <summary>
/// The model-controlled acoustic fields of one syllable, produced by
/// <see cref="SyllableTargetDecoder.DecodeTargets"/> before any procedural post-processing.
/// Trainers and tests use it to check that a target vector means what they think it means.
/// </summary>
/// <param name="StartFrequency">Glide start frequency in Hz.</param>
/// <param name="MidFrequency">Contour midpoint in Hz, or <c>0</c> without pitch curvature.</param>
/// <param name="EndFrequency">Glide end frequency in Hz.</param>
/// <param name="LengthSamples">Syllable length in whole samples.</param>
/// <param name="Gain">Syllable gain.</param>
/// <param name="FmDepth">First modulator depth.</param>
/// <param name="FmDepth2">Second modulator depth.</param>
/// <param name="FmRatio">First modulator ratio.</param>
/// <param name="Growl">Growl modulation depth.</param>
/// <param name="Noise">Sibilant noise amount.</param>
/// <param name="RingMod">Ring modulation mix.</param>
/// <param name="Formant1">First formant in Hz.</param>
/// <param name="Formant2">Second formant in Hz.</param>
/// <param name="FormantBlend">Formant bank mix.</param>
/// <param name="AttackSeconds">Word-envelope attack in seconds.</param>
/// <param name="ReleaseSeconds">Word-envelope release in seconds.</param>
/// <param name="Drive">Soft-clip drive.</param>
/// <param name="Saturation">Soft-clip amount.</param>
/// <param name="PauseAfterSamples">Phrase-boundary pause in whole samples.</param>
public readonly record struct PredictedFrame(
    float StartFrequency,
    float MidFrequency,
    float EndFrequency,
    int LengthSamples,
    float Gain,
    float FmDepth,
    float FmDepth2,
    float FmRatio,
    float Growl,
    float Noise,
    float RingMod,
    float Formant1,
    float Formant2,
    float FormantBlend,
    float AttackSeconds,
    float ReleaseSeconds,
    float Drive,
    float Saturation,
    int PauseAfterSamples);
