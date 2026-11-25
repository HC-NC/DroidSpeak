using DroidSpeak.DSP;
using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.Neural;

/// <summary>
/// A prosody model driven by a <see cref="MlpModel"/>.
/// <para>
/// Each syllable is encoded with <see cref="SyllableFeatureEncoder"/>, pushed through the network by
/// <see cref="MlpModel.Predict"/>, and the resulting normalized vector is decoded into a
/// <see cref="SyllableFrame"/> whose values are scaled by the active
/// <see cref="DroidVoiceProfile"/>. The encoder window is <c>[previous, current, next]</c>, so the
/// network sees enough context to shape its own contours: this predictor deliberately does not apply
/// the rule engine's question rises or falling endings on top of the model, because that would fight
/// it. Punctuation still steers the rhythm through the pause and tempo features.
/// </para>
/// <para>
/// Timing and structure stay procedural. Phrase leads, word gaps, hyphen breaks, word spans and the
/// whistle and chatter events come from the same deterministic rules as
/// <see cref="RuleBasedProsodyPredictor"/>, scaled by the model's own duration and pause outputs.
/// That keeps <c>DroidSynthesizer.GetSampleCount</c> honest: it runs this exact method on pooled
/// buffers and only reads the sample counts back.
/// </para>
/// <para>
/// The hot path allocates nothing. The feature vector and the output vector are both
/// <c>stackalloc</c> buffers, and every value read from the network is clamped before it reaches a
/// frame, so even a badly trained model cannot emit non-finite or out-of-range acoustics.
/// </para>
/// </summary>
public sealed class NeuralProsodyPredictor : IProsodyPredictor
{
    /// <summary>Number of normalized values the model must produce per syllable.</summary>
    public const int OutputSize = 17;

    /// <summary>Index of the start-frequency output.</summary>
    public const int StartFrequencyIndex = 0;

    /// <summary>Index of the pitch-curve (contour midpoint) output.</summary>
    public const int MidFrequencyIndex = 1;

    /// <summary>Index of the end-frequency output.</summary>
    public const int EndFrequencyIndex = 2;

    /// <summary>Index of the syllable duration output.</summary>
    public const int DurationIndex = 3;

    /// <summary>Index of the gain output.</summary>
    public const int GainIndex = 4;

    /// <summary>Index of the FM depth output.</summary>
    public const int FmDepthIndex = 5;

    /// <summary>Index of the modulator ratio output.</summary>
    public const int FmRatioIndex = 6;

    /// <summary>Index of the growl depth output.</summary>
    public const int GrowlIndex = 7;

    /// <summary>Index of the sibilant noise output.</summary>
    public const int NoiseIndex = 8;

    /// <summary>Index of the ring-modulation output.</summary>
    public const int RingModIndex = 9;

    /// <summary>Index of the first formant output.</summary>
    public const int Formant1Index = 10;

    /// <summary>Index of the second formant output.</summary>
    public const int Formant2Index = 11;

    /// <summary>Index of the formant blend output.</summary>
    public const int FormantMixIndex = 12;

    /// <summary>Index of the word-envelope attack output.</summary>
    public const int AttackIndex = 13;

    /// <summary>Index of the word-envelope release output.</summary>
    public const int ReleaseIndex = 14;

    /// <summary>Index of the soft-clipping drive output.</summary>
    public const int SaturationIndex = 15;

    /// <summary>Index of the phrase pause output.</summary>
    public const int PauseAfterIndex = 16;

    /// <summary>Width of the hidden layers <see cref="CreateDefault"/> builds.</summary>
    public const int DefaultHiddenLayerWidth = 64;

    private const float FirstPhraseLead = 0.04f;
    private const float PhraseLead = 0.09f;
    private const float WordGapSeconds = 0.016f;
    private const float WordGapJitter = 0.028f;
    private const float CommaGapSeconds = 0.060f;
    private const float CommaGapJitter = 0.045f;
    private const float HyphenGapSeconds = 0.018f;
    private const float HyphenGapJitter = 0.014f;
    private const float MaxTrillChance = 0.75f;
    private const float TrillSeedSalt = 0x721115u;

    private readonly MlpModel _model;

    /// <summary>Creates a predictor backed by <paramref name="model"/>.</summary>
    /// <param name="model">
    /// A network with <see cref="SyllableFeatureEncoder.InputSize"/> inputs and
    /// <see cref="OutputSize"/> outputs. An output layer that lands inside <c>[0, 1]</c> (sigmoid, or
    /// a squashed linear head) is the expected shape; anything wider is clamped anyway.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="model"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// The network shape does not match the encoder window or the frame vector.
    /// </exception>
    public NeuralProsodyPredictor(MlpModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        if (model.InputSize != SyllableFeatureEncoder.InputSize)
        {
            throw new ArgumentException(
                $"The model takes {model.InputSize} inputs, but the syllable window produces " +
                $"{SyllableFeatureEncoder.InputSize}.",
                nameof(model));
        }

        if (model.OutputSize != OutputSize)
        {
            throw new ArgumentException(
                $"The model produces {model.OutputSize} outputs, but a syllable frame needs " +
                $"{OutputSize}.",
                nameof(model));
        }

        _model = model;
    }

    /// <summary>The backing network.</summary>
    public MlpModel Model => _model;

    /// <summary>
    /// Builds a predictor around a deterministically initialized network sized for the encoder
    /// window: an untrained but perfectly valid voice. Useful for tests, demos and as the starting
    /// point a v0.7 trainer fine-tunes.
    /// </summary>
    /// <param name="seed">Initialization seed; the same seed always yields the same voice.</param>
    public static NeuralProsodyPredictor CreateDefault(uint seed = MlpModel.DefaultInitSeed) =>
        new(MlpModel.CreateInitialized(
        [
            new LayerSpec(
                SyllableFeatureEncoder.InputSize,
                DefaultHiddenLayerWidth,
                ActivationType.ModRelu),
            new LayerSpec(
                DefaultHiddenLayerWidth,
                DefaultHiddenLayerWidth / 2,
                ActivationType.Tanh),
            new LayerSpec(
                DefaultHiddenLayerWidth / 2,
                OutputSize,
                ActivationType.Sigmoid),
        ], seed));

    /// <inheritdoc />
    public void Predict(
        ReadOnlySpan<SyllableContext> contexts,
        Span<SyllableFrame> frames,
        DroidVoiceProfile profile)
    {
        if (contexts.IsEmpty)
        {
            return;
        }

        if (frames.Length < contexts.Length)
        {
            throw new ArgumentException(
                "The frame buffer must be at least as long as the context buffer.",
                nameof(frames));
        }

        profile = profile.Normalized();

        var sampleRate = contexts[0].SampleRate;
        var seed = contexts[0].GlobalSeed;

        Span<float> features = stackalloc float[SyllableFeatureEncoder.InputSize];
        Span<float> output = stackalloc float[OutputSize];

        var frameCount = 0;
        var cursor = 0;
        var previousPhraseIndex = -1;
        var previousWordIndex = -1;
        var segmentStartFrame = 0;
        var segmentStartSample = 0;
        var segmentGrowlFrequency = 0f;
        var pauseBeforeSamples = 0;
        var pendingChatter = default(ChatterSpec);
        var pendingChatterStart = 0;
        var pendingPauseAmount = 0f;

        for (var i = 0; i < contexts.Length; i++)
        {
            var context = contexts[i];

            if (context.PhraseIndex != previousPhraseIndex)
            {
                if (previousPhraseIndex >= 0)
                {
                    CloseWord(
                        frames,
                        segmentStartFrame,
                        frameCount,
                        segmentStartSample,
                        cursor,
                        segmentGrowlFrequency,
                        sampleRate,
                        profile);

                    // pendingPauseAmount still holds the value of the last syllable of the phrase
                    // that just ended, which is the only one that gets a trailing silence.
                    var phrasePause = Samples(
                        SyllableTargetDecoder.PauseSeconds(
                            pendingPauseAmount, contexts[i - 1].Punctuation, profile),
                        sampleRate);
                    cursor += phrasePause;
                    frames[frameCount - 1].PauseAfterSamples = phrasePause;
                    frames[frameCount - 1].PauseAfterSeconds = phrasePause / (float)sampleRate;
                }

                cursor += Samples(
                    context.IsFirstSpokenPhrase ? FirstPhraseLead : PhraseLead,
                    sampleRate);
                previousWordIndex = -1;
                previousPhraseIndex = context.PhraseIndex;
                pendingChatter = default;
                pendingPauseAmount = 0f;
            }

            pauseBeforeSamples = 0;

            if (context.WordIndexInPhrase != previousWordIndex)
            {
                if (previousWordIndex >= 0)
                {
                    CloseWord(
                        frames,
                        segmentStartFrame,
                        frameCount,
                        segmentStartSample,
                        cursor,
                        segmentGrowlFrequency,
                        sampleRate,
                        profile);
                }

                if (context.WordIndexInPhrase > 0)
                {
                    var gapRandom = new XorShift32(
                        RuleBasedProsodyPredictor.MixSeed(seed, context.WordHash, 0x51ED270Bu));
                    var gap = context.PreviousWordHadComma
                        ? CommaGapSeconds + gapRandom.NextFloat() * CommaGapJitter
                        : WordGapSeconds + gapRandom.NextFloat() * WordGapJitter;
                    var gapSamples = Samples(gap, sampleRate);
                    pendingChatterStart = cursor;
                    pendingChatter = RuleBasedProsodyPredictor.BuildChatter(
                        context,
                        gapSamples,
                        sampleRate,
                        seed,
                        profile);
                    cursor += gapSamples;
                    pauseBeforeSamples = gapSamples;
                }

                segmentStartFrame = frameCount;
                segmentStartSample = cursor;
                segmentGrowlFrequency = 26f + (context.WordHash & 0x0Fu);
                previousWordIndex = context.WordIndexInPhrase;
            }
            else if (context.SyllableIndex > 0 && context.IsHyphenBoundary)
            {
                CloseWord(
                    frames,
                    segmentStartFrame,
                    frameCount,
                    segmentStartSample,
                    cursor,
                    segmentGrowlFrequency,
                    sampleRate,
                    profile);

                var hyphenRandom = new XorShift32(
                    RuleBasedProsodyPredictor.MixSeed(
                        seed,
                        context.WordHash,
                        (uint)(0x900 + context.SyllableIndex)));
                var hyphenGap = Samples(
                    HyphenGapSeconds + hyphenRandom.NextFloat() * HyphenGapJitter,
                    sampleRate);
                pendingChatterStart = cursor;
                pendingChatter = RuleBasedProsodyPredictor.BuildChatter(
                    context,
                    hyphenGap,
                    sampleRate,
                    seed,
                    profile);
                cursor += hyphenGap;
                pauseBeforeSamples = hyphenGap;

                segmentStartFrame = frameCount;
                segmentStartSample = cursor;
                segmentGrowlFrequency = 26f + ((context.WordHash >> 4) & 0x0Fu);
            }

            SyllableFeatureEncoder.Encode(contexts, i, features);
            _model.Predict(features, output);
            pendingPauseAmount = Read(output, PauseAfterIndex);

            var syllableSeed = RuleBasedProsodyPredictor.MixSeed(
                seed,
                context.WordHash,
                (uint)context.SyllableIndex);
            var length = Math.Max(
                1,
                Samples(
                    SyllableTargetDecoder.DurationSeconds(
                        Read(output, DurationIndex), context.TempoScale, profile),
                    sampleRate));
            var isExclamation = context.Punctuation == PhrasePunctuation.Exclamation;
            var energy = Energy(context, isExclamation);
            var isSegmentStart = frameCount == segmentStartFrame;

            var endFrequency = SyllableTargetDecoder.Frequency(
                Read(output, EndFrequencyIndex), profile);

            // Inside a word the glide continues from where the previous syllable landed, exactly as
            // the rule engine does. Without that joint the oscillator would jump and click, and the
            // model would only control word onsets anyway. A frame that is not a segment start always
            // has a predecessor, so the indexed reads below are safe.
            var startFrequency = isSegmentStart
                ? SyllableTargetDecoder.Frequency(Read(output, StartFrequencyIndex), profile)
                : frames[frameCount - 1].EndFreq;

            var gain = SyllableTargetDecoder.Gain(Read(output, GainIndex));
            var fmDepth = SyllableTargetDecoder.FmDepth(Read(output, FmDepthIndex), profile);
            var fmDepth2 = SyllableTargetDecoder.FmDepth2(fmDepth);
            var fmRatio = SyllableTargetDecoder.FmRatio(Read(output, FmRatioIndex));
            var growl = SyllableTargetDecoder.Growl(Read(output, GrowlIndex));
            var saturation = Read(output, SaturationIndex);
            var formant1 = SyllableTargetDecoder.Formant1(
                Read(output, Formant1Index), sampleRate);

            var frame = new SyllableFrame
            {
                StartSample = cursor,
                LengthSamples = length,
                DurationSeconds = length / (float)sampleRate,
                PauseBeforeSamples = pauseBeforeSamples,
                PauseBeforeSeconds = pauseBeforeSamples / (float)sampleRate,
                StartFreq = startFrequency,
                MidFreq = SyllableTargetDecoder.CurveCenter(
                    Read(output, MidFrequencyIndex), startFrequency, endFrequency, profile),
                EndFreq = endFrequency,
                StartGain = isSegmentStart ? gain : frames[frameCount - 1].Gain,
                Gain = gain,
                StartFmDepth = isSegmentStart ? fmDepth : frames[frameCount - 1].FmDepth,
                FmDepth = fmDepth,
                StartFmDepth2 = isSegmentStart ? fmDepth2 : frames[frameCount - 1].FmDepth2,
                FmDepth2 = fmDepth2,
                FmRatio = fmRatio,
                FmRatio2 = SyllableTargetDecoder.FmRatio2(fmRatio),
                StartGrowlIntensity = isSegmentStart ? growl : frames[frameCount - 1].GrowlIntensity,
                GrowlIntensity = growl,
                GrowlFrequency = segmentGrowlFrequency,
                NoiseMix = SyllableTargetDecoder.Noise(Read(output, NoiseIndex)),
                RingModMix = SyllableTargetDecoder.RingMod(Read(output, RingModIndex), profile),
                FormantF1 = formant1,
                FormantF2 = SyllableTargetDecoder.Formant2(
                    Read(output, Formant2Index), formant1, sampleRate),
                FormantMix = SyllableTargetDecoder.FormantMix(
                    Read(output, FormantMixIndex), profile),
                AttackChirp = AttackChirp(context, syllableSeed, energy, profile),
                AttackTime = SyllableTargetDecoder.AttackSeconds(
                    Read(output, AttackIndex), profile),
                DecayTime = profile.DecaySeconds,
                ReleaseTime = SyllableTargetDecoder.ReleaseSeconds(
                    Read(output, ReleaseIndex), profile),
                SustainLevel = profile.SustainLevel,
                HasVoicedOnset = context.HasVoicedOnset,
                Seed = syllableSeed,
                Drive = SyllableTargetDecoder.Drive(saturation),
                Saturation = SyllableTargetDecoder.Saturation(saturation),
                StartsLegatoSegment = isSegmentStart,
            };

            ApplyTrill(ref frame, syllableSeed, energy, profile);

            if (pendingChatter.Present)
            {
                pendingChatter.StartOffsetSamples = pendingChatterStart - cursor;
                frame.Chatter = pendingChatter;
                pendingChatter = default;
            }

            frame.Whistle = RuleBasedProsodyPredictor.BuildWhistle(
                context,
                length,
                isExclamation,
                sampleRate,
                seed,
                profile);

            frames[frameCount] = frame;
            frameCount++;
            cursor += length;
        }

        if (frameCount == 0)
        {
            return;
        }

        CloseWord(
            frames,
            segmentStartFrame,
            frameCount,
            segmentStartSample,
            cursor,
            segmentGrowlFrequency,
            sampleRate,
            profile);

        var finalPause = Samples(
            SyllableTargetDecoder.PauseSeconds(
                pendingPauseAmount, contexts[^1].Punctuation, profile),
            sampleRate);
        frames[frameCount - 1].PauseAfterSamples = finalPause;
        frames[frameCount - 1].PauseAfterSeconds = finalPause / (float)sampleRate;
    }

    /// <summary>
    /// Reads one normalized output and folds non-finite values back into <c>[0, 1]</c>, so a damaged
    /// or badly trained network can never push a garbage number into a frame.
    /// </summary>
    private static float Read(ReadOnlySpan<float> output, int index) =>
        NeuralMathF.Clamp01(output[index]);

    /// <summary>
    /// The same energy measure the rule engine uses to weight whistles and trills, expressed through
    /// the syllable's own context so the neural path reacts to emotion identically.
    /// </summary>
    private static float Energy(in SyllableContext context, bool isExclamation)
    {
        var valence = MathF.Max(0f, context.EmotionValence);
        var arousal = MathF.Max(0f, context.EmotionArousal);
        return Math.Clamp(
            0.55f * valence +
            0.90f * arousal +
            (context.IsStressed ? 0.22f : 0f) +
            (context.IsShout ? 0.32f : 0f) +
            (isExclamation ? 0.22f : 0f),
            0f,
            1.1f);
    }

    private static float AttackChirp(
        in SyllableContext context,
        uint syllableSeed,
        float energy,
        DroidVoiceProfile profile)
    {
        if (profile.PitchCurve <= 0f)
        {
            return 0f;
        }

        var magnitude = profile.PitchCurve * (0.05f + 0.10f * Math.Clamp(energy, 0f, 1f));

        if (context.HasSibilantOnset)
        {
            magnitude *= 1.35f;
        }

        if (context.HasVoicedOnset)
        {
            magnitude *= 1.15f;
        }

        var direction = (syllableSeed & 0x10u) == 0u ? 1f : -1f;
        return direction * magnitude;
    }

    /// <summary>
    /// Keeps the signature trill from the rule engine: the syllable seed decides, the local energy
    /// scales it, and the voice profile's chatter probability gates it.
    /// </summary>
    private static void ApplyTrill(
        ref SyllableFrame frame,
        uint syllableSeed,
        float energy,
        DroidVoiceProfile profile)
    {
        if (profile.ChatterProbability <= 0f)
        {
            return;
        }

        var random = new XorShift32(syllableSeed ^ 0x721115u);
        var chance = profile.ChatterProbability * Math.Clamp(
            0.06f + 0.34f * energy + (frame.DurationSeconds >= 0.10f ? 0.08f : 0f),
            0f,
            MaxTrillChance);

        if (random.NextFloat() >= chance)
        {
            return;
        }

        frame.TrillDepth = 16f + 34f * Math.Clamp(energy, 0f, 1f) + random.NextFloat() * 10f;
        frame.TrillRate = 27f + random.NextFloat() * 8f;
    }

    /// <summary>
    /// Finalizes a word (or hyphen part): publishes its sample span and shrinks the envelope of every
    /// syllable inside it proportionally when the word is too short to hold the requested attack,
    /// decay and release, which is the rule engine's contract for click-free edges.
    /// </summary>
    private static void CloseWord(
        Span<SyllableFrame> frames,
        int startIndex,
        int endIndex,
        int wordStartSample,
        int wordEndSample,
        float growlFrequency,
        int sampleRate,
        DroidVoiceProfile profile)
    {
        if (endIndex <= startIndex)
        {
            return;
        }

        var duration = Math.Max(1, wordEndSample - wordStartSample) / (float)sampleRate;

        for (var i = startIndex; i < endIndex; i++)
        {
            ref var frame = ref frames[i];
            frame.WordStartSample = wordStartSample;
            frame.WordEndSample = wordEndSample;
            frame.GrowlFrequency = growlFrequency;

            var attack = frame.AttackTime;
            var decay = frame.DecayTime;
            var release = frame.ReleaseTime;
            var envelopeTotal = attack + decay + release;

            if (envelopeTotal > duration * 0.90f)
            {
                var scale = duration * 0.90f / envelopeTotal;
                frame.AttackTime = attack * scale;
                frame.DecayTime = decay * scale;
                frame.ReleaseTime = Math.Max(0.001f, release * scale);
            }
        }
    }

    private static int Samples(float seconds, int sampleRate) =>
        Math.Max(0, (int)MathF.Round(seconds * sampleRate));
}
