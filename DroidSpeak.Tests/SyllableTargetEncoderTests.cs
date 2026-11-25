using DroidSpeak.DSP;
using DroidSpeak.Neural;
using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.Tests;

/// <summary>
/// Verifies that <see cref="SyllableTargetEncoder"/> really inverts the decoding performed by
/// <see cref="NeuralProsodyPredictor"/> (via the shared <see cref="SyllableTargetDecoder"/>), which
/// is what lets a v0.7 trainer turn teacher frames into ground-truth targets without drift.
/// </summary>
public class SyllableTargetEncoderTests
{
    private const int SampleRate = DroidSynthesizer.DefaultSampleRate;

    // Float32 math plus whole-sample quantization: every exact-inverse assertion lands inside this.
    private const float Tolerance = 2e-3f;

    [Theory]
    [InlineData(0f)]
    [InlineData(0.17f)]
    [InlineData(0.5f)]
    [InlineData(0.83f)]
    [InlineData(1f)]
    public void EncodeTargets_InvertsEveryOutputForConstantVectors(float value)
    {
        // Index 1 (the pitch-curve midpoint) is excluded: at saturated endpoints the decoded arc
        // hits the profile band clamp, where the forward map stops being injective. The curve
        // inverse has its own round-trip and clamp tests.
        var targets = Vector(value);
        var context = Context();
        SyllableTargetDecoder.DecodeTargets(targets, context, DroidVoiceProfile.R2, out var decoded);
        var frame = Frame(decoded, context);

        Span<float> roundTrip = stackalloc float[SyllableTargetEncoder.TargetSize];
        SyllableTargetEncoder.EncodeTargets(frame, context, DroidVoiceProfile.R2, roundTrip);

        for (var i = 0; i < targets.Length; i++)
        {
            if (i == NeuralProsodyPredictor.MidFrequencyIndex)
            {
                continue;
            }

            Assert.True(
                MathF.Abs(targets[i] - roundTrip[i]) <= Tolerance,
                $"Output {i}: decoded then re-encoded {targets[i]:F4} became {roundTrip[i]:F4}.");
        }
    }

    [Fact]
    public void EncodeTargets_InvertsRandomVectorsInsideTheSafeBand()
    {
        // Endpoints stay inside [0.1, 0.9] and the curve amount inside [0.25, 0.75] so the decoded
        // arc cannot reach the profile band clamp, keeping the forward map injective everywhere.
        var random = new SeededFloats(0xD157u);
        var context = Context();
        Span<float> roundTrip = stackalloc float[SyllableTargetEncoder.TargetSize];

        for (var trial = 0; trial < 64; trial++)
        {
            var targets = new float[SyllableTargetEncoder.TargetSize];
            targets[NeuralProsodyPredictor.StartFrequencyIndex] = random.Value(0.1f, 0.9f);
            targets[NeuralProsodyPredictor.MidFrequencyIndex] = random.Value(0.25f, 0.75f);
            targets[NeuralProsodyPredictor.EndFrequencyIndex] = random.Value(0.1f, 0.9f);
            for (var i = 3; i < targets.Length; i++)
            {
                targets[i] = random.Next();
            }

            SyllableTargetDecoder.DecodeTargets(
                targets, context, DroidVoiceProfile.R2, out var decoded);
            var frame = Frame(decoded, context);

            SyllableTargetEncoder.EncodeTargets(frame, context, DroidVoiceProfile.R2, roundTrip);

            for (var i = 0; i < targets.Length; i++)
            {
                Assert.True(
                    MathF.Abs(targets[i] - roundTrip[i]) <= Tolerance,
                    $"Trial {trial}, output {i}: {targets[i]:F4} became {roundTrip[i]:F4}.");
            }
        }
    }

    [Fact]
    public void EncodeTargets_StaysIdempotentWhenTheCurveHitsTheBand()
    {
        // Endpoints near the top of the band push the arc past MaxFreq and decoding clamps it. The
        // inverse of a clamped value is not the original amount, but it must decode to the very
        // same (clamped) frame, so a second round trip changes nothing.
        var targets = Vector(0.5f);
        targets[NeuralProsodyPredictor.StartFrequencyIndex] = 0.95f;
        targets[NeuralProsodyPredictor.EndFrequencyIndex] = 0.95f;
        targets[NeuralProsodyPredictor.MidFrequencyIndex] = 1f;

        var context = Context();
        SyllableTargetDecoder.DecodeTargets(targets, context, DroidVoiceProfile.R2, out var decoded);
        Assert.Equal(DroidVoiceProfile.R2.MaxFreq, decoded.MidFrequency, 3);

        var frame = Frame(decoded, context);
        Span<float> first = stackalloc float[SyllableTargetEncoder.TargetSize];
        SyllableTargetEncoder.EncodeTargets(frame, context, DroidVoiceProfile.R2, first);

        // The saturated arc maps back below 1 (information was lost at the clamp), but never above.
        Assert.InRange(first[NeuralProsodyPredictor.MidFrequencyIndex], 0f, 1f);
        Assert.NotEqual(1f, first[NeuralProsodyPredictor.MidFrequencyIndex]);

        SyllableTargetDecoder.DecodeTargets(first, context, DroidVoiceProfile.R2, out var again);
        var secondFrame = Frame(again, context);
        Assert.Equal(decoded.MidFrequency, again.MidFrequency, 3);

        Span<float> second = stackalloc float[SyllableTargetEncoder.TargetSize];
        SyllableTargetEncoder.EncodeTargets(secondFrame, context, DroidVoiceProfile.R2, second);

        for (var i = 0; i < first.Length; i++)
        {
            Assert.True(
                MathF.Abs(first[i] - second[i]) <= Tolerance,
                $"Output {i} drifted on the second pass: {first[i]:F4} vs {second[i]:F4}.");
        }
    }

    [Theory]
    [InlineData(PhrasePunctuation.Statement, 0.50f)]
    [InlineData(PhrasePunctuation.Question, 0.17f)]
    [InlineData(PhrasePunctuation.Ellipsis, 0.83f)]
    [InlineData(PhrasePunctuation.Exclamation, 0.31f)]
    public void EncodeTargets_InvertsTheExactPauseFormula(PhrasePunctuation punctuation, float amount)
    {
        // The pause decode is: clamp(Lerp(0.5, 1.5, amount) * base / speed, 0.01, 1.0). Every
        // combination below stays clear of both clamps, so the inverse must return the amount for
        // the rounded, rendered pause exactly as the predictor would have written it.
        Span<float> pauseTargets = stackalloc float[SyllableTargetEncoder.TargetSize];
        foreach (var speed in new[] { 0.5f, 1f, 2f })
        {
            var profile = DroidVoiceProfile.R2 with { SpeedMultiplier = speed };
            var context = Context(punctuation: punctuation);
            var seconds = SyllableTargetDecoder.PauseSeconds(amount, punctuation, profile);
            var samples = (int)MathF.Round(seconds * SampleRate);

            var frame = new SyllableFrame
            {
                PauseAfterSamples = samples,
                PauseAfterSeconds = samples / (float)SampleRate,
            };

            SyllableTargetEncoder.EncodeTargets(frame, context, profile, pauseTargets);

            Assert.True(
                MathF.Abs(pauseTargets[NeuralProsodyPredictor.PauseAfterIndex] - amount) <= Tolerance,
                $"{punctuation} at speed {speed}: decoded {seconds:F4}s re-encoded to " +
                $"{pauseTargets[NeuralProsodyPredictor.PauseAfterIndex]:F4}, expected {amount:F4}.");
        }
    }

    [Fact]
    public void EncodeTargets_GivesNonPhraseFinalSyllablesTheNeutralPause()
    {
        var context = Context();
        var frame = new SyllableFrame { PauseAfterSamples = 0, PauseAfterSeconds = 0f };

        Span<float> targets = stackalloc float[SyllableTargetEncoder.TargetSize];
        SyllableTargetEncoder.EncodeTargets(frame, context, DroidVoiceProfile.R2, targets);

        Assert.Equal(SyllableTargetEncoder.NeutralTarget, targets[NeuralProsodyPredictor.PauseAfterIndex]);
    }

    [Fact]
    public void EncodeTargets_MatchesTheNeuralPredictorEndToEnd()
    {
        // The frame the predictor emits for a target vector must re-encode to that same vector:
        // encoder and predictor decode against identical formulas, including the CloseWord pass
        // (which, with a long syllable, leaves the envelope untouched).
        var targets = Vector(0.72f);
        targets[NeuralProsodyPredictor.DurationIndex] = 1f;

        var contexts = new[] { Context() };
        var frames = new SyllableFrame[1];
        new NeuralProsodyPredictor(ConstantModel(targets)).Predict(
            contexts, frames, DroidVoiceProfile.R2);

        Span<float> roundTrip = stackalloc float[SyllableTargetEncoder.TargetSize];
        SyllableTargetEncoder.EncodeTargets(frames[0], contexts[0], DroidVoiceProfile.R2, roundTrip);

        for (var i = 0; i < targets.Length; i++)
        {
            Assert.True(
                MathF.Abs(targets[i] - roundTrip[i]) <= Tolerance,
                $"Predictor output {i}: fed {targets[i]:F4}, frame re-encoded to {roundTrip[i]:F4}.");
        }
    }

    [Fact]
    public void EncodeTargets_ImitatesRuleBasedTeacherFrames()
    {
        // Decode(Encode(teacher)) must reproduce every model-representable teacher field exactly,
        // or land on the nearest representable value where the rule engine went outside the neural
        // output's reachable band (shrunk envelopes, mixes past the profile scale). That is the
        // contract knowledge distillation rests on.
        var phrases = TextParser.Parse(
            "Привет, мир! Reactor coolant pressure... STONDER? Why do the engines stop, little droid.");
        var contexts = new SyllableContext[SyllableContextBuilder.Count(phrases)];
        SyllableContextBuilder.Build(phrases, SampleRate, 42u, contexts);
        var teacher = new SyllableFrame[contexts.Length];
        RuleBasedProsodyPredictor.Shared.Predict(contexts, teacher, DroidVoiceProfile.R2);

        var profile = DroidVoiceProfile.R2;
        var checkedFields = 0;
        Span<float> targets = stackalloc float[SyllableTargetEncoder.TargetSize];

        for (var i = 0; i < contexts.Length; i++)
        {
            SyllableTargetEncoder.EncodeTargets(teacher[i], contexts[i], profile, targets);
            SyllableTargetDecoder.DecodeTargets(targets, contexts[i], profile, out var decoded);

            var frame = teacher[i];
            AssertClamped(decoded.StartFrequency, frame.StartFreq, profile.MinFreq, profile.MaxFreq, HzTolerance(frame.StartFreq), ref checkedFields);
            AssertClamped(decoded.EndFrequency, frame.EndFreq, profile.MinFreq, profile.MaxFreq, HzTolerance(frame.EndFreq), ref checkedFields);
            AssertClamped(decoded.MidFrequency, frame.MidFreq, profile.MinFreq, profile.MaxFreq, HzTolerance(frame.MidFreq), ref checkedFields);

            // Whole-sample quantization can flip the rounding boundary by one sample.
            Assert.True(Math.Abs(frame.LengthSamples - decoded.LengthSamples) <= 1,
                $"Length {frame.LengthSamples} vs {decoded.LengthSamples}.");

            AssertClamped(decoded.Gain, frame.Gain, SyllableTargetDecoder.MinGain, SyllableTargetDecoder.MaxGain, 2e-3f, ref checkedFields);
            AssertClamped(decoded.FmDepth, frame.FmDepth, SyllableTargetDecoder.MinFmDepth, SyllableTargetDecoder.MaxFmDepth, 5e-3f, ref checkedFields);
            AssertClamped(decoded.FmRatio, frame.FmRatio, SyllableTargetDecoder.MinFmRatio, SyllableTargetDecoder.MaxFmRatio, 2e-3f, ref checkedFields);
            AssertClamped(decoded.Growl, frame.GrowlIntensity, 0f, SyllableTargetDecoder.MaxGrowl, 2e-3f, ref checkedFields);
            AssertClamped(decoded.Noise, frame.NoiseMix, 0f, SyllableTargetDecoder.MaxNoise, 2e-3f, ref checkedFields);
            AssertClamped(decoded.RingMod, frame.RingModMix, 0f, profile.RingModMix, 2e-3f, ref checkedFields);
            AssertClamped(decoded.Formant1, frame.FormantF1, SyllableTargetDecoder.MinFormantHz, SyllableTargetDecoder.FormantCeiling(SampleRate) * 0.70f, HzTolerance(frame.FormantF1), ref checkedFields);
            AssertClamped(decoded.Formant2, frame.FormantF2, frame.FormantF1 + SyllableTargetDecoder.FormantGuardHz, SyllableTargetDecoder.FormantCeiling(SampleRate), HzTolerance(frame.FormantF2), ref checkedFields);
            AssertClamped(decoded.FormantBlend, frame.FormantMix, 0f, profile.FormantMix, 2e-3f, ref checkedFields);
            AssertClamped(decoded.Saturation, frame.Saturation, 0f, SyllableTargetDecoder.MaxSaturation, 2e-3f, ref checkedFields);

            // Only phrase-final syllables carry a trailing pause. Non-final ones encode to the
            // neutral target, which is exactly the rule engine's fixed pause (Lerp(0.5,1.5,0.5)=1).
            if (frame.PauseAfterSamples > 0)
            {
                Assert.Equal(frame.PauseAfterSamples, decoded.PauseAfterSamples);
            }
            else
            {
                Assert.Equal(SyllableTargetEncoder.NeutralTarget, targets[NeuralProsodyPredictor.PauseAfterIndex]);
            }

            // Envelopes: the rule engine writes the raw profile seconds (possibly shrunk for very
            // short words); the neural window is profile * [0.6, 1.4] folded into the legal range.
            var attackLow = MathF.Max(profile.AttackSeconds * SyllableTargetDecoder.EnvelopeScaleMin, SyllableTargetDecoder.AttackSecondsFloor);
            var attackHigh = MathF.Min(profile.AttackSeconds * SyllableTargetDecoder.EnvelopeScaleMax, SyllableTargetDecoder.AttackSecondsCeiling);
            AssertClamped(decoded.AttackSeconds, frame.AttackTime, attackLow, attackHigh, 1e-4f, ref checkedFields);

            var releaseLow = MathF.Max(profile.ReleaseSeconds * SyllableTargetDecoder.EnvelopeScaleMin, SyllableTargetDecoder.ReleaseSecondsFloor);
            var releaseHigh = MathF.Min(profile.ReleaseSeconds * SyllableTargetDecoder.EnvelopeScaleMax, SyllableTargetDecoder.ReleaseSecondsCeiling);
            AssertClamped(decoded.ReleaseSeconds, frame.ReleaseTime, releaseLow, releaseHigh, 1e-4f, ref checkedFields);
        }

        Assert.True(contexts.Length >= 12);
        Assert.True(checkedFields > 0);
    }

    [Fact]
    public void EncodeTargets_FallsBackToNeutralForUnrepresentableFrames()
    {
        // Pitch curve off: MidFreq is not network-controlled. FM intensity zero: depth is pinned.
        // Mix stages zero: those fields are silent. All land on the neutral point instead of
        // asking the network for values it cannot express.
        var neutral = DroidVoiceProfile.R2 with
        {
            PitchCurve = 0f,
            FmIntensity = 0f,
            RingModMix = 0f,
            FormantMix = 0f,
        };

        var context = Context();
        var frame = new SyllableFrame
        {
            StartFreq = 800f,
            EndFreq = 900f,
            MidFreq = 900f,
            FmDepth = 3f,
            RingModMix = 0.5f,
            FormantMix = 0.5f,
            PauseAfterSeconds = 0.12f,
            PauseAfterSamples = (int)MathF.Round(0.12f * SampleRate),
        };

        Span<float> targets = stackalloc float[SyllableTargetEncoder.TargetSize];
        SyllableTargetEncoder.EncodeTargets(frame, context, neutral, targets);

        Assert.Equal(SyllableTargetEncoder.NeutralTarget, targets[NeuralProsodyPredictor.MidFrequencyIndex]);
        Assert.Equal(SyllableTargetEncoder.NeutralTarget, targets[NeuralProsodyPredictor.FmDepthIndex]);
        Assert.Equal(SyllableTargetEncoder.NeutralTarget, targets[NeuralProsodyPredictor.RingModIndex]);
        Assert.Equal(SyllableTargetEncoder.NeutralTarget, targets[NeuralProsodyPredictor.FormantMixIndex]);
    }

    [Fact]
    public void EncodeTargets_ClampsOutOfRangeFrameValues()
    {
        var context = Context();

        // Frequencies outside the profile band, out-of-band formants and an over-max saturation all
        // clamp to the edge of the representable range; training data must stay inside [0, 1].
        var frame = new SyllableFrame
        {
            StartFreq = 10f,
            EndFreq = 9000f,
            Gain = 2f,
            FmRatio = 0.1f,
            GrowlIntensity = 5f,
            NoiseMix = -1f,
            Saturation = 1.5f,
            FormantF1 = 10f,
            FormantF2 = 99_000f,
            PauseAfterSeconds = 5f,
            PauseAfterSamples = (int)(5f * SampleRate),
        };

        Span<float> targets = stackalloc float[SyllableTargetEncoder.TargetSize];
        SyllableTargetEncoder.EncodeTargets(frame, context, DroidVoiceProfile.R2, targets);

        foreach (var target in targets)
        {
            Assert.InRange(target, 0f, 1f);
            Assert.True(float.IsFinite(target));
        }

        Assert.Equal(0f, targets[NeuralProsodyPredictor.StartFrequencyIndex]);
        Assert.Equal(1f, targets[NeuralProsodyPredictor.EndFrequencyIndex]);
        Assert.Equal(1f, targets[NeuralProsodyPredictor.GainIndex]);
        Assert.Equal(0f, targets[NeuralProsodyPredictor.FmRatioIndex]);
        Assert.Equal(1f, targets[NeuralProsodyPredictor.GrowlIndex]);
        Assert.Equal(0f, targets[NeuralProsodyPredictor.NoiseIndex]);
        Assert.Equal(1f, targets[NeuralProsodyPredictor.SaturationIndex]);
        Assert.Equal(1f, targets[NeuralProsodyPredictor.Formant2Index]);
    }

    [Fact]
    public void EncodeTargets_RejectsBadArguments()
    {
        var targets = new float[SyllableTargetEncoder.TargetSize - 1];
        Assert.Throws<ArgumentException>(() =>
            SyllableTargetEncoder.EncodeTargets(
                default, Context(), DroidVoiceProfile.R2, targets));

        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SyllableTargetEncoder.EncodeTargets(
                default,
                Context(sampleRate: 4000),
                DroidVoiceProfile.R2,
                stackalloc float[SyllableTargetEncoder.TargetSize]));
    }

    [Fact]
    public void DecodeTargets_RejectsAShortTargetBuffer()
    {
        Assert.Throws<ArgumentException>(() =>
            SyllableTargetDecoder.DecodeTargets(
                new float[SyllableTargetEncoder.TargetSize - 1],
                Context(),
                DroidVoiceProfile.R2,
                out _));
    }

    [Fact]
    public void EncodeTargets_IsAllocationFree()
    {
        var frame = new SyllableFrame
        {
            EndFreq = 900f,
            StartFreq = 800f,
            Gain = 0.4f,
            PauseAfterSeconds = 0.12f,
            PauseAfterSamples = 5292,
        };

        Span<float> targets = stackalloc float[SyllableTargetEncoder.TargetSize];
        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 500; i++)
        {
            SyllableTargetEncoder.EncodeTargets(frame, Context(), DroidVoiceProfile.R2, targets);
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    /// <summary>
    /// The decoded value must equal the teacher value, or the teacher value must sit outside the
    /// reachable band and the decoded value must land exactly on that band's edge (a clamped
    /// inverse cannot recover more than the nearest representable point).
    /// </summary>
    private static void AssertClamped(
        float decoded,
        float teacher,
        float low,
        float high,
        float tolerance,
        ref int checkedFields)
    {
        checkedFields++;

        if (teacher >= low - tolerance && teacher <= high + tolerance)
        {
            Assert.True(
                MathF.Abs(decoded - teacher) <= tolerance,
                $"Representable value {teacher:F4} in [{low:F4}, {high:F4}] re-decoded to {decoded:F4}.");
            return;
        }

        var edge = teacher < low ? low : high;
        Assert.True(
            MathF.Abs(decoded - edge) <= tolerance,
            $"Out-of-band value {teacher:F4} should clamp to {edge:F4}, got {decoded:F4}.");
    }

    /// <summary>
    /// Float32 ratios lose precision through the divide-and-multiply inverse, so Hz assertions
    /// scale with the magnitude of the frequency.
    /// </summary>
    private static float HzTolerance(float value) => MathF.Max(0.05f, value * 1e-3f);

    private static float[] Vector(float value)
    {
        var targets = new float[SyllableTargetEncoder.TargetSize];
        Array.Fill(targets, value);
        return targets;
    }

    private static SyllableContext Context(
        int sampleRate = SampleRate,
        PhrasePunctuation punctuation = PhrasePunctuation.Statement) =>
        new()
        {
            SampleRate = sampleRate,
            GlobalSeed = 42u,
            WordHash = 0x1234_5678u,
            WordSyllableCount = 2,
            PhraseSyllableCount = 6,
            WordTextLength = 7,
            TempoScale = 1f,
            Punctuation = punctuation,
            IsFirstSpokenPhrase = true,
            Nucleus = 'o',
        };

    /// <summary>
    /// Lifts a decoded snapshot into the frame fields the encoder reads. Timing and procedural
    /// fields (whistle, chatter, trill, chirp, start values) stay at their defaults.
    /// </summary>
    private static SyllableFrame Frame(in PredictedFrame decoded, in SyllableContext context) =>
        new()
        {
            LengthSamples = decoded.LengthSamples,
            DurationSeconds = decoded.LengthSamples / (float)context.SampleRate,
            StartFreq = decoded.StartFrequency,
            MidFreq = decoded.MidFrequency,
            EndFreq = decoded.EndFrequency,
            Gain = decoded.Gain,
            FmDepth = decoded.FmDepth,
            FmRatio = decoded.FmRatio,
            GrowlIntensity = decoded.Growl,
            NoiseMix = decoded.Noise,
            RingModMix = decoded.RingMod,
            FormantF1 = decoded.Formant1,
            FormantF2 = decoded.Formant2,
            FormantMix = decoded.FormantBlend,
            AttackTime = decoded.AttackSeconds,
            ReleaseTime = decoded.ReleaseSeconds,
            Saturation = decoded.Saturation,
            Drive = decoded.Drive,
            PauseAfterSamples = decoded.PauseAfterSamples,
            PauseAfterSeconds = decoded.PauseAfterSamples / (float)context.SampleRate,
        };

    /// <summary>A one-layer linear network whose biases emit the fixed target vector.</summary>
    private static MlpModel ConstantModel(float[] targets)
    {
        var biases = new float[targets.Length];
        Array.Copy(targets, biases, targets.Length);

        return new MlpModel(
        [
            DenseLayer.Create(
                new float[SyllableFeatureEncoder.InputSize * targets.Length],
                biases,
                SyllableFeatureEncoder.InputSize,
                targets.Length,
                ActivationType.Linear),
        ]);
    }

    private struct SeededFloats(uint seed)
    {
        private uint _state = seed == 0u ? 0x9E3779B9u : seed;

        public float Next()
        {
            var value = _state;
            value ^= value << 13;
            value ^= value >> 17;
            value ^= value << 5;
            _state = value;
            return (value >> 8) * (1f / 16777216f);
        }

        public float Value(float min, float max) => min + (max - min) * Next();
    }
}
