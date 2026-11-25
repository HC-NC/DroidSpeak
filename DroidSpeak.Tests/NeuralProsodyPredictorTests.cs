using DroidSpeak.DSP;
using DroidSpeak.Neural;
using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.Tests;

public class NeuralProsodyPredictorTests
{
    private const int SampleRate = DroidSynthesizer.DefaultSampleRate;
    private const string SampleText = "Hello, world! R2-D2 is... fine? BEEP BOOP!";

    // Formant1 tops out at 0.7 * max(400, sampleRate * 0.42).
    private static readonly float Formant1Ceiling = MathF.Max(400f, SampleRate * 0.42f) * 0.70f;

    [Fact]
    public void OutputSize_MatchesTheDocumentedLayout()
    {
        // The output vector is part of the model contract: renumbering it would silently reinterpret
        // every saved .dsbrain file.
        Assert.Equal(17, NeuralProsodyPredictor.OutputSize);
        Assert.Equal(0, NeuralProsodyPredictor.StartFrequencyIndex);
        Assert.Equal(1, NeuralProsodyPredictor.MidFrequencyIndex);
        Assert.Equal(2, NeuralProsodyPredictor.EndFrequencyIndex);
        Assert.Equal(3, NeuralProsodyPredictor.DurationIndex);
        Assert.Equal(4, NeuralProsodyPredictor.GainIndex);
        Assert.Equal(5, NeuralProsodyPredictor.FmDepthIndex);
        Assert.Equal(6, NeuralProsodyPredictor.FmRatioIndex);
        Assert.Equal(7, NeuralProsodyPredictor.GrowlIndex);
        Assert.Equal(8, NeuralProsodyPredictor.NoiseIndex);
        Assert.Equal(9, NeuralProsodyPredictor.RingModIndex);
        Assert.Equal(10, NeuralProsodyPredictor.Formant1Index);
        Assert.Equal(11, NeuralProsodyPredictor.Formant2Index);
        Assert.Equal(12, NeuralProsodyPredictor.FormantMixIndex);
        Assert.Equal(13, NeuralProsodyPredictor.AttackIndex);
        Assert.Equal(14, NeuralProsodyPredictor.ReleaseIndex);
        Assert.Equal(15, NeuralProsodyPredictor.SaturationIndex);
        Assert.Equal(16, NeuralProsodyPredictor.PauseAfterIndex);
    }

    [Fact]
    public void Constructor_RequiresAModelWithTheEncoderShape()
    {
        var matching = CreateModel(SyllableFeatureEncoder.InputSize, NeuralProsodyPredictor.OutputSize);
        var tooNarrow = CreateModel(10, NeuralProsodyPredictor.OutputSize);
        var wrongHead = CreateModel(SyllableFeatureEncoder.InputSize, 8);

        var predictor = new NeuralProsodyPredictor(matching);
        Assert.Same(matching, predictor.Model);

        var inputError = Assert.Throws<ArgumentException>(() => new NeuralProsodyPredictor(tooNarrow));
        Assert.Contains("inputs", inputError.Message, StringComparison.Ordinal);

        var outputError = Assert.Throws<ArgumentException>(() => new NeuralProsodyPredictor(wrongHead));
        Assert.Contains("outputs", outputError.Message, StringComparison.Ordinal);

        Assert.Throws<ArgumentNullException>(() => new NeuralProsodyPredictor(null!));
    }

    [Fact]
    public void CreateDefault_BuildsAValidUntrainedModel()
    {
        var predictor = NeuralProsodyPredictor.CreateDefault(seed: 123u);

        Assert.Equal(SyllableFeatureEncoder.InputSize, predictor.Model.InputSize);
        Assert.Equal(NeuralProsodyPredictor.OutputSize, predictor.Model.OutputSize);
        Assert.Equal(3, predictor.Model.LayerCount);
        Assert.Equal(NeuralProsodyPredictor.DefaultHiddenLayerWidth, predictor.Model[0].OutputSize);
        Assert.Equal(NeuralProsodyPredictor.DefaultHiddenLayerWidth / 2, predictor.Model[1].OutputSize);
        Assert.Equal(ActivationType.ModRelu, predictor.Model[0].Activation);
        Assert.Equal(ActivationType.Tanh, predictor.Model[1].Activation);
        Assert.Equal(ActivationType.Sigmoid, predictor.Model[2].Activation);
    }

    [Fact]
    public void Predict_WritesRendererSafeFrames()
    {
        var contexts = Contexts();
        var frames = new SyllableFrame[contexts.Length];
        var profile = DroidVoiceProfile.R2;

        NeuralProsodyPredictor.CreateDefault().Predict(contexts, frames, profile);

        for (var i = 0; i < contexts.Length; i++)
        {
            ref var frame = ref frames[i];

            Assert.True(frame.LengthSamples > 0);
            Assert.Equal(frame.LengthSamples / (float)SampleRate, frame.DurationSeconds, 4);
            Assert.InRange(frame.DurationSeconds, 0.045f, 0.50f);

            Assert.InRange(frame.StartFreq, profile.MinFreq, profile.MaxFreq);
            Assert.InRange(frame.EndFreq, profile.MinFreq, profile.MaxFreq);
            Assert.InRange(frame.Gain, 0.18f, 0.62f);
            Assert.InRange(frame.StartGain, 0.18f, 0.62f);
            Assert.Equal(frame.Gain, frame.Gain);
            Assert.InRange(frame.FmDepth, 0.20f, 6.50f);
            Assert.InRange(frame.FmRatio, 1.00f, 3.50f);
            Assert.True(frame.FmRatio2 > 0f);
            Assert.InRange(frame.NoiseMix, 0f, 0.25f);
            Assert.InRange(frame.GrowlIntensity, 0f, 0.35f);
            Assert.InRange(frame.RingModMix, 0f, 1f);
            Assert.InRange(frame.FormantMix, 0f, 1f);
            Assert.InRange(frame.FormantF1, 150f, Formant1Ceiling);
            Assert.True(frame.FormantF2 > frame.FormantF1);
            Assert.True(frame.AttackTime > 0f);
            Assert.True(frame.ReleaseTime > 0f);
            Assert.Equal(profile.SustainLevel, frame.SustainLevel);
            Assert.True(frame.GrowlFrequency > 0f);
            Assert.Equal(contexts[i].HasVoicedOnset, frame.HasVoicedOnset);

            // Seeds are procedural, so the noise sources stay reproducible across predictor swaps.
            Assert.Equal(
                RuleBasedProsodyPredictor.MixSeed(
                    contexts[i].GlobalSeed,
                    contexts[i].WordHash,
                    (uint)contexts[i].SyllableIndex),
                frame.Seed);
        }
    }

    [Fact]
    public void Predict_KeepsTimingMonotonicAndWordsIntact()
    {
        var contexts = Contexts();
        var frames = new SyllableFrame[contexts.Length];

        NeuralProsodyPredictor.CreateDefault().Predict(contexts, frames, DroidVoiceProfile.R2);

        for (var i = 1; i < frames.Length; i++)
        {
            ref var frame = ref frames[i];
            ref var previous = ref frames[i - 1];

            Assert.True(frame.StartSample >= previous.StartSample + previous.LengthSamples);

            // Silence between two frames: the word gap the predictor published, plus the phrase lead
            // and pause that a phrase boundary adds on top of it.
            var gap = frame.StartSample - previous.StartSample - previous.LengthSamples;
            Assert.True(gap >= frame.PauseBeforeSamples);

            if (frame.StartsLegatoSegment)
            {
                if (contexts[i].PhraseIndex == contexts[i - 1].PhraseIndex)
                {
                    // A new word or hyphen part inside the same phrase: the whole gap is its pause.
                    Assert.Equal(gap, frame.PauseBeforeSamples);
                }

                Assert.True(frame.WordStartSample >= previous.WordStartSample);
            }
            else
            {
                // Syllables sharing a segment are back to back with no silence between them.
                Assert.Equal(previous.StartSample + previous.LengthSamples, frame.StartSample);
                Assert.Equal(0, frame.PauseBeforeSamples);
                Assert.Equal(previous.WordStartSample, frame.WordStartSample);
                Assert.Equal(previous.WordEndSample, frame.WordEndSample);
            }

            Assert.True(frame.WordStartSample <= frame.StartSample);
            Assert.True(frame.WordEndSample >= frame.StartSample + frame.LengthSamples);
        }

        Assert.True(frames[0].StartsLegatoSegment);
        Assert.True(frames[^1].PauseAfterSamples > 0);
        Assert.Equal(
            frames[^1].PauseAfterSamples / (float)SampleRate,
            frames[^1].PauseAfterSeconds,
            4);

        // Only a phrase-final syllable carries a trailing pause.
        for (var i = 0; i < frames.Length - 1; i++)
        {
            var endsPhrase = contexts[i].PhraseIndex != contexts[i + 1].PhraseIndex;
            Assert.Equal(endsPhrase, frames[i].PauseAfterSamples > 0);
        }
    }

    [Fact]
    public void Predict_JoinsTheGlideInsideASegment()
    {
        var contexts = Contexts();
        var frames = new SyllableFrame[contexts.Length];

        NeuralProsodyPredictor.CreateDefault().Predict(contexts, frames, DroidVoiceProfile.R2);

        var joined = 0;
        for (var i = 1; i < frames.Length; i++)
        {
            if (frames[i].StartsLegatoSegment)
            {
                continue;
            }

            joined++;
            Assert.Equal(frames[i - 1].EndFreq, frames[i].StartFreq, 4);
            Assert.Equal(frames[i - 1].Gain, frames[i].StartGain, 4);
            Assert.Equal(frames[i - 1].FmDepth, frames[i].StartFmDepth, 4);
            Assert.Equal(frames[i - 1].FmDepth2, frames[i].StartFmDepth2, 4);
            Assert.Equal(frames[i - 1].GrowlIntensity, frames[i].StartGrowlIntensity, 4);
        }

        Assert.True(joined > 0, "The sample text must contain multi-syllable segments to join.");
    }

    [Fact]
    public void Predict_KeepsFrequenciesInsideEveryProfile()
    {
        var contexts = Contexts();

        foreach (var profile in new[]
        {
            DroidVoiceProfile.R2,
            DroidVoiceProfile.Bass,
            DroidVoiceProfile.Tiny,
            DroidVoiceProfile.Grumpy,
        })
        {
            var frames = new SyllableFrame[contexts.Length];
            NeuralProsodyPredictor.CreateDefault().Predict(contexts, frames, profile);

            foreach (var frame in frames)
            {
                Assert.InRange(frame.StartFreq, profile.MinFreq, profile.MaxFreq);
                Assert.InRange(frame.EndFreq, profile.MinFreq, profile.MaxFreq);

                if (frame.MidFreq > 0f)
                {
                    Assert.InRange(frame.MidFreq, profile.MinFreq, profile.MaxFreq);
                }

                Assert.True(frame.FormantF1 < MathF.Max(400f, frame.FormantF2));
            }
        }
    }

    [Fact]
    public void Predict_SilencesExpressiveStagesTheProfileDisables()
    {
        var contexts = Contexts();

        // A profile with the v0.5 stages off must see silence in those fields, exactly like the rule
        // engine, so the classic voice never picks up an unintended colour.
        var neutral = DroidVoiceProfile.R2 with
        {
            FormantMix = 0f,
            RingModMix = 0f,
            PitchCurve = 0f,
            ChatterProbability = 0f,
            WhistleProbability = 0f,
        };

        var frames = new SyllableFrame[contexts.Length];
        NeuralProsodyPredictor.CreateDefault().Predict(contexts, frames, neutral);

        foreach (var frame in frames)
        {
            Assert.Equal(0f, frame.FormantMix);
            Assert.Equal(0f, frame.RingModMix);
            Assert.Equal(0f, frame.MidFreq);
            Assert.Equal(0f, frame.AttackChirp);
            Assert.Equal(0f, frame.TrillDepth);
            Assert.Equal(0f, frame.TrillRate);
            Assert.False(frame.Whistle.Present);
            Assert.False(frame.Chatter.Present);
        }

        // R2 enables them, so the model does drive those fields.
        frames = new SyllableFrame[contexts.Length];
        NeuralProsodyPredictor.CreateDefault().Predict(contexts, frames, DroidVoiceProfile.R2);

        Assert.Contains(frames, frame => frame.FormantMix > 0f);
        Assert.Contains(frames, frame => frame.RingModMix > 0f);
        Assert.Contains(frames, frame => frame.MidFreq > 0f);
        Assert.Contains(frames, frame => frame.AttackChirp != 0f);
    }

    [Fact]
    public void Predict_MapsEveryContinuousOutput()
    {
        var contexts = Contexts();
        var low = PredictWithConstantOutput(contexts, 0f);
        var mid = PredictWithConstantOutput(contexts, 0.5f);
        var high = PredictWithConstantOutput(contexts, 1f);

        var profile = DroidVoiceProfile.R2;

        // Frequency head: 0 sits on MinFreq, 1 on MaxFreq.
        Assert.Equal(profile.MinFreq, low[^1].EndFreq, 3);
        Assert.True(mid[^1].EndFreq > low[^1].EndFreq);
        Assert.Equal(profile.MaxFreq, high[^1].EndFreq, 3);

        // Gain head: 0 is quiet, 1 is loud.
        Assert.Equal(0.18f, low[^1].Gain, 3);
        Assert.Equal(0.62f, high[^1].Gain, 3);

        // Duration head: the model value is scaled by the phrase tempo and then clamped, so only the
        // shortest syllable lands exactly on the floor. What matters is that the head is monotonic
        // and stays inside the legal synthesis range.
        Assert.Equal(0.045f, low[^1].DurationSeconds, 3);
        Assert.True(mid[^1].DurationSeconds > low[^1].DurationSeconds);
        Assert.True(high[^1].DurationSeconds > mid[^1].DurationSeconds);
        Assert.InRange(high[^1].DurationSeconds, 0.045f, 0.50f);

        // Noise and growl heads are plain scales.
        Assert.Equal(0f, low[^1].NoiseMix, 3);
        Assert.Equal(0.25f, high[^1].NoiseMix, 3);
        Assert.Equal(0f, low[^1].GrowlIntensity, 3);
        Assert.Equal(0.35f, high[^1].GrowlIntensity, 3);

        // Saturation drives both the per-note soft clip and the drive amount.
        Assert.Equal(0f, low[^1].Saturation, 3);
        Assert.Equal(0.72f, high[^1].Saturation, 3);
        Assert.True(high[^1].Drive > low[^1].Drive);

        // The pitch-curve head is bipolar around the geometric midpoint.
        Assert.True(float.IsFinite(mid[^1].MidFreq));
        Assert.NotEqual(high[^1].MidFreq, low[^1].MidFreq);
    }

    [Fact]
    public void Predict_MapsThePauseOutputIntoPhraseSilence()
    {
        var contexts = Contexts();
        var shortPauses = PredictWithConstantOutput(contexts, 0f);
        var longPauses = PredictWithConstantOutput(contexts, 1f);

        // Lerp(0.5, 1.5, output) over the rule engine's 0.12 s statement pause.
        Assert.Equal(0.5f * 0.12f, shortPauses[^1].PauseAfterSeconds, 3);
        Assert.Equal(1.5f * 0.12f, longPauses[^1].PauseAfterSeconds, 3);
        Assert.True(shortPauses[^1].PauseAfterSamples < longPauses[^1].PauseAfterSamples);
    }

    [Fact]
    public void PauseOutput_ShiftsTheTotalSampleCount()
    {
        // GetSampleCount reads the frames back, so a model that wants longer silence must make the
        // utterance longer, otherwise the measure pass and the render pass would disagree.
        var synthesizer = new DroidSynthesizer(
            seed: 7,
            predictor: new NeuralProsodyPredictor(CreateConstantModel(0f)));
        var pauses = new DroidSynthesizer(
            seed: 7,
            predictor: new NeuralProsodyPredictor(CreateConstantModel(1f)));

        var phrases = TextParser.Parse(SampleText);

        Assert.True(synthesizer.GetSampleCount(phrases) < pauses.GetSampleCount(phrases));
    }

    [Fact]
    public void Predict_HandlesASingleSyllableUtterance()
    {
        var contexts = new[]
        {
            new SyllableContext
            {
                SampleRate = SampleRate,
                GlobalSeed = 99u,
                WordHash = 0x1234_5678u,
                WordSyllableCount = 1,
                PhraseSyllableCount = 1,
                WordTextLength = 4,
                TempoScale = 1f,
                Punctuation = PhrasePunctuation.Exclamation,
                IsFirstSpokenPhrase = true,
                Nucleus = 'o',
            },
        };

        var frames = new SyllableFrame[1];
        NeuralProsodyPredictor.CreateDefault().Predict(contexts, frames, DroidVoiceProfile.R2);

        Assert.Equal(0, frames[0].PauseBeforeSamples);
        Assert.True(frames[0].StartsLegatoSegment);
        Assert.True(frames[0].PauseAfterSamples > 0);
        Assert.True(frames[0].StartSample > 0, "The first phrase keeps its lead-in silence.");
    }

    [Fact]
    public void Predict_LeavesSpareFramesUntouched()
    {
        // GetSampleCount rents pooled buffers that are longer than the context span, so the predictor
        // must never write past contexts.Length.
        var contexts = Contexts();
        var frames = new SyllableFrame[contexts.Length + 3];
        frames[^1] = new SyllableFrame { LengthSamples = -7, StartFreq = -11f };

        NeuralProsodyPredictor.CreateDefault().Predict(contexts, frames, DroidVoiceProfile.R2);

        Assert.Equal(-7, frames[^1].LengthSamples);
        Assert.Equal(-11f, frames[^1].StartFreq);
    }

    [Fact]
    public void Predict_IgnoresEmptyContexts()
    {
        var frames = new SyllableFrame[4];
        NeuralProsodyPredictor.CreateDefault().Predict(
            ReadOnlySpan<SyllableContext>.Empty,
            frames,
            DroidVoiceProfile.R2);

        Assert.Equal(default, frames[0]);
    }

    [Fact]
    public void Predict_RejectsAFrameBufferShorterThanTheContexts()
    {
        var contexts = Contexts();
        var frames = new SyllableFrame[contexts.Length - 1];

        Assert.Throws<ArgumentException>(() =>
            NeuralProsodyPredictor.CreateDefault().Predict(contexts, frames, DroidVoiceProfile.R2));
    }

    [Fact]
    public void Predict_IsBitDeterministic()
    {
        var contexts = Contexts();
        var first = new SyllableFrame[contexts.Length];
        var second = new SyllableFrame[contexts.Length];
        var predictor = NeuralProsodyPredictor.CreateDefault(seed: 999u);

        predictor.Predict(contexts, first, DroidVoiceProfile.R2);
        predictor.Predict(contexts, second, DroidVoiceProfile.R2);

        for (var i = 0; i < first.Length; i++)
        {
            Assert.Equal(
                Fingerprint(first[i]),
                Fingerprint(second[i]));
        }

        static uint[] Fingerprint(in SyllableFrame frame) =>
        [
            unchecked((uint)frame.StartSample),
            unchecked((uint)frame.LengthSamples),
            unchecked((uint)frame.PauseBeforeSamples),
            unchecked((uint)frame.PauseAfterSamples),
            unchecked((uint)frame.WordStartSample),
            unchecked((uint)frame.WordEndSample),
            BitConverter.SingleToUInt32Bits(frame.StartFreq),
            BitConverter.SingleToUInt32Bits(frame.MidFreq),
            BitConverter.SingleToUInt32Bits(frame.EndFreq),
            BitConverter.SingleToUInt32Bits(frame.Gain),
            BitConverter.SingleToUInt32Bits(frame.FmDepth),
            BitConverter.SingleToUInt32Bits(frame.FmRatio),
            BitConverter.SingleToUInt32Bits(frame.FormantF1),
            BitConverter.SingleToUInt32Bits(frame.NoiseMix),
            BitConverter.SingleToUInt32Bits(frame.AttackChirp),
            BitConverter.SingleToUInt32Bits(frame.TrillDepth),
            frame.Seed,
        ];
    }

    [Fact]
    public void Predict_DifferentSeedsGiveDifferentVoices()
    {
        var contexts = Contexts();
        var first = new SyllableFrame[contexts.Length];
        var second = new SyllableFrame[contexts.Length];

        NeuralProsodyPredictor.CreateDefault(seed: 1u).Predict(contexts, first, DroidVoiceProfile.R2);
        NeuralProsodyPredictor.CreateDefault(seed: 2u).Predict(contexts, second, DroidVoiceProfile.R2);

        var left = first.Select(frame => frame.EndFreq).ToArray();
        var right = second.Select(frame => frame.EndFreq).ToArray();

        Assert.False(left.SequenceEqual(right));
    }

    [Fact]
    public void Predict_ClampsExtremeNetworkOutputs()
    {
        // A linear head with a huge finite bias saturates every normalized output. Loading NaN or
        // infinity is refused by the format, so clamping is what stands between a wild network and
        // the renderer.
        var model = CreateModel(
            SyllableFeatureEncoder.InputSize,
            NeuralProsodyPredictor.OutputSize,
            ActivationType.Linear,
            bias: 1e30f);

        var contexts = Contexts();
        var frames = new SyllableFrame[contexts.Length];
        new NeuralProsodyPredictor(model).Predict(contexts, frames, DroidVoiceProfile.R2);

        foreach (var frame in frames)
        {
            Assert.True(float.IsFinite(frame.StartFreq));
            Assert.True(float.IsFinite(frame.MidFreq));
            Assert.True(float.IsFinite(frame.EndFreq));
            Assert.True(float.IsFinite(frame.Gain));
            Assert.True(float.IsFinite(frame.FmDepth));
            Assert.True(float.IsFinite(frame.FmRatio));
            Assert.True(float.IsFinite(frame.NoiseMix));
            Assert.True(float.IsFinite(frame.FormantF1));
            Assert.True(float.IsFinite(frame.FormantF2));
            Assert.True(float.IsFinite(frame.DurationSeconds));
            Assert.True(float.IsFinite(frame.PauseAfterSeconds));
            Assert.True(frame.LengthSamples > 0);
            Assert.InRange(frame.StartFreq, DroidVoiceProfile.R2.MinFreq, DroidVoiceProfile.R2.MaxFreq);
            Assert.InRange(frame.DurationSeconds, 0.045f, 0.50f);
        }
    }

    [Fact]
    public void Predict_RespectsSpeedAndTempoWithoutBreakingBounds()
    {
        var contexts = Contexts();

        foreach (var speed in new[] { 0.25f, 1f, 4f })
        {
            var profile = DroidVoiceProfile.R2 with { SpeedMultiplier = speed };
            var frames = new SyllableFrame[contexts.Length];
            NeuralProsodyPredictor.CreateDefault().Predict(contexts, frames, profile);

            foreach (var frame in frames)
            {
                Assert.InRange(frame.DurationSeconds, 0.045f, 0.50f);
                Assert.InRange(frame.PauseAfterSeconds, 0f, 1f);
            }
        }
    }

    [Fact]
    public void Predict_AllocatesNothingOnTheWarmPath()
    {
        var contexts = Contexts();
        var frames = new SyllableFrame[contexts.Length];
        var predictor = NeuralProsodyPredictor.CreateDefault();

        for (var i = 0; i < 32; i++)
        {
            predictor.Predict(contexts, frames, DroidVoiceProfile.R2);
        }

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 200; i++)
        {
            predictor.Predict(contexts, frames, DroidVoiceProfile.R2);
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    [Fact]
    public void NeuralPredictor_SynthesisAndSizingAllocateNothingOnTheWarmPath()
    {
        var phrases = TextParser.Parse(SampleText);
        var synthesizer = new DroidSynthesizer(
            seed: 7,
            profile: DroidVoiceProfile.R2,
            predictor: NeuralProsodyPredictor.CreateDefault(seed: 5u));

        var destination = new float[synthesizer.GetSampleCount(phrases)];

        // Warm up the array pool, tier-0 JIT and the static caches.
        synthesizer.Synthesize(phrases, destination);
        synthesizer.GetSampleCount(phrases);
        synthesizer.Synthesize(phrases, destination);

        var before = GC.GetAllocatedBytesForCurrentThread();
        var written = synthesizer.Synthesize(phrases, destination);
        var count = synthesizer.GetSampleCount(phrases);
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(destination.Length, written);
        Assert.Equal(destination.Length, count);
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void NeuralPredictor_RendersAudibleNonClippingSpeech()
    {
        var phrases = TextParser.Parse(SampleText);
        var synthesizer = new DroidSynthesizer(
            seed: 3,
            profile: DroidVoiceProfile.R2,
            predictor: NeuralProsodyPredictor.CreateDefault(seed: 11u));

        var audio = synthesizer.Synthesize(phrases);

        Assert.True(audio.Length > 0);
        Assert.DoesNotContain(audio, float.IsNaN);
        Assert.DoesNotContain(audio, float.IsInfinity);

        var peak = 0f;
        var quiet = 0;
        foreach (var sample in audio)
        {
            Assert.InRange(sample, -1f, 1f);
            peak = MathF.Max(peak, MathF.Abs(sample));
            if (MathF.Abs(sample) < 1e-4f)
            {
                quiet++;
            }
        }

        Assert.True(peak > 0.05f, "The neural voice must actually produce signal.");
        Assert.True(peak <= 0.95f, "The engine keeps a 0.95 peak ceiling.");
        Assert.True(
            quiet > audio.Length / 40,
            "Word gaps and phrase pauses must leave audible silence.");
    }

    [Fact]
    public void NeuralPredictor_StreamMatchesOfflineLengthAndIsChunkSizeInvariant()
    {
        var phrases = TextParser.Parse(SampleText);
        var synthesizer = new DroidSynthesizer(
            seed: 5,
            predictor: NeuralProsodyPredictor.CreateDefault(seed: 17u));

        var whole = synthesizer.Synthesize(phrases);

        float[]? reference = null;
        foreach (var chunk in new[] { 1, 17, 256, 4096 })
        {
            var stream = synthesizer.CreateStream(phrases);
            Assert.Equal(whole.Length, stream.TotalSamples);

            var result = new float[whole.Length];
            var written = 0;
            var buffer = new float[chunk];

            while (!stream.IsCompleted)
            {
                var taken = stream.Read(buffer);
                Assert.True(taken > 0);
                Array.Copy(buffer, 0, result, written, taken);
                written += taken;
            }

            Assert.Equal(whole.Length, written);
            Assert.All(result, sample => Assert.True(float.IsFinite(sample)));

            reference ??= result;
            Assert.Equal(reference, result);
        }
    }

    [Fact]
    public void NeuralPredictor_ModelSurvivesDsBrainRoundTrip()
    {
        var predictor = NeuralProsodyPredictor.CreateDefault(seed: 23u);

        using var stream = new MemoryStream();
        predictor.Model.Save(stream);

        var restored = new NeuralProsodyPredictor(MlpModel.Load(stream.ToArray().AsSpan()));

        var contexts = Contexts();
        var original = new SyllableFrame[contexts.Length];
        var loaded = new SyllableFrame[contexts.Length];

        predictor.Predict(contexts, original, DroidVoiceProfile.R2);
        restored.Predict(contexts, loaded, DroidVoiceProfile.R2);

        for (var i = 0; i < original.Length; i++)
        {
            Assert.Equal(original[i].StartSample, loaded[i].StartSample);
            Assert.Equal(original[i].LengthSamples, loaded[i].LengthSamples);
            Assert.Equal(original[i].PauseAfterSamples, loaded[i].PauseAfterSamples);
            Assert.Equal(
                BitConverter.SingleToUInt32Bits(original[i].StartFreq),
                BitConverter.SingleToUInt32Bits(loaded[i].StartFreq));
            Assert.Equal(
                BitConverter.SingleToUInt32Bits(original[i].Gain),
                BitConverter.SingleToUInt32Bits(loaded[i].Gain));
        }
    }

    [Fact]
    public void NeuralPredictor_KeepsTheRuleBasedDefaultUntouched()
    {
        // Swapping in a neural model must not leak into synthesizers that never asked for one.
        var classic = new DroidSynthesizer(seed: 7, profile: DroidVoiceProfile.R2);
        var before = classic.Synthesize(TextParser.Parse(SampleText));

        _ = new DroidSynthesizer(
            seed: 7,
            profile: DroidVoiceProfile.R2,
            predictor: NeuralProsodyPredictor.CreateDefault());

        var after = classic.Synthesize(TextParser.Parse(SampleText));

        Assert.Equal(before, after);
    }

    private static SyllableContext[] Contexts()
    {
        var phrases = TextParser.Parse(SampleText);
        var contexts = new SyllableContext[SyllableContextBuilder.Count(phrases)];
        SyllableContextBuilder.Build(phrases, SampleRate, 42u, contexts);
        Assert.True(contexts.Length > 4);
        return contexts;
    }

    private static SyllableFrame[] PredictWithConstantOutput(
        ReadOnlySpan<SyllableContext> contexts,
        float value)
    {
        var frames = new SyllableFrame[contexts.Length];
        new NeuralProsodyPredictor(CreateConstantModel(value))
            .Predict(contexts, frames, DroidVoiceProfile.R2);
        return frames;
    }

    /// <summary>
    /// Builds a network whose every output equals <paramref name="value"/>: a linear head with zero
    /// weights and one bias. That pins each decoded field to a known point in its range.
    /// </summary>
    private static MlpModel CreateConstantModel(float value) =>
        CreateModel(
            SyllableFeatureEncoder.InputSize,
            NeuralProsodyPredictor.OutputSize,
            ActivationType.Linear,
            bias: value);

    private static MlpModel CreateModel(
        int inputSize,
        int outputSize,
        ActivationType activation = ActivationType.Sigmoid,
        float bias = 0f)
    {
        var biases = new float[outputSize];
        if (bias != 0f)
        {
            Array.Fill(biases, bias);
        }

        return new MlpModel(
        [
            DenseLayer.Create(
                new float[inputSize * outputSize],
                biases,
                inputSize,
                outputSize,
                activation),
        ]);
    }
}
