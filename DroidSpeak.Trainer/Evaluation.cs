using System.Globalization;
using DroidSpeak.DSP;
using DroidSpeak.Neural;
using DroidSpeak.NLP;
using DroidSpeak.Prosody;
using DroidSpeak.Trainer.Datasets;

namespace DroidSpeak.Trainer;

/// <summary>
/// Scores a trained <c>.dsbrain</c> file three ways: how closely it reproduces the teacher's target
/// vector, how much of that survives the forward pipeline into frames, and how the rendered audio
/// compares to the rule engine's.
/// <para>
/// The three numbers answer different questions and are reported separately on purpose. The
/// direct-output error is the training objective and will always look best. The frame check catches
/// targets that are statistically close but decode into the wrong syllable length or pause. The
/// audio comparison is the only one a listener cares about, and it is expected to trail the other
/// two: <see cref="NeuralProsodyPredictor"/> deliberately does not reproduce the rule engine's
/// question and ellipsis contours, so identical frames would still not be identical audio.
/// </para>
/// </summary>
internal static class Evaluation
{
    private const int EnvelopeWindows = 16;

    /// <summary>Human-readable names of the 17 outputs, in output order.</summary>
    private static readonly string[] OutputNames =
    [
        "startFreq", "midFreq", "endFreq", "duration", "gain", "fmDepth", "fmRatio", "growl",
        "noise", "ringMod", "formant1", "formant2", "formantMix", "attack", "release",
        "saturation", "pause",
    ];

    /// <summary>Fixed utterances the acoustic comparison renders. Deliberately unlike the corpus.</summary>
    private static readonly string[] Utterances =
    [
        "Привет, маленький дроид! Как проходит день?",
        "Осторожно... давление в реакторе падает, но системы держат.",
        "BEEP BOOP. Diagnostics complete, master. All systems nominal.",
        "Why do the engines stop, little one? Nobody knows.",
        "СРОЧНО! Перегрев кислородных баков, отключите подачу!",
        "coolant pressure rising, coolant pressure rising... again.",
    ];

    /// <summary>
    /// Evaluates a model and writes a report. Returns <c>false</c> when the model is unusable, which
    /// the CLI turns into a non-zero exit code so a bad checkpoint fails CI.
    /// </summary>
    /// <param name="model">The distilled network.</param>
    /// <param name="profile">The voice the network was distilled for.</param>
    /// <param name="sampleRate">Render sample rate, matching the corpus.</param>
    /// <param name="seed">Corpus seed; a validator should pass one the training run never used.</param>
    /// <param name="samples">Rows of held-out corpus to score.</param>
    public static bool Report(
        MlpModel model,
        DroidVoiceProfile profile,
        int sampleRate,
        uint seed,
        int samples)
    {
        if (model.InputSize != SyllableFeatureEncoder.InputSize ||
            model.OutputSize != SyllableTargetEncoder.TargetSize)
        {
            Console.Error.WriteLine(string.Format(
                CultureInfo.InvariantCulture,
                "error: the model is {0} -> {1}, but a prosody network must be {2} -> {3}.",
                model.InputSize,
                model.OutputSize,
                SyllableFeatureEncoder.InputSize,
                SyllableTargetEncoder.TargetSize));
            return false;
        }

        var data = DistillationCorpus.Build(samples, profile, seed, sampleRate);
        var targetMse = MeasureTargetError(model, data);
        var frameErrors = FrameFidelity(model, profile, sampleRate, seed);
        var acoustic = AcousticSimilarity(model, profile, sampleRate);

        ReportTargetError(targetMse);
        ReportFrameErrors(frameErrors);
        ReportAcoustic(acoustic);

        var usable = float.IsFinite(targetMse.Overall) && targetMse.Overall < 0.02f;
        Console.WriteLine(usable
            ? $"model looks usable: validation MSE {targetMse.Overall:E3}, " +
                $"envelope correlation {acoustic.EnvelopeCorrelation:F3}."
            : $"model does NOT look usable: validation MSE {targetMse.Overall:E3} is too high.");

        return usable;
    }

    private readonly record struct TargetError(float Overall, float[] Mae, float[] Rmse);

    private static TargetError MeasureTargetError(MlpModel model, TrainingSet data)
    {
        var squared = new double[data.OutputSize];
        var absolute = new double[data.OutputSize];
        Span<float> output = stackalloc float[data.OutputSize];

        for (var index = 0; index < data.Count; index++)
        {
            model.Predict(data.Features(index), output);
            var target = data.Targets(index);

            for (var i = 0; i < output.Length; i++)
            {
                var difference = output[i] - target[i];
                squared[i] += (double)difference * difference;
                absolute[i] += Math.Abs((double)difference);
            }
        }

        var mae = new float[data.OutputSize];
        var rmse = new float[data.OutputSize];
        var total = 0d;

        for (var i = 0; i < rmse.Length; i++)
        {
            mae[i] = (float)(absolute[i] / data.Count);
            rmse[i] = (float)Math.Sqrt(squared[i] / data.Count);
            total += squared[i];
        }

        return new TargetError(
            (float)(total / ((double)data.Count * data.OutputSize)),
            mae,
            rmse);
    }

    private static void ReportTargetError(TargetError error)
    {
        Console.WriteLine(string.Format(
            CultureInfo.InvariantCulture,
            "target error: MSE {0:E3} over held-out syllables",
            error.Overall));

        for (var i = 0; i < error.Rmse.Length; i++)
        {
            Console.WriteLine(string.Format(
                CultureInfo.InvariantCulture,
                "  {0,-11} MAE {1:F4}  RMSE {2:F4}",
                OutputNames[i],
                error.Mae[i],
                error.Rmse[i]));
        }

        Console.WriteLine();
    }

    private readonly record struct FrameError(
        int Syllables,
        double DurationSamples,
        double PauseSamples,
        double FrequencyPercent,
        double GainAbsolute);

    /// <summary>
    /// Compares the teacher's frames with the network's frames syllable by syllable. Durations and
    /// pauses are whole samples, so they are reported in samples; frequencies are relative errors
    /// because the band is hundreds of Hz wide; gain is small enough to report absolutely.
    /// </summary>
    private static FrameError FrameFidelity(
        MlpModel model,
        DroidVoiceProfile profile,
        int sampleRate,
        uint seed)
    {
        var predictor = new NeuralProsodyPredictor(model);

        var syllables = 0;
        var durationError = 0d;
        var pauseError = 0d;
        var frequencyError = 0d;
        var frequencyCount = 0;
        var gainError = 0d;

        for (var utterance = 0; utterance < Utterances.Length; utterance++)
        {
            var parsed = TextParser.Parse(Utterances[utterance]);
            if (parsed.Count == 0)
            {
                continue;
            }

            var count = SyllableContextBuilder.Count(parsed);
            if (count == 0)
            {
                continue;
            }

            var contexts = new SyllableContext[count];
            SyllableContextBuilder.Build(parsed, sampleRate, seed + (uint)utterance, contexts);

            var teacher = new SyllableFrame[count];
            RuleBasedProsodyPredictor.Shared.Predict(contexts, teacher, profile);

            var neural = new SyllableFrame[count];
            predictor.Predict(contexts, neural, profile);

            for (var i = 0; i < count; i++)
            {
                syllables++;
                durationError += Math.Abs(teacher[i].LengthSamples - neural[i].LengthSamples);
                pauseError += Math.Abs(teacher[i].PauseAfterSamples - neural[i].PauseAfterSamples);
                gainError += Math.Abs(teacher[i].Gain - neural[i].Gain);

                frequencyError += RelativeError(teacher[i].StartFreq, neural[i].StartFreq);
                frequencyError += RelativeError(teacher[i].EndFreq, neural[i].EndFreq);
                frequencyCount += 2;
            }
        }

        if (syllables == 0)
        {
            return default;
        }

        return new FrameError(
            syllables,
            durationError / syllables,
            pauseError / syllables,
            frequencyCount == 0 ? 0 : frequencyError / frequencyCount,
            gainError / syllables);

        static double RelativeError(double teacher, double neural) =>
            teacher <= 0 ? 0 : Math.Abs(teacher - neural) / teacher;
    }

    private static void ReportFrameErrors(FrameError error)
    {
        Console.WriteLine(string.Format(
            CultureInfo.InvariantCulture,
            "frame fidelity over {0} syllables:",
            error.Syllables));
        Console.WriteLine(string.Format(
            CultureInfo.InvariantCulture,
            "  mean duration error {0:F1} samples, pause {1:F1} samples",
            error.DurationSamples,
            error.PauseSamples));
        Console.WriteLine(string.Format(
            CultureInfo.InvariantCulture,
            "  mean frequency error {0:P2}, gain error {1:F4}",
            error.FrequencyPercent,
            error.GainAbsolute));
        Console.WriteLine();
    }

    private readonly record struct AcousticMetric(
        double LengthRatio,
        double RmsRatio,
        double PeakRatio,
        double EnvelopeCorrelation,
        double CentroidRatio);

    /// <summary>
    /// Renders the same sentences twice - once through the rule engine, once through the distilled
    /// network - and compares aggregate acoustics rather than samples.
    /// </summary>
    /// <remarks>
    /// Sample-exact comparison would be meaningless here: the two engines choose different phase
    /// trajectories by design, and the point is whether the distilled voice sounds like the teacher,
    /// not whether it is bit-identical. Envelope correlation over time is the closest cheap proxy for
    /// that, and it is reported next to the ratios so a short, quiet-but-similar render cannot score
    /// well by accident.
    /// </remarks>
    private static AcousticMetric AcousticSimilarity(
        MlpModel model,
        DroidVoiceProfile profile,
        int sampleRate)
    {
        var teacherSynth = new DroidSynthesizer(sampleRate, 0x51DE, profile);
        var neuralSynth = new DroidSynthesizer(
            sampleRate,
            0x51DE,
            profile,
            new NeuralProsodyPredictor(model));

        var lengthRatio = 0d;
        var rmsRatio = 0d;
        var peakRatio = 0d;
        var correlation = 0d;
        var centroidRatio = 0d;
        var counted = 0;

        foreach (var utterance in Utterances)
        {
            var teacher = teacherSynth.Synthesize(utterance, profile);
            var neural = neuralSynth.Synthesize(utterance, profile);

            if (teacher.Length == 0 || neural.Length == 0)
            {
                continue;
            }

            var teacherEnvelope = Envelope(teacher);
            var neuralEnvelope = Envelope(neural);

            lengthRatio += Ratio(teacher.Length, neural.Length);
            rmsRatio += Ratio(Rms(teacher), Rms(neural));
            peakRatio += Ratio(Peak(teacher), Peak(neural));
            correlation += Correlation(teacherEnvelope, neuralEnvelope);
            centroidRatio += Ratio(Centroid(teacher, sampleRate), Centroid(neural, sampleRate));
            counted++;
        }

        if (counted == 0)
        {
            return new AcousticMetric(0, 0, 0, 0, 0);
        }

        return new AcousticMetric(
            lengthRatio / counted,
            rmsRatio / counted,
            peakRatio / counted,
            correlation / counted,
            centroidRatio / counted);

        static double Ratio(double teacher, double neural) =>
            teacher <= 0 ? 0 : neural / teacher;
    }

    private static void ReportAcoustic(AcousticMetric metric)
    {
        Console.WriteLine("acoustic similarity to the rule-based teacher (rendered audio):");
        Console.WriteLine(string.Format(
            CultureInfo.InvariantCulture,
            "  length {0:P1} of teacher, RMS {1:P1}, peak {2:P1}, centroid {3:P1}",
            metric.LengthRatio,
            metric.RmsRatio,
            metric.PeakRatio,
            metric.CentroidRatio));
        Console.WriteLine(string.Format(
            CultureInfo.InvariantCulture,
            "  envelope correlation {0:F3} (1.0 is identical rhythm and loudness)",
            metric.EnvelopeCorrelation));
        Console.WriteLine();
    }

    private static double[] Envelope(ReadOnlySpan<float> samples)
    {
        var envelope = new double[EnvelopeWindows];
        var window = Math.Max(1, samples.Length / EnvelopeWindows);

        for (var i = 0; i < EnvelopeWindows; i++)
        {
            var start = i * window;
            var end = Math.Min(samples.Length, start + window);
            var sum = 0d;

            for (var s = start; s < end; s++)
            {
                sum += (double)samples[s] * samples[s];
            }

            envelope[i] = end > start ? Math.Sqrt(sum / (end - start)) : 0;
        }

        return envelope;
    }

    private static double Correlation(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        var count = Math.Min(a.Length, b.Length);
        var meanA = 0d;
        var meanB = 0d;

        for (var i = 0; i < count; i++)
        {
            meanA += a[i];
            meanB += b[i];
        }

        meanA /= count;
        meanB /= count;

        var covariance = 0d;
        var varianceA = 0d;
        var varianceB = 0d;

        for (var i = 0; i < count; i++)
        {
            var da = a[i] - meanA;
            var db = b[i] - meanB;
            covariance += da * db;
            varianceA += da * da;
            varianceB += db * db;
        }

        var denominator = Math.Sqrt(varianceA * varianceB);
        return denominator > 1e-12 ? covariance / denominator : 0;
    }

    private static double Rms(ReadOnlySpan<float> samples)
    {
        var sum = 0d;
        foreach (var sample in samples)
        {
            sum += (double)sample * sample;
        }

        return samples.IsEmpty ? 0 : Math.Sqrt(sum / samples.Length);
    }

    private static double Peak(ReadOnlySpan<float> samples)
    {
        var peak = 0d;
        foreach (var sample in samples)
        {
            peak = Math.Max(peak, Math.Abs((double)sample));
        }

        return peak;
    }

    private static double Centroid(ReadOnlySpan<float> samples, int sampleRate)
    {
        // A single-bin Goertzel sweep is too slow for a report, so the centroid is estimated from a
        // coarse autocorrelation-free proxy: the mean absolute first difference, which rises with
        // high-frequency content and is cheap and stable across platforms.
        if (samples.Length < 2)
        {
            return 0;
        }

        var sum = 0d;
        for (var i = 1; i < samples.Length; i++)
        {
            sum += Math.Abs(samples[i] - samples[i - 1]);
        }

        return sampleRate * (sum / (samples.Length - 1));
    }
}
