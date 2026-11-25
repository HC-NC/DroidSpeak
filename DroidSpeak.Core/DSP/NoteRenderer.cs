namespace DroidSpeak.DSP;

/// <summary>
/// Renders a single syllable note. The offline <see cref="DroidSynthesizer"/> and the streaming
/// <see cref="DroidSpeechStream"/> share this one implementation, so splitting an utterance into
/// arbitrary chunks can never change a single sample.
/// <para>
/// The legacy carrier/FM path is untouched: every stage added in v0.5 is skipped outright while
/// its mix value is zero, which keeps neutral profiles bit-identical to v0.4.
/// </para>
/// </summary>
internal static class NoteRenderer
{
    private const float TwoPi = MathF.PI * 2f;

    /// <summary>Formant smoothing rate per coefficient block.</summary>
    private const float FormantSmoothing = 0.30f;

    /// <summary>Number of samples between formant coefficient updates.</summary>
    private const int FormantBlockSize = 16;

    /// <summary>Duration of the pitch-attack chirp in seconds.</summary>
    private const float AttackChirpSeconds = 0.014f;

    /// <summary>Makeup gain that compensates for the narrow band-pass responses.</summary>
    private const float FormantGain = 1.55f;

    /// <summary>Bandwidth (Q) of the first three formants.</summary>
    private const float Formant1Q = 5.5f;
    private const float Formant2Q = 7.5f;
    private const float Formant3Q = 8.5f;

    /// <summary>
    /// Renders <paramref name="note"/> into <paramref name="target"/>, starting at
    /// <paramref name="startOffset"/> samples inside the note.
    /// </summary>
    public static void Render(
        in DroidSynthesizer.NoteEvent note,
        int startOffset,
        Span<float> target,
        ref VoiceState state,
        float triangleMix,
        int sampleRate)
    {
        // Noise is consumed once per sample, so skipping N samples is exactly N PRNG steps.
        var random = new XorShift32(note.Seed);
        if (note.NoiseAmount > 0f)
        {
            for (var skip = 0; skip < startOffset; skip++)
            {
                random.NextUInt();
            }
        }

        var wordDuration = Math.Max(1, note.WordEndSample - note.WordStartSample) / (float)sampleRate;
        var frequencyRatio = note.EndFrequency / note.StartFrequency;
        var lastSample = Math.Max(1, note.Length - 1);
        var clampedTriangleMix = Math.Clamp(triangleMix, 0f, 0.60f);
        var formantMix = Math.Clamp(note.FormantMix, 0f, 1f);
        var ringModMix = Math.Clamp(note.RingModMix, 0f, 1f);

        // A three-point contour is expressed as a logarithmic bulge away from the straight
        // Start -> End glide. A zero bulge is the legacy path and stays bit-identical.
        var midBulge = 0f;
        if (note.MidFrequency > 0f && note.StartFrequency > 0f && note.EndFrequency > 0f)
        {
            midBulge = MathF.Log(note.MidFrequency)
                - 0.5f * (MathF.Log(note.StartFrequency) + MathF.Log(note.EndFrequency));
            if (MathF.Abs(midBulge) < 1e-6f)
            {
                midBulge = 0f;
            }
        }

        for (var k = 0; k < target.Length; k++)
        {
            var n = startOffset + k;
            var noteTime = n / (float)sampleRate;
            var absoluteTime = (note.StartSample + n) / (float)sampleRate;
            var timeInWord = (note.StartSample + n - note.WordStartSample) / (float)sampleRate;
            var position = n / (float)lastSample;

            // Smoothstep makes the glide ease in and out instead of sounding like a ramp.
            var curveT = position * position * (3f - 2f * position);

            // Pitch is perceived logarithmically, so glide exponentially between frequencies.
            var frequency = note.StartFrequency * MathF.Pow(frequencyRatio, curveT);

            if (midBulge != 0f)
            {
                // sin(pi * t) is zero at both ends and peaks at the midpoint, so the contour still
                // starts on StartFrequency and lands exactly on EndFrequency while passing through
                // MidFreq at t = 0.5.
                var bend = MathF.Sin(MathF.PI * position);
                frequency = MathF.Exp(MathF.Log(frequency) + midBulge * bend);
            }

            if (note.AttackChirp != 0f)
            {
                // Fast, energy-driven onset chirp. The bell shape starts and ends at zero, so the
                // pitch never jumps and no click is introduced.
                if (timeInWord < AttackChirpSeconds)
                {
                    var chirpPhase = timeInWord / AttackChirpSeconds;
                    frequency *= 1f + note.AttackChirp * MathF.Sin(MathF.PI * chirpPhase);
                }
            }
            else if (timeInWord < 0.022f)
            {
                // Bell-shaped legacy onset scoop (starts and ends at zero, so no click).
                var scoopPhase = timeInWord / 0.022f;
                var direction = (note.Seed & 1u) == 0u ? -0.045f : 0.040f;
                frequency *= 1f + direction * MathF.Sin(MathF.PI * scoopPhase);
            }

            if (note.TrillDepth > 0f)
            {
                // Smooth 15 ms fade-in prevents an instantaneous frequency jump (click) at onset.
                var trillFade = Math.Clamp(noteTime / 0.015f, 0f, 1f);
                frequency += note.TrillDepth * trillFade * MathF.Sin(TwoPi * note.TrillRate * absoluteTime);
            }

            var growlDepth = note.StartGrowlDepth
                + (note.GrowlDepth - note.StartGrowlDepth) * curveT;
            var growlWave = MathF.Sin(TwoPi * note.GrowlFrequency * absoluteTime);
            if (growlDepth > 0f)
            {
                frequency += 45f * growlDepth * growlWave;
            }

            frequency = MathF.Max(20f, frequency);

            state.CarrierPhase += TwoPi * frequency / sampleRate;
            state.ModulatorPhase += TwoPi * frequency * note.ModulatorRatio / sampleRate;
            state.ModulatorPhase2 += TwoPi * frequency * note.ModulatorRatio2 / sampleRate;

            var modulator = MathF.Sin((float)state.ModulatorPhase);
            var modulator2 = MathF.Sin((float)state.ModulatorPhase2);

            // Dynamic FM attack decay: bright consonant bite at onset, warming into a pure whistle.
            var fmBurst = 0.42f + 0.58f * MathF.Exp(-noteTime * 28f);
            var fmIndex = (note.StartFmIndex + (note.FmIndex - note.StartFmIndex) * curveT) * fmBurst;
            var fmIndex2 = (note.StartFmIndex2 + (note.FmIndex2 - note.StartFmIndex2) * curveT) * fmBurst;

            var carrier = MathF.Sin((float)state.CarrierPhase + fmIndex * modulator + fmIndex2 * modulator2);
            var localTriangleMix = clampedTriangleMix * fmBurst;
            var wave =
                (0.85f - localTriangleMix) * carrier +
                localTriangleMix * Triangle(state.CarrierPhase) +
                0.15f * fmBurst * MathF.Sin((float)(state.CarrierPhase * 3.01 + 0.55 * modulator2));

            if (ringModMix > 0f)
            {
                // Inharmonic ring carrier at a fixed ratio to the note: classic metallic sidebands.
                var ringFrequency = 40f + 0.62f * frequency;
                state.RingPhase += TwoPi * ringFrequency / sampleRate;
                var ring = MathF.Sin((float)state.RingPhase);
                wave += (wave * ring - wave) * ringModMix;
            }

            var envelope = AdsrEnvelope(
                timeInWord,
                wordDuration,
                note.WordAttack,
                note.WordDecay,
                note.SustainLevel,
                note.WordRelease);

            var amplitude = note.StartAmplitude + (note.Amplitude - note.StartAmplitude) * curveT;
            var growl = 1f + growlDepth * growlWave;
            var sample = wave * amplitude * envelope * growl;

            if (note.NoiseAmount > 0f)
            {
                var noise = random.NextFloat() * 2f - 1f;
                var noiseEnvelope = MathF.Exp(-noteTime * 110f);
                sample += noise * note.NoiseAmount * envelope * noiseEnvelope;
            }

            if (formantMix > 0f)
            {
                sample = ApplyFormants(ref state, note, sample, n, formantMix, sampleRate);
            }

            target[k] += SoftSaturate(sample, note.Drive, note.Saturation);
        }
    }

    /// <summary>
    /// Runs the parallel vowel formant bank. Coefficients glide every
    /// <see cref="FormantBlockSize"/> samples so consecutive syllables morph smoothly instead of
    /// clicking, while the audio itself is processed at full rate.
    /// </summary>
    private static float ApplyFormants(
        ref VoiceState state,
        in DroidSynthesizer.NoteEvent note,
        float sample,
        int n,
        float mix,
        int sampleRate)
    {
        if ((n & (FormantBlockSize - 1)) == 0)
        {
            UpdateFormantCoefficients(ref state, note, sampleRate);
        }

        var first = state.Formant1.Process(sample);
        var second = state.Formant2.Process(sample);
        var third = state.Formant3.Process(sample);

        var voiced = (0.64f * first + 0.28f * second + 0.08f * third) * FormantGain;
        return sample + (voiced - sample) * mix;
    }

    private static void UpdateFormantCoefficients(
        ref VoiceState state,
        in DroidSynthesizer.NoteEvent note,
        int sampleRate)
    {
        var target1 = note.FormantF1 > 0f ? note.FormantF1 : 500f;
        var target2 = note.FormantF2 > 0f ? note.FormantF2 : 1500f;

        if (!state.FormantsInitialized)
        {
            state.Formant1Frequency = target1;
            state.Formant2Frequency = target2;
            state.FormantsInitialized = true;
        }
        else
        {
            state.Formant1Frequency += (target1 - state.Formant1Frequency) * FormantSmoothing;
            state.Formant2Frequency += (target2 - state.Formant2Frequency) * FormantSmoothing;
        }

        var third = Math.Clamp(
            state.Formant2Frequency * 1.85f,
            2000f,
            sampleRate * 0.42f);

        state.Formant1.SetBandPass(state.Formant1Frequency, Formant1Q, sampleRate);
        state.Formant2.SetBandPass(state.Formant2Frequency, Formant2Q, sampleRate);
        state.Formant3.SetBandPass(third, Formant3Q, sampleRate);
    }

    private static float SoftSaturate(float sample, float drive, float amount)
    {
        if (amount <= 0f || drive <= 1f)
        {
            return sample;
        }

        var shaped = MathF.Tanh(sample * drive);
        return sample + (shaped - sample) * Math.Clamp(amount, 0f, 0.9f);
    }

    private static float AdsrEnvelope(
        float time,
        float duration,
        float attack,
        float decay,
        float sustain,
        float release)
    {
        if (time <= 0f || time >= duration)
        {
            return 0f;
        }

        if (time < attack)
        {
            return time / attack;
        }

        var decayEnd = attack + decay;
        if (time < decayEnd)
        {
            return 1f - (1f - sustain) * ((time - attack) / decay);
        }

        var releaseStart = duration - release;
        if (time >= releaseStart)
        {
            return sustain * ((duration - time) / release);
        }

        return sustain;
    }

    private static float Triangle(double phase)
    {
        var normalized = phase / (2d * Math.PI);
        normalized -= Math.Floor(normalized);
        return (float)(4d * Math.Abs(normalized - 0.5d) - 1d);
    }
}
