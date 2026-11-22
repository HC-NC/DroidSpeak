using System;
using System.Collections.Generic;
using DroidSpeak.DSP;

namespace DroidSpeak.Music
{
    /// <summary>
    /// Polyphonic music synthesizer that renders a <see cref="DroidScore"/> using a droid voice timbre.
    /// Separates the upper melody line as an expressive lead droid voice (with portamento glides
    /// and high whistle chirps) while keeping lower harmony notes soft so the melody stays clear.
    /// </summary>
    public sealed class DroidMusicSynthesizer
    {
        public const int DefaultSampleRate = DroidSynthesizer.DefaultSampleRate;

        private const float TwoPi = MathF.PI * 2f;
        private const float PeakCeiling = 0.95f;

        private readonly int _sampleRate;
        private readonly DroidVoiceProfile _profile;

        public DroidMusicSynthesizer(
            int sampleRate = DefaultSampleRate,
            DroidVoiceProfile? profile = null)
        {
            if (sampleRate is < 8000 or > 192000)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(sampleRate),
                    sampleRate,
                    "Sample rate must be between 8000 and 192000 Hz.");
            }

            _sampleRate = sampleRate;
            _profile = (profile ?? DroidVoiceProfile.R2).Normalized();
        }

        public int SampleRate => _sampleRate;

        public DroidVoiceProfile Profile => _profile;

        /// <summary>
        /// Returns the exact number of samples required to render <paramref name="score"/>.
        /// </summary>
        public int GetSampleCount(DroidScore score, DroidVoiceProfile? profile = null)
        {
            ArgumentNullException.ThrowIfNull(score);
            if (score.Notes.Count == 0)
            {
                return 0;
            }

            var activeProfile = (profile ?? _profile).Normalized();
            var durationSeconds = (score.TotalDurationSeconds / activeProfile.SpeedMultiplier) + 0.15f;
            return Math.Max(1, (int)MathF.Ceiling(durationSeconds * _sampleRate));
        }

        /// <summary>
        /// Renders a musical score into a newly allocated mono PCM buffer.
        /// </summary>
        public float[] Render(
            DroidScore score,
            DroidVoiceProfile? profile = null,
            bool soloMelodyOnly = false)
        {
            ArgumentNullException.ThrowIfNull(score);

            var count = GetSampleCount(score, profile);
            var buffer = new float[count];
            Render(score, buffer, profile, soloMelodyOnly);
            return buffer;
        }

        /// <summary>
        /// Convenience overload that parses and renders a file via <paramref name="parser"/>.
        /// </summary>
        public float[] Render(
            IScoreParser parser,
            string filePath,
            DroidVoiceProfile? profile = null,
            bool soloMelodyOnly = false)
        {
            ArgumentNullException.ThrowIfNull(parser);
            return Render(parser.Load(filePath), profile, soloMelodyOnly);
        }

        /// <summary>
        /// Renders a musical score directly into <paramref name="destination"/> and returns the number of samples written.
        /// </summary>
        public int Render(
            DroidScore score,
            Span<float> destination,
            DroidVoiceProfile? profile = null,
            bool soloMelodyOnly = false)
        {
            ArgumentNullException.ThrowIfNull(score);

            var totalSamples = GetSampleCount(score, profile);
            var count = Math.Min(destination.Length, totalSamples);
            if (count == 0)
            {
                return 0;
            }

            var activeProfile = (profile ?? _profile).Normalized();
            var target = destination[..count];
            target.Clear();

            var speed = activeProfile.SpeedMultiplier;
            var seedCounter = 1u;

            var isLeadFlags = ClassifyLeadNotes(score.Notes);

            float? prevLeadFreq = null;
            float prevLeadEndSec = -10f;

            for (int i = 0; i < score.Notes.Count; i++)
            {
                var note = score.Notes[i];
                var startSec = note.StartSeconds / speed;
                var durationSec = note.DurationSeconds / speed;

                var startSample = Math.Max(0, (int)MathF.Round(startSec * _sampleRate));
                if (startSample >= target.Length)
                {
                    continue;
                }

                var noteLength = Math.Max(1, (int)MathF.Round(durationSec * _sampleRate));
                var length = Math.Min(noteLength, target.Length - startSample);
                var slice = target.Slice(startSample, length);

                if (note.IsPercussion || note.FrequencyHz <= 0f)
                {
                    if (!soloMelodyOnly)
                    {
                        RenderPercussion(slice, note.Velocity, seedCounter++);
                    }
                    continue;
                }

                var freq = FitFrequencyToProfile(note.FrequencyHz, activeProfile);
                bool isLead = isLeadFlags[i];

                if (isLead)
                {
                    float? glideFrom = (prevLeadFreq.HasValue && (startSec - prevLeadEndSec) < 0.09f)
                        ? prevLeadFreq.Value
                        : null;

                    RenderLeadNote(slice, freq, glideFrom, note.Velocity, activeProfile, seedCounter++);

                    prevLeadFreq = freq;
                    prevLeadEndSec = startSec + durationSec;
                }
                else if (!soloMelodyOnly)
                {
                    RenderHarmonyNote(slice, freq, note.Velocity, activeProfile);
                }
            }

            ApplyPeakCeiling(target);
            return count;
        }

        private static bool[] ClassifyLeadNotes(IReadOnlyList<ScoreNote> notes)
        {
            var flags = new bool[notes.Count];
            for (int i = 0; i < notes.Count; i++)
            {
                var current = notes[i];
                if (current.IsPercussion || current.FrequencyHz <= 0f)
                {
                    continue;
                }

                bool hasHigherSimultaneous = false;
                // Check nearby notes in time for higher pitches sounding at the same moment
                for (int j = i - 1; j >= 0; j--)
                {
                    var other = notes[j];
                    if (current.StartSeconds - other.StartSeconds > 0.5f) break;
                    if (!other.IsPercussion &&
                        other.StartSeconds <= current.StartSeconds + 0.025f &&
                        other.StartSeconds + other.DurationSeconds > current.StartSeconds + 0.02f &&
                        other.FrequencyHz > current.FrequencyHz * 1.02f)
                    {
                        hasHigherSimultaneous = true;
                        break;
                    }
                }

                if (!hasHigherSimultaneous)
                {
                    for (int j = i + 1; j < notes.Count; j++)
                    {
                        var other = notes[j];
                        if (other.StartSeconds - current.StartSeconds > 0.025f) break;
                        if (!other.IsPercussion && other.FrequencyHz > current.FrequencyHz * 1.02f)
                        {
                            hasHigherSimultaneous = true;
                            break;
                        }
                    }
                }

                flags[i] = !hasHigherSimultaneous;
            }

            return flags;
        }

        private void RenderLeadNote(
            Span<float> target,
            float frequency,
            float? glideFromFreq,
            float velocity,
            DroidVoiceProfile profile,
            uint noteSeed)
        {
            var duration = target.Length / (float)_sampleRate;
            var attack = Math.Clamp(duration * 0.07f, 0.004f, 0.012f);
            var decay = Math.Clamp(duration * 0.16f, 0.010f, 0.035f);
            var release = Math.Clamp(duration * 0.20f, 0.012f, 0.040f);
            const float sustain = 0.82f;

            var totalEnv = attack + decay + release;
            if (totalEnv > duration * 0.90f)
            {
                var scale = duration * 0.90f / totalEnv;
                attack *= scale;
                decay *= scale;
                release *= scale;
            }

            uint hash = noteSeed * 0x9E3779B1u + 0x85EBCA77u;
            hash ^= hash >> 15;
            float r1 = (hash & 0xFFFF) / 65535f;
            float r2 = ((hash >> 16) & 0xFFFF) / 65535f;

            // Organic droid detune (±7 cents)
            float targetFreq = frequency * (1f + (r1 - 0.5f) * 0.008f);

            // Portamento glide from previous note, or expressive pitch scoop on fresh onset
            float startFreq;
            float glideDuration;
            if (glideFromFreq.HasValue && MathF.Abs(glideFromFreq.Value - targetFreq) > 5f)
            {
                startFreq = glideFromFreq.Value;
                glideDuration = Math.Clamp(duration * 0.32f, 0.022f, 0.055f);
            }
            else
            {
                float scoop = (r2 > 0.35f ? -1f : 1f) * (0.035f + 0.04f * velocity);
                startFreq = targetFreq * (1f + scoop);
                glideDuration = Math.Clamp(duration * 0.22f, 0.014f, 0.032f);
            }

            float freqRatio = targetFreq / MathF.Max(20f, startFreq);

            var carrierPhase = 0d;
            var modPhase1 = 0d;
            var modPhase2 = 0d;

            var baseFm = Math.Clamp((1.35f + 1.35f * velocity) * profile.FmIntensity, 0.35f, 5.8f);
            var baseFm2 = baseFm * 0.42f;
            var amplitude = 0.36f * Math.Clamp(velocity, 0.15f, 1f);
            var triangleMix = Math.Clamp(profile.TriangleMix, 0f, 0.60f);

            // Signature R2 trill / vibrato on sustained or accented lead notes
            bool hasTrill = duration > 0.16f && r1 < (0.25f + profile.ChatterProbability * 0.70f);
            float trillRate = hasTrill && velocity > 0.78f ? 28f : 7.5f;
            float trillDepth = hasTrill ? targetFreq * (velocity > 0.78f ? 0.022f : 0.011f) : 0f;
            // Occasional high astromech chirp overtone on upward leaps
            bool hasUpwardChirp = glideFromFreq.HasValue &&
            targetFreq > glideFromFreq.Value * 1.18f &&
            r2 < (0.25f + profile.ChatterProbability * 0.65f);
            var chirpPhase = 0d;
            for (var n = 0; n < target.Length; n++)
            {
                var time = n / (float)_sampleRate;
                var currentFreq = targetFreq;
                // Exponential portamento / scoop into target frequency
                if (time < glideDuration)
                {
                    float t = time / glideDuration;
                    float smooth = t * t * (3f - 2f * t);
                    currentFreq = startFreq * MathF.Pow(freqRatio, smooth);
                }
                if (hasTrill && time > 0.05f)
                {
                    var vibFade = Math.Clamp((time - 0.05f) / 0.07f, 0f, 1f);
                    currentFreq += trillDepth * vibFade * MathF.Sin(TwoPi * trillRate * time);
                }
                carrierPhase += TwoPi * currentFreq / _sampleRate;
                modPhase1 += TwoPi * (currentFreq * 2.0f) / _sampleRate;
                modPhase2 += TwoPi * (currentFreq * 3.07f) / _sampleRate;
                var attackDecay = MathF.Exp(-time * 14f);
                var fm1 = baseFm * (0.42f + 0.58f * attackDecay);
                var fm2 = baseFm2 * (0.42f + 0.58f * attackDecay);
                var mod1 = MathF.Sin((float)modPhase1);
                var mod2 = MathF.Sin((float)modPhase2);
                var carrier = MathF.Sin((float)carrierPhase + fm1 * mod1 + fm2 * mod2);
                var wave =
                (0.85f - triangleMix) * carrier +
                triangleMix * Triangle(carrierPhase) +
                0.15f * MathF.Sin((float)(carrierPhase * 2.01));
                var env = AdsrEnvelope(time, duration, attack, decay, sustain, release);
                var sample = wave * amplitude * env;
                // Short octave whistle chirp on upward jumps
                if (hasUpwardChirp && time < 0.045f)
                {
                    float chirpT = time / 0.045f;
                    float chirpFreq = MathF.Min(targetFreq * (1.85f + 0.35f * chirpT), _sampleRate * 0.45f);
                    chirpPhase += TwoPi * chirpFreq / _sampleRate;
                    float chirpEnv = MathF.Sin(MathF.PI * chirpT);
                    sample += MathF.Sin((float)chirpPhase) * chirpEnv * amplitude * 0.28f;
                }
                target[n] += sample;
            }
        }
        private void RenderHarmonyNote(
        Span<float> target,
        float frequency,
        float velocity,
        DroidVoiceProfile profile)
        {
            var duration = target.Length / (float)_sampleRate;
            var attack = Math.Clamp(duration * 0.12f, 0.008f, 0.025f);
            var decay = Math.Clamp(duration * 0.20f, 0.015f, 0.050f);
            var release = Math.Clamp(duration * 0.25f, 0.015f, 0.055f);
            const float sustain = 0.65f;
            var totalEnv = attack + decay + release;
            if (totalEnv > duration * 0.90f)
            {
                var scale = duration * 0.90f / totalEnv;
                attack *= scale;
                decay *= scale;
                release *= scale;
            }
            var carrierPhase = 0d;
            var modPhase = 0d;
            // Soft accompaniment: 3x quieter than lead and gentle FM so chords stay in the background
            var amplitude = 0.11f * Math.Clamp(velocity, 0.1f, 1f);
            var fm = Math.Clamp(0.45f * profile.FmIntensity, 0.1f, 1.5f);
            for (var n = 0; n < target.Length; n++)
            {
                var time = n / (float)_sampleRate;
                carrierPhase += TwoPi * frequency / _sampleRate;
                modPhase += TwoPi * (frequency * 2.0f) / _sampleRate;
                var sample = MathF.Sin((float)carrierPhase + fm * MathF.Sin((float)modPhase));
                var env = AdsrEnvelope(time, duration, attack, decay, sustain, release);
                target[n] += sample * amplitude * env;
            }
        }
        private void RenderPercussion(Span<float> target, float velocity, uint seed)
        {
            var maxSamples = Math.Min(target.Length, (int)(0.035f * _sampleRate));
            if (maxSamples <= 0)
            {
                return;
            }
            var state = seed * 0x9E3779B9u + 1u;
            var phase = 0d;
            var startFreq = 1600f + (seed % 5) * 250f;
            var endFreq = 160f;
            var ratio = endFreq / startFreq;
            var last = Math.Max(1, maxSamples - 1);
            var amp = 0.22f * Math.Clamp(velocity, 0.1f, 1f);
            for (var n = 0; n < maxSamples; n++)
            {
                var pos = n / (float)last;
                var freq = startFreq * MathF.Pow(ratio, pos);
                phase += TwoPi * freq / _sampleRate;
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                var noise = ((state >> 8) * (1f / 16777216f)) * 2f - 1f;
                var env = MathF.Exp(-pos * 12f);
                var sample = (MathF.Sin((float)phase) * 0.5f + noise * 0.7f) * env * amp;
                target[n] += sample;
            }
        }
        private static float FitFrequencyToProfile(float freq, DroidVoiceProfile profile)
        {
            var minAllowed = profile.MinFreq * 0.65f;
            var maxAllowed = profile.MaxFreq * 1.45f;
            while (freq < minAllowed && freq * 2f <= maxAllowed)
            {
                freq *= 2f;
            }
            while (freq > maxAllowed && freq * 0.5f >= minAllowed)
            {
                freq *= 0.5f;
            }
            return freq;
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
        private static void ApplyPeakCeiling(Span<float> buffer)
        {
            var peak = 0f;
            foreach (var sample in buffer)
            {
                if (float.IsFinite(sample))
                {
                    peak = MathF.Max(peak, MathF.Abs(sample));
                }
            }
            if (peak > PeakCeiling)
            {
                var gain = PeakCeiling / peak;
                for (var i = 0; i < buffer.Length; i++)
                {
                    buffer[i] *= gain;
                }
            }
            for (var i = 0; i < buffer.Length; i++)
            {
                buffer[i] = float.IsFinite(buffer[i])
                ? Math.Clamp(buffer[i], -1f, 1f)
                : 0f;
            }
        }
    }
}