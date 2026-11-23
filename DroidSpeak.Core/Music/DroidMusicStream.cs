using System;
using System.Collections.Generic;
using DroidSpeak.DSP;

namespace DroidSpeak.Music
{
    /// <summary>
    /// Stateful pull-based stream for real-time, chunked music synthesis.
    /// <para>
    /// The score is classified once when the stream is created (lead-line detection and
    /// glide-from pitches are resolved up front), then each <see cref="Read"/> renders only the
    /// notes and percussion hits that overlap the requested sample window. Oscillator phase is
    /// preserved per active note across chunks, so arbitrarily small reads sound exactly like one
    /// continuous render without allocating a full-score scratch buffer.
    /// </para>
    /// </summary>
    public sealed class DroidMusicStream
    {
        private const float TwoPi = MathF.PI * 2f;
        private const float PeakCeiling = 0.95f;

        // Soft-knee limiter: unity gain below the knee, smooth compression above it, asymptote at
        // PeakCeiling. This keeps already-pulled chunks byte-stable no matter how they are sliced.
        private const float Knee = 0.57f;

        private const float PercussionMaxSeconds = 0.035f;

        private readonly int _sampleRate;
        private readonly DroidVoiceProfile _profile;
        private readonly int _totalSamples;
        private readonly NoteInfo[] _notes;
        private readonly int[] _order;
        private readonly VoiceState[] _states;

        private int _position;
        private int _orderCursor;

        internal DroidMusicStream(
            int sampleRate,
            int totalSamples,
            DroidVoiceProfile profile,
            NoteInfo[] notes,
            int[] order)
        {
            _sampleRate = sampleRate;
            _totalSamples = totalSamples;
            _profile = profile;
            _notes = notes;
            _order = order;
            _states = new VoiceState[notes.Length];
        }

        /// <summary>Output sample rate in Hz.</summary>
        public int SampleRate => _sampleRate;

        /// <summary>Voice profile used to render the score.</summary>
        public DroidVoiceProfile Profile => _profile;

        /// <summary>Total length of the rendered score in samples.</summary>
        public int TotalSamples => _totalSamples;

        /// <summary>Current playback/render cursor position in samples.</summary>
        public int Position => _position;

        /// <summary>True when every sample has been pulled.</summary>
        public bool IsCompleted => _position >= _totalSamples;

        /// <summary>Resets the stream back to sample 0 and clears all per-note voice state.</summary>
        public void Reset()
        {
            _position = 0;
            _orderCursor = 0;
            Array.Clear(_states, 0, _states.Length);
        }

        /// <summary>
        /// Renders the next chunk of mono PCM samples into <paramref name="destination"/> and
        /// returns the number of samples written (0 once the score is complete).
        /// </summary>
        public int Read(Span<float> destination)
        {
            if (destination.IsEmpty || _position >= _totalSamples)
            {
                return 0;
            }

            var chunkStart = _position;
            var count = Math.Min(destination.Length, _totalSamples - chunkStart);
            var chunkEnd = chunkStart + count;
            var target = destination[..count];
            target.Clear();

            // Drop notes that ended before this window (render order is sorted by start sample).
            while (_orderCursor < _order.Length &&
                   _notes[_order[_orderCursor]].SampleEnd <= chunkStart)
            {
                _orderCursor++;
            }

            for (var oi = _orderCursor; oi < _order.Length; oi++)
            {
                var index = _order[oi];
                var note = _notes[index];
                if (note.StartSample >= chunkEnd)
                {
                    break;
                }

                var overlapStart = Math.Max(chunkStart, note.StartSample);
                var overlapEnd = Math.Min(chunkEnd, note.SampleEnd);
                if (overlapEnd <= overlapStart)
                {
                    continue;
                }

                var noteOffset = overlapStart - note.StartSample;
                ref var state = ref _states[index];

                // Fast-forward a voice that is first reached mid-note (defensive; chunked reads
                // are contiguous so this normally only runs if a chunk starts inside a note).
                if (!state.Started && noteOffset > 0)
                {
                    RenderSlice(index, ref state, 0, noteOffset, Span<float>.Empty);
                }

                state.Started = true;
                RenderSlice(
                    index,
                    ref state,
                    noteOffset,
                    overlapEnd - overlapStart,
                    target.Slice(overlapStart - chunkStart, overlapEnd - overlapStart));
            }

            ApplySoftCeiling(target);

            _position = chunkEnd;
            return count;
        }

        private void RenderSlice(
            int index,
            ref VoiceState state,
            int startOffset,
            int length,
            Span<float> target)
        {
            var note = _notes[index];
            if (note.IsPercussion)
            {
                RenderPercussionSlice(note, ref state, startOffset, length, target);
            }
            else if (note.IsLead)
            {
                RenderLeadSlice(note, ref state, startOffset, length, target);
            }
            else
            {
                RenderHarmonySlice(note, ref state, startOffset, length, target);
            }
        }

        private void RenderLeadSlice(
            in NoteInfo note,
            ref VoiceState state,
            int startOffset,
            int length,
            Span<float> target)
        {
            var invRate = 1f / _sampleRate;

            for (var k = 0; k < length; k++)
            {
                var n = startOffset + k;
                var time = n * invRate;
                var currentFreq = note.TargetFrequency;

                if (time < note.GlideDuration)
                {
                    var t = time / note.GlideDuration;
                    var smooth = t * t * (3f - 2f * t);
                    currentFreq = note.StartFrequency * MathF.Pow(note.FrequencyRatio, smooth);
                }

                if (note.HasTrill && time > 0.05f)
                {
                    var vibFade = Math.Clamp((time - 0.05f) / 0.07f, 0f, 1f);
                    currentFreq += note.TrillDepth * vibFade * MathF.Sin(TwoPi * note.TrillRate * time);
                }

                state.CarrierPhase += TwoPi * currentFreq * invRate;
                state.ModPhase1 += TwoPi * (currentFreq * 2.0f) * invRate;
                state.ModPhase2 += TwoPi * (currentFreq * 3.07f) * invRate;

                var attackDecay = MathF.Exp(-time * 14f);
                var fm1 = note.BaseFm * (0.42f + 0.58f * attackDecay);
                var fm2 = note.BaseFm2 * (0.42f + 0.58f * attackDecay);
                var mod1 = MathF.Sin((float)state.ModPhase1);
                var mod2 = MathF.Sin((float)state.ModPhase2);
                var carrier = MathF.Sin((float)state.CarrierPhase + fm1 * mod1 + fm2 * mod2);
                var wave =
                    (0.85f - note.TriangleMix) * carrier +
                    note.TriangleMix * Triangle(state.CarrierPhase) +
                    0.15f * MathF.Sin((float)(state.CarrierPhase * 2.01));
                var env = AdsrEnvelope(time, note.Duration, note.Attack, note.Decay, 0.82f, note.Release);
                var sample = wave * note.Amplitude * env;

                if (note.HasUpwardChirp && time < 0.045f)
                {
                    var chirpT = time / 0.045f;
                    var chirpFreq = MathF.Min(
                        note.TargetFrequency * (1.85f + 0.35f * chirpT),
                        _sampleRate * 0.45f);
                    state.ChirpPhase += TwoPi * chirpFreq * invRate;
                    var chirpEnv = MathF.Sin(MathF.PI * chirpT);
                    sample += MathF.Sin((float)state.ChirpPhase) * chirpEnv * note.Amplitude * 0.28f;
                }

                if (!target.IsEmpty)
                {
                    target[k] += sample;
                }
            }
        }

        private void RenderHarmonySlice(
            in NoteInfo note,
            ref VoiceState state,
            int startOffset,
            int length,
            Span<float> target)
        {
            var invRate = 1f / _sampleRate;
            var phase = state.CarrierPhase;
            var modPhase = state.ModPhase1;

            for (var k = 0; k < length; k++)
            {
                var time = (startOffset + k) * invRate;
                phase += TwoPi * note.TargetFrequency * invRate;
                modPhase += TwoPi * (note.TargetFrequency * 2.0f) * invRate;
                var sample = MathF.Sin((float)phase + note.HarmonyFm * MathF.Sin((float)modPhase));
                var env = AdsrEnvelope(time, note.Duration, note.Attack, note.Decay, 0.65f, note.Release);
                if (!target.IsEmpty)
                {
                    target[k] += sample * note.Amplitude * env;
                }
            }

            state.CarrierPhase = phase;
            state.ModPhase1 = modPhase;
        }

        private void RenderPercussionSlice(
            in NoteInfo note,
            ref VoiceState state,
            int startOffset,
            int length,
            Span<float> target)
        {
            var invRate = 1f / _sampleRate;
            var phase = state.CarrierPhase;
            var randomState = state.NoiseState;
            if (randomState == 0)
            {
                randomState = note.PercussionSeed;
            }

            var ratio = note.PercussionEndFrequency / note.PercussionStartFrequency;

            for (var k = 0; k < length; k++)
            {
                var n = startOffset + k;
                var pos = n / note.PercussionLastSample;
                var freq = note.PercussionStartFrequency * MathF.Pow(ratio, pos);
                phase += TwoPi * freq * invRate;

                randomState ^= randomState << 13;
                randomState ^= randomState >> 17;
                randomState ^= randomState << 5;
                var noise = ((randomState >> 8) * (1f / 16777216f)) * 2f - 1f;

                var env = MathF.Exp(-pos * 12f);
                var sample = (MathF.Sin((float)phase) * 0.5f + noise * 0.7f) * env * note.Amplitude;
                if (!target.IsEmpty)
                {
                    target[k] += sample;
                }
            }

            state.CarrierPhase = phase;
            state.NoiseState = randomState;
        }

        private static void ApplySoftCeiling(Span<float> buffer)
        {
            for (var i = 0; i < buffer.Length; i++)
            {
                var sample = buffer[i];
                if (!float.IsFinite(sample))
                {
                    buffer[i] = 0f;
                    continue;
                }

                buffer[i] = Math.Clamp(SoftLimit(sample), -1f, 1f);
            }
        }

        private static float SoftLimit(float sample)
        {
            var magnitude = MathF.Abs(sample);
            if (magnitude <= Knee)
            {
                return sample;
            }

            var headroom = PeakCeiling - Knee;
            var limited = Knee + headroom * MathF.Tanh((magnitude - Knee) / headroom);
            return sample < 0f ? -limited : limited;
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

        private static void ScaleEnvelope(ref float attack, ref float decay, ref float release, float duration)
        {
            var total = attack + decay + release;
            if (total > duration * 0.90f)
            {
                var scale = duration * 0.90f / total;
                attack *= scale;
                decay *= scale;
                release *= scale;
            }
        }

        /// <summary>
        /// Pre-classified, immutable per-note render data. Built once by
        /// <see cref="DroidMusicSynthesizer.CreateStream"/>.
        /// </summary>
        internal readonly struct NoteInfo
        {
            public int StartSample { get; init; }
            public int Length { get; init; }
            public int SampleEnd => StartSample + Length;
            public bool IsPercussion { get; init; }
            public bool IsLead { get; init; }

            // Tone notes
            public float TargetFrequency { get; init; }
            public float StartFrequency { get; init; }
            public float FrequencyRatio { get; init; }
            public float GlideDuration { get; init; }
            public float Duration { get; init; }
            public float Attack { get; init; }
            public float Decay { get; init; }
            public float Release { get; init; }
            public float Amplitude { get; init; }
            public float TriangleMix { get; init; }
            public float BaseFm { get; init; }
            public float BaseFm2 { get; init; }
            public float HarmonyFm { get; init; }
            public bool HasTrill { get; init; }
            public float TrillRate { get; init; }
            public float TrillDepth { get; init; }
            public bool HasUpwardChirp { get; init; }

            // Percussion
            public float PercussionStartFrequency { get; init; }
            public float PercussionEndFrequency { get; init; }
            public float PercussionLastSample { get; init; }
            public uint PercussionSeed { get; init; }
        }

        private struct VoiceState
        {
            public double CarrierPhase;
            public double ModPhase1;
            public double ModPhase2;
            public double ChirpPhase;
            public uint NoiseState;
            public bool Started;
        }

        internal static NoteInfo BuildNoteInfo(
            in ScoreNote note,
            int startSample,
            int length,
            bool isLead,
            float? glideFrom,
            uint seed,
            DroidVoiceProfile profile,
            int sampleRate)
        {
            if (note.IsPercussion || note.FrequencyHz <= 0f)
            {
                var maxSamples = Math.Min(length, Math.Max(1, (int)(PercussionMaxSeconds * sampleRate)));
                var percussionSeed = seed * 0x9E3779B9u + 1u;
                return new NoteInfo
                {
                    StartSample = startSample,
                    Length = maxSamples,
                    IsPercussion = true,
                    Duration = maxSamples / (float)sampleRate,
                    Amplitude = 0.22f * Math.Clamp(note.Velocity, 0.1f, 1f),
                    PercussionStartFrequency = 1600f + (seed % 5) * 250f,
                    PercussionEndFrequency = 160f,
                    PercussionLastSample = Math.Max(1, maxSamples - 1),
                    PercussionSeed = percussionSeed,
                };
            }

            var frequency = DroidMusicSynthesizer.FitFrequencyToProfile(note.FrequencyHz, profile);
            var duration = length / (float)sampleRate;
            var velocity = Math.Clamp(note.Velocity, 0.1f, 1f);

            if (isLead)
            {
                var attack = Math.Clamp(duration * 0.07f, 0.004f, 0.012f);
                var decay = Math.Clamp(duration * 0.16f, 0.010f, 0.035f);
                var release = Math.Clamp(duration * 0.20f, 0.012f, 0.040f);
                ScaleEnvelope(ref attack, ref decay, ref release, duration);

                var hash = seed * 0x9E3779B1u + 0x85EBCA77u;
                hash ^= hash >> 15;
                var r1 = (hash & 0xFFFF) / 65535f;
                var r2 = ((hash >> 16) & 0xFFFF) / 65535f;

                // Organic droid detune (±7 cents).
                var targetFreq = frequency * (1f + (r1 - 0.5f) * 0.008f);

                // Portamento glide from the previous lead note, or an expressive scoop on a fresh onset.
                float startFreq;
                float glideDuration;
                if (glideFrom.HasValue && MathF.Abs(glideFrom.Value - targetFreq) > 5f)
                {
                    startFreq = glideFrom.Value;
                    glideDuration = Math.Clamp(duration * 0.32f, 0.022f, 0.055f);
                }
                else
                {
                    var scoop = (r2 > 0.35f ? -1f : 1f) * (0.035f + 0.04f * velocity);
                    startFreq = targetFreq * (1f + scoop);
                    glideDuration = Math.Clamp(duration * 0.22f, 0.014f, 0.032f);
                }

                var baseFm = Math.Clamp((1.35f + 1.35f * velocity) * profile.FmIntensity, 0.35f, 5.8f);
                var hasTrill = duration > 0.16f && r1 < (0.25f + profile.ChatterProbability * 0.70f);
                var trillRate = hasTrill && velocity > 0.78f ? 28f : 7.5f;
                var trillDepth = hasTrill
                    ? targetFreq * (velocity > 0.78f ? 0.022f : 0.011f)
                    : 0f;

                // Occasional high astromech chirp overtone on upward leaps.
                var hasUpwardChirp = glideFrom.HasValue &&
                    targetFreq > glideFrom.Value * 1.18f &&
                    r2 < (0.25f + profile.ChatterProbability * 0.65f);

                return new NoteInfo
                {
                    StartSample = startSample,
                    Length = length,
                    IsLead = true,
                    TargetFrequency = targetFreq,
                    StartFrequency = startFreq,
                    FrequencyRatio = targetFreq / MathF.Max(20f, startFreq),
                    GlideDuration = glideDuration,
                    Duration = duration,
                    Attack = attack,
                    Decay = decay,
                    Release = release,
                    Amplitude = 0.36f * Math.Clamp(note.Velocity, 0.15f, 1f),
                    TriangleMix = Math.Clamp(profile.TriangleMix, 0f, 0.60f),
                    BaseFm = baseFm,
                    BaseFm2 = baseFm * 0.42f,
                    HasTrill = hasTrill,
                    TrillRate = trillRate,
                    TrillDepth = trillDepth,
                    HasUpwardChirp = hasUpwardChirp,
                };
            }

            var harmonyAttack = Math.Clamp(duration * 0.12f, 0.008f, 0.025f);
            var harmonyDecay = Math.Clamp(duration * 0.20f, 0.015f, 0.050f);
            var harmonyRelease = Math.Clamp(duration * 0.25f, 0.015f, 0.055f);
            ScaleEnvelope(ref harmonyAttack, ref harmonyDecay, ref harmonyRelease, duration);

            return new NoteInfo
            {
                StartSample = startSample,
                Length = length,
                TargetFrequency = frequency,
                Duration = duration,
                Attack = harmonyAttack,
                Decay = harmonyDecay,
                Release = harmonyRelease,
                Amplitude = 0.11f * velocity,
                HarmonyFm = Math.Clamp(0.45f * profile.FmIntensity, 0.1f, 1.5f),
            };
        }
    }
}