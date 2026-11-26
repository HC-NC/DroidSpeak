using System;
using System.Collections.Generic;
using System.Text;

namespace DroidSpeak.Music
{
    /// <summary>
    /// A single musical note or percussion hit inside a <see cref="DroidScore"/>.
    /// </summary>
    /// <param name="StartSeconds">Start offset from the beginning of the score in seconds.</param>
    /// <param name="DurationSeconds">Duration of the note in seconds.</param>
    /// <param name="FrequencyHz">Pitch in Hz (0 for pure percussion).</param>
    /// <param name="Velocity">Normalized loudness/intensity (0–1).</param>
    /// <param name="IsPercussion">True for drum/percussion events (rendered as droid relay clicks).</param>
    public readonly record struct ScoreNote(
        float StartSeconds,
        float DurationSeconds,
        float FrequencyHz,
        float Velocity = 0.8f,
        bool IsPercussion = false);

    /// <summary>
    /// Format-agnostic musical score ready to be rendered by <see cref="DroidMusicSynthesizer"/>.
    /// </summary>
    public sealed class DroidScore
    {
        /// <summary>Creates a score from already parsed notes.</summary>
        /// <param name="notes">Notes and percussion events in the score.</param>
        /// <param name="title">Optional human-readable title (for example the RTTTL or SMF name).</param>
        public DroidScore(IReadOnlyList<ScoreNote> notes, string? title = null)
        {
            ArgumentNullException.ThrowIfNull(notes);
            Notes = notes;
            Title = title;

            var maxEnd = 0f;
            foreach (var note in notes)
            {
                var end = note.StartSeconds + note.DurationSeconds;
                if (end > maxEnd)
                {
                    maxEnd = end;
                }
            }

            TotalDurationSeconds = maxEnd;
        }

        /// <summary>The optional title of the piece, or <c>null</c> when the source carried none.</summary>
        public string? Title { get; }

        /// <summary>All notes and percussion events, in parsing order.</summary>
        public IReadOnlyList<ScoreNote> Notes { get; }

        /// <summary>The offset of the last note end, in seconds.</summary>
        public float TotalDurationSeconds { get; }

        /// <summary>
        /// Converts a MIDI note number to its equal-tempered frequency (A4 = 69 = 440 Hz).
        /// </summary>
        /// <param name="midiNote">MIDI note number.</param>
        public static float MidiNoteToFrequency(int midiNote) =>
            440f * MathF.Pow(2f, (midiNote - 69) / 12f);
    }
}
