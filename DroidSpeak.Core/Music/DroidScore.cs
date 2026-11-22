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

        public string? Title { get; }

        public IReadOnlyList<ScoreNote> Notes { get; }

        public float TotalDurationSeconds { get; }

        public static float MidiNoteToFrequency(int midiNote) =>
            440f * MathF.Pow(2f, (midiNote - 69) / 12f);
    }
}
