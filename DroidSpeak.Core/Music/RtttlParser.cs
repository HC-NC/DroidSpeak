using System;
using System.Collections.Generic;
using System.Text;

namespace DroidSpeak.Music
{
    /// <summary>
    /// Parses Ring Tone Text Transfer Language (RTTTL) strings into a <see cref="DroidScore"/>.
    /// </summary>
    public sealed class RtttlParser : ITextScoreParser
    {
        public static RtttlParser Shared { get; } = new();

        public DroidScore Parse(string text)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(text);

            var sections = text.Split(':', 3, StringSplitOptions.TrimEntries);
            if (sections.Length != 3)
            {
                throw new FormatException("Invalid RTTTL format: expected 'Name:Defaults:Notes'.");
            }

            var title = string.IsNullOrWhiteSpace(sections[0]) ? null : sections[0];
            var defaultDuration = 4;
            var defaultOctave = 5;
            var bpm = 120;

            foreach (var param in sections[1].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                var kv = param.Split('=', 2, StringSplitOptions.TrimEntries);
                if (kv.Length != 2 || kv[0].Length == 0 || !int.TryParse(kv[1], out var value) || value <= 0)
                {
                    continue;
                }

                switch (char.ToLowerInvariant(kv[0][0]))
                {
                    case 'd':
                        defaultDuration = value;
                        break;
                    case 'o':
                        defaultOctave = value;
                        break;
                    case 'b':
                        bpm = value;
                        break;
                }
            }

            var wholeNoteSeconds = 240f / bpm;
            var notes = new List<ScoreNote>();
            var cursor = 0f;

            foreach (var rawToken in sections[2].Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            {
                var token = rawToken.AsSpan().Trim();
                if (token.IsEmpty)
                {
                    continue;
                }

                var index = 0;
                var durationValue = ReadDigits(token, ref index, defaultDuration);
                var durationSeconds = wholeNoteSeconds / Math.Max(1, durationValue);

                if (index >= token.Length)
                {
                    continue;
                }

                var noteChar = char.ToLowerInvariant(token[index++]);
                var semitone = noteChar switch
                {
                    'c' => 0,
                    'd' => 2,
                    'e' => 4,
                    'f' => 5,
                    'g' => 7,
                    'a' => 9,
                    'b' or 'h' => 11,
                    'p' or 'r' => -1,
                    _ => throw new FormatException($"Unknown RTTTL note '{noteChar}' in token '{rawToken}'."),
                };

                if (index < token.Length && (token[index] == '#' || token[index] == '_'))
                {
                    semitone++;
                    index++;
                }

                var dotted = false;
                if (index < token.Length && token[index] == '.')
                {
                    dotted = true;
                    index++;
                }

                var octave = ReadDigits(token, ref index, defaultOctave);

                // In some RTTTL files the dot appears after the octave number (e.g. "4c6.")
                if (index < token.Length && token[index] == '.')
                {
                    dotted = true;
                }

                if (dotted)
                {
                    durationSeconds *= 1.5f;
                }

                if (semitone >= 0)
                {
                    var midiNote = (octave + 1) * 12 + semitone;
                    var frequency = DroidScore.MidiNoteToFrequency(midiNote);
                    // Slight staccato gap (92% gate) keeps repeated notes articulate
                    notes.Add(new ScoreNote(cursor, durationSeconds * 0.92f, frequency, Velocity: 0.85f));
                }

                cursor += durationSeconds;
            }

            return new DroidScore(notes, title);
        }

        private static int ReadDigits(ReadOnlySpan<char> span, ref int index, int fallback)
        {
            var start = index;
            while (index < span.Length && char.IsAsciiDigit(span[index]))
            {
                index++;
            }

            return index > start && int.TryParse(span[start..index], out var parsed) && parsed > 0
                ? parsed
                : fallback;
        }
    }

}
