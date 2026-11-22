using System;
using System.Collections.Generic;
using System.Text;
using System.Buffers.Binary;

namespace DroidSpeak.Music
{
    /// <summary>
    /// Zero-dependency parser for Standard MIDI Files (.mid, SMF Format 0 and 1).
    /// </summary>
    public sealed class MidiParser : IScoreParser
    {
        public static MidiParser Shared { get; } = new();

        public DroidScore Parse(ReadOnlySpan<byte> data)
        {
            var reader = new MidiReader(data);

            if (!reader.ReadTag("MThd"))
            {
                throw new FormatException("Invalid MIDI file: missing MThd header.");
            }

            var headerLength = reader.ReadUInt32BigEndian();
            if (headerLength < 6)
            {
                throw new FormatException("Invalid MIDI header length.");
            }

            var format = reader.ReadUInt16BigEndian();
            if (format > 1)
            {
                throw new NotSupportedException($"SMF format {format} is not supported (only Format 0 and 1).");
            }

            var trackCount = reader.ReadUInt16BigEndian();
            var division = reader.ReadUInt16BigEndian();

            if ((division & 0x8000) != 0)
            {
                throw new NotSupportedException("SMPTE timecode division is not supported.");
            }

            int ticksPerQuarterNote = division;
            if (ticksPerQuarterNote <= 0)
            {
                throw new FormatException("Invalid MIDI ticks per quarter note.");
            }

            // Skip any extra header bytes if headerLength > 6
            if (headerLength > 6)
            {
                reader.Skip((int)(headerLength - 6));
            }

            var rawNotes = new List<RawMidiNote>();
            var tempoChanges = new List<TempoChange> { new(0, 500_000) }; // Default 120 BPM (500,000 us/quarter)
            string? title = null;

            for (var track = 0; track < trackCount && !reader.IsEnd; track++)
            {
                if (!reader.ReadTag("MTrk"))
                {
                    throw new FormatException($"Invalid MIDI file: missing MTrk chunk at track {track}.");
                }

                var trackLength = reader.ReadUInt32BigEndian();
                var trackEnd = Math.Min(data.Length, reader.Position + (int)trackLength);
                var trackReader = new MidiReader(data[..trackEnd], reader.Position);
                reader.Skip((int)trackLength);

                long currentTick = 0;
                byte runningStatus = 0;
                var activeNotes = new Dictionary<(int Channel, int Note), (long StartTick, int Velocity)>();

                while (!trackReader.IsEnd)
                {
                    var delta = trackReader.ReadVlq();
                    currentTick += delta;

                    var statusByte = trackReader.PeekByte();
                    if ((statusByte & 0x80) != 0)
                    {
                        statusByte = trackReader.ReadByte();
                        if (statusByte < 0xF0)
                        {
                            runningStatus = statusByte;
                        }
                    }
                    else
                    {
                        statusByte = runningStatus;
                    }

                    if (statusByte == 0xFF) // Meta Event
                    {
                        var metaType = trackReader.ReadByte();
                        var metaLength = (int)trackReader.ReadVlq();
                        var metaData = trackReader.ReadBytes(metaLength);

                        if (metaType == 0x51 && metaData.Length == 3) // Set Tempo
                        {
                            int microsecondsPerQuarter =
                                (metaData[0] << 16) |
                                (metaData[1] << 8) |
                                metaData[2];
                            tempoChanges.Add(new TempoChange(currentTick, microsecondsPerQuarter));
                        }
                        else if (metaType == 0x03 && title is null && metaData.Length > 0) // Track Name
                        {
                            title = Encoding.UTF8.GetString(metaData).Trim();
                        }
                        else if (metaType == 0x2F) // End of Track
                        {
                            break;
                        }
                    }
                    else if (statusByte == 0xF0 || statusByte == 0xF7) // SysEx
                    {
                        var sysExLength = (int)trackReader.ReadVlq();
                        trackReader.Skip(sysExLength);
                    }
                    else
                    {
                        var command = statusByte & 0xF0;
                        var channel = statusByte & 0x0F;

                        switch (command)
                        {
                            case 0x80: // Note Off
                                {
                                    var note = trackReader.ReadByte() & 0x7F;
                                    _ = trackReader.ReadByte(); // release velocity
                                    CloseNote(activeNotes, rawNotes, channel, note, currentTick);
                                    break;
                                }
                            case 0x90: // Note On
                                {
                                    var note = trackReader.ReadByte() & 0x7F;
                                    var velocity = trackReader.ReadByte() & 0x7F;
                                    if (velocity == 0)
                                    {
                                        CloseNote(activeNotes, rawNotes, channel, note, currentTick);
                                    }
                                    else
                                    {
                                        // Close previous overlapping note on the same pitch if any
                                        CloseNote(activeNotes, rawNotes, channel, note, currentTick);
                                        activeNotes[(channel, note)] = (currentTick, velocity);
                                    }

                                    break;
                                }
                            case 0xA0: // Polyphonic Key Pressure
                            case 0xB0: // Control Change
                            case 0xE0: // Pitch Bend
                                trackReader.Skip(2);
                                break;

                            case 0xC0: // Program Change
                            case 0xD0: // Channel Pressure
                                trackReader.Skip(1);
                                break;
                        }
                    }
                }

                // Close any unclosed notes at the end of the track
                foreach (var ((channel, note), (startTick, velocity)) in activeNotes)
                {
                    if (currentTick > startTick)
                    {
                        rawNotes.Add(new RawMidiNote(startTick, currentTick, note, velocity, channel == 9));
                    }
                }
            }

            tempoChanges.Sort((a, b) => a.Tick.CompareTo(b.Tick));
            rawNotes.Sort((a, b) => a.StartTick.CompareTo(b.StartTick));

            var notes = new List<ScoreNote>(rawNotes.Count);
            foreach (var raw in rawNotes)
            {
                var startSec = TicksToSeconds(raw.StartTick, ticksPerQuarterNote, tempoChanges);
                var endSec = TicksToSeconds(raw.EndTick, ticksPerQuarterNote, tempoChanges);
                var durationSec = MathF.Max(0.015f, endSec - startSec);
                var frequency = raw.IsPercussion ? 0f : DroidScore.MidiNoteToFrequency(raw.MidiNote);
                var velocity = Math.Clamp(raw.Velocity / 127f, 0.05f, 1f);

                notes.Add(new ScoreNote(startSec, durationSec, frequency, velocity, raw.IsPercussion));
            }

            return new DroidScore(notes, title);
        }

        private static void CloseNote(
            Dictionary<(int Channel, int Note), (long StartTick, int Velocity)> activeNotes,
            List<RawMidiNote> output,
            int channel,
            int note,
            long endTick)
        {
            if (activeNotes.Remove((channel, note), out var started) && endTick > started.StartTick)
            {
                output.Add(new RawMidiNote(started.StartTick, endTick, note, started.Velocity, channel == 9));
            }
        }

        private static float TicksToSeconds(long targetTick, int ticksPerQuarter, List<TempoChange> tempos)
        {
            double seconds = 0;
            long lastTick = 0;
            int currentTempo = 500_000;

            foreach (var change in tempos)
            {
                if (change.Tick >= targetTick)
                {
                    break;
                }

                if (change.Tick > lastTick)
                {
                    seconds += (change.Tick - lastTick) * (currentTempo / 1_000_000.0) / ticksPerQuarter;
                    lastTick = change.Tick;
                }

                currentTempo = change.MicrosecondsPerQuarter;
            }

            if (targetTick > lastTick)
            {
                seconds += (targetTick - lastTick) * (currentTempo / 1_000_000.0) / ticksPerQuarter;
            }

            return (float)seconds;
        }

        private readonly record struct TempoChange(long Tick, int MicrosecondsPerQuarter);

        private readonly record struct RawMidiNote(
            long StartTick,
            long EndTick,
            int MidiNote,
            int Velocity,
            bool IsPercussion);

        private ref struct MidiReader
        {
            private readonly ReadOnlySpan<byte> _data;
            private int _pos;

            public MidiReader(ReadOnlySpan<byte> data, int startPos = 0)
            {
                _data = data;
                _pos = startPos;
            }

            public readonly int Position => _pos;
            public readonly bool IsEnd => _pos >= _data.Length;

            public readonly byte PeekByte() =>
                _pos < _data.Length ? _data[_pos] : throw new FormatException("Unexpected end of MIDI data.");

            public byte ReadByte() =>
                _pos < _data.Length ? _data[_pos++] : throw new FormatException("Unexpected end of MIDI data.");

            public ReadOnlySpan<byte> ReadBytes(int count)
            {
                if (count < 0 || _pos + count > _data.Length)
                {
                    throw new FormatException("Unexpected end of MIDI data.");
                }

                var slice = _data.Slice(_pos, count);
                _pos += count;
                return slice;
            }

            public void Skip(int count)
            {
                if (count < 0 || _pos + count > _data.Length)
                {
                    throw new FormatException("Unexpected end of MIDI data.");
                }
                _pos += count;
            }
            public bool ReadTag(string expected)
            {
                if (_pos + 4 > _data.Length)
                {
                    return false;
                }
                var tag = Encoding.ASCII.GetString(_data.Slice(_pos, 4));
                _pos += 4;
                return tag == expected;
            }

            public ushort ReadUInt16BigEndian()
            {
                var value = BinaryPrimitives.ReadUInt16BigEndian(ReadBytes(2));
                return value;
            }
            public uint ReadUInt32BigEndian()
            {
                var value = BinaryPrimitives.ReadUInt32BigEndian(ReadBytes(4));
                return value;
            }
            public uint ReadVlq()
            {
                uint value = 0;
                for (var i = 0; i < 4; i++)
                {
                    var b = ReadByte();
                    value = (value << 7) | (uint)(b & 0x7F);
                    if ((b & 0x80) == 0)
                    {
                        return value;
                    }
                }
                throw new FormatException("Invalid MIDI variable-length quantity.");
            }
        }
    }
}
