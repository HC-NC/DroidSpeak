using System;
using System.Collections.Generic;
using System.Text;

namespace DroidSpeak.Music
{
        /// <summary>
    /// Parses raw binary data into a format-agnostic <see cref="DroidScore"/>.
    /// </summary>
    public interface IScoreParser
    {
        /// <summary>Parses raw binary score data into a <see cref="DroidScore"/>.</summary>
        /// <param name="data">The file bytes (for example a Standard MIDI File).</param>
        /// <exception cref="FormatException">The data is not a recognizable score in this format.</exception>
        /// <exception cref="NotSupportedException">The format is recognized but uses an unsupported variant.</exception>
        DroidScore Parse(ReadOnlySpan<byte> data);
    }

    /// <summary>
    /// Parses text-based musical notations (such as RTTTL or MML) into a <see cref="DroidScore"/>.
    /// </summary>
    public interface ITextScoreParser : IScoreParser
    {
        /// <summary>Parses a text score into a <see cref="DroidScore"/>.</summary>
        /// <param name="text">The notation source text.</param>
        /// <exception cref="FormatException">The text is not a recognizable score in this format.</exception>
        DroidScore Parse(string text);

        DroidScore IScoreParser.Parse(ReadOnlySpan<byte> data) =>
            Parse(Encoding.UTF8.GetString(data));
    }

    /// <summary>
    /// Convenience helpers for loading scores from streams and files.
    /// </summary>
    public static class ScoreParserExtensions
    {
        /// <summary>Reads <paramref name="stream"/> fully and parses it as a score.</summary>
        /// <param name="parser">The format parser.</param>
        /// <param name="stream">Readable stream positioned at the start of the score data.</param>
        /// <exception cref="ArgumentNullException"><paramref name="parser"/> or <paramref name="stream"/> is null.</exception>
        public static DroidScore Parse(this IScoreParser parser, Stream stream)
        {
            ArgumentNullException.ThrowIfNull(parser);
            ArgumentNullException.ThrowIfNull(stream);

            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            return parser.Parse(ms.GetBuffer().AsSpan(0, (int)ms.Length));
        }

        /// <summary>Reads the file at <paramref name="filePath"/> and parses it as a score.</summary>
        /// <param name="parser">The format parser.</param>
        /// <param name="filePath">Path to the score file.</param>
        /// <exception cref="ArgumentNullException"><paramref name="parser"/> is null.</exception>
        /// <exception cref="ArgumentException"><paramref name="filePath"/> is null or whitespace.</exception>
        /// <exception cref="System.IO.FileNotFoundException">No file exists at <paramref name="filePath"/>.</exception>
        public static DroidScore Load(this IScoreParser parser, string filePath)
        {
            ArgumentNullException.ThrowIfNull(parser);
            ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

            byte[] bytes = File.ReadAllBytes(filePath);
            return parser.Parse(bytes);
        }
    }

}
