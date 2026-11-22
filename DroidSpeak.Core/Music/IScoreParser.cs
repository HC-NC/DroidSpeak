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
        DroidScore Parse(ReadOnlySpan<byte> data);
    }

    /// <summary>
    /// Parses text-based musical notations (such as RTTTL or MML) into a <see cref="DroidScore"/>.
    /// </summary>
    public interface ITextScoreParser : IScoreParser
    {
        DroidScore Parse(string text);

        DroidScore IScoreParser.Parse(ReadOnlySpan<byte> data) =>
            Parse(Encoding.UTF8.GetString(data));
    }

    /// <summary>
    /// Convenience helpers for loading scores from streams and files.
    /// </summary>
    public static class ScoreParserExtensions
    {
        public static DroidScore Parse(this IScoreParser parser, Stream stream)
        {
            ArgumentNullException.ThrowIfNull(parser);
            ArgumentNullException.ThrowIfNull(stream);

            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            return parser.Parse(ms.GetBuffer().AsSpan(0, (int)ms.Length));
        }

        public static DroidScore Load(this IScoreParser parser, string filePath)
        {
            ArgumentNullException.ThrowIfNull(parser);
            ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

            byte[] bytes = File.ReadAllBytes(filePath);
            return parser.Parse(bytes);
        }
    }

}
