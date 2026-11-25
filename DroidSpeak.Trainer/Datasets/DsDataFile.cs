using System.Text;
using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.Trainer.Datasets;

/// <summary>
/// The <c>.dsdata</c> container: a little-endian, self-describing dump of extracted acoustic clips.
/// <para>
/// A dataset is stored as its clips rather than as encoded rows because the encoding depends on the
/// voice profile and the sample rate, both of which are cheap to re-apply but expensive to bake in
/// wrongly. Keeping contexts and frames also means a dataset can be re-encoded against a different
/// profile without re-reading the source WAVs, and a test can compare the exact structs that were
/// extracted.
/// </para>
/// <para>
/// The layout is versioned and intentionally plain: a fixed header, then a length-prefixed path and
/// a run of fixed-size records per clip. There is no compression and no alignment padding, so a
/// dataset can be inspected with any binary tool and a future version can add fields behind a version
/// bump instead of a second magic number.
/// </para>
/// </summary>
internal static class DsDataFile
{
    /// <summary>Four-byte container signature, "DSDA".</summary>
    public static ReadOnlySpan<byte> Magic => "DSDA"u8;

    /// <summary>Current container version.</summary>
    public const byte Version = 1;

    /// <summary>Size of the fixed header in bytes: magic(4) + version(1) + flags(1) + five ints.</summary>
    public const int HeaderSize = 4 + 1 + 1 + (5 * 4);

    /// <summary>Serialized size of one context record.</summary>
    public const int ContextBytes = (12 * 4) + 1 + 1 + 2;

    /// <summary>Serialized size of one frame record.</summary>
    public const int FrameBytes = (31 * 4) + (6 * 4) + 4 + 1;

    /// <summary>Largest path a dataset record may carry, in UTF-8 bytes.</summary>
    public const int MaxPathBytes = 4096;

    /// <summary>Largest clip count accepted on load.</summary>
    public const int MaxClips = 1_000_000;

    /// <summary>Largest syllable count accepted on load.</summary>
    public const int MaxSyllables = 20_000_000;

    private static readonly UTF8Encoding Utf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    /// <summary>Writes a dataset to disk, truncating any existing file.</summary>
    public static void Save(AcousticDataset dataset, string path)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentNullException.ThrowIfNull(path);

        using var stream = File.Create(path);
        Save(dataset, stream);
    }

    /// <summary>Writes a dataset to a writable stream, which is left open.</summary>
    public static void Save(AcousticDataset dataset, Stream stream)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentNullException.ThrowIfNull(stream);

        using var writer = new BinaryWriter(stream, Utf8, leaveOpen: true);

        writer.Write(Magic);
        writer.Write(Version);
        writer.Write((byte)0);
        writer.Write(dataset.RenderRate);
        writer.Write(dataset.Seed);
        writer.Write(dataset.ClipCount);
        writer.Write(dataset.SyllableCount);

        foreach (var clip in dataset.Clips)
        {
            var pathBytes = Utf8.GetBytes(clip.Path);
            if (pathBytes.Length > MaxPathBytes)
            {
                throw new ArgumentException(
                    $"Clip path '{clip.Path}' is longer than {MaxPathBytes} UTF-8 bytes.",
                    nameof(dataset));
            }

            var contexts = clip.Contexts;
            var frames = clip.Frames;
            if (contexts.Length != frames.Length)
            {
                throw new ArgumentException(
                    $"Clip '{clip.Path}' has mismatched context and frame counts.",
                    nameof(dataset));
            }

            writer.Write(clip.ClipId);
            writer.Write(pathBytes.Length);
            writer.Write(pathBytes);
            writer.Write(contexts.Length);

            for (var i = 0; i < contexts.Length; i++)
            {
                WriteContext(writer, contexts[i]);
            }

            for (var i = 0; i < frames.Length; i++)
            {
                WriteFrame(writer, frames[i]);
            }
        }
    }

    /// <summary>Reads and validates a dataset from disk.</summary>
    /// <exception cref="InvalidDataException">The file is not a valid <c>.dsdata</c> container.</exception>
    public static AcousticDataset Load(string path)
    {
        ArgumentNullException.ThrowIfNull(path);

        using var stream = File.OpenRead(path);
        return Load(stream);
    }

    /// <summary>Reads and validates a dataset from a stream, which is left open.</summary>
    /// <exception cref="InvalidDataException">The stream is not a valid <c>.dsdata</c> container.</exception>
    public static AcousticDataset Load(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);

        using var reader = new BinaryReader(stream, Utf8, leaveOpen: true);

        try
        {
            return Read(reader, stream);
        }
        catch (EndOfStreamException exception)
        {
            throw new InvalidDataException("The dataset ends before its declared length.", exception);
        }
    }

    private static AcousticDataset Read(BinaryReader reader, Stream stream)
    {
        Span<byte> magic = stackalloc byte[4];
        if (reader.Read(magic) != magic.Length || !magic.SequenceEqual(Magic))
        {
            throw new InvalidDataException("The file is not a .dsdata container.");
        }

        var version = reader.ReadByte();
        if (version != Version)
        {
            throw new InvalidDataException(
                $"Unsupported .dsdata version {version}; this build reads version {Version}.");
        }

        var flags = reader.ReadByte();
        if (flags != 0)
        {
            throw new InvalidDataException($"Unsupported .dsdata flags 0x{flags:X2}.");
        }

        var renderRate = reader.ReadInt32();
        if (renderRate is < 8000 or > 192000)
        {
            throw new InvalidDataException(
                $"The dataset declares a render rate of {renderRate} Hz, outside 8000..192000.");
        }

        var seed = reader.ReadUInt32();
        var clipCount = reader.ReadInt32();
        var declaredSyllables = reader.ReadInt32();

        if (clipCount is < 1 or > MaxClips)
        {
            throw new InvalidDataException(
                $"The dataset declares {clipCount} clips, outside 1..{MaxClips}.");
        }

        if (declaredSyllables is < 0 or > MaxSyllables)
        {
            throw new InvalidDataException(
                $"The dataset declares {declaredSyllables} syllables, outside 0..{MaxSyllables}.");
        }

        var clips = new List<AcousticClip>(clipCount);
        var totalSyllables = 0;

        for (var clipIndex = 0; clipIndex < clipCount; clipIndex++)
        {
            var clipId = reader.ReadUInt32();
            var pathLength = reader.ReadInt32();

            if (pathLength is < 0 or > MaxPathBytes)
            {
                throw new InvalidDataException(
                    $"Clip {clipIndex} declares a path of {pathLength} bytes, outside 0..{MaxPathBytes}.");
            }

            var pathBytes = reader.ReadBytes(pathLength);
            if (pathBytes.Length != pathLength)
            {
                throw new InvalidDataException($"Clip {clipIndex} has a truncated path.");
            }

            var path = Utf8.GetString(pathBytes);
            var syllableCount = reader.ReadInt32();

            if (syllableCount is < 0 or > MaxSyllables)
            {
                throw new InvalidDataException(
                    $"Clip {clipIndex} declares {syllableCount} syllables, outside 0..{MaxSyllables}.");
            }

            totalSyllables += syllableCount;
            if (totalSyllables > MaxSyllables)
            {
                throw new InvalidDataException(
                    $"The dataset holds more than {MaxSyllables} syllables.");
            }

            var contexts = new SyllableContext[syllableCount];
            var frames = new SyllableFrame[syllableCount];

            for (var i = 0; i < syllableCount; i++)
            {
                contexts[i] = ReadContext(reader, clipIndex, i, renderRate);
            }

            for (var i = 0; i < syllableCount; i++)
            {
                frames[i] = ReadFrame(reader, clipIndex, i);
            }

            clips.Add(new AcousticClip(clipId, path, contexts, frames));
        }

        if (totalSyllables != declaredSyllables)
        {
            throw new InvalidDataException(
                $"The header declared {declaredSyllables} syllables but the clips hold " +
                $"{totalSyllables}.");
        }

        if (stream.CanSeek && stream.Position != stream.Length)
        {
            throw new InvalidDataException(
                $"The dataset has {stream.Length - stream.Position} trailing bytes.");
        }

        return new AcousticDataset(clips, renderRate, seed);
    }

    private static void WriteContext(BinaryWriter writer, in SyllableContext context)
    {
        writer.Write(context.SampleRate);
        writer.Write(context.GlobalSeed);
        writer.Write(context.WordHash);
        writer.Write(context.PhraseIndex);
        writer.Write(context.WordIndexInPhrase);
        writer.Write(context.SyllableIndex);
        writer.Write(context.WordSyllableCount);
        writer.Write(context.PhraseSyllableCount);
        writer.Write(context.WordTextLength);
        writer.Write(context.TempoScale);
        writer.Write(context.EmotionValence);
        writer.Write(context.EmotionArousal);
        writer.Write((byte)context.Punctuation);
        writer.Write(ContextFlags(in context));
        writer.Write((ushort)context.Nucleus);
    }

    private static SyllableContext ReadContext(
        BinaryReader reader,
        int clipIndex,
        int syllableIndex,
        int renderRate)
    {
        var context = new SyllableContext
        {
            SampleRate = reader.ReadInt32(),
            GlobalSeed = reader.ReadUInt32(),
            WordHash = reader.ReadUInt32(),
            PhraseIndex = reader.ReadInt32(),
            WordIndexInPhrase = reader.ReadInt32(),
            SyllableIndex = reader.ReadInt32(),
            WordSyllableCount = reader.ReadInt32(),
            PhraseSyllableCount = reader.ReadInt32(),
            WordTextLength = reader.ReadInt32(),
            TempoScale = reader.ReadSingle(),
            EmotionValence = reader.ReadSingle(),
            EmotionArousal = reader.ReadSingle(),
        };

        var punctuation = reader.ReadByte();
        var flags = reader.ReadByte();
        var nucleus = reader.ReadUInt16();

        if (context.SampleRate != renderRate)
        {
            throw new InvalidDataException(
                $"Clip {clipIndex} syllable {syllableIndex} has a {context.SampleRate} Hz context, " +
                $"but the dataset renders at {renderRate} Hz.");
        }

        if (context.WordSyllableCount < 1 ||
            context.SyllableIndex < 0 ||
            context.SyllableIndex >= context.WordSyllableCount)
        {
            throw new InvalidDataException(
                $"Clip {clipIndex} syllable {syllableIndex} has an invalid position " +
                $"{context.SyllableIndex}/{context.WordSyllableCount}.");
        }

        if (punctuation > (byte)PhrasePunctuation.Ellipsis)
        {
            throw new InvalidDataException(
                $"Clip {clipIndex} syllable {syllableIndex} has punctuation {punctuation}.");
        }

        if (!float.IsFinite(context.TempoScale) ||
            !float.IsFinite(context.EmotionValence) ||
            !float.IsFinite(context.EmotionArousal))
        {
            throw new InvalidDataException(
                $"Clip {clipIndex} syllable {syllableIndex} carries a non-finite context.");
        }

        context.Punctuation = (PhrasePunctuation)punctuation;
        context.IsFirstSpokenPhrase = (flags & 0x01) != 0;
        context.PreviousWordHadComma = (flags & 0x02) != 0;
        context.IsShout = (flags & 0x04) != 0;
        context.IsStressed = (flags & 0x08) != 0;
        context.HasVoicedOnset = (flags & 0x10) != 0;
        context.HasSibilantOnset = (flags & 0x20) != 0;
        context.IsHyphenBoundary = (flags & 0x40) != 0;
        context.Nucleus = (char)nucleus;
        return context;
    }

    private static byte ContextFlags(in SyllableContext context)
    {
        var flags = 0;
        if (context.IsFirstSpokenPhrase) flags |= 0x01;
        if (context.PreviousWordHadComma) flags |= 0x02;
        if (context.IsShout) flags |= 0x04;
        if (context.IsStressed) flags |= 0x08;
        if (context.HasVoicedOnset) flags |= 0x10;
        if (context.HasSibilantOnset) flags |= 0x20;
        if (context.IsHyphenBoundary) flags |= 0x40;
        return (byte)flags;
    }

    private static void WriteFrame(BinaryWriter writer, in SyllableFrame frame)
    {
        writer.Write(frame.DurationSeconds);
        writer.Write(frame.StartFreq);
        writer.Write(frame.MidFreq);
        writer.Write(frame.EndFreq);
        writer.Write(frame.StartGain);
        writer.Write(frame.Gain);
        writer.Write(frame.StartFmDepth);
        writer.Write(frame.FmDepth);
        writer.Write(frame.StartFmDepth2);
        writer.Write(frame.FmDepth2);
        writer.Write(frame.FmRatio);
        writer.Write(frame.FmRatio2);
        writer.Write(frame.StartGrowlIntensity);
        writer.Write(frame.GrowlIntensity);
        writer.Write(frame.GrowlFrequency);
        writer.Write(frame.NoiseMix);
        writer.Write(frame.RingModMix);
        writer.Write(frame.AttackChirp);
        writer.Write(frame.FormantF1);
        writer.Write(frame.FormantF2);
        writer.Write(frame.FormantMix);
        writer.Write(frame.AttackTime);
        writer.Write(frame.DecayTime);
        writer.Write(frame.ReleaseTime);
        writer.Write(frame.SustainLevel);
        writer.Write(frame.TrillDepth);
        writer.Write(frame.TrillRate);
        writer.Write(frame.Drive);
        writer.Write(frame.Saturation);
        writer.Write(frame.PauseBeforeSeconds);
        writer.Write(frame.PauseAfterSeconds);

        writer.Write(frame.StartSample);
        writer.Write(frame.LengthSamples);
        writer.Write(frame.PauseBeforeSamples);
        writer.Write(frame.PauseAfterSamples);
        writer.Write(frame.WordStartSample);
        writer.Write(frame.WordEndSample);

        writer.Write(frame.Seed);
        writer.Write(FrameFlags(in frame));
    }

    private static SyllableFrame ReadFrame(BinaryReader reader, int clipIndex, int syllableIndex)
    {
        var frame = new SyllableFrame
        {
            DurationSeconds = reader.ReadSingle(),
            StartFreq = reader.ReadSingle(),
            MidFreq = reader.ReadSingle(),
            EndFreq = reader.ReadSingle(),
            StartGain = reader.ReadSingle(),
            Gain = reader.ReadSingle(),
            StartFmDepth = reader.ReadSingle(),
            FmDepth = reader.ReadSingle(),
            StartFmDepth2 = reader.ReadSingle(),
            FmDepth2 = reader.ReadSingle(),
            FmRatio = reader.ReadSingle(),
            FmRatio2 = reader.ReadSingle(),
            StartGrowlIntensity = reader.ReadSingle(),
            GrowlIntensity = reader.ReadSingle(),
            GrowlFrequency = reader.ReadSingle(),
            NoiseMix = reader.ReadSingle(),
            RingModMix = reader.ReadSingle(),
            AttackChirp = reader.ReadSingle(),
            FormantF1 = reader.ReadSingle(),
            FormantF2 = reader.ReadSingle(),
            FormantMix = reader.ReadSingle(),
            AttackTime = reader.ReadSingle(),
            DecayTime = reader.ReadSingle(),
            ReleaseTime = reader.ReadSingle(),
            SustainLevel = reader.ReadSingle(),
            TrillDepth = reader.ReadSingle(),
            TrillRate = reader.ReadSingle(),
            Drive = reader.ReadSingle(),
            Saturation = reader.ReadSingle(),
            PauseBeforeSeconds = reader.ReadSingle(),
            PauseAfterSeconds = reader.ReadSingle(),
            StartSample = reader.ReadInt32(),
            LengthSamples = reader.ReadInt32(),
            PauseBeforeSamples = reader.ReadInt32(),
            PauseAfterSamples = reader.ReadInt32(),
            WordStartSample = reader.ReadInt32(),
            WordEndSample = reader.ReadInt32(),
            Seed = reader.ReadUInt32(),
        };

        var flags = reader.ReadByte();

        if (frame.LengthSamples < 1)
        {
            throw new InvalidDataException(
                $"Clip {clipIndex} syllable {syllableIndex} has a length of {frame.LengthSamples}.");
        }

        if (frame.PauseBeforeSamples < 0 || frame.PauseAfterSamples < 0)
        {
            throw new InvalidDataException(
                $"Clip {clipIndex} syllable {syllableIndex} has a negative pause.");
        }

        if (!IsFinite(in frame))
        {
            throw new InvalidDataException(
                $"Clip {clipIndex} syllable {syllableIndex} carries a non-finite frame value.");
        }

        frame.HasVoicedOnset = (flags & 0x01) != 0;
        frame.StartsLegatoSegment = (flags & 0x02) != 0;
        return frame;
    }

    private static byte FrameFlags(in SyllableFrame frame)
    {
        var flags = 0;
        if (frame.HasVoicedOnset) flags |= 0x01;
        if (frame.StartsLegatoSegment) flags |= 0x02;
        return (byte)flags;
    }

    private static bool IsFinite(in SyllableFrame frame) =>
        float.IsFinite(frame.DurationSeconds) &&
        float.IsFinite(frame.StartFreq) &&
        float.IsFinite(frame.MidFreq) &&
        float.IsFinite(frame.EndFreq) &&
        float.IsFinite(frame.StartGain) &&
        float.IsFinite(frame.Gain) &&
        float.IsFinite(frame.StartFmDepth) &&
        float.IsFinite(frame.FmDepth) &&
        float.IsFinite(frame.StartFmDepth2) &&
        float.IsFinite(frame.FmDepth2) &&
        float.IsFinite(frame.FmRatio) &&
        float.IsFinite(frame.FmRatio2) &&
        float.IsFinite(frame.StartGrowlIntensity) &&
        float.IsFinite(frame.GrowlIntensity) &&
        float.IsFinite(frame.GrowlFrequency) &&
        float.IsFinite(frame.NoiseMix) &&
        float.IsFinite(frame.RingModMix) &&
        float.IsFinite(frame.AttackChirp) &&
        float.IsFinite(frame.FormantF1) &&
        float.IsFinite(frame.FormantF2) &&
        float.IsFinite(frame.FormantMix) &&
        float.IsFinite(frame.AttackTime) &&
        float.IsFinite(frame.DecayTime) &&
        float.IsFinite(frame.ReleaseTime) &&
        float.IsFinite(frame.SustainLevel) &&
        float.IsFinite(frame.TrillDepth) &&
        float.IsFinite(frame.TrillRate) &&
        float.IsFinite(frame.Drive) &&
        float.IsFinite(frame.Saturation) &&
        float.IsFinite(frame.PauseBeforeSeconds) &&
        float.IsFinite(frame.PauseAfterSeconds);
}