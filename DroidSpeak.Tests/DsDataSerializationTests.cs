using DroidSpeak;
using DroidSpeak.Trainer.Datasets;

namespace DroidSpeak.Tests;

/// <summary>
/// The <c>.dsdata</c> container: a saved dataset must load back field for field, and a corrupted
/// container must fail loudly rather than produce a dataset with silently missing or shifted rows.
/// </summary>
public class DsDataSerializationTests
{
    [Fact]
    public void SaveLoad_RoundTripsEverySerializedField()
    {
        var dataset = AcousticFixtures.BuildDataset(3, DroidVoiceProfile.R2, AcousticFixtures.Rate, 42);
        var bytes = SaveToArray(dataset);

        using var stream = new MemoryStream(bytes);
        var restored = DsDataFile.Load(stream);

        Assert.Equal(dataset.RenderRate, restored.RenderRate);
        Assert.Equal(dataset.Seed, restored.Seed);
        Assert.Equal(dataset.ClipCount, restored.ClipCount);
        Assert.Equal(dataset.SyllableCount, restored.SyllableCount);

        for (var clip = 0; clip < dataset.ClipCount; clip++)
        {
            var expected = dataset.Clips[clip];
            var actual = restored.Clips[clip];

            Assert.Equal(expected.ClipId, actual.ClipId);
            Assert.Equal(expected.Path, actual.Path);
            Assert.Equal(expected.Contexts.Length, actual.Contexts.Length);
            Assert.Equal(expected.Frames.Length, actual.Frames.Length);

            for (var i = 0; i < expected.Contexts.Length; i++)
            {
                AcousticAssert.ContextEqual(expected.Contexts[i], actual.Contexts[i]);
                AcousticAssert.FrameEqual(expected.Frames[i], actual.Frames[i]);
            }
        }
    }

    [Fact]
    public void Load_RejectsBadMagic()
    {
        var bytes = SaveToArray(AcousticFixtures.BuildDataset(2, DroidVoiceProfile.R2, AcousticFixtures.Rate, 1));
        bytes[0] = (byte)'X';

        Assert.Throws<InvalidDataException>(() => DsDataFile.Load(new MemoryStream(bytes)));
    }

    [Fact]
    public void Load_RejectsUnsupportedVersion()
    {
        var bytes = SaveToArray(AcousticFixtures.BuildDataset(2, DroidVoiceProfile.R2, AcousticFixtures.Rate, 1));
        bytes[4] = byte.MaxValue;

        Assert.Throws<InvalidDataException>(() => DsDataFile.Load(new MemoryStream(bytes)));
    }

    [Fact]
    public void Load_RejectsTrailingBytes()
    {
        var bytes = SaveToArray(AcousticFixtures.BuildDataset(2, DroidVoiceProfile.R2, AcousticFixtures.Rate, 1));
        var longer = new byte[bytes.Length + 1];
        bytes.CopyTo(longer, 0);

        Assert.Throws<InvalidDataException>(() => DsDataFile.Load(new MemoryStream(longer)));
    }

    [Fact]
    public void Load_RejectsATruncatedContainer()
    {
        var bytes = SaveToArray(AcousticFixtures.BuildDataset(2, DroidVoiceProfile.R2, AcousticFixtures.Rate, 1));
        var truncated = bytes[..(bytes.Length - 1)];

        Assert.Throws<InvalidDataException>(() => DsDataFile.Load(new MemoryStream(truncated)));
    }

    private static byte[] SaveToArray(AcousticDataset dataset)
    {
        using var stream = new MemoryStream();
        DsDataFile.Save(dataset, stream);
        return stream.ToArray();
    }
}