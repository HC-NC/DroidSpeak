using System.Text;
using DroidSpeak.Neural;
using DroidSpeak.Web;

namespace DroidSpeak.Tests;

// Coverage for the playground's manifest-driven brain loading. The parser and catalog are plain
// classes (no Blazor rendering), so a stub HttpMessageHandler exercises the exact HTTP paths the
// browser will take, including the app-relative URLs that must survive the GitHub Pages base path.
public class BrainManifestTests
{
    [Fact]
    public void Parse_EmptyArray_ReturnsNoFiles()
    {
        var result = BrainManifest.Parse("[]");

        Assert.Empty(result.Files);
        Assert.Null(result.Warning);
    }

    [Fact]
    public void Parse_ValidNames_AreKeptInOrder()
    {
        var result = BrainManifest.Parse("""["r2.dsbrain", "Bass.DSBRAIN"]""");

        Assert.Equal(["r2.dsbrain", "Bass.DSBRAIN"], result.Files);
        Assert.Null(result.Warning);
    }

    [Theory]
    [InlineData("brains/r2.dsbrain")]
    [InlineData("../r2.dsbrain")]
    [InlineData("http://example.com/r2.dsbrain")]
    [InlineData("r2.dsbrain?cache=1")]
    [InlineData(".hidden.dsbrain")]
    [InlineData("r2.dsbrain%2F..%2Fsecret")]
    [InlineData("r2.txt")]
    public void NormalizeFileName_RejectsUnsafeOrNonBrainEntries(string entry)
    {
        Assert.Null(BrainManifest.NormalizeFileName(entry));
    }

    [Theory]
    [InlineData("  r2.dsbrain  ", "r2.dsbrain")]
    [InlineData("R2.DSBRAIN", "R2.DSBRAIN")]
    [InlineData(null, null)]
    [InlineData("", null)]
    public void NormalizeFileName_TrimsAndValidates(string? entry, string? expected)
    {
        Assert.Equal(expected, BrainManifest.NormalizeFileName(entry));
    }

    [Fact]
    public void Parse_SkipsInvalidEntriesAndReportsOneWarning()
    {
        var result = BrainManifest.Parse("""["good.dsbrain", "../evil.dsbrain", 42]""");

        Assert.Equal(["good.dsbrain"], result.Files);
        Assert.NotNull(result.Warning);
        Assert.Contains("2", result.Warning, StringComparison.Ordinal);
    }

    [Fact]
    public void Parse_DeduplicatesNames()
    {
        var result = BrainManifest.Parse("""["a.dsbrain", "a.dsbrain", "A.DSBRAIN"]""");

        Assert.Single(result.Files);
        Assert.Null(result.Warning);
    }

    [Fact]
    public void Parse_NonArrayRoot_ReportsWarning()
    {
        var result = BrainManifest.Parse("""{"file":"a.dsbrain"}""");

        Assert.Empty(result.Files);
        Assert.NotNull(result.Warning);
    }

    [Fact]
    public void Parse_MalformedJson_Throws()
    {
        // JsonReaderException derives from JsonException, so assert on the base contract.
        Assert.ThrowsAny<System.Text.Json.JsonException>(() => BrainManifest.Parse("["));
    }

    [Fact]
    public void Parse_Stream_MatchesStringParse()
    {
        const string json = """["a.dsbrain"]""";
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(json));

        var fromStream = BrainManifest.Parse(stream);
        var fromText = BrainManifest.Parse(json);

        Assert.Equal(fromText.Files, fromStream.Files);
    }
}