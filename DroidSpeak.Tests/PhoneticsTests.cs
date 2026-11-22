using DroidSpeak.NLP;

namespace DroidSpeak.Tests;

public class PhoneticsTests
{
    [Theory]
    [InlineData('а', true)]
    [InlineData('ё', true)]
    [InlineData('ю', true)]
    [InlineData('a', true)]
    [InlineData('y', true)]
    [InlineData('б', false)]
    [InlineData('x', false)]
    public void IsVowel_RecognizesCyrillicAndLatinVowels(char value, bool expected)
    {
        Assert.Equal(expected, Phonetics.IsVowel(value));
    }

    [Theory]
    [InlineData('с', true)]
    [InlineData('ш', true)]
    [InlineData('щ', true)]
    [InlineData('ч', true)]
    [InlineData('ф', true)]
    [InlineData('s', true)]
    [InlineData('f', true)]
    [InlineData('к', false)]
    [InlineData('k', false)]
    public void IsSibilant_RecognizesHissingConsonants(char value, bool expected)
    {
        Assert.Equal(expected, Phonetics.IsSibilant(value));
    }

    [Theory]
    [InlineData('б', true)]
    [InlineData('д', true)]
    [InlineData('г', true)]
    [InlineData('р', true)]
    [InlineData('ж', true)]
    [InlineData('з', true)]
    [InlineData('b', true)]
    [InlineData('d', true)]
    [InlineData('g', true)]
    [InlineData('r', true)]
    [InlineData('z', true)]
    [InlineData('п', false)]
    [InlineData('p', false)]
    public void IsVoicedConsonant_RecognizesVoicedConsonants(char value, bool expected)
    {
        Assert.Equal(expected, Phonetics.IsVoicedConsonant(value));
    }

    [Theory]
    [InlineData("sh", true)]
    [InlineData("ch", true)]
    [InlineData("zh", false)]
    [InlineData("стр", true)]
    [InlineData("br", false)]
    [InlineData("кл", false)]
    public void ContainsSibilant_ChecksClustersAndDigraphs(string onset, bool expected)
    {
        Assert.Equal(expected, Phonetics.ContainsSibilant(onset));
    }

    [Fact]
    public void StableHash_IsCaseInsensitiveAndDeterministic()
    {
        Assert.Equal(Phonetics.StableHash("Hello"), Phonetics.StableHash("HELLO"));
        Assert.Equal(Phonetics.StableHash("привет"), Phonetics.StableHash("ПРИВЕТ"));
        Assert.NotEqual(Phonetics.StableHash("beep"), Phonetics.StableHash("boop"));
    }

    [Fact]
    public void StableHash_MatchesFnv1a32ForHello()
    {
        // FNV-1a 32-bit reference value for the ASCII string "hello".
        Assert.Equal(0x4F9F2CABu, Phonetics.StableHash("hello"));
    }

    [Fact]
    public void NormalizeWord_TrimsSurroundingPunctuation()
    {
        Assert.Equal("Привет", Phonetics.NormalizeWord("«Привет!»"));
        Assert.Equal("R2", Phonetics.NormalizeWord(",R2."));
    }
}
