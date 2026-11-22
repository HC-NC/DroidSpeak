using DroidSpeak.NLP;

namespace DroidSpeak.Tests;

public class SyllableSplitterTests
{
    [Theory]
    [InlineData("привет", 2)]
    [InlineData("сломалось", 3)]
    [InlineData("молоко", 3)]
    [InlineData("hello", 2)]
    [InlineData("extra", 2)]
    [InlineData("R2", 1)]
    [InlineData("брр", 1)]
    public void Split_ReturnsExpectedSyllableCount(string word, int expected)
    {
        var syllables = SyllableSplitter.Split(word);

        Assert.Equal(expected, syllables.Count);
    }

    [Theory]
    [InlineData("привет")]
    [InlineData("сломалось")]
    [InlineData("hello")]
    [InlineData("extra")]
    [InlineData("R2")]
    public void Split_PreservesEveryCharacter(string word)
    {
        var syllables = SyllableSplitter.Split(word);

        Assert.Equal(word, string.Concat(syllables.Select(s => s.Text)));
        foreach (var syllable in syllables)
        {
            if (syllable.HasNucleus)
            {
                Assert.Equal(syllable.Text, syllable.Onset + syllable.Nucleus + syllable.Coda);
            }
            else
            {
                Assert.Equal(syllable.Text, syllable.Onset);
            }
        }
    }

    [Fact]
    public void Split_IsDeterministicAndMarksOneStress()
    {
        var first = SyllableSplitter.Split("привет");
        var second = SyllableSplitter.Split("привет");

        Assert.Equal(first, second);
        Assert.Single(first, syllable => syllable.IsStressed);
    }

    [Fact]
    public void Split_DetectsSibilantAndVoicedOnsets()
    {
        var hissing = SyllableSplitter.Split("шип");
        var voiced = SyllableSplitter.Split("дом");

        Assert.True(hissing[0].HasSibilantOnset);
        Assert.False(hissing[0].HasVoicedOnset);
        Assert.True(voiced[0].HasVoicedOnset);
    }

    [Fact]
    public void Split_HandlesEmptyWord()
    {
        Assert.Empty(SyllableSplitter.Split(string.Empty));
    }
}
