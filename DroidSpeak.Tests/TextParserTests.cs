using DroidSpeak.NLP;

namespace DroidSpeak.Tests;

public class TextParserTests
{
    [Fact]
    public void Parse_SplitsPhrasesAndTracksPunctuation()
    {
        var phrases = TextParser.Parse("Привет, R2! Как дела?");

        Assert.Equal(2, phrases.Count);
        Assert.Equal(PhrasePunctuation.Exclamation, phrases[0].Punctuation);
        Assert.Equal(PhrasePunctuation.Question, phrases[1].Punctuation);
        Assert.Equal(2, phrases[0].Words.Count);
        Assert.Equal(2, phrases[1].Words.Count);
    }

    [Fact]
    public void Exclamation_RaisesArousal()
    {
        var calm = TextParser.Parse("Тише.")[0];
        var loud = TextParser.Parse("Тише!")[0];

        Assert.True(loud.Emotion.Arousal > calm.Emotion.Arousal);
    }

    [Fact]
    public void Question_AddsRisingGlide()
    {
        var phrase = TextParser.Parse("Как дела?")[0];

        Assert.Equal(PhrasePunctuation.Question, phrase.Punctuation);
        Assert.True(phrase.RisingGlide);
        Assert.False(phrase.FallingGlide);
    }

    [Fact]
    public void Ellipsis_SlowsTempoAndFalls()
    {
        var statement = TextParser.Parse("Ну вот.")[0];
        var trailing = TextParser.Parse("Ну вот...")[0];

        Assert.Equal(PhrasePunctuation.Ellipsis, trailing.Punctuation);
        Assert.True(trailing.FallingGlide);
        Assert.True(trailing.TempoScale > statement.TempoScale);
        Assert.True(trailing.Emotion.Arousal < statement.Emotion.Arousal);
    }

    [Fact]
    public void Uppercase_AddsShoutAndArousal()
    {
        var quiet = TextParser.Parse("тревога.")[0];
        var shout = TextParser.Parse("ТРЕВОГА!")[0];

        Assert.False(quiet.IsShout);
        Assert.True(shout.IsShout);
        Assert.Equal(1f, shout.ShoutRatio);
        Assert.True(shout.Emotion.Arousal > quiet.Emotion.Arousal);
    }

    [Fact]
    public void Comma_MarksPauseAfterWord()
    {
        var phrase = TextParser.Parse("Привет, R2!")[0];
        var spacedComma = TextParser.Parse("Привет , R2!")[0];

        Assert.True(phrase.Words[0].IsFollowedByComma);
        Assert.False(phrase.Words[1].IsFollowedByComma);
        Assert.True(spacedComma.Words[0].IsFollowedByComma);
        Assert.False(spacedComma.Words[1].IsFollowedByComma);
    }

    [Fact]
    public void Parse_ReturnsNoPhrasesForWhitespace()
    {
        Assert.Empty(TextParser.Parse("   \r\n\t "));
    }

    [Fact]
    public void Parse_SplitsWordsIntoSyllables()
    {
        var phrase = TextParser.Parse("Привет, R2!")[0];

        Assert.Equal(2, phrase.Words[0].Syllables.Count);
        Assert.Single(phrase.Words[1].Syllables);
        Assert.Equal(3, phrase.SyllableCount);
    }
}
