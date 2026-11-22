using DroidSpeak.NLP;

namespace DroidSpeak.Tests;

public class EmotionVectorTests
{
    [Fact]
    public void Constructor_ClampsComponentsToValidRange()
    {
        var emotion = new EmotionVector(2.5f, -3.0f);

        Assert.Equal(1f, emotion.Valence);
        Assert.Equal(-1f, emotion.Arousal);
    }

    [Fact]
    public void Addition_ClampsResult()
    {
        var result = new EmotionVector(0.8f, -0.8f) + new EmotionVector(0.8f, -0.8f);

        Assert.Equal(1f, result.Valence);
        Assert.Equal(-1f, result.Arousal);
    }

    [Fact]
    public void Lerp_InterpolatesBothComponents()
    {
        var result = EmotionVector.Neutral.Lerp(new EmotionVector(1f, -1f), 0.25f);

        Assert.Equal(0.25f, result.Valence, precision: 5);
        Assert.Equal(-0.25f, result.Arousal, precision: 5);
    }

    [Fact]
    public void WithMethods_ReturnClampedCopies()
    {
        var emotion = new EmotionVector(0.5f, 0.5f);

        var updated = emotion.WithValence(-4f).WithArousal(4f);

        Assert.Equal(-1f, updated.Valence);
        Assert.Equal(1f, updated.Arousal);
        Assert.Equal(0.5f, emotion.Valence);
        Assert.Equal(0.5f, emotion.Arousal);
    }
}
