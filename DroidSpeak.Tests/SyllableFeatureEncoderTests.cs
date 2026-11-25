using DroidSpeak.DSP;
using DroidSpeak.Neural;
using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.Tests;

public class SyllableFeatureEncoderTests
{
    private const int Slot = SyllableFeatureEncoder.FeaturesPerSyllable;

    [Fact]
    public void WindowLayout_HasTheDocumentedWidth()
    {
        // Renumbering these would silently misalign every saved .dsbrain model.
        Assert.Equal(20, SyllableFeatureEncoder.FeaturesPerSyllable);
        Assert.Equal(3, SyllableFeatureEncoder.WindowSize);
        Assert.Equal(60, SyllableFeatureEncoder.InputSize);
        Assert.Equal(20, SyllableFeatureEncoder.CurrentSlotOffset);
        Assert.Equal(40, SyllableFeatureEncoder.NextSlotOffset);

        Assert.Equal(6, SyllableFeatureEncoder.StressIndex);
        Assert.Equal(7, SyllableFeatureEncoder.VoicedOnsetIndex);
        Assert.Equal(8, SyllableFeatureEncoder.SibilantOnsetIndex);
        Assert.Equal(9, SyllableFeatureEncoder.ShoutIndex);
        Assert.Equal(10, SyllableFeatureEncoder.HyphenBoundaryIndex);
        Assert.Equal(11, SyllableFeatureEncoder.PreviousCommaIndex);
        Assert.Equal(12, SyllableFeatureEncoder.SyllablePositionIndex);
        Assert.Equal(13, SyllableFeatureEncoder.WordPositionIndex);
        Assert.Equal(14, SyllableFeatureEncoder.WordLengthIndex);
        Assert.Equal(15, SyllableFeatureEncoder.ValenceIndex);
        Assert.Equal(16, SyllableFeatureEncoder.ArousalIndex);
        Assert.Equal(17, SyllableFeatureEncoder.TempoIndex);
        Assert.Equal(18, SyllableFeatureEncoder.QuestionIndex);
        Assert.Equal(19, SyllableFeatureEncoder.EmphasisIndex);
    }

    [Fact]
    public void Encode_ProducesEveryValueBetweenZeroAndOne()
    {
        var contexts = CreateContexts();
        Span<float> features = stackalloc float[SyllableFeatureEncoder.InputSize];

        for (var i = 0; i < contexts.Length; i++)
        {
            SyllableFeatureEncoder.Encode(contexts, i, features);

            for (var n = 0; n < SyllableFeatureEncoder.InputSize; n++)
            {
                Assert.InRange(features[n], 0f, 1f);
            }
        }
    }

    [Fact]
    public void Encode_EncodesTheCurrentSlotFromTheRequestedIndex()
    {
        var contexts = CreateContexts();
        Span<float> features = stackalloc float[SyllableFeatureEncoder.InputSize];

        SyllableFeatureEncoder.Encode(contexts, 1, features);

        var single = new[] { contexts[1] };
        Span<float> reference = stackalloc float[SyllableFeatureEncoder.InputSize];
        SyllableFeatureEncoder.Encode(single, 0, reference);

        Assert.True(features[Slot..(Slot * 2)].SequenceEqual(reference[Slot..(Slot * 2)]));
    }

    [Fact]
    public void Encode_FillsNeighbourSlots()
    {
        var contexts = CreateContexts();
        Span<float> features = stackalloc float[SyllableFeatureEncoder.InputSize];

        SyllableFeatureEncoder.Encode(contexts, 1, features);

        // Syllable 0 is unstressed, syllable 1 is stressed in the fixture.
        Assert.Equal(0f, features[SyllableFeatureEncoder.StressIndex]);
        Assert.Equal(1f, features[Slot + SyllableFeatureEncoder.StressIndex]);
        Assert.Equal(1f, features[2 * Slot + SyllableFeatureEncoder.VoicedOnsetIndex]);
    }

    [Fact]
    public void Encode_LeavesMissingNeighboursAtZero()
    {
        var contexts = CreateContexts();
        Span<float> features = stackalloc float[SyllableFeatureEncoder.InputSize];

        SyllableFeatureEncoder.Encode(contexts, 0, features);

        Assert.Equal(0f, features[..Slot].ToArray().Sum());

        SyllableFeatureEncoder.Encode(contexts, contexts.Length - 1, features);

        Assert.Equal(0f, features[(2 * Slot)..].ToArray().Sum());
    }

    [Fact]
    public void Encode_IsDeterministicForEqualContexts()
    {
        var first = CreateContexts();
        var second = CreateContexts();
        Span<float> a = stackalloc float[SyllableFeatureEncoder.InputSize];
        Span<float> b = stackalloc float[SyllableFeatureEncoder.InputSize];

        SyllableFeatureEncoder.Encode(first, 1, a);
        SyllableFeatureEncoder.Encode(second, 1, b);

        Assert.True(a.SequenceEqual(b));
    }

    [Fact]
    public void EncodeSyllable_SelectsExactlyOneVowelClass()
    {
        Span<float> slot = stackalloc float[Slot];

        foreach (var nucleus in new[] { 'a', 'e', 'i', 'o', 'u', 'ы', '\0', 'q' })
        {
            var context = CreateContext();
            context.Nucleus = nucleus;

            SyllableFeatureEncoder.EncodeSyllable(context, slot);

            var sum = 0f;
            for (var v = 0; v <= SyllableFeatureEncoder.OtherVowelClass; v++)
            {
                sum += slot[v];
            }

            Assert.Equal(1f, sum);
        }
    }

    [Theory]
    [InlineData('a', 0)]
    [InlineData('A', 0)]
    [InlineData('я', 0)]
    [InlineData('e', 1)]
    [InlineData('ё', 1)]
    [InlineData('и', 2)]
    [InlineData('y', 2)]
    [InlineData('o', 3)]
    [InlineData('u', 4)]
    [InlineData('ю', 4)]
    [InlineData('\0', 5)]
    [InlineData('q', 5)]
    public void VowelClass_MatchesTheDocumentedMapping(char nucleus, int expected)
    {
        Assert.Equal(expected, SyllableFeatureEncoder.VowelClass(nucleus));
    }

    [Fact]
    public void EncodeSyllable_WritesTheBooleanFlags()
    {
        var context = CreateContext();
        context.IsStressed = true;
        context.HasVoicedOnset = true;
        context.IsShout = true;
        context.IsHyphenBoundary = true;
        context.PreviousWordHadComma = true;

        Span<float> slot = stackalloc float[Slot];
        SyllableFeatureEncoder.EncodeSyllable(context, slot);

        Assert.Equal(1f, slot[SyllableFeatureEncoder.StressIndex]);
        Assert.Equal(1f, slot[SyllableFeatureEncoder.VoicedOnsetIndex]);
        Assert.Equal(0f, slot[SyllableFeatureEncoder.SibilantOnsetIndex]);
        Assert.Equal(1f, slot[SyllableFeatureEncoder.ShoutIndex]);
        Assert.Equal(1f, slot[SyllableFeatureEncoder.HyphenBoundaryIndex]);
        Assert.Equal(1f, slot[SyllableFeatureEncoder.PreviousCommaIndex]);
    }

    [Fact]
    public void EncodeSyllable_NormalizesPositions()
    {
        Span<float> slot = stackalloc float[Slot];

        var first = CreateContext();
        first.SyllableIndex = 0;
        first.WordSyllableCount = 3;
        SyllableFeatureEncoder.EncodeSyllable(first, slot);
        Assert.Equal(0f, slot[SyllableFeatureEncoder.SyllablePositionIndex]);

        var middle = first;
        middle.SyllableIndex = 1;
        SyllableFeatureEncoder.EncodeSyllable(middle, slot);
        Assert.Equal(0.5f, slot[SyllableFeatureEncoder.SyllablePositionIndex], 5);

        var last = first;
        last.SyllableIndex = 2;
        SyllableFeatureEncoder.EncodeSyllable(last, slot);
        Assert.Equal(1f, slot[SyllableFeatureEncoder.SyllablePositionIndex]);

        // A one-syllable word has no interior position, and must not divide by zero.
        var single = CreateContext();
        single.WordSyllableCount = 1;
        SyllableFeatureEncoder.EncodeSyllable(single, slot);
        Assert.Equal(0f, slot[SyllableFeatureEncoder.SyllablePositionIndex]);
    }

    [Fact]
    public void EncodeSyllable_SaturatesWordLength()
    {
        Span<float> slot = stackalloc float[Slot];

        var context = CreateContext();
        context.WordSyllableCount = 4;
        SyllableFeatureEncoder.EncodeSyllable(context, slot);
        Assert.Equal(0.5f, slot[SyllableFeatureEncoder.WordLengthIndex], 5);

        context.WordSyllableCount = 64;
        SyllableFeatureEncoder.EncodeSyllable(context, slot);
        Assert.Equal(1f, slot[SyllableFeatureEncoder.WordLengthIndex]);
    }

    [Theory]
    [InlineData(-1f, 0f)]
    [InlineData(0f, 0.5f)]
    [InlineData(1f, 1f)]
    [InlineData(3f, 1f)]
    [InlineData(-3f, 0f)]
    public void EncodeSyllable_MapsEmotionIntoTheUnitRange(float raw, float expected)
    {
        Span<float> slot = stackalloc float[Slot];
        var context = CreateContext();
        context.EmotionValence = raw;
        context.EmotionArousal = raw;

        SyllableFeatureEncoder.EncodeSyllable(context, slot);

        Assert.Equal(expected, slot[SyllableFeatureEncoder.ValenceIndex], 5);
        Assert.Equal(expected, slot[SyllableFeatureEncoder.ArousalIndex], 5);
    }

    [Theory]
    [InlineData(0.55f, 0f)]
    [InlineData(1.0f, 0.3103f)]
    [InlineData(2.00f, 1f)]
    [InlineData(9f, 1f)]
    public void EncodeSyllable_NormalizesTempo(float tempo, float expected)
    {
        Span<float> slot = stackalloc float[Slot];
        var context = CreateContext();
        context.TempoScale = tempo;

        SyllableFeatureEncoder.EncodeSyllable(context, slot);

        Assert.Equal(expected, slot[SyllableFeatureEncoder.TempoIndex], 4);
    }

    [Theory]
    [InlineData(PhrasePunctuation.Statement, 0, 0)]
    [InlineData(PhrasePunctuation.Question, 1, 0)]
    [InlineData(PhrasePunctuation.Exclamation, 0, 1)]
    [InlineData(PhrasePunctuation.Ellipsis, 0, 1)]
    public void EncodeSyllable_EncodesPunctuation(
        PhrasePunctuation punctuation,
        float question,
        float emphasis)
    {
        Span<float> slot = stackalloc float[Slot];
        var context = CreateContext();
        context.Punctuation = punctuation;

        SyllableFeatureEncoder.EncodeSyllable(context, slot);

        Assert.Equal(question, slot[SyllableFeatureEncoder.QuestionIndex]);
        Assert.Equal(emphasis, slot[SyllableFeatureEncoder.EmphasisIndex]);
    }

    [Fact]
    public void EncodeSyllable_OverwritesThePreviousSlotContents()
    {
        Span<float> slot = stackalloc float[Slot];
        slot.Fill(0.75f);

        SyllableFeatureEncoder.EncodeSyllable(CreateContext(), slot);

        Assert.Equal(0f, slot[SyllableFeatureEncoder.ShoutIndex]);
        Assert.Equal(1f, slot[SyllableFeatureEncoder.ValenceIndex] * 2f - 1f, 4);
    }

    [Fact]
    public void Encode_RejectsTooShortBuffers()
    {
        var contexts = CreateContexts();
        var features = new float[SyllableFeatureEncoder.InputSize - 1];

        Assert.Throws<ArgumentException>(() => SyllableFeatureEncoder.Encode(contexts, 0, features));
    }

    [Fact]
    public void Encode_RejectsIndexesOutsideTheUtterance()
    {
        var contexts = CreateContexts();
        var features = new float[SyllableFeatureEncoder.InputSize];

        Assert.Throws<ArgumentOutOfRangeException>(() => SyllableFeatureEncoder.Encode(contexts, -1, features));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SyllableFeatureEncoder.Encode(contexts, contexts.Length, features));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            SyllableFeatureEncoder.Encode(ReadOnlySpan<SyllableContext>.Empty, 0, features));
    }

    [Fact]
    public void EncodeSyllable_RejectsShortSlots()
    {
        var slot = new float[Slot - 1];

        Assert.Throws<ArgumentException>(() => SyllableFeatureEncoder.EncodeSyllable(CreateContext(), slot));
    }

    [Fact]
    public void Encode_AllocatesNothingOnTheWarmPath()
    {
        var contexts = CreateContexts();
        Span<float> features = stackalloc float[SyllableFeatureEncoder.InputSize];

        for (var i = 0; i < 16; i++)
        {
            SyllableFeatureEncoder.Encode(contexts, 1, features);
        }

        var before = GC.GetAllocatedBytesForCurrentThread();
        for (var i = 0; i < 500; i++)
        {
            SyllableFeatureEncoder.Encode(contexts, i % contexts.Length, features);
        }

        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    [Fact]
    public void Encode_SurvivesNonFiniteContextValues()
    {
        var context = CreateContext();
        context.EmotionValence = float.NaN;
        context.EmotionArousal = float.PositiveInfinity;
        context.TempoScale = float.NegativeInfinity;

        Span<float> slot = stackalloc float[Slot];
        SyllableFeatureEncoder.EncodeSyllable(context, slot);

        Assert.Equal(0f, slot[SyllableFeatureEncoder.ValenceIndex]);
        Assert.Equal(1f, slot[SyllableFeatureEncoder.ArousalIndex]);
        Assert.Equal(0f, slot[SyllableFeatureEncoder.TempoIndex]);
    }

    private static SyllableContext[] CreateContexts()
    {
        var first = CreateContext();
        first.Nucleus = 'a';
        first.WordSyllableCount = 2;

        var second = CreateContext();
        second.Nucleus = 'o';
        second.IsStressed = true;
        second.EmotionValence = 0.6f;
        second.EmotionArousal = -0.2f;
        second.TempoScale = 1.4f;
        second.Punctuation = PhrasePunctuation.Question;

        var third = CreateContext();
        third.Nucleus = 'i';
        third.HasVoicedOnset = true;
        third.HasSibilantOnset = true;

        return [first, second, third];
    }

    private static SyllableContext CreateContext() => new()
    {
        SampleRate = 24000,
        GlobalSeed = 42u,
        WordHash = 0xDEAD_BEEFu,
        PhraseIndex = 0,
        WordIndexInPhrase = 0,
        SyllableIndex = 0,
        WordSyllableCount = 1,
        PhraseSyllableCount = 3,
        WordTextLength = 4,
        TempoScale = 1f,
        EmotionValence = 1f,
        EmotionArousal = 1f,
        Punctuation = PhrasePunctuation.Statement,
        IsFirstSpokenPhrase = true,
        Nucleus = 'e',
    };
}
