using DroidSpeak.Neural;

namespace DroidSpeak.Tests;

public class NeuralMathTests
{
    [Theory]
    [InlineData(0f, 0.5f)]
    [InlineData(2f, 0.8807971f)]
    [InlineData(-2f, 0.1192029f)]
    public void Sigmoid_MatchesReferenceValues(float value, float expected)
    {
        Assert.Equal(expected, NeuralMathF.Sigmoid(value), 5);
    }

    [Fact]
    public void Sigmoid_SaturatesForExtremeInputs()
    {
        Assert.Equal(1f, NeuralMathF.Sigmoid(60f), 6);
        Assert.Equal(0f, NeuralMathF.Sigmoid(-60f), 6);
        Assert.Equal(1f, NeuralMathF.Sigmoid(float.PositiveInfinity), 6);
        Assert.Equal(0f, NeuralMathF.Sigmoid(float.NegativeInfinity), 6);
    }

    [Theory]
    [InlineData(0f, 0f)]
    [InlineData(1f, 0.7615942f)]
    [InlineData(-1f, -0.7615942f)]
    public void Tanh_MatchesReferenceValues(float value, float expected)
    {
        Assert.Equal(expected, NeuralMathF.Tanh(value), 5);
    }

    [Theory]
    [InlineData(-3f, 0f)]
    [InlineData(0f, 0f)]
    [InlineData(0.25f, 0.25f)]
    [InlineData(7.5f, 7.5f)]
    public void Relu_ClampsNegatives(float value, float expected)
    {
        Assert.Equal(expected, NeuralMathF.Relu(value));
    }

    [Theory]
    [InlineData(-3f, -0.03f)]
    [InlineData(0f, 0f)]
    [InlineData(4f, 4f)]
    public void LeakyRelu_KeepsASmallNegativeSlope(float value, float expected)
    {
        Assert.Equal(expected, NeuralMathF.LeakyRelu(value), 6);
    }

    [Theory]
    [InlineData(-2f, -0.2f)]
    [InlineData(-0.5f, -0.05f)]
    [InlineData(0f, 0f)]
    [InlineData(0.5f, 0.5f)]
    [InlineData(1f, 1f)]
    [InlineData(3f, 1.3f)]
    [InlineData(10f, 2f)]
    public void ModRelu_IsIdentityOnUnitBandAndLeakyOutside(float value, float expected)
    {
        Assert.Equal(expected, NeuralMathF.ModRelu(value), 6);
    }

    [Fact]
    public void ModRelu_MatchesTheLegacyEngineFormula()
    {
        for (var i = -20; i <= 40; i++)
        {
            var x = i * 0.1f;
            var legacy = x < 0f ? 0.1f * x : x > 1f ? 0.1f * x + 1f : x;
            Assert.Equal(legacy, NeuralMathF.ModRelu(x));
        }
    }

    [Fact]
    public void Linear_IsIdentity()
    {
        Assert.Equal(-12.5f, NeuralMathF.Linear(-12.5f));
        Assert.Equal(0f, NeuralMathF.Linear(0f));
    }

    [Theory]
    [InlineData(ActivationType.Sigmoid)]
    [InlineData(ActivationType.Tanh)]
    [InlineData(ActivationType.Relu)]
    [InlineData(ActivationType.LeakyRelu)]
    [InlineData(ActivationType.ModRelu)]
    [InlineData(ActivationType.Linear)]
    public void Apply_DelegatesToTheMatchingHelper(ActivationType activation)
    {
        const float value = -1.75f;
        var expected = activation switch
        {
            ActivationType.Sigmoid => NeuralMathF.Sigmoid(value),
            ActivationType.Tanh => NeuralMathF.Tanh(value),
            ActivationType.Relu => NeuralMathF.Relu(value),
            ActivationType.LeakyRelu => NeuralMathF.LeakyRelu(value),
            ActivationType.ModRelu => NeuralMathF.ModRelu(value),
            _ => NeuralMathF.Linear(value),
        };

        Assert.Equal(expected, NeuralMathF.Apply(value, activation));
    }

    [Fact]
    public void ApplyInPlace_TransformsEveryElement()
    {
        Span<float> values = [-2f, 0f, 3f];
        NeuralMathF.ApplyInPlace(values, ActivationType.Relu);

        Assert.Equal(0f, values[0]);
        Assert.Equal(0f, values[1]);
        Assert.Equal(3f, values[2]);
    }

    [Fact]
    public void Sanitize_LeavesFiniteValuesAlone()
    {
        Assert.Equal(1.25f, NeuralMathF.Sanitize(1.25f, ActivationType.Tanh));
        Assert.Equal(-9f, NeuralMathF.Sanitize(-9f, ActivationType.Linear));
    }

    [Fact]
    public void Sanitize_FoldsNonFiniteValuesIntoTheActivationRange()
    {
        Assert.Equal(0.5f, NeuralMathF.Sanitize(float.NaN, ActivationType.Sigmoid));
        Assert.Equal(0f, NeuralMathF.Sanitize(float.NaN, ActivationType.Tanh));
        Assert.Equal(1f, NeuralMathF.Sanitize(float.PositiveInfinity, ActivationType.Sigmoid));
        Assert.Equal(0f, NeuralMathF.Sanitize(float.NegativeInfinity, ActivationType.Sigmoid));
        Assert.Equal(1f, NeuralMathF.Sanitize(float.PositiveInfinity, ActivationType.Tanh));
        Assert.Equal(-1f, NeuralMathF.Sanitize(float.NegativeInfinity, ActivationType.Tanh));
        Assert.Equal(
            NeuralMathF.UnboundedLimit,
            NeuralMathF.Sanitize(float.PositiveInfinity, ActivationType.Linear));
        Assert.Equal(
            -NeuralMathF.UnboundedLimit,
            NeuralMathF.Sanitize(float.NegativeInfinity, ActivationType.ModRelu));
    }

    [Fact]
    public void Clamp01_HandlesOutOfRangeAndNonFiniteInputs()
    {
        Assert.Equal(0f, NeuralMathF.Clamp01(-4f));
        Assert.Equal(0f, NeuralMathF.Clamp01(0f));
        Assert.Equal(0.4f, NeuralMathF.Clamp01(0.4f));
        Assert.Equal(1f, NeuralMathF.Clamp01(4f));
        Assert.Equal(1f, NeuralMathF.Clamp01(float.PositiveInfinity));
        Assert.Equal(0f, NeuralMathF.Clamp01(float.NegativeInfinity));
        Assert.Equal(0f, NeuralMathF.Clamp01(float.NaN));
    }

    [Fact]
    public void ClampFinite_UsesTheFallbackOnlyForNaN()
    {
        Assert.Equal(0.5f, NeuralMathF.ClampFinite(0.2f, 0.5f, 2f, 1f));
        Assert.Equal(2f, NeuralMathF.ClampFinite(9f, 0.5f, 2f, 1f));
        Assert.Equal(2f, NeuralMathF.ClampFinite(float.PositiveInfinity, 0.5f, 2f, 1f));
        Assert.Equal(0.5f, NeuralMathF.ClampFinite(float.NegativeInfinity, 0.5f, 2f, 1f));
        Assert.Equal(1f, NeuralMathF.ClampFinite(float.NaN, 0.5f, 2f, 1f));
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(0.5f)]
    [InlineData(1f)]
    public void Lerp_InterpolatesAndClampsTheAmount(float amount)
    {
        Assert.Equal(10f + 20f * amount, NeuralMathF.Lerp(10f, 30f, amount), 5);
    }

    [Fact]
    public void Lerp_ClampsOutOfRangeAmounts()
    {
        Assert.Equal(10f, NeuralMathF.Lerp(10f, 30f, -3f));
        Assert.Equal(30f, NeuralMathF.Lerp(10f, 30f, 3f));
        Assert.Equal(10f, NeuralMathF.Lerp(10f, 30f, float.NaN));
    }

    [Fact]
    public void ActivationType_HasStableNumericCodes()
    {
        // The codes are written into saved models, so renumbering them would corrupt old files.
        Assert.Equal(0, (byte)ActivationType.Sigmoid);
        Assert.Equal(1, (byte)ActivationType.Tanh);
        Assert.Equal(2, (byte)ActivationType.Relu);
        Assert.Equal(3, (byte)ActivationType.LeakyRelu);
        Assert.Equal(4, (byte)ActivationType.ModRelu);
        Assert.Equal(5, (byte)ActivationType.Linear);
    }
}
