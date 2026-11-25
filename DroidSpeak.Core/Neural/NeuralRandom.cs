namespace DroidSpeak.Neural;

/// <summary>
/// Tiny deterministic generator used only while building a model. Inference never draws random
/// numbers, so this type stays out of the hot path.
/// <para>
/// The stream is an xorshift32 with a hashed seed, which gives the same weight tensor for the same
/// seed on every platform, unlike <see cref="System.Random"/> whose sequence is documented to
/// change between runtime versions.
/// </para>
/// </summary>
internal struct NeuralRandom
{
    private const float Unit = 1f / 16777216f;

    private uint _state;
    private float _spare;
    private bool _hasSpare;

    public NeuralRandom(uint seed)
    {
        // Never allow the degenerate all-zero xorshift state.
        _state = seed == 0u ? 0x9E3779B9u : seed;
        _spare = 0f;
        _hasSpare = false;
    }

    public uint NextUInt()
    {
        var value = _state;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        _state = value;
        return value;
    }

    /// <summary>Uniform value in <c>[0, 1)</c> with 24 bits of precision.</summary>
    public float NextFloat() => (NextUInt() >> 8) * Unit;

    /// <summary>Uniform value in <c>[-magnitude, +magnitude)</c>.</summary>
    public float NextSigned(float magnitude) => (NextFloat() * 2f - 1f) * magnitude;

    /// <summary>Standard normal sample, produced with the polar Box-Muller transform.</summary>
    public float NextGaussian()
    {
        if (_hasSpare)
        {
            _hasSpare = false;
            return _spare;
        }

        float u;
        float v;
        float squared;
        do
        {
            u = NextFloat() * 2f - 1f;
            v = NextFloat() * 2f - 1f;
            squared = u * u + v * v;
        }
        while (squared is 0f || squared >= 1f);

        var scale = MathF.Sqrt(-2f * MathF.Log(squared) / squared);
        _spare = v * scale;
        _hasSpare = true;
        return u * scale;
    }
}
