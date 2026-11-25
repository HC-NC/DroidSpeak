namespace DroidSpeak.Trainer;

/// <summary>
/// A small deterministic generator, independent of <see cref="System.Random"/>, so a training run is
/// reproducible from its seed on any runtime.
/// <para>
/// xorshift32 has no state beyond one 32-bit word and no allocation, which is all a trainer needs:
/// shuffling sample order and jittering the corpus must be repeatable, not cryptographically sound.
/// The sequence is identical to the one the legacy engine used, so a dataset built with a given seed
/// can be rebuilt byte for byte when a checkpoint is compared.
/// </para>
/// </summary>
internal struct RandomSource
{
    private uint _state;

    public RandomSource(uint seed)
    {
        // Zero is a fixed point of xorshift, so it is nudged into the cycle instead of accepted.
        _state = seed == 0u ? 0x9E3779B9u : seed;
    }

    /// <summary>Next 32-bit value.</summary>
    public uint NextUInt()
    {
        var value = _state;
        value ^= value << 13;
        value ^= value >> 17;
        value ^= value << 5;
        _state = value;
        return value;
    }

    /// <summary>Next value in <c>[0, 1)</c>, using the upper 24 bits for exact float resolution.</summary>
    public float NextFloat() => (NextUInt() >> 8) * (1f / 16777216f);

    /// <summary>Next value in <c>[min, max)</c>.</summary>
    public float NextFloat(float min, float max) => min + ((max - min) * NextFloat());

    /// <summary>Next index in <c>[0, exclusiveMax)</c>.</summary>
    public int NextInt(int exclusiveMax) => (int)(NextUInt() % (uint)exclusiveMax);

    /// <summary>Bernoulli draw with the given success probability.</summary>
    public bool NextBool(float probability) => NextFloat() < probability;
}
