namespace DroidSpeak.DSP;

/// <summary>
/// Deterministic xorshift32 generator. Every noise source in the engine owns its own instance
/// seeded from stable text hashes, so output never depends on call order across sources.
/// </summary>
internal struct XorShift32
{
    private uint _state;

    public XorShift32(uint seed)
    {
        _state = seed == 0 ? 0x9E3779B9u : seed;
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

    public float NextFloat() => (NextUInt() >> 8) * (1f / 16777216f);
}
