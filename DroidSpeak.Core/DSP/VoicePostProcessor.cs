namespace DroidSpeak.DSP;

/// <summary>
/// Whole-buffer stages that run after the notes, whistles and chatter have been mixed but before
/// the dynamic low-pass and the peak ceiling.
/// <para>
/// Both stages are strictly in place and allocation free. They are no-ops at their neutral
/// settings, so a profile that leaves the v0.5 parameters at zero renders exactly like v0.4.
/// </para>
/// </summary>
internal static class VoicePostProcessor
{
    /// <summary>First early-reflection tap in seconds (~7 ms).</summary>
    public const float FirstTapSeconds = 0.007f;

    /// <summary>Second early-reflection tap in seconds (~13 ms).</summary>
    public const float SecondTapSeconds = 0.013f;

    private const float FirstTapGain = 0.38f;
    private const float SecondTapGain = 0.24f;

    /// <summary>Returns the number of samples needed for the reflection delay line.</summary>
    public static int ReflectionLineLength(int sampleRate) => SecondTapSamples(sampleRate) + 1;

    /// <summary>Returns the first tap distance in samples.</summary>
    public static int FirstTapSamples(int sampleRate) =>
        Math.Max(1, (int)MathF.Round(FirstTapSeconds * sampleRate));

    /// <summary>Returns the second tap distance in samples.</summary>
    public static int SecondTapSamples(int sampleRate)
    {
        var first = FirstTapSamples(sampleRate);
        return Math.Max(first + 1, (int)MathF.Round(SecondTapSeconds * sampleRate));
    }

    /// <summary>
    /// Subtle analog warmth: a tanh soft-clip blended into the dry signal. <paramref name="drive"/>
    /// is the profile's <c>SaturationDrive</c>; <c>0</c> is a bit-exact no-op.
    /// </summary>
    public static void ApplyAnalogSaturation(Span<float> buffer, float drive)
    {
        if (drive <= 0f || buffer.IsEmpty)
        {
            return;
        }

        var amount = Math.Clamp(drive * 0.30f, 0f, 0.85f);
        var gain = 1f + drive * 2f;

        for (var i = 0; i < buffer.Length; i++)
        {
            var sample = buffer[i];
            var shaped = MathF.Tanh(sample * gain);
            buffer[i] = sample + (shaped - sample) * amount;
        }
    }

    /// <summary>
    /// Two-tap early-reflection / micro-reverb stage (~7 ms and ~13 ms). The delay line is owned
    /// by the caller: the offline renderer stack-allocates it, while the stream keeps one field so
    /// chunk boundaries stay invisible.
    /// </summary>
    public static void ApplyEarlyReflections(
        Span<float> buffer,
        Span<float> line,
        ref int index,
        float roomMix,
        int firstTap,
        int secondTap)
    {
        if (roomMix <= 0f || buffer.IsEmpty)
        {
            return;
        }

        var length = line.Length;
        var mix = Math.Clamp(roomMix, 0f, 1f);

        for (var i = 0; i < buffer.Length; i++)
        {
            var dry = buffer[i];

            var read1 = index - firstTap;
            if (read1 < 0)
            {
                read1 += length;
            }

            var read2 = index - secondTap;
            if (read2 < 0)
            {
                read2 += length;
            }

            var wet = FirstTapGain * line[read1] + SecondTapGain * line[read2];
            line[index] = dry;
            buffer[i] = dry + wet * mix;

            index++;
            if (index == length)
            {
                index = 0;
            }
        }
    }
}
