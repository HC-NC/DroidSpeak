namespace DroidSpeak.Neural;

/// <summary>
/// Constants of the <c>.dsbrain</c> binary model container.
/// <para>
/// The layout is fixed, little-endian and free of any object graph, so a file written by
/// <see cref="MlpModel.Save"/> can be read by any runtime, including NativeAOT and Blazor
/// WebAssembly, without reflection, formatters or serialization contracts.
/// </para>
/// <para>
/// All integers are unsigned and all floats are IEEE-754 binary32 written least-significant byte
/// first. Sizes are deliberately small fields: <see cref="MlpModel.MaxSupportedLayerWidth"/> keeps
/// every dimension inside 16 bits, and the tensor element counts fit in 32 bits.
/// </para>
/// </summary>
public static class DsBrainFormat
{
    /// <summary>The four magic bytes every container starts with.</summary>
    public static ReadOnlySpan<byte> Magic => "DSBR"u8;

    /// <summary>The only format version this library writes, and the only one it accepts.</summary>
    public const byte Version = 1;

    /// <summary>Recommended file extension for a saved model.</summary>
    public const string FileExtension = ".dsbrain";

    /// <summary>Size of the fixed header in bytes.</summary>
    /// <remarks>
    /// <c>magic (4) + version (1) + flags (1) + inputSize (2) + outputSize (2) + layerCount (2)</c>.
    /// </remarks>
    public const int HeaderSize = 12;

    /// <summary>
    /// Size of the fixed part of a layer record, before its tensors.
    /// <c>inputSize (2) + outputSize (2) + activation (1) + flags (1) + weightCount (4) + biasCount (4)</c>.
    /// </summary>
    public const int LayerHeaderSize = 14;

    /// <summary>Size of a single tensor element on disk.</summary>
    public const int ElementSize = sizeof(float);

    /// <summary>Reserved flags written by this version.</summary>
    internal const byte ReservedFlags = 0;

    /// <summary>Largest value the 16-bit size fields can carry.</summary>
    internal const int MaxWidthField = ushort.MaxValue;
}
