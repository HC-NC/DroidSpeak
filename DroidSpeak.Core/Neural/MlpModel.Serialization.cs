using System.Buffers.Binary;

namespace DroidSpeak.Neural;

/// <summary>
/// Binary persistence of <see cref="MlpModel"/> in the <c>.dsbrain</c> container.
/// <para>
/// Loading is the safety boundary of the whole neural pipeline: a file is only accepted when its
/// header is intact, its layer chain is continuous, every tensor holds exactly as many values as
/// its shape demands, no value is <see cref="float.NaN"/> or infinite, and nothing follows the last
/// tensor. Anything else throws <see cref="InvalidDataException"/> instead of producing a model that
/// could emit non-finite prosody.
/// </para>
/// <para>
/// Writing is deterministic: the same model always produces the same bytes, which is what lets the
/// test suite pin a golden file.
/// </para>
/// </summary>
public sealed partial class MlpModel
{
    /// <summary>Bytes used as a chunk when streaming tensors.</summary>
    private const int TensorChunkBytes = 4 * 128;

    /// <summary>
    /// Writes the model to <paramref name="stream"/> as a <c>.dsbrain</c> container. The stream is
    /// not closed.
    /// </summary>
    /// <param name="stream">Writable destination positioned where the file should start.</param>
    /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
    public void Save(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanWrite)
        {
            throw new ArgumentException("The stream is not writable.", nameof(stream));
        }

        Span<byte> header = stackalloc byte[DsBrainFormat.HeaderSize];
        DsBrainFormat.Magic.CopyTo(header);
        header[4] = DsBrainFormat.Version;
        header[5] = DsBrainFormat.ReservedFlags;
        BinaryPrimitives.WriteUInt16LittleEndian(header[6..], (ushort)InputSize);
        BinaryPrimitives.WriteUInt16LittleEndian(header[8..], (ushort)OutputSize);
        BinaryPrimitives.WriteUInt16LittleEndian(header[10..], (ushort)_layers.Length);
        stream.Write(header);

        Span<byte> record = stackalloc byte[DsBrainFormat.LayerHeaderSize];

        for (var i = 0; i < _layers.Length; i++)
        {
            var layer = _layers[i];

            record.Clear();
            BinaryPrimitives.WriteUInt16LittleEndian(record, (ushort)layer.InputSize);
            BinaryPrimitives.WriteUInt16LittleEndian(record[2..], (ushort)layer.OutputSize);
            record[4] = (byte)layer.Activation;
            BinaryPrimitives.WriteUInt32LittleEndian(
                record[6..],
                (uint)(layer.InputSize * layer.OutputSize));
            BinaryPrimitives.WriteUInt32LittleEndian(record[10..], (uint)layer.OutputSize);
            stream.Write(record);

            WriteTensor(stream, layer.WeightStorage);
            WriteTensor(stream, layer.BiasStorage);
        }
    }

    /// <summary>
    /// Reads a model from a <c>.dsbrain</c> container. The stream is not closed and is consumed
    /// until the end of the model, which must be the end of the readable data.
    /// </summary>
    /// <param name="stream">Source positioned at the start of the file.</param>
    /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
    /// <exception cref="InvalidDataException">
    /// The data is truncated, carries an unknown version, describes an inconsistent or oversized
    /// network, or holds a non-finite weight.
    /// </exception>
    public static MlpModel Load(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead)
        {
            throw new ArgumentException("The stream is not readable.", nameof(stream));
        }

        Span<byte> header = stackalloc byte[DsBrainFormat.HeaderSize];
        if (!ReadFully(stream, header))
        {
            throw new InvalidDataException(
                $"The data is shorter than the {DsBrainFormat.HeaderSize}-byte DSBR header.");
        }

        ValidateHeader(header, out var inputSize, out var outputSize, out var layerCount);

        var layers = new DenseLayer[layerCount];
        Span<byte> record = stackalloc byte[DsBrainFormat.LayerHeaderSize];

        for (var i = 0; i < layerCount; i++)
        {
            if (!ReadFully(stream, record))
            {
                throw new InvalidDataException(
                    $"The model is truncated inside the header of layer {i}.");
            }

            ReadLayerHeader(record, i, out var layerInput, out var layerOutput, out var activation);
            ValidateChainLink(i, layerInput, i == 0 ? -1 : layers[i - 1].OutputSize);

            var weights = ReadTensor(stream, layerInput * layerOutput, i, "weights");
            var biases = ReadTensor(stream, layerOutput, i, "biases");

            layers[i] = new DenseLayer(weights, biases, layerInput, layerOutput, activation);
        }

        if (stream.ReadByte() >= 0)
        {
            throw new InvalidDataException("The model has unexpected trailing bytes.");
        }

        ValidateAgainstHeader(layers, inputSize, outputSize);
        return new MlpModel(layers);
    }

    /// <summary>Reads a model from an in-memory <c>.dsbrain</c> container.</summary>
    /// <param name="data">The complete file contents.</param>
    /// <exception cref="InvalidDataException">
    /// The data is truncated, carries an unknown version, describes an inconsistent or oversized
    /// network, or holds a non-finite weight.
    /// </exception>
    public static MlpModel Load(ReadOnlySpan<byte> data)
    {
        if (data.Length < DsBrainFormat.HeaderSize)
        {
            throw new InvalidDataException(
                $"The data is shorter than the {DsBrainFormat.HeaderSize}-byte DSBR header.");
        }

        ValidateHeader(data[..DsBrainFormat.HeaderSize], out var inputSize, out var outputSize, out var layerCount);
        data = data[DsBrainFormat.HeaderSize..];

        var layers = new DenseLayer[layerCount];

        for (var i = 0; i < layerCount; i++)
        {
            if (data.Length < DsBrainFormat.LayerHeaderSize)
            {
                throw new InvalidDataException(
                    $"The model is truncated inside the header of layer {i}.");
            }

            ReadLayerHeader(
                data[..DsBrainFormat.LayerHeaderSize],
                i,
                out var layerInput,
                out var layerOutput,
                out var activation);
            ValidateChainLink(
                i,
                layerInput,
                i == 0 ? -1 : layers[i - 1].OutputSize);
            data = data[DsBrainFormat.LayerHeaderSize..];

            var weightCount = layerInput * layerOutput;
            var weights = ReadTensor(data, weightCount, i, "weights");
            data = data[(weightCount * DsBrainFormat.ElementSize)..];

            var biases = ReadTensor(data, layerOutput, i, "biases");
            data = data[(layerOutput * DsBrainFormat.ElementSize)..];

            layers[i] = new DenseLayer(weights, biases, layerInput, layerOutput, activation);
        }

        if (!data.IsEmpty)
        {
            throw new InvalidDataException("The model has unexpected trailing bytes.");
        }

        ValidateAgainstHeader(layers, inputSize, outputSize);
        return new MlpModel(layers);
    }

    /// <summary>
    /// Measures the byte size a model occupies without allocating its tensors, and confirms the
    /// buffer really holds that many bytes. Useful for validating a download before loading it.
    /// </summary>
    /// <param name="data">The complete file contents.</param>
    /// <returns>The exact byte length of the container described by <paramref name="data"/>.</returns>
    /// <exception cref="InvalidDataException">
    /// The header is malformed or oversized, the layer chain is inconsistent, or the buffer is
    /// shorter than the tensors it declares.
    /// </exception>
    public static int Measure(ReadOnlySpan<byte> data)
    {
        if (data.Length < DsBrainFormat.HeaderSize)
        {
            throw new InvalidDataException(
                $"The data is shorter than the {DsBrainFormat.HeaderSize}-byte DSBR header.");
        }

        ValidateHeader(data[..DsBrainFormat.HeaderSize], out _, out _, out var layerCount);

        var total = DsBrainFormat.HeaderSize;
        var previousOutput = -1;

        for (var i = 0; i < layerCount; i++)
        {
            if (data.Length < total + DsBrainFormat.LayerHeaderSize)
            {
                throw new InvalidDataException(
                    $"The model is truncated inside the header of layer {i}.");
            }

            ReadLayerHeader(
                data.Slice(total, DsBrainFormat.LayerHeaderSize),
                i,
                out var layerInput,
                out var layerOutput,
                out _);
            ValidateChainLink(i, layerInput, previousOutput);

            total += DsBrainFormat.LayerHeaderSize;
            total = checked(total + (layerInput * layerOutput + layerOutput) * DsBrainFormat.ElementSize);
            previousOutput = layerOutput;
        }

        if (data.Length < total)
        {
            throw new InvalidDataException(
                $"The model declares {total} bytes, but only {data.Length} are present.");
        }

        return total;
    }

    /// <summary>
    /// Makes sure the layer chain actually spans the input and output widths the header promised, so
    /// a file whose header and body disagree is rejected instead of silently resized.
    /// </summary>
    private static void ValidateAgainstHeader(DenseLayer[] layers, int inputSize, int outputSize)
    {
        if (layers[0].InputSize != inputSize)
        {
            throw new InvalidDataException(
                $"The header declares an input size of {inputSize}, but the first layer expects " +
                $"{layers[0].InputSize}.");
        }

        if (layers[^1].OutputSize != outputSize)
        {
            throw new InvalidDataException(
                $"The header declares an output size of {outputSize}, but the last layer produces " +
                $"{layers[^1].OutputSize}.");
        }
    }

    private static void ValidateHeader(
        ReadOnlySpan<byte> header,
        out int inputSize,
        out int outputSize,
        out int layerCount)
    {
        if (!header[..DsBrainFormat.Magic.Length].SequenceEqual(DsBrainFormat.Magic))
        {
            throw new InvalidDataException("This is not a DSBR model file.");
        }

        if (header[4] != DsBrainFormat.Version)
        {
            throw new InvalidDataException(
                $"Model version {header[4]} is not supported; this build reads version " +
                $"{DsBrainFormat.Version}.");
        }

        if (header[5] != DsBrainFormat.ReservedFlags)
        {
            throw new InvalidDataException("The model header sets an unknown flag.");
        }

        inputSize = BinaryPrimitives.ReadUInt16LittleEndian(header[6..]);
        outputSize = BinaryPrimitives.ReadUInt16LittleEndian(header[8..]);
        var layers = BinaryPrimitives.ReadUInt16LittleEndian(header[10..]);

        if (inputSize is < 1 or > MaxSupportedLayerWidth)
        {
            throw new InvalidDataException(
                $"The model declares an input size of {inputSize}, which is outside " +
                $"1..{MaxSupportedLayerWidth}.");
        }

        if (outputSize is < 1 or > MaxSupportedLayerWidth)
        {
            throw new InvalidDataException(
                $"The model declares an output size of {outputSize}, which is outside " +
                $"1..{MaxSupportedLayerWidth}.");
        }

        if (layers is < 1 or > MaxLayerCount)
        {
            throw new InvalidDataException(
                $"The model declares {layers} layers, which is outside 1..{MaxLayerCount}.");
        }

        layerCount = layers;
    }

    private static void ReadLayerHeader(
        ReadOnlySpan<byte> record,
        int index,
        out int inputSize,
        out int outputSize,
        out ActivationType activation)
    {
        inputSize = BinaryPrimitives.ReadUInt16LittleEndian(record);
        outputSize = BinaryPrimitives.ReadUInt16LittleEndian(record[2..]);
        var code = record[4];

        if (record[5] != DsBrainFormat.ReservedFlags)
        {
            throw new InvalidDataException($"Layer {index} sets an unknown flag.");
        }

        if (code > (byte)ActivationType.Linear)
        {
            throw new InvalidDataException($"Layer {index} uses unknown activation code {code}.");
        }

        activation = (ActivationType)code;

        if (inputSize is < 1 or > MaxSupportedLayerWidth)
        {
            throw new InvalidDataException(
                $"Layer {index} declares an input size of {inputSize}, which is outside " +
                $"1..{MaxSupportedLayerWidth}.");
        }

        if (outputSize is < 1 or > MaxSupportedLayerWidth)
        {
            throw new InvalidDataException(
                $"Layer {index} declares an output size of {outputSize}, which is outside " +
                $"1..{MaxSupportedLayerWidth}.");
        }

        var weightCount = BinaryPrimitives.ReadUInt32LittleEndian(record[6..]);
        var biasCount = BinaryPrimitives.ReadUInt32LittleEndian(record[10..]);

        // 16-bit sizes can multiply past int.MaxValue, so the comparison stays in unsigned space and
        // never throws OverflowException on a hostile file.
        var expectedWeights = (uint)inputSize * (uint)outputSize;

        if (weightCount != expectedWeights)
        {
            throw new InvalidDataException(
                $"Layer {index} declares {weightCount} weights, but a " +
                $"{inputSize}x{outputSize} layer needs {expectedWeights}.");
        }

        if (biasCount != outputSize)
        {
            throw new InvalidDataException(
                $"Layer {index} declares {biasCount} biases, but has {outputSize} neurons.");
        }
    }

    /// <summary>
    /// Confirms a freshly read layer continues the chain. Sizes have already been bounded by
    /// <see cref="ReadLayerHeader"/>, so this runs before any tensor is allocated.
    /// </summary>
    /// <param name="index">Zero-based layer index.</param>
    /// <param name="inputSize">Declared input width of the layer.</param>
    /// <param name="previousOutput">
    /// Output width of the preceding layer, or <c>-1</c> for the first layer.
    /// </param>
    private static void ValidateChainLink(int index, int inputSize, int previousOutput)
    {
        if (previousOutput >= 0 && previousOutput != inputSize)
        {
            throw new InvalidDataException(
                $"Layer {index - 1} produces {previousOutput} values but layer {index} expects " +
                $"{inputSize}.");
        }
    }

    private static void WriteTensor(Stream stream, float[] values)
    {
        Span<byte> chunk = stackalloc byte[TensorChunkBytes];
        var offset = 0;

        while (offset < values.Length)
        {
            var take = Math.Min(values.Length - offset, TensorChunkBytes / DsBrainFormat.ElementSize);
            var span = chunk[..(take * DsBrainFormat.ElementSize)];
            var source = values.AsSpan(offset, take);

            for (var i = 0; i < take; i++)
            {
                BinaryPrimitives.WriteSingleLittleEndian(span[(i * 4)..], source[i]);
            }

            stream.Write(span);
            offset += take;
        }
    }

    private static float[] ReadTensor(
        Stream stream,
        int count,
        int layerIndex,
        string what)
    {
        var values = new float[count];
        Span<byte> buffer = stackalloc byte[TensorChunkBytes];
        var offset = 0;

        while (offset < count)
        {
            var take = Math.Min(count - offset, TensorChunkBytes / DsBrainFormat.ElementSize);
            var span = buffer[..(take * DsBrainFormat.ElementSize)];

            if (!ReadFully(stream, span))
            {
                throw new InvalidDataException(
                    $"The model is truncated inside the {what} of layer {layerIndex}.");
            }

            for (var i = 0; i < take; i++)
            {
                values[offset + i] = BinaryPrimitives.ReadSingleLittleEndian(span[(i * 4)..]);
            }

            offset += take;
        }

        ValidateFinite(values, layerIndex, what);
        return values;
    }

    private static float[] ReadTensor(ReadOnlySpan<byte> data, int count, int layerIndex, string what)
    {
        var bytes = count * DsBrainFormat.ElementSize;
        if (data.Length < bytes)
        {
            throw new InvalidDataException(
                $"The model is truncated inside the {what} of layer {layerIndex}.");
        }

        var values = new float[count];

        for (var i = 0; i < count; i++)
        {
            values[i] = BinaryPrimitives.ReadSingleLittleEndian(data[(i * 4)..]);
        }

        ValidateFinite(values, layerIndex, what);
        return values;
    }

    private static void ValidateFinite(float[] values, int layerIndex, string what)
    {
        for (var i = 0; i < values.Length; i++)
        {
            if (!float.IsFinite(values[i]))
            {
                throw new InvalidDataException(
                    $"{char.ToUpperInvariant(what[0])}{what[1..]} of layer {layerIndex} contain a " +
                    $"non-finite value at index {i}.");
            }
        }
    }

    private static bool ReadFully(Stream stream, Span<byte> destination)
    {
        var read = 0;

        while (read < destination.Length)
        {
            var taken = stream.Read(destination[read..]);
            if (taken == 0)
            {
                return false;
            }

            read += taken;
        }

        return true;
    }
}
