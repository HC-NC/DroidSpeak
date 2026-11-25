using System.Buffers.Binary;
using DroidSpeak.Neural;

namespace DroidSpeak.Tests;

public class DsBrainSerializationTests
{
    [Fact]
    public void Save_Load_RoundTripsThroughAStream()
    {
        var model = CreateModel();
        using var stream = new MemoryStream();

        model.Save(stream);
        stream.Position = 0;

        var restored = MlpModel.Load(stream);

        Assert.Equal(model.InputSize, restored.InputSize);
        Assert.Equal(model.OutputSize, restored.OutputSize);
        Assert.Equal(model.LayerCount, restored.LayerCount);
        Assert.Equal(model.ParameterCount, restored.ParameterCount);
        AssertPredictionsMatch(model, restored);
    }

    [Fact]
    public void Save_Load_RoundTripsThroughASpan()
    {
        var model = CreateModel();
        using var stream = new MemoryStream();
        model.Save(stream);

        var restored = MlpModel.Load(stream.ToArray().AsSpan());

        AssertPredictionsMatch(model, restored);
    }

    [Fact]
    public void Save_IsByteDeterministic()
    {
        using var first = new MemoryStream();
        using var second = new MemoryStream();
        var model = CreateModel();

        model.Save(first);
        model.Save(second);

        Assert.Equal(first.ToArray(), second.ToArray());
    }

    [Fact]
    public void Save_WritesTheDocumentedHeaderLayout()
    {
        var bytes = Serialize(CreateModel());

        Assert.Equal("DSBR"u8.ToArray(), bytes[..4]);
        Assert.Equal(DsBrainFormat.Version, bytes[4]);
        Assert.Equal(DsBrainFormat.ReservedFlags, bytes[5]);
        Assert.Equal(3, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(6)));
        Assert.Equal(2, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(8)));
        Assert.Equal(2, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(10)));
    }

    [Fact]
    public void Save_WritesLayerRecordsInTheDocumentedOrder()
    {
        var bytes = Serialize(CreateModel());
        var offset = DsBrainFormat.HeaderSize;

        // Layer 0 is 3x4 tanh, layer 1 is 4x2 sigmoid.
        Assert.Equal(3, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(offset)));
        Assert.Equal(4, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(offset + 2)));
        Assert.Equal((byte)ActivationType.Tanh, bytes[offset + 4]);
        Assert.Equal(0, bytes[offset + 5]);
        Assert.Equal(12u, BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset + 6)));
        Assert.Equal(4u, BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(offset + 10)));

        var second = offset + DsBrainFormat.LayerHeaderSize
            + (12 + 4) * DsBrainFormat.ElementSize;
        Assert.Equal(4, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(second)));
        Assert.Equal(2, BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(second + 2)));
        Assert.Equal((byte)ActivationType.Sigmoid, bytes[second + 4]);
        Assert.Equal(8u, BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(second + 6)));
        Assert.Equal(2u, BinaryPrimitives.ReadUInt32LittleEndian(bytes.AsSpan(second + 10)));

        var expectedLength = second
            + DsBrainFormat.LayerHeaderSize
            + (8 + 2) * DsBrainFormat.ElementSize;
        Assert.Equal(expectedLength, bytes.Length);
        Assert.Equal(bytes.Length, MlpModel.Measure(bytes));
    }

    [Fact]
    public void Save_WritesFloatsLittleEndian()
    {
        // The first weight of layer 0 is 0.25f, whose little-endian IEEE-754 form is known exactly.
        var bytes = Serialize(CreateModel());
        var tensorStart = DsBrainFormat.HeaderSize + DsBrainFormat.LayerHeaderSize;

        Assert.Equal(new byte[] { 0x00, 0x00, 0x80, 0x3E }, bytes[tensorStart..(tensorStart + 4)].ToArray());
    }

    [Fact]
    public void Measure_RejectsATruncatedFile()
    {
        var bytes = Serialize(CreateModel());

        Assert.Throws<InvalidDataException>(() => MlpModel.Measure(bytes[..8]));
        Assert.Throws<InvalidDataException>(() => MlpModel.Measure(bytes[..(bytes.Length - 4)]));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(4)]
    [InlineData(DsBrainFormat.HeaderSize)]
    [InlineData(DsBrainFormat.HeaderSize + DsBrainFormat.LayerHeaderSize + 4)]
    public void Load_RejectsTruncatedData(int length)
    {
        var bytes = Serialize(CreateModel());
        var truncated = bytes[..Math.Min(length, bytes.Length)];

        Assert.Throws<InvalidDataException>(() => MlpModel.Load(truncated));

        using var stream = new MemoryStream(truncated);
        Assert.Throws<InvalidDataException>(() => MlpModel.Load(stream));
    }

    [Fact]
    public void Load_RejectsAnEmptyBuffer()
    {
        Assert.Throws<InvalidDataException>(() => MlpModel.Load(ReadOnlySpan<byte>.Empty));

        using var stream = new MemoryStream();
        Assert.Throws<InvalidDataException>(() => MlpModel.Load(stream));
    }

    [Fact]
    public void Load_RejectsAWrongMagic()
    {
        var bytes = Serialize(CreateModel());
        bytes[1] = (byte)'X';

        Assert.Throws<InvalidDataException>(() => MlpModel.Load(bytes));
    }

    [Fact]
    public void Load_RejectsAnUnknownVersion()
    {
        var bytes = Serialize(CreateModel());
        bytes[4] = 2;

        var error = Assert.Throws<InvalidDataException>(() => MlpModel.Load(bytes));
        Assert.Contains("version", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_RejectsReservedFlags()
    {
        var bytes = Serialize(CreateModel());
        bytes[5] = 1;

        Assert.Throws<InvalidDataException>(() => MlpModel.Load(bytes));
    }

    [Fact]
    public void Load_RejectsZeroSizedHeaderFields()
    {
        var bytes = Serialize(CreateModel());
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(6), 0);

        Assert.Throws<InvalidDataException>(() => MlpModel.Load(bytes));
    }

    [Fact]
    public void Load_RejectsAnImpossibleLayerCount()
    {
        var bytes = Serialize(CreateModel());
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(10), 999);

        Assert.Throws<InvalidDataException>(() => MlpModel.Load(bytes));
    }

    [Fact]
    public void Load_RejectsAnUnknownActivationCode()
    {
        var bytes = Serialize(CreateModel());
        bytes[DsBrainFormat.HeaderSize + 4] = 77;

        var error = Assert.Throws<InvalidDataException>(() => MlpModel.Load(bytes));
        Assert.Contains("activation", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Load_RejectsWidthsAboveTheForwardPassLimit()
    {
        var bytes = Serialize(CreateModel());

        // Claim a 900-wide input without touching the tensor sizes: the width check must fire first.
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(DsBrainFormat.HeaderSize), 900);

        Assert.Throws<InvalidDataException>(() => MlpModel.Load(bytes));

        using var stream = new MemoryStream(bytes);
        Assert.Throws<InvalidDataException>(() => MlpModel.Load(stream));
    }

    [Fact]
    public void Load_RejectsTensorCountsThatDisagreeWithTheShape()
    {
        var bytes = Serialize(CreateModel());
        BinaryPrimitives.WriteUInt32LittleEndian(
            bytes.AsSpan(DsBrainFormat.HeaderSize + 6),
            11u);

        Assert.Throws<InvalidDataException>(() => MlpModel.Load(bytes));
    }

    [Fact]
    public void Load_RejectsANonFiniteWeight()
    {
        var bytes = Serialize(CreateModel());
        var tensorStart = DsBrainFormat.HeaderSize + DsBrainFormat.LayerHeaderSize;
        WriteSingle(bytes, tensorStart, float.NaN);

        var error = Assert.Throws<InvalidDataException>(() => MlpModel.Load(bytes));
        Assert.Contains("non-finite", error.Message, StringComparison.Ordinal);

        WriteSingle(bytes, tensorStart, float.PositiveInfinity);
        Assert.Throws<InvalidDataException>(() => MlpModel.Load(bytes));
    }

    [Fact]
    public void Load_RejectsTrailingBytes()
    {
        var bytes = Serialize(CreateModel());
        byte[] withTrailer = [.. bytes, (byte)0xAA];

        Assert.Throws<InvalidDataException>(() => MlpModel.Load(withTrailer));

        using var stream = new MemoryStream(withTrailer);
        Assert.Throws<InvalidDataException>(() => MlpModel.Load(stream));
    }

    [Fact]
    public void Load_RejectsADiscontinuousChain()
    {
        // Rewire the second layer so its input width no longer matches the first layer's output,
        // while keeping the record itself internally consistent.
        var bytes = RebuildWithMismatchedChain();

        var error = Assert.Throws<InvalidDataException>(() => MlpModel.Load(bytes));
        Assert.Contains("but layer 1 expects", error.Message, StringComparison.Ordinal);

        using var stream = new MemoryStream(bytes);
        Assert.Throws<InvalidDataException>(() => MlpModel.Load(stream));

        Assert.Throws<InvalidDataException>(() => MlpModel.Measure(bytes));
    }

    [Fact]
    public void Load_RejectsAHeaderThatDisagreesWithTheLayers()
    {
        var bytes = Serialize(CreateModel());

        // Every record stays internally valid, but the header promises three outputs while the last
        // layer only produces two.
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(8), 3);

        Assert.Throws<InvalidDataException>(() => MlpModel.Load(bytes));

        using var stream = new MemoryStream(bytes);
        Assert.Throws<InvalidDataException>(() => MlpModel.Load(stream));
    }

    [Fact]
    public void Load_StreamsByteByByte()
    {
        var model = CreateModel();
        using var source = new MemoryStream();
        model.Save(source);

        using var dribbling = new OneByteAtATimeStream(source.ToArray());
        var restored = MlpModel.Load(dribbling);

        AssertPredictionsMatch(model, restored);
    }

    [Fact]
    public void SaveAndLoad_RejectStreamsTheyCannotUse()
    {
        var model = CreateModel();

        Assert.Throws<ArgumentNullException>(() => model.Save(null!));
        Assert.Throws<ArgumentNullException>(() => MlpModel.Load((Stream)null!));
        Assert.Throws<ArgumentException>(() => model.Save(new NonWritableStream()));
        Assert.Throws<ArgumentException>(() => MlpModel.Load(new NonReadableStream()));
    }

    [Fact]
    public void FormatConstants_DescribeTheContainer()
    {
        Assert.Equal(12, DsBrainFormat.HeaderSize);
        Assert.Equal(14, DsBrainFormat.LayerHeaderSize);
        Assert.Equal(4, DsBrainFormat.ElementSize);
        Assert.Equal(".dsbrain", DsBrainFormat.FileExtension, StringComparer.OrdinalIgnoreCase);
        Assert.Equal("DSBR"u8.ToArray(), DsBrainFormat.Magic.ToArray());
    }

    [Fact]
    public void Load_PreservesEveryActivationType()
    {
        foreach (ActivationType activation in Enum.GetValues<ActivationType>())
        {
            var model = new MlpModel(
            [
                DenseLayer.Create([0.5f, -0.25f], [0.1f], 2, 1, activation),
            ]);

            using var stream = new MemoryStream();
            model.Save(stream);
            var restored = MlpModel.Load(stream.ToArray().AsSpan());

            Assert.Equal(activation, restored[0].Activation);
            AssertPredictionsMatch(model, restored);
        }
    }

    private static MlpModel CreateModel() => new(
    [
        DenseLayer.Create(
            [0.25f, -0.5f, 1f, 0.125f, -2f, 3f, 0.5f, -0.25f, 1.5f, -1f, 2f, 0.75f],
            [0.1f, -0.2f, 0.3f, -0.4f],
            inputSize: 3,
            outputSize: 4,
            activation: ActivationType.Tanh),
        DenseLayer.Create(
            [1f, -1f, 0.5f, 0.25f, 2f, -0.5f, 0.125f, 3f],
            [0.05f, -0.05f],
            inputSize: 4,
            outputSize: 2,
            activation: ActivationType.Sigmoid),
    ]);

    private static byte[] Serialize(MlpModel model)
    {
        using var stream = new MemoryStream();
        model.Save(stream);
        return stream.ToArray();
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        var buffer = new byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(buffer, value);
        Array.Copy(buffer, 0, bytes, offset, 4);
    }

    private static byte[] RebuildWithMismatchedChain()
    {
        // 3x4 tanh followed by a 3x2 sigmoid: the record is consistent, but the chain is not.
        var bytes = Serialize(CreateModel());
        var second = DsBrainFormat.HeaderSize + DsBrainFormat.LayerHeaderSize
            + (12 + 4) * DsBrainFormat.ElementSize;

        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(second), 3);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(second + 6), 6u);

        return bytes;
    }

    private static void AssertPredictionsMatch(MlpModel expected, MlpModel actual)
    {
        Assert.Equal(expected.InputSize, actual.InputSize);

        var samples = new float[] { 0f, 0.5f, -0.5f, 1f, 2.5f, -3f };

        for (var i = 0; i < samples.Length; i++)
        {
            var input = new float[expected.InputSize];
            for (var n = 0; n < input.Length; n++)
            {
                input[n] = samples[(i + n) % samples.Length];
            }

            var left = new float[expected.OutputSize];
            var right = new float[actual.OutputSize];

            expected.Predict(input, left);
            actual.Predict(input, right);

            Assert.Equal(left, right);
        }
    }

    private sealed class OneByteAtATimeStream : MemoryStream
    {
        public OneByteAtATimeStream(byte[] buffer)
            : base(buffer)
        {
        }

        public override int Read(Span<byte> destination)
        {
            if (destination.IsEmpty)
            {
                return 0;
            }

            return base.Read(destination[..1]);
        }
    }

    private sealed class NonWritableStream : MemoryStream
    {
        public override bool CanWrite => false;

        public override void Write(ReadOnlySpan<byte> buffer) =>
            throw new NotSupportedException();
    }

    private sealed class NonReadableStream : MemoryStream
    {
        public override bool CanRead => false;

        public override int Read(Span<byte> destination) => throw new NotSupportedException();
    }
}
