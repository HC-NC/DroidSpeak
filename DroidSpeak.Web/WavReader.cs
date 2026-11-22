using System.Buffers.Binary;

namespace DroidSpeak.Web;

public static class WavReader
{
    public static (float[] Samples, int SampleRate) ReadMonoPcm16(ReadOnlySpan<byte> wavBytes)
    {
        if (wavBytes.Length < 44 ||
            wavBytes[0] != 'R' || wavBytes[1] != 'I' || wavBytes[2] != 'F' || wavBytes[3] != 'F')
        {
            throw new FormatException("Invalid WAV file (missing RIFF header).");
        }

        int sampleRate = 44100;
        short channels = 1;
        short bitsPerSample = 16;
        int pos = 12;

        while (pos + 8 <= wavBytes.Length)
        {
            uint chunkId = BinaryPrimitives.ReadUInt32LittleEndian(wavBytes.Slice(pos, 4));
            int chunkSize = BinaryPrimitives.ReadInt32LittleEndian(wavBytes.Slice(pos + 4, 4));
            pos += 8;

            if (chunkSize < 0 || pos + chunkSize > wavBytes.Length) break;

            if (chunkId == 0x20746D66) // "fmt "
            {
                channels = BinaryPrimitives.ReadInt16LittleEndian(wavBytes.Slice(pos + 2, 2));
                sampleRate = BinaryPrimitives.ReadInt32LittleEndian(wavBytes.Slice(pos + 4, 4));
                bitsPerSample = BinaryPrimitives.ReadInt16LittleEndian(wavBytes.Slice(pos + 14, 2));
            }
            else if (chunkId == 0x61746164) // "data"
            {
                if (bitsPerSample != 16)
                {
                    throw new NotSupportedException($"Only 16-bit PCM WAV is supported (got {bitsPerSample}-bit).");
                }

                int bytesPerFrame = Math.Max(1, (int)channels) * 2;
                int frameCount = chunkSize / bytesPerFrame;
                var samples = new float[frameCount];

                for (int i = 0; i < frameCount; i++)
                {
                    short pcm = BinaryPrimitives.ReadInt16LittleEndian(wavBytes.Slice(pos + i * bytesPerFrame, 2));
                    samples[i] = pcm / 32768f;
                }

                return (samples, sampleRate);
            }

            pos += chunkSize + (chunkSize & 1);
        }

        throw new FormatException("WAV file contains no audio data chunk.");
    }
}
