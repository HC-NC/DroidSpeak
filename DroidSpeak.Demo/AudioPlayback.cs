using SoundFlow.Backends.MiniAudio;
using SoundFlow.Components;
using SoundFlow.Enums;
using SoundFlow.Providers;
using SoundFlow.Structs;

namespace DroidSpeak.Demo;

/// <summary>
/// Plays synthesized mono samples through the default output device using SoundFlow.
/// Playback failures (headless CI, missing device, unsupported format) are reported instead of
/// crashing the demo.
/// </summary>
internal static class AudioPlayback
{
    private static bool _unavailable;

    /// <summary>
    /// Plays <paramref name="samples"/> and blocks until playback finishes or times out.
    /// </summary>
    /// <param name="samples">Mono samples in <c>[-1, 1]</c>.</param>
    /// <param name="sampleRate">Sample rate of <paramref name="samples"/>.</param>
    /// <param name="error">A short diagnostic when playback could not be started.</param>
    /// <returns><c>true</c> when playback completed.</returns>
    public static bool TryPlay(float[] samples, int sampleRate, out string? error)
    {
        error = null;

        if (samples.Length == 0)
        {
            return true;
        }

        if (_unavailable)
        {
            error = "playback is unavailable in this environment";
            return false;
        }

        try
        {
            // SoundFlow reads interleaved data, so duplicate the mono signal into stereo.
            var interleaved = new float[samples.Length * 2];
            for (var i = 0; i < samples.Length; i++)
            {
                var sample = samples[i];
                interleaved[i * 2] = sample;
                interleaved[i * 2 + 1] = sample;
            }

            var format = new AudioFormat
            {
                Format = SampleFormat.F32,
                Channels = 2,
                Layout = ChannelLayout.Stereo,
                SampleRate = sampleRate,
            };

            using var engine = new MiniAudioEngine();
            var devices = engine.PlaybackDevices;
            var deviceInfo = devices.FirstOrDefault(device => device.IsDefault);
            if (deviceInfo.Id == IntPtr.Zero)
            {
                deviceInfo = devices.FirstOrDefault();
            }

            if (deviceInfo.Id == IntPtr.Zero)
            {
                _unavailable = true;
                error = "no playback device was found";
                return false;
            }

            var finished = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            using var device = engine.InitializePlaybackDevice(deviceInfo, format);
            using var provider = new RawDataProvider(interleaved, sampleRate);
            using var player = new SoundPlayer(engine, format, provider);

            player.PlaybackEnded += (_, _) => finished.TrySetResult(true);
            device.Start();
            device.MasterMixer.AddComponent(player);
            player.Play();

            var timeout = TimeSpan.FromSeconds(Math.Max(2.0, player.Duration + 2.0));
            if (!finished.Task.Wait(timeout))
            {
                error = "playback timed out";
                return false;
            }

            return true;
        }
        catch (Exception exception)
        {
            _unavailable = true;
            error = exception.Message;
            return false;
        }
    }
}
