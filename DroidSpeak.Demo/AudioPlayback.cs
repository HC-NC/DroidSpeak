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

            player.PlaybackEnded += (_, _) => finished.TrySetResult(true);
            device.Start();
            device.MasterMixer.AddComponent(player);
            player.Play();

            var allowInterrupt = !Console.IsInputRedirected && player.Duration > 1.2f;
            if (allowInterrupt)
            {
                Console.WriteLine("  [Press any key to stop...]");
                while (Console.KeyAvailable)
                {
                    Console.ReadKey(intercept: true);
                }
            }

            var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(Math.Max(2.0, player.Duration + 2.0));
            while (!finished.Task.Wait(30))
            {
                if (allowInterrupt && Console.KeyAvailable)
                {
                    Console.ReadKey(intercept: true);
                    player.Stop();
                    Console.WriteLine("  [Stopped]");
                    return true;
                }

                if (DateTime.UtcNow > deadline)
                {
                    player.Stop();
                    error = "playback timed out";
                    return false;
                }
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

    /// <summary>
    /// Plays mono samples pulled on demand from <paramref name="pull"/>, starting playback as soon
    /// as the first chunk arrives. The producer runs on a background thread and feeds a bounded
    /// queue, so long audio never has to be rendered up front.
    /// </summary>
    /// <param name="pull">Fills the supplied buffer with the next mono chunk and returns its length (0 at the end).</param>
    /// <param name="sampleRate">Sample rate of the pulled samples.</param>
    /// <param name="totalSamples">Total number of samples the puller will produce.</param>
    /// <param name="error">A short diagnostic when playback could not be started or completed.</param>
    /// <returns><c>true</c> when playback completed.</returns>
    public static bool TryPlayStream(
        Func<float[], int> pull,
        int sampleRate,
        int totalSamples,
        out string? error)
    {
        ArgumentNullException.ThrowIfNull(pull);
        error = null;

        if (totalSamples <= 0)
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

            // About 2 seconds of interleaved stereo: the producer blocks when the queue is full
            // instead of quietly buffering the entire utterance in memory.
            var maxQueuedSamples = Math.Max(4096, sampleRate * 4);
            using var provider = new QueueDataProvider(format, maxQueuedSamples, QueueFullBehavior.Block);
            using var device = engine.InitializePlaybackDevice(deviceInfo, format);
            using var player = new SoundPlayer(engine, format, provider);

            var finished = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            player.PlaybackEnded += (_, _) => finished.TrySetResult(true);

            using var cancellation = new CancellationTokenSource();
            Exception? producerError = null;

            var producer = Task.Run(() =>
            {
                try
                {
                    var mono = new float[1024];
                    var stereo = new float[mono.Length * 2];

                    while (!cancellation.IsCancellationRequested)
                    {
                        var count = pull(mono);
                        if (count <= 0)
                        {
                            break;
                        }

                        count = Math.Min(count, mono.Length);
                        for (var i = 0; i < count; i++)
                        {
                            var sample = mono[i];
                            stereo[i * 2] = sample;
                            stereo[i * 2 + 1] = sample;
                        }

                        provider.AddSamples(stereo.AsSpan(0, count * 2));
                    }
                }
                catch (Exception exception)
                {
                    producerError = exception;
                }
                finally
                {
                    try
                    {
                        provider.CompleteAdding();
                    }
                    catch (ObjectDisposedException)
                    {
                        // Playback was already torn down.
                    }
                }
            });

            device.Start();
            device.MasterMixer.AddComponent(player);
            player.Play();

            var durationSeconds = totalSamples / (double)sampleRate;
            var allowInterrupt = !Console.IsInputRedirected && durationSeconds > 1.2;
            if (allowInterrupt)
            {
                Console.WriteLine("  [Streaming — press any key to stop...]");
            }

            // A QueueDataProvider reports Length == -1 (unknown, live stream), so SoundFlow never
            // raises PlaybackEnded for it. Watch the dequeued sample count instead: once the producer
            // is done and the device has consumed every interleaved sample, playback is complete.
            var expectedSamples = (long)totalSamples * format.Channels;
            var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(Math.Max(2.0, durationSeconds + 5.0));
            var stopped = false;

            while (true)
            {
                if (finished.Task.IsCompleted)
                {
                    break;
                }

                if (producer.IsCompleted && provider.Position >= expectedSamples)
                {
                    // Give the device a moment to flush its final hardware buffer.
                    Thread.Sleep(120);
                    break;
                }

                if (allowInterrupt && Console.KeyAvailable)
                {
                    Console.ReadKey(intercept: true);
                    stopped = true;
                    break;
                }

                if (DateTime.UtcNow > deadline)
                {
                    error = "streaming playback timed out";
                    break;
                }

                Thread.Sleep(10);
            }

            if (stopped || error is not null)
            {
                player.Stop();
                cancellation.Cancel();
                provider.Reset(); // Unblocks a producer waiting on a full queue.
                if (stopped)
                {
                    Console.WriteLine("  [Stopped]");
                }
            }
            else
            {
                player.Stop();
            }

            producer.Wait(TimeSpan.FromSeconds(2));

            if (producerError is not null && error is null)
            {
                error = producerError.Message;
            }

            return error is null;
        }
        catch (Exception exception)
        {
            _unavailable = true;
            error = exception.Message;
            return false;
        }
    }
}
