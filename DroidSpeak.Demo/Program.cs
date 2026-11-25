using System.Globalization;
using System.Text;
using DroidSpeak;
using DroidSpeak.Demo;
using DroidSpeak.DSP;
using DroidSpeak.Modem;
using DroidSpeak.NLP;
using DroidSpeak.Music;
using DroidSpeak.Neural;

try
{
    Console.OutputEncoding = Encoding.UTF8;

    if (Console.IsInputRedirected)
    {
        // Piped input is usually UTF-8, while the console code page may be something else.
        Console.InputEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
    }
}
catch (IOException)
{
    // Redirected streams may not support changing the encoding.
}

var options = DemoCommandLine.ParseArguments(args);

if (options.Error is not null)
{
    Console.Error.WriteLine($"error: {options.Error}");
    Console.Error.WriteLine();
    PrintUsage();
    return 1;
}

if (options.ShowHelp)
{
    PrintUsage();
    return 0;
}

var initialProfile = ResolveInitialProfile(options.Voice, out var profileError);
if (profileError is not null)
{
    Console.Error.WriteLine($"error: {profileError}");
    return 1;
}

// A --brain model is loaded before anything speaks, so a broken file fails fast with a one-line
// diagnostic instead of a stack trace deep inside the first synthesis.
NeuralProsodyPredictor? brainPredictor = null;
if (options.Brain is not null)
{
    var (predictor, brainError) = DemoCommandLine.TryLoadBrain(options.Brain);
    if (brainError is not null || predictor is null)
    {
        Console.Error.WriteLine($"error: {brainError ?? "the model could not be loaded"}");
        return 1;
    }

    brainPredictor = predictor;
}

var session = new VoiceSession(
    initialProfile,
    ResolveVoiceName(options.Voice),
    brainPredictor,
    brainPredictor is null ? null : Path.GetFullPath(options.Brain!));

if (options.Music is not null)
{
    var musicSamples = RunMusic(
        VoiceSession.SampleRate,
        session.CurrentProfile,
        options.Music,
        options.Play,
        options.Solo,
        options.Stream);
    if (musicSamples is null)
    {
        return 1;
    }

    if (options.OutputPath is not null)
    {
        SaveSamplesToPath(musicSamples, VoiceSession.SampleRate, options.OutputPath);
    }

    return 0;
}

if (options.Text is null)
{
    RunRepl(session, options);
    return 0;
}

RunUtterance(session.Synth, options.Text, options.OutputPath, options.Play, options.Stream);
return 0;


static float[] RunUtterance(
    DroidSynthesizer synthesizer,
    string text,
    string? outputPath,
    bool play,
    bool stream,
    DroidVoiceProfile? profile = null)
{
    var samples = stream
        ? RenderSpeechStream(synthesizer, text, play, profile)
        : synthesizer.Synthesize(text, profile);

    PrintAnalysis(text, samples, synthesizer.SampleRate);

    if (outputPath is not null)
    {
        var fullPath = Path.GetFullPath(outputPath);
        var directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory))
        {
            Directory.CreateDirectory(directory);
        }

        WavWriter.Write(fullPath, samples, synthesizer.SampleRate);
        Console.WriteLine($"Saved: {fullPath} ({new FileInfo(fullPath).Length} bytes)");
    }

    if (play && !stream)
    {
        Play(samples, synthesizer.SampleRate);
    }

    return samples;
}

/// <summary>
/// Demonstrates <see cref="DroidSpeechStream"/> by pulling 1024-sample chunks on the fly and
/// feeding them straight to the playback queue, so audio starts before the whole utterance exists.
/// </summary>
static float[] RenderSpeechStream(DroidSynthesizer synthesizer, string text, bool play, DroidVoiceProfile? profile)
{
    var stream = synthesizer.CreateStream(text, profile);
    var samples = new float[stream.TotalSamples];
    var offset = 0;
    var seconds = (samples.Length / (double)stream.SampleRate).ToString("0.00", CultureInfo.InvariantCulture);
    Console.WriteLine($"  Streaming speech in 1024-sample chunks ({seconds} s)...");

    int Pull(float[] buffer)
    {
        if (offset >= samples.Length)
        {
            return 0;
        }

        var want = Math.Min(buffer.Length, samples.Length - offset);
        var read = stream.Read(buffer.AsSpan(0, want));
        if (read > 0)
        {
            buffer.AsSpan(0, read).CopyTo(samples.AsSpan(offset));
            offset += read;
        }

        return read;
    }

    if (play && !AudioPlayback.TryPlayStream(Pull, stream.SampleRate, stream.TotalSamples, out var error))
    {
        Console.WriteLine($"  Playback skipped: {error}");
    }

    // Finish the stream even when playback was unavailable or interrupted, so the returned
    // buffer is always complete (used by /save and --output).
    if (offset < samples.Length)
    {
        Drain(Pull);
    }

    return samples;
}

static void Drain(Func<float[], int> pull)
{
    var buffer = new float[1024];
    while (pull(buffer) > 0)
    {
    }
}

/// <summary>
/// Runs the interactive loop. The session owns the mutable voice and prosody state, so every
/// command re-reads <see cref="VoiceSession.Synth"/> instead of caching a synthesizer locally:
/// /voice and /brain invalidate it, and the next utterance picks up the change.
/// </summary>
static void RunRepl(VoiceSession session, DemoOptions options)
{
    Console.WriteLine($"DroidSpeak — R2-D2 voice | {VoiceSession.SampleRate} Hz");
    Console.WriteLine("Type text to speak, or /help for commands.");
    Console.WriteLine();

    float[]? lastSamples = null;

    while (true)
    {
        Console.Write($"{session.VoiceLabel}> ");
        var line = Console.ReadLine();

        if (line is null)
        {
            break;
        }

        line = line.Trim();
        if (line.Length == 0)
        {
            break;
        }

        if (line[0] == '/')
        {
            var separator = line.IndexOf(' ');
            var command = (separator < 0 ? line : line[..separator]).ToLowerInvariant();
            var argument = separator < 0 ? string.Empty : line[(separator + 1)..].Trim();

            switch (command)
            {
                case "/exit":
                case "/quit":
                    return;

                case "/help":
                    PrintReplHelp();
                    continue;

                case "/voice":
                    HandleVoice(argument);
                    continue;

                case "/brain":
                    HandleBrain(argument);
                    continue;

                case "/demo":
                    lastSamples = RunDemo(session, options.Play);
                    continue;

                case "/save":
                    SaveLast(session, lastSamples, argument);
                    continue;

                case "/transmit":
                    lastSamples = RunTransmit(session, argument, options.Play);
                    continue;

                case "/music":
                    lastSamples = RunMusic(
                        VoiceSession.SampleRate,
                        session.CurrentProfile,
                        argument,
                        options.Play,
                        options.Solo,
                        options.Stream);
                    continue;

                case "/stream":
                    if (string.IsNullOrWhiteSpace(argument))
                    {
                        Console.WriteLine("Usage: /stream <text>");
                        continue;
                    }

                    lastSamples = RunUtterance(session.Synth, argument, null, options.Play, stream: true, session.CurrentProfile);
                    continue;

                default:
                    Console.WriteLine($"Unknown command '{command}'. Type /help for available commands.");
                    continue;
            }
        }

        lastSamples = RunUtterance(session.Synth, line, null, options.Play, options.Stream, session.CurrentProfile);
    }

    void HandleVoice(string argument)
    {
        if (string.IsNullOrWhiteSpace(argument))
        {
            Console.WriteLine(
                $"Current voice: {session.VoiceLabel} " +
                $"({session.CurrentProfile.MinFreq.ToString("0", CultureInfo.InvariantCulture)}-" +
                $"{session.CurrentProfile.MaxFreq.ToString("0", CultureInfo.InvariantCulture)} Hz, " +
                $"x{session.CurrentProfile.SpeedMultiplier.ToString("0.00", CultureInfo.InvariantCulture)}).");
            Console.WriteLine("  Presets: r2, grumpy, bass, tiny");
            Console.WriteLine("  /voice json <file>  Load a profile from JSON.");
            Console.WriteLine("  /voice save <file>  Save the current profile as JSON.");
            return;
        }

        var parts = argument.Split(
            ' ',
            2,
            StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        var name = parts[0].ToLowerInvariant();

        if (name is "json" or "save")
        {
            if (parts.Length < 2)
            {
                Console.WriteLine($"Usage: /voice {name} <file.json>");
                return;
            }

            var path = Path.GetFullPath(parts[1]);
            if (name == "save")
            {
                var directory = Path.GetDirectoryName(path);
                if (!string.IsNullOrEmpty(directory))
                {
                    Directory.CreateDirectory(directory);
                }

                File.WriteAllText(path, session.CurrentProfile.ToJson());
                Console.WriteLine($"Voice profile saved: {path}");
                return;
            }

            try
            {
                session.CurrentProfile = DroidVoiceProfile.FromJson(File.ReadAllText(path));
                session.VoiceName = Path.GetFileNameWithoutExtension(path);
                Console.WriteLine($"Voice loaded from {path} as '{session.VoiceName}'.");
            }
            catch (Exception exception) when (exception is IOException
                or UnauthorizedAccessException
                or System.Text.Json.JsonException
                or ArgumentException)
            {
                Console.WriteLine($"Could not load voice: {exception.Message}");
            }

            return;
        }

        var preset = name switch
        {
            "r2" => DroidVoiceProfile.R2,
            "grumpy" => DroidVoiceProfile.Grumpy,
            "bass" => DroidVoiceProfile.Bass,
            "tiny" => DroidVoiceProfile.Tiny,
            _ => null,
        };

        if (preset is null)
        {
            Console.WriteLine($"Unknown voice '{name}'. Available: r2, grumpy, bass, tiny.");
            return;
        }

        session.CurrentProfile = preset;
        session.VoiceName = name;
        Console.WriteLine(
            $"Voice set to {name} " +
            $"({preset.MinFreq.ToString("0", CultureInfo.InvariantCulture)}-" +
            $"{preset.MaxFreq.ToString("0", CultureInfo.InvariantCulture)} Hz, " +
            $"x{preset.SpeedMultiplier.ToString("0.00", CultureInfo.InvariantCulture)}).");
    }

    void HandleBrain(string argument)
    {
        if (string.IsNullOrWhiteSpace(argument))
        {
            PrintBrainStatus();
            Console.WriteLine();
            Console.WriteLine("Usage:");
            Console.WriteLine("  /brain <path.dsbrain>  Load a neural prosody model and use it.");
            Console.WriteLine("  /brain off             Switch back to the rule-based engine.");
            return;
        }

        if (argument.Equals("off", StringComparison.OrdinalIgnoreCase))
        {
            session.UseRuleBased();
            Console.WriteLine("Prosody engine: rule-based.");
            return;
        }

        var (predictor, error) = DemoCommandLine.TryLoadBrain(argument);
        if (predictor is null || error is not null)
        {
            Console.WriteLine($"  {error ?? "the model could not be loaded"}");
            Console.WriteLine("  Keeping the previous prosody engine.");
            return;
        }

        session.UseNeural(predictor, Path.GetFullPath(argument));
        PrintBrainStatus();
    }

    void PrintBrainStatus()
    {
        var neural = session.NeuralPredictor;
        if (neural is null)
        {
            Console.WriteLine("Prosody engine: rule-based (no .dsbrain model loaded).");
            return;
        }

        var model = neural.Model;
        Console.WriteLine(
            $"Prosody engine: neural — {session.BrainPath} " +
            $"({model.InputSize} inputs -> {model.OutputSize} outputs, " +
            $"{model.LayerCount} layers, {model.ParameterCount} parameters).");
    }
}

static float[]? RunDemo(VoiceSession session, bool play)
{
    var demos = new (string Label, string Text)[]
    {
        ("greeting", "Hello there, R2! How are you doing?"),
        ("alert", "ALERT! Reactor core overheating!"),
        ("sad", "Oh no... everything is broken again..."),
        ("curious", "Beep boop! Wait, what is this over here?"),
    };

    float[]? lastSamples = null;

    foreach (var (label, text) in demos)
    {
        Console.WriteLine($"[{label}] {text} ({session.VoiceLabel})");
        lastSamples = session.Synth.Synthesize(text, session.CurrentProfile);
        PrintAnalysis(text, lastSamples, session.Synth.SampleRate);

        if (play)
        {
            Play(lastSamples, session.Synth.SampleRate);
        }

        Console.WriteLine();
    }

    return lastSamples;
}

static float[]? RunTransmit(VoiceSession session, string text, bool play)
{
    if (string.IsNullOrWhiteSpace(text))
    {
        Console.WriteLine("Usage: /transmit <text>");
        return null;
    }

    var synthesizer = session.Synth;
    var speech = synthesizer.Synthesize(text, session.CurrentProfile);
    float[] mixed;
    try
    {
        mixed = DroidModemEncoder.EncodeIntoSpeech(text, speech, synthesizer.SampleRate);
    }
    catch (ArgumentException exception)
    {
        Console.WriteLine($"  Cannot transmit: {exception.Message}");
        return null;
    }

    var path = Path.GetFullPath("transmit.wav");
    WavWriter.Write(path, mixed, synthesizer.SampleRate);
    Console.WriteLine($"  Transmitting {text.Length} chars ({mixed.Length} samples).");
    Console.WriteLine($"  Saved: {path} ({new FileInfo(path).Length} bytes)");

    if (play)
    {
        Play(mixed, synthesizer.SampleRate);
    }

    var decoded = DroidModemDecoder.Decode(mixed, synthesizer.SampleRate);
    if (decoded is null)
    {
        Console.WriteLine("[Decoded from audio]: <failed to decode>");
    }
    else
    {
        var suffix = decoded == text ? string.Empty : "  (mismatch!)";
        Console.WriteLine($"[Decoded from audio]: {decoded}{suffix}");
    }

    return mixed;
}

static void SaveLast(VoiceSession session, float[]? samples, string argument)
{
    if (samples is null)
    {
        Console.WriteLine("Nothing to save yet. Speak something first.");
        return;
    }

    if (string.IsNullOrWhiteSpace(argument))
    {
        Console.WriteLine("Usage: /save <filename.wav>");
        return;
    }

    var fullPath = Path.GetFullPath(argument);
    var directory = Path.GetDirectoryName(fullPath);
    if (!string.IsNullOrEmpty(directory))
    {
        Directory.CreateDirectory(directory);
    }

    WavWriter.Write(fullPath, samples, session.Synth.SampleRate);
    Console.WriteLine($"Saved: {fullPath} ({new FileInfo(fullPath).Length} bytes)");
}

static DroidVoiceProfile ResolveInitialProfile(string? voiceArg, out string? error)
{
    error = null;
    if (string.IsNullOrWhiteSpace(voiceArg))
    {
        return DroidVoiceProfile.R2;
    }

    var key = voiceArg.Trim();
    return key.ToLowerInvariant() switch
    {
        "r2" => DroidVoiceProfile.R2,
        "grumpy" => DroidVoiceProfile.Grumpy,
        "bass" => DroidVoiceProfile.Bass,
        "tiny" => DroidVoiceProfile.Tiny,
        _ when File.Exists(key) => DroidVoiceProfile.FromJson(File.ReadAllText(key)).Normalized(),
        _ => SetError(out error, $"unknown voice '{voiceArg}'. Use r2, grumpy, bass, tiny, or a JSON file path.")
    };

    static DroidVoiceProfile SetError(out string? err, string msg)
    {
        err = msg;
        return DroidVoiceProfile.R2;
    }
}

/// <summary>
/// The prompt label for the startup voice: the preset name, or the JSON profile's file stem. Purely
/// presentation — the profile itself is validated by <see cref="ResolveInitialProfile"/>.
/// </summary>
static string ResolveVoiceName(string? voiceArg)
{
    if (string.IsNullOrWhiteSpace(voiceArg))
    {
        return "r2";
    }

    var key = voiceArg.Trim();
    var preset = key.ToLowerInvariant();
    return preset is "r2" or "grumpy" or "bass" or "tiny"
        ? preset
        : Path.GetFileNameWithoutExtension(key);
}

static void SaveSamplesToPath(float[] samples, int sampleRate, string outputPath)
{
    var fullPath = Path.GetFullPath(outputPath);
    var directory = Path.GetDirectoryName(fullPath);
    if (!string.IsNullOrEmpty(directory))
    {
        Directory.CreateDirectory(directory);
    }

    WavWriter.Write(fullPath, samples, sampleRate);
    Console.WriteLine($"Saved: {fullPath} ({new FileInfo(fullPath).Length} bytes)");
}


static void PrintAnalysis(string text, float[] samples, int sampleRate)
{
    foreach (var phrase in TextParser.Parse(text))
    {
        var valence = phrase.Emotion.Valence.ToString("0.00", CultureInfo.InvariantCulture);
        var arousal = phrase.Emotion.Arousal.ToString("0.00", CultureInfo.InvariantCulture);
        var tempo = phrase.TempoScale.ToString("0.00", CultureInfo.InvariantCulture);

        Console.WriteLine(
            $"  ({valence}, {arousal})  {phrase.Punctuation}  " +
            $"tempo={tempo}  syllables={phrase.SyllableCount}");
    }

    var seconds = (samples.Length / (double)sampleRate).ToString("0.00", CultureInfo.InvariantCulture);
    Console.WriteLine($"  Duration: {seconds} s ({samples.Length} samples)");
}

static void Play(float[] samples, int sampleRate)
{
    if (!AudioPlayback.TryPlay(samples, sampleRate, out var error))
    {
        Console.WriteLine($"  Playback skipped: {error}");
    }
}

static float[]? RunMusic(
    int sampleRate,
    DroidVoiceProfile profile,
    string argument,
    bool play,
    bool solo = false,
    bool stream = false)
{
    var key = argument.Trim();
    if (key.StartsWith("--solo", StringComparison.OrdinalIgnoreCase))
    {
        solo = true;
        key = key["--solo".Length..].Trim();
    }
    else if (key.EndsWith("--solo", StringComparison.OrdinalIgnoreCase))
    {
        solo = true;
        key = key[..^"--solo".Length].Trim();
    }

    if (string.IsNullOrWhiteSpace(key))
    {
        Console.WriteLine("Usage:");
        Console.WriteLine("  /music [--solo] <file.mid>       Play a Standard MIDI file (.mid).");
        Console.WriteLine("  /music [--solo] <rtttl string>   Play a raw RTTTL ringtone string.");
        return null;
    }

    DroidScore score;
    try
    {
        if (File.Exists(key))
        {
            var ext = Path.GetExtension(key).ToLowerInvariant();
            score = ext == ".rtttl"
                ? RtttlParser.Shared.Load(key)
                : MidiParser.Shared.Load(key);
        }
        else
        {
            score = RtttlParser.Shared.Parse(key);
        }
    }
    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FormatException or NotSupportedException)
    {
        Console.WriteLine($"  Music error: {ex.Message}");
        return null;
    }

    var musicSynth = new DroidMusicSynthesizer(sampleRate, profile);
    var title = string.IsNullOrWhiteSpace(score.Title) ? "Untitled" : score.Title;
    var modeLabel = solo ? ", solo" : string.Empty;

    if (stream)
    {
        // Pull 1024-sample chunks from DroidMusicStream while playback is already running.
        var musicStream = musicSynth.CreateStream(score, profile, solo);
        var streamed = new float[musicStream.TotalSamples];
        var offset = 0;
        var streamSeconds = (streamed.Length / (double)sampleRate).ToString("0.00", CultureInfo.InvariantCulture);
        Console.WriteLine(
            $"  Streaming: {title} ({score.Notes.Count} notes, {streamSeconds} s{modeLabel}, 1024-sample chunks)");

        int Pull(float[] buffer)
        {
            if (offset >= streamed.Length)
            {
                return 0;
            }

            var want = Math.Min(buffer.Length, streamed.Length - offset);
            var read = musicStream.Read(buffer.AsSpan(0, want));
            if (read > 0)
            {
                buffer.AsSpan(0, read).CopyTo(streamed.AsSpan(offset));
                offset += read;
            }

            return read;
        }

        if (play && !AudioPlayback.TryPlayStream(Pull, sampleRate, musicStream.TotalSamples, out var error))
        {
            Console.WriteLine($"  Playback skipped: {error}");
        }

        if (offset < streamed.Length)
        {
            Drain(Pull);
        }

        return streamed;
    }

    var samples = musicSynth.Render(score, soloMelodyOnly: solo);

    var seconds = (samples.Length / (double)sampleRate).ToString("0.00", CultureInfo.InvariantCulture);
    Console.WriteLine($"  Playing: {title} ({score.Notes.Count} notes, {seconds} s{modeLabel})");

    if (play)
    {
        Play(samples, sampleRate);
    }

    return samples;
}

static void PrintUsage()
{
    Console.WriteLine("DroidSpeak — procedural astromech-style text-to-beep & music demo.");
    Console.WriteLine();
    Console.WriteLine("Usage:");
    Console.WriteLine("  dotnet run --project DroidSpeak.Demo -- \"<text>\" [-v voice] [--brain file.dsbrain] [-o file.wav] [--stream] [--no-play]");
    Console.WriteLine("  dotnet run --project DroidSpeak.Demo -- -m <file.mid|rtttl> [-v voice] [--solo] [--stream] [-o file.wav]");
    Console.WriteLine("  dotnet run --project DroidSpeak.Demo");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  -m, --music <arg>    Play or export a melody (.mid file or RTTTL string).");
    Console.WriteLine("  -v, --voice <voice>  Voice preset (r2, grumpy, bass, tiny) or path to a JSON profile.");
    Console.WriteLine("  --brain <file>       Load a .dsbrain neural prosody model (60->17) for this run.");
    Console.WriteLine("  -o, --output <file>  Save the synthesized audio as a 16-bit PCM WAV.");
    Console.WriteLine("  --solo               Play only the upper lead melody when rendering music.");
    Console.WriteLine("  --stream             Pull audio in 1024-sample chunks and start playback immediately.");
    Console.WriteLine("  --play               Play audio (default).");
    Console.WriteLine("  --no-play            Do not play audio.");
    Console.WriteLine("  -h, --help           Show this help.");
    Console.WriteLine();
    Console.WriteLine("Without text or --music, an interactive REPL starts:");
    PrintReplCommands();
}

static void PrintReplHelp()
{
    Console.WriteLine("Commands:");
    PrintReplCommands();
}

static void PrintReplCommands()
{
    Console.WriteLine("  /voice [r2|grumpy|bass|tiny]         Show or change the droid voice.");
    Console.WriteLine("  /voice json <file>                   Load a voice profile from JSON.");
    Console.WriteLine("  /voice save <file>                   Save the current voice profile as JSON.");
    Console.WriteLine("  /brain [file.dsbrain|off]            Show, load, or disable a neural prosody model.");
    Console.WriteLine("  /music [--solo] <file.mid|rtttl>     Play a melody using the current droid voice.");
    Console.WriteLine("  /stream <text>                       Speak text by pulling real-time 1024-sample chunks.");
    Console.WriteLine("  /transmit <text>                     Hide text in speech, save transmit.wav and decode it.");
    Console.WriteLine("  /demo                                Play four test phrases.");
    Console.WriteLine("  /save <file.wav>                     Save the last utterance or melody.");
    Console.WriteLine("  /help                                Show REPL commands.");
    Console.WriteLine("  /exit                                Quit (empty input also quits).");
}
