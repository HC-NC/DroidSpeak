using System.Globalization;
using System.Text;
using DroidSpeak;
using DroidSpeak.Demo;
using DroidSpeak.DSP;
using DroidSpeak.Modem;
using DroidSpeak.NLP;
using DroidSpeak.Music;

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

var options = ParseOptions(args);

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

var synthesizer = new DroidSynthesizer(profile: initialProfile);

if (options.Music is not null)
{
    var musicSamples = RunMusic(synthesizer.SampleRate, initialProfile, options.Music, options.Play, options.Solo);
    if (musicSamples is null)
    {
        return 1;
    }

    if (options.OutputPath is not null)
    {
        SaveSamplesToPath(musicSamples, synthesizer.SampleRate, options.OutputPath);
    }

    return 0;
}

if (options.Text is null)
{
    RunRepl(synthesizer, options);
    return 0;
}

RunUtterance(synthesizer, options.Text, options.OutputPath, options.Play);
return 0;


static CliOptions ParseOptions(string[] arguments)
{
    string? outputPath = null;
    string? music = null;
    string? voice = null;
    string? error = null;
    var play = true;
    var solo = false;
    var showHelp = false;
    var textParts = new List<string>();

    for (var i = 0; i < arguments.Length; i++)
    {
        var argument = arguments[i];
        switch (argument)
        {
            case "-h":
            case "--help":
                showHelp = true;
                break;

            case "-o":
            case "--output":
                if (i + 1 >= arguments.Length)
                    error = $"missing filename after '{argument}'";
                else
                    outputPath = arguments[++i];
                break;

            case "-m":
            case "--music":
                if (i + 1 >= arguments.Length)
                    error = $"missing melody preset, file, or RTTTL string after '{argument}'";
                else
                    music = arguments[++i];
                break;

            case "-v":
            case "--voice":
                if (i + 1 >= arguments.Length)
                    error = $"missing voice preset or JSON path after '{argument}'";
                else
                    voice = arguments[++i];
                break;

            case "--play":
                play = true;
                break;

            case "--solo":
                solo = true;
                break;

            case "--no-play":
                play = false;
                break;

            default:
                if (argument.StartsWith('-') && argument.Length > 1)
                    error = $"unknown option '{argument}'";
                else
                    textParts.Add(argument);
                break;
        }
    }

    if (music is not null && textParts.Count > 0)
    {
        error = "cannot combine text input with --music in a single CLI call";
    }

    var text = textParts.Count == 0 ? null : string.Join(' ', textParts);
    return new CliOptions(text, music, voice, outputPath, play, solo, showHelp, error);
}

static void RunUtterance(
    DroidSynthesizer synthesizer,
    string text,
    string? outputPath,
    bool play)
{
    var samples = synthesizer.Synthesize(text);
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

    if (play)
    {
        Play(samples, synthesizer.SampleRate);
    }
}

static void RunRepl(DroidSynthesizer synthesizer, CliOptions options)
{
    Console.WriteLine($"DroidSpeak — R2-D2 voice | {synthesizer.SampleRate} Hz");
    Console.WriteLine("Type text to speak, or /help for commands.");
    Console.WriteLine();

    float[]? lastSamples = null;
    var currentProfile = synthesizer.Profile;
    var voiceLabel = "r2";

    while (true)
    {
        Console.Write($"{voiceLabel}> ");
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

                case "/demo":
                    lastSamples = RunDemo(synthesizer, currentProfile, options.Play, voiceLabel);
                    continue;

                case "/save":
                    SaveLast(synthesizer, lastSamples, argument);
                    continue;

                case "/transmit":
                    lastSamples = RunTransmit(synthesizer, currentProfile, argument, options.Play);
                    continue;

                case "/music":
                    lastSamples = RunMusic(synthesizer.SampleRate, currentProfile, argument, options.Play, options.Solo);
                    continue;

                default:
                    Console.WriteLine($"Unknown command '{command}'. Type /help for available commands.");
                    continue;
            }
        }

        lastSamples = synthesizer.Synthesize(line, currentProfile);
        PrintAnalysis(line, lastSamples, synthesizer.SampleRate);

        if (options.Play)
        {
            Play(lastSamples, synthesizer.SampleRate);
        }
    }

    void HandleVoice(string argument)
    {
        if (string.IsNullOrWhiteSpace(argument))
        {
            Console.WriteLine(
                $"Current voice: {voiceLabel} " +
                $"({currentProfile.MinFreq.ToString("0", CultureInfo.InvariantCulture)}-" +
                $"{currentProfile.MaxFreq.ToString("0", CultureInfo.InvariantCulture)} Hz, " +
                $"x{currentProfile.SpeedMultiplier.ToString("0.00", CultureInfo.InvariantCulture)}).");
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

                File.WriteAllText(path, currentProfile.ToJson());
                Console.WriteLine($"Voice profile saved: {path}");
                return;
            }

            try
            {
                currentProfile = DroidVoiceProfile.FromJson(File.ReadAllText(path)).Normalized();
                voiceLabel = Path.GetFileNameWithoutExtension(path);
                Console.WriteLine($"Voice loaded from {path} as '{voiceLabel}'.");
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

        currentProfile = preset;
        voiceLabel = name;
        Console.WriteLine(
            $"Voice set to {name} " +
            $"({preset.MinFreq.ToString("0", CultureInfo.InvariantCulture)}-" +
            $"{preset.MaxFreq.ToString("0", CultureInfo.InvariantCulture)} Hz, " +
            $"x{preset.SpeedMultiplier.ToString("0.00", CultureInfo.InvariantCulture)}).");
    }
}

static float[]? RunDemo(
    DroidSynthesizer synthesizer,
    DroidVoiceProfile profile,
    bool play,
    string voiceLabel)
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
        Console.WriteLine($"[{label}] {text} ({voiceLabel})");
        lastSamples = synthesizer.Synthesize(text, profile);
        PrintAnalysis(text, lastSamples, synthesizer.SampleRate);

        if (play)
        {
            Play(lastSamples, synthesizer.SampleRate);
        }

        Console.WriteLine();
    }

    return lastSamples;
}

static float[]? RunTransmit(
    DroidSynthesizer synthesizer,
    DroidVoiceProfile profile,
    string text,
    bool play)
{
    if (string.IsNullOrWhiteSpace(text))
    {
        Console.WriteLine("Usage: /transmit <text>");
        return null;
    }

    var speech = synthesizer.Synthesize(text, profile);
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

static void SaveLast(DroidSynthesizer synthesizer, float[]? samples, string argument)
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

    WavWriter.Write(fullPath, samples, synthesizer.SampleRate);
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
    bool solo = false)
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
    var samples = musicSynth.Render(score, soloMelodyOnly: solo);

    var title = string.IsNullOrWhiteSpace(score.Title) ? "Untitled" : score.Title;
    var modeLabel = solo ? ", solo" : string.Empty;
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
    Console.WriteLine("  dotnet run --project DroidSpeak.Demo -- \"<text>\" [-v voice] [-o file.wav] [--no-play]");
    Console.WriteLine("  dotnet run --project DroidSpeak.Demo -- -m <file.mid|rtttl> [-v voice] [--solo] [-o file.wav]");
    Console.WriteLine("  dotnet run --project DroidSpeak.Demo");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  -m, --music <arg>    Play or export a melody (.mid file or RTTTL string).");
    Console.WriteLine("  -v, --voice <voice>  Voice preset (r2, grumpy, bass, tiny) or path to a JSON profile.");
    Console.WriteLine("  -o, --output <file>  Save the synthesized audio as a 16-bit PCM WAV.");
    Console.WriteLine("  --solo               Play only the upper lead melody when rendering music.");
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
    Console.WriteLine("  /music [--solo] <file.mid|rtttl>     Play a melody using the current droid voice.");
    Console.WriteLine("  /transmit <text>                     Hide text in speech, save transmit.wav and decode it.");
    Console.WriteLine("  /demo                                Play four test phrases.");
    Console.WriteLine("  /save <file.wav>                     Save the last utterance or melody.");
    Console.WriteLine("  /help                                Show REPL commands.");
    Console.WriteLine("  /exit                                Quit (empty input also quits).");
}


internal sealed record CliOptions(
    string? Text,
    string? Music,
    string? Voice,
    string? OutputPath,
    bool Play,
    bool Solo,
    bool ShowHelp,
    string? Error);
