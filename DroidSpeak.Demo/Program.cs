using System.Globalization;
using System.Text;
using DroidSpeak;
using DroidSpeak.Demo;
using DroidSpeak.DSP;
using DroidSpeak.Modem;
using DroidSpeak.NLP;

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

var synthesizer = new DroidSynthesizer();

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
    string? error = null;
    var play = true;
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
                {
                    error = $"missing filename after '{argument}'";
                }
                else
                {
                    outputPath = arguments[++i];
                }

                break;

            case "--play":
                play = true;
                break;

            case "--no-play":
                play = false;
                break;

            default:
                if (argument.StartsWith('-') && argument.Length > 1)
                {
                    error = $"unknown option '{argument}'";
                }
                else
                {
                    textParts.Add(argument);
                }

                break;
        }
    }

    var text = textParts.Count == 0 ? null : string.Join(' ', textParts);
    return new CliOptions(text, outputPath, play, showHelp, error);
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
        ("greeting", "Привет, R2! Как дела?"),
        ("alert", "ТРЕВОГА! Перегрев реактора!"),
        ("sad", "Ну вот... опять все сломалось..."),
        ("english", "Beep boop! System is ready."),
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
        Console.WriteLine("[Распознано из звука]: <не удалось декодировать>");
    }
    else
    {
        var suffix = decoded == text ? string.Empty : "  (несовпадение!)";
        Console.WriteLine($"[Распознано из звука]: {decoded}{suffix}");
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

static void PrintUsage()
{
    Console.WriteLine("DroidSpeak — procedural R2-D2-style text-to-beep demo.");
    Console.WriteLine();
    Console.WriteLine("Usage:");
    Console.WriteLine("  dotnet run --project DroidSpeak.Demo -- \"<text>\" [-o file.wav] [--no-play]");
    Console.WriteLine("  dotnet run --project DroidSpeak.Demo");
    Console.WriteLine();
    Console.WriteLine("Options:");
    Console.WriteLine("  -o, --output <file>  Save the synthesized audio as a 16-bit PCM WAV.");
    Console.WriteLine("  --play               Play audio (default).");
    Console.WriteLine("  --no-play            Do not play audio.");
    Console.WriteLine("  -h, --help           Show this help.");
    Console.WriteLine();
    Console.WriteLine("Without text, an interactive REPL starts:");
    Console.WriteLine("  /voice [r2|grumpy|bass|tiny]  Show or change the droid voice.");
    Console.WriteLine("  /voice json <file>   Load a voice profile from JSON.");
    Console.WriteLine("  /voice save <file>   Save the current voice profile as JSON.");
    Console.WriteLine("  /transmit <text>     Hide text in droid speech, save transmit.wav and decode it.");
    Console.WriteLine("  /demo                Play four test phrases.");
    Console.WriteLine("  /save <file.wav>     Save the last utterance.");
    Console.WriteLine("  /help                Show REPL commands.");
    Console.WriteLine("  /exit                Quit (empty input also quits).");
}

static void PrintReplHelp()
{
    Console.WriteLine("Commands:");
    Console.WriteLine("  /voice [r2|grumpy|bass|tiny]  Show or change the droid voice.");
    Console.WriteLine("  /voice json <file>   Load a voice profile from JSON.");
    Console.WriteLine("  /voice save <file>   Save the current voice profile as JSON.");
    Console.WriteLine("  /transmit <text>     Hide text in droid speech, save transmit.wav and decode it.");
    Console.WriteLine("  /demo                Play four test phrases.");
    Console.WriteLine("  /save <file.wav>     Save the last utterance.");
    Console.WriteLine("  /help                Show this help.");
    Console.WriteLine("  /exit                Quit (empty input also quits).");
}

internal sealed record CliOptions(
    string? Text,
    string? OutputPath,
    bool Play,
    bool ShowHelp,
    string? Error);
