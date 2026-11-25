using DroidSpeak.Neural;

namespace DroidSpeak.Demo;

/// <summary>
/// The immutable result of parsing the demo command line.
/// <para>
/// Unlike the old <c>CliOptions</c> a parse never exits the process: a problem is reported through
/// <see cref="Error"/> so the parsing rules stay testable in isolation. <see cref="Brain"/> is
/// appended last on purpose, so every position that existed before the neural flag was added keeps
/// its meaning.
/// </para>
/// </summary>
/// <param name="Text">Text to speak, or <c>null</c> for the interactive REPL.</param>
/// <param name="Music">Melody preset, MIDI file or RTTTL string, or <c>null</c>.</param>
/// <param name="Voice">Voice preset name or JSON profile path, or <c>null</c> for the default.</param>
/// <param name="OutputPath">Where to write the rendered WAV, or <c>null</c>.</param>
/// <param name="Play">Whether to play audio.</param>
/// <param name="Solo">Whether to render only the music lead line.</param>
/// <param name="Stream">Whether to synthesize through the streaming path.</param>
/// <param name="ShowHelp">Whether <c>--help</c> was requested.</param>
/// <param name="Error">A concise problem description, or <c>null</c> when the line is valid.</param>
/// <param name="Brain">Path to a <c>.dsbrain</c> model, or <c>null</c> for rule-based prosody.</param>
internal sealed record DemoOptions(
    string? Text,
    string? Music,
    string? Voice,
    string? OutputPath,
    bool Play,
    bool Solo,
    bool Stream,
    bool ShowHelp,
    string? Error,
    string? Brain);

/// <summary>
/// Pure command-line helpers shared by <c>Program.cs</c> and the test suite. Nothing here writes to
/// the console or calls <see cref="Environment.Exit(int)"/>: failures come back as strings.
/// </summary>
internal static class DemoCommandLine
{
    /// <summary>
    /// Parses the demo arguments the same way the top-level program always did, plus the new
    /// <c>--brain</c> flag.
    /// </summary>
    /// <param name="arguments">Arguments without the executable name.</param>
    /// <returns>
    /// A <see cref="DemoOptions"/> whose <see cref="DemoOptions.Error"/> is <c>null</c> when the
    /// arguments are valid. The last problem found wins, matching the previous behaviour.
    /// </returns>
    internal static DemoOptions ParseArguments(string[] arguments)
    {
        ArgumentNullException.ThrowIfNull(arguments);

        string? outputPath = null;
        string? music = null;
        string? voice = null;
        string? brain = null;
        string? error = null;
        var play = true;
        var solo = false;
        var stream = false;
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

                case "--brain":
                    if (i + 1 >= arguments.Length)
                        error = $"missing .dsbrain model path after '{argument}'";
                    else
                        brain = arguments[++i];
                    break;

                case "--play":
                    play = true;
                    break;

                case "--solo":
                    solo = true;
                    break;

                case "--stream":
                    stream = true;
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
        return new DemoOptions(text, music, voice, outputPath, play, solo, stream, showHelp, error, brain);
    }

    /// <summary>
    /// Loads a <c>.dsbrain</c> model and wraps it in a <see cref="NeuralProsodyPredictor"/>.
    /// </summary>
    /// <param name="path">Path to the model file.</param>
    /// <returns>
    /// The predictor on success; otherwise <c>null</c> plus a one-line diagnostic suitable for
    /// printing to stderr. Expected failures (missing file, truncated/corrupt container, a model
    /// whose shape does not match the syllable window) are converted into that string, so no
    /// exception ever escapes and no stack trace reaches the user.
    /// </returns>
    internal static (NeuralProsodyPredictor? Predictor, string? Error) TryLoadBrain(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
        {
            return (null, "no .dsbrain model path was supplied");
        }

        try
        {
            var fullPath = Path.GetFullPath(path);
            if (!File.Exists(fullPath))
            {
                return (null, $"model file not found: {fullPath}");
            }

            using var stream = File.OpenRead(fullPath);
            var model = MlpModel.Load(stream);

            // The constructor is the shape check: it only accepts 60 inputs and 17 outputs.
            var predictor = new NeuralProsodyPredictor(model);
            return (predictor, null);
        }
        catch (Exception exception) when (exception is InvalidDataException
            or IOException
            or UnauthorizedAccessException
            or ArgumentException
            or NotSupportedException)
        {
            return (null, $"cannot load '{path}': {exception.Message}");
        }
    }
}