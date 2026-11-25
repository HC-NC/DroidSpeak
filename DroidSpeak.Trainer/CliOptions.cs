using System.Globalization;

namespace DroidSpeak.Trainer;

/// <summary>
/// The tiny option grammar the trainer's CLI speaks: <c>--name value</c> pairs and bare
/// <c>--flag</c> switches, after the subcommand.
/// </summary>
/// <remarks>
/// Hand-rolled instead of a command-line library because the trainer ships with zero package
/// references. The grammar is deliberately strict: an unknown option is an error, not something to
/// ignore, because a typo'd <c>--epochs</c> silently training with the default is the kind of
/// mistake that wastes an afternoon.
/// </remarks>
internal sealed class CliOptions
{
    private readonly Dictionary<string, string> _values = new(StringComparer.Ordinal);
    private readonly List<string> _flags = [];

    /// <summary>Parses the arguments that follow the subcommand.</summary>
    /// <param name="arguments">Options only; the subcommand has already been consumed.</param>
    /// <param name="known">
    /// Every option the command understands. Leaving them unchecked would let a typo through.
    /// </param>
    /// <exception cref="ArgumentException">An option is unknown or missing its value.</exception>
    public static CliOptions Parse(string[] arguments, params string[] known)
    {
        var options = new CliOptions();
        var allowed = new HashSet<string>(known, StringComparer.Ordinal);

        for (var i = 0; i < arguments.Length; i++)
        {
            var argument = arguments[i];

            if (!argument.StartsWith("--", StringComparison.Ordinal))
            {
                throw new ArgumentException($"Unexpected argument '{argument}'.");
            }

            var name = argument[2..];
            var separator = name.IndexOf('=');
            string key;
            string? value = null;

            if (separator >= 0)
            {
                key = name[..separator];
                value = name[(separator + 1)..];
            }
            else
            {
                key = name;

                if (i + 1 < arguments.Length && !arguments[i + 1].StartsWith("--", StringComparison.Ordinal))
                {
                    value = arguments[++i];
                }
            }

            if (!allowed.Contains(key))
            {
                throw new ArgumentException($"Unknown option '--{key}'.");
            }

            if (value is null)
            {
                options._flags.Add(key);
            }
            else
            {
                options._values[key] = value;
            }
        }

        return options;
    }

    /// <summary>Value of <paramref name="name"/>, or <c>null</c> when it was not supplied.</summary>
    public string? Value(string name) => _values.GetValueOrDefault(name);

    /// <summary>Whether the bare flag <paramref name="name"/> was supplied.</summary>
    public bool Has(string name) => _flags.Contains(name);

    /// <summary>Integer option, parsed invariantly.</summary>
    /// <exception cref="ArgumentException">The value is not an integer.</exception>
    public int Int(string name, int fallback)
    {
        var value = Value(name);
        if (value is null)
        {
            return fallback;
        }

        return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : throw new ArgumentException($"--{name} must be an integer, but was '{value}'.");
    }

    /// <summary>Single-precision option, parsed invariantly.</summary>
    /// <exception cref="ArgumentException">The value is not a number.</exception>
    public float Single(string name, float fallback)
    {
        var value = Value(name);
        if (value is null)
        {
            return fallback;
        }

        return float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : throw new ArgumentException($"--{name} must be a number, but was '{value}'.");
    }

    /// <summary>Unsigned integer option, parsed invariantly.</summary>
    /// <exception cref="ArgumentException">The value is not an unsigned integer.</exception>
    public uint UInt(string name, uint fallback)
    {
        var value = Value(name);
        if (value is null)
        {
            return fallback;
        }

        return uint.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : throw new ArgumentException(
                $"--{name} must be an unsigned integer, but was '{value}'.");
    }

    /// <summary>Comma-separated integer list, e.g. <c>--arch 64,48</c>.</summary>
    /// <exception cref="ArgumentException">A segment is not an integer.</exception>
    public int[] IntList(string name, int[] fallback)
    {
        var value = Value(name);
        if (string.IsNullOrWhiteSpace(value))
        {
            return fallback;
        }

        var segments = value.Split(',', StringSplitOptions.RemoveEmptyEntries |
            StringSplitOptions.TrimEntries);
        var result = new int[segments.Length];

        for (var i = 0; i < segments.Length; i++)
        {
            if (!int.TryParse(
                segments[i],
                NumberStyles.Integer,
                CultureInfo.InvariantCulture,
                out result[i]))
            {
                throw new ArgumentException(
                    $"--{name} expects comma-separated integers, but '{segments[i]}' is not one.");
            }
        }

        return result;
    }

    /// <summary>
    /// Resolves a voice profile name: a built-in preset name or a path to a JSON profile the library
    /// can already read with <see cref="DroidVoiceProfile.FromJson"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The name is neither a preset nor an existing file.</exception>
    public DroidVoiceProfile Profile(string name)
    {
        switch (name.ToLowerInvariant())
        {
            case "r2":
                return DroidVoiceProfile.R2;
            case "grumpy":
                return DroidVoiceProfile.Grumpy;
            case "bass":
                return DroidVoiceProfile.Bass;
            case "tiny":
                return DroidVoiceProfile.Tiny;
            default:
                if (File.Exists(name))
                {
                    return DroidVoiceProfile.FromJson(File.ReadAllText(name));
                }

                throw new ArgumentException(
                    $"'{name}' is neither a preset (r2, grumpy, bass, tiny) nor a profile file.");
        }
    }
}
