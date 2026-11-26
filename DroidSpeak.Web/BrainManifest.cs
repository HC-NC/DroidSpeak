using System.Text.Json;

namespace DroidSpeak.Web;

/// <summary>
/// Parsing and sanitization of the optional <c>wwwroot/brains/manifest.json</c> list that tells the
/// playground which <c>.dsbrain</c> files the host is serving.
/// <para>
/// The manifest is a flat JSON array of plain file names, for example <c>["r2-custom.dsbrain"]</c>.
/// Entries are never treated as URLs or paths: anything that is not a bare <c>*.dsbrain</c> file name
/// is skipped, so a malformed manifest can never steer the browser to fetch an arbitrary location.
/// The file itself is tracked in git (it ships as <c>[]</c>); the <c>.dsbrain</c> blobs it names are
/// ignored, which is what keeps drop-in model support free of repository noise.
/// </para>
/// </summary>
public static class BrainManifest
{
    /// <summary>
    /// Location of the manifest, relative to the app base address. It must stay relative so the same
    /// build works when served from <c>/</c> and from the <c>/DroidSpeak/</c> GitHub Pages base path.
    /// </summary>
    public const string ManifestUrl = "brains/manifest.json";

    /// <summary>Required extension of every listed file, matched case-insensitively.</summary>
    public const string BrainExtension = ".dsbrain";

    /// <summary>Longest file name the manifest accepts (a guard against absurd entries).</summary>
    public const int MaxFileNameLength = 200;

    private static readonly string[] NoFiles = [];

    /// <summary>
    /// Parses a manifest document and returns the sanitized file names, plus a non-fatal warning when
    /// the document is unusable or holds entries that had to be skipped.
    /// </summary>
    /// <param name="json">Raw JSON text of the manifest.</param>
    /// <exception cref="JsonException">The text is not valid JSON.</exception>
    public static BrainManifestResult Parse(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        using var document = JsonDocument.Parse(json);
        return ParseRoot(document.RootElement);
    }

    /// <summary>
    /// Streamed form of <see cref="Parse(string)"/>, used for the manifest as it arrives over HTTP.
    /// </summary>
    /// <param name="stream">Readable manifest content.</param>
    /// <exception cref="ArgumentNullException"><paramref name="stream"/> is null.</exception>
    /// <exception cref="JsonException">The content is not valid JSON.</exception>
    public static BrainManifestResult Parse(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var document = JsonDocument.Parse(stream);
        return ParseRoot(document.RootElement);
    }

    /// <summary>
    /// Normalizes one manifest entry to a bare <c>.dsbrain</c> file name.
    /// </summary>
    /// <param name="entry">Candidate entry; may be null or whitespace.</param>
    /// <returns>The entry when it is a safe file name, otherwise <c>null</c> to signal "ignore".</returns>
    public static string? NormalizeFileName(string? entry)
    {
        var name = entry?.Trim();
        if (string.IsNullOrEmpty(name) || name.Length > MaxFileNameLength)
        {
            return null;
        }

        // Reject path separators, drive colons and URL fragments before anything else: a manifest
        // entry must be exactly one file name below brains/, never a traversal or an absolute URL.
        if (name.IndexOfAny(['/', '\\', ':', '?', '#', '%']) >= 0)
        {
            return null;
        }

        if (name.Contains("..", StringComparison.Ordinal) || name[0] == '.')
        {
            return null;
        }

        return name.EndsWith(BrainExtension, StringComparison.OrdinalIgnoreCase) ? name : null;
    }

    private static BrainManifestResult ParseRoot(JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Array)
        {
            return new BrainManifestResult(
                NoFiles,
                "The brain manifest is not a JSON array of .dsbrain file names.");
        }

        var files = new List<string>();
        var skipped = 0;

        foreach (var element in root.EnumerateArray())
        {
            var name = NormalizeFileName(
                element.ValueKind == JsonValueKind.String ? element.GetString() : null);

            if (name is null)
            {
                skipped++;
            }
            else if (!files.Contains(name, StringComparer.OrdinalIgnoreCase))
            {
                files.Add(name);
            }
        }

        var warning = skipped > 0
            ? $"Ignored {skipped} invalid manifest entr{(skipped == 1 ? "y" : "ies")}; expected bare .dsbrain file names."
            : null;

        return new BrainManifestResult(files, warning);
    }
}

/// <summary>
/// Outcome of reading the brain manifest: the sanitized list of hosted <c>.dsbrain</c> file names
/// (possibly empty) and an optional non-fatal warning the UI can surface while the classic voice
/// keeps working.
/// <param name="Files">Hosted brain file names, safe to combine with <c>brains/</c>.</param>
/// <param name="Warning">Human-readable note about what went wrong, or <c>null</c> on a clean read.</param>
/// </summary>
public sealed record BrainManifestResult(IReadOnlyList<string> Files, string? Warning);
