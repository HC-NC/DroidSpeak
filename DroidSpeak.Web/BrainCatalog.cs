using System.Net;
using System.Text.Json;
using DroidSpeak.Neural;

namespace DroidSpeak.Web;

/// <summary>
/// Fetches the hosted <c>.dsbrain</c> models that live in <c>wwwroot/brains/</c> over plain HTTP, so
/// dropping a new model plus a manifest entry onto the static host is enough to make it appear in
/// the playground — no C# code changes and no recompile.
/// <para>
/// Every failure mode degrades to the rule-based voice instead of breaking the page: a missing or
/// malformed manifest becomes an empty list with an optional warning, and a failed model download
/// surfaces as an exception the caller is expected to catch before falling back to
/// <c>RuleBasedProsodyPredictor.Shared</c>. Requests use app-relative URLs, which keeps them correct
/// under both <c>/</c> and the <c>/DroidSpeak/</c> GitHub Pages base path.
/// </para>
/// </summary>
public sealed class BrainCatalog
{
    /// <summary>
    /// Largest brain payload accepted, in bytes. The <c>.dsbrain</c> format itself caps out near
    /// 8.4 MB (<c>32</c> layers of <c>256×256</c> floats), so 16 MB is generous headroom that still
    /// refuses a malicious or corrupted download early.
    /// </summary>
    public const long MaxBrainBytes = 16L * 1024 * 1024;

    private static readonly IReadOnlyList<string> NoFiles = [];

    private readonly HttpClient _httpClient;

    /// <summary>Creates a catalog reading through <paramref name="httpClient"/>.</summary>
    /// <param name="httpClient">
    /// Client whose <see cref="HttpClient.BaseAddress"/> is the app root; the playground injects the
    /// one registered from <c>Program.cs</c>, tests supply a stub handler.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="httpClient"/> is null.</exception>
    public BrainCatalog(HttpClient httpClient) =>
        _httpClient = httpClient ?? throw new ArgumentNullException(nameof(httpClient));

    /// <summary>
    /// Downloads and parses <c>brains/manifest.json</c>. A 404 is the normal "no hosted brains" state
    /// and yields an empty list without a warning; network, timeout and JSON problems also yield an
    /// empty list, but carry a warning the UI can show while the classic voice stays selected.
    /// </summary>
    /// <param name="cancellationToken">Token cancelling an in-flight fetch.</param>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was raised.</exception>
    public async Task<BrainManifestResult> ListBrainsAsync(CancellationToken cancellationToken = default)
    {
        HttpResponseMessage? response = null;
        try
        {
            response = await _httpClient.GetAsync(
                BrainManifest.ManifestUrl,
                HttpCompletionOption.ResponseContentRead,
                cancellationToken);

            if (response.StatusCode == HttpStatusCode.NotFound)
            {
                return new BrainManifestResult(NoFiles, null);
            }

            response.EnsureSuccessStatusCode();
            await using var content = await response.Content.ReadAsStreamAsync(cancellationToken);
            return BrainManifest.Parse(content);
        }
        catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
        {
            return new BrainManifestResult(NoFiles, "The brain manifest request timed out.");
        }
        catch (JsonException)
        {
            return new BrainManifestResult(NoFiles, "The brain manifest is not valid JSON.");
        }
        catch (HttpRequestException ex)
        {
            return new BrainManifestResult(NoFiles, $"Could not fetch the brain manifest: {ex.Message}");
        }
        finally
        {
            response?.Dispose();
        }
    }

    /// <summary>
    /// Downloads one hosted model from <c>brains/&lt;fileName&gt;</c> and loads it through
    /// <see cref="MlpModel.Load(ReadOnlySpan{byte})"/>, which fully validates the container.
    /// The transfer is size-capped by <see cref="MaxBrainBytes"/> before the model is parsed.
    /// </summary>
    /// <param name="fileName">A bare <c>.dsbrain</c> file name, normally from the manifest.</param>
    /// <param name="cancellationToken">Token cancelling an in-flight download.</param>
    /// <exception cref="ArgumentException"><paramref name="fileName"/> is not a safe bare file name.</exception>
    /// <exception cref="HttpRequestException">The file is missing or the host refused it.</exception>
    /// <exception cref="InvalidOperationException">The payload exceeds <see cref="MaxBrainBytes"/>.</exception>
    /// <exception cref="InvalidDataException">The bytes are not a valid <c>.dsbrain</c> container.</exception>
    /// <exception cref="OperationCanceledException"><paramref name="cancellationToken"/> was raised.</exception>
    public async Task<MlpModel> LoadBrainAsync(string fileName, CancellationToken cancellationToken = default)
    {
        var safeName = BrainManifest.NormalizeFileName(fileName)
            ?? throw new ArgumentException(
                $"'{fileName}' is not a bare {BrainManifest.BrainExtension} file name.",
                nameof(fileName));

        using var response = await _httpClient.GetAsync(
            "brains/" + Uri.EscapeDataString(safeName),
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);

        response.EnsureSuccessStatusCode();

        if (response.Content.Headers.ContentLength > MaxBrainBytes)
        {
            throw TooLarge(safeName);
        }

        await using var source = await response.Content.ReadAsStreamAsync(cancellationToken);
        using var buffer = new MemoryStream();
        var chunk = new byte[80 * 1024];
        int read;
        while ((read = await source.ReadAsync(chunk, cancellationToken)) > 0)
        {
            buffer.Write(chunk, 0, read);
            if (buffer.Length > MaxBrainBytes)
            {
                throw TooLarge(safeName);
            }
        }

        return MlpModel.Load(buffer.GetBuffer().AsSpan(0, (int)buffer.Length));
    }

    /// <summary>
    /// Loads an in-memory <c>.dsbrain</c> container — the path a browser-uploaded file takes after
    /// its bytes were read client-side. It enforces the same size cap as a hosted download.
    /// </summary>
    /// <param name="data">The complete file contents picked by the user.</param>
    /// <exception cref="InvalidOperationException">The payload exceeds <see cref="MaxBrainBytes"/>.</exception>
    /// <exception cref="InvalidDataException">The bytes are not a valid <c>.dsbrain</c> container.</exception>
    public static MlpModel LoadBrainBytes(ReadOnlySpan<byte> data)
    {
        if (data.Length > MaxBrainBytes)
        {
            throw new InvalidOperationException(
                $"The uploaded brain is {data.Length} bytes, above the {MaxBrainBytes} byte limit.");
        }

        return MlpModel.Load(data);
    }

    private static InvalidOperationException TooLarge(string fileName) => new(
        $"The brain '{fileName}' is larger than the {MaxBrainBytes / (1024 * 1024)} MB limit.");
}
