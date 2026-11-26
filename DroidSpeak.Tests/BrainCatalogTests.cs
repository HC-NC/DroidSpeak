using System.Net;
using System.Text;
using DroidSpeak.Neural;
using DroidSpeak.Web;

namespace DroidSpeak.Tests;

// Exercises BrainCatalog against a stubbed transport. The base address mirrors GitHub Pages
// (/DroidSpeak/) so the tests prove the manifest and model URLs stay app-relative rather than
// root-absolute, which is the difference between working under both deployments and only one.
public class BrainCatalogTests
{
    private const string BaseAddress = "https://example.test/DroidSpeak/";

    private static MlpModel CreateValidBrain() => MlpModel.CreateInitialized(
    [
        new LayerSpec(SyllableFeatureEncoder.InputSize, 8, ActivationType.Tanh),
        new LayerSpec(8, NeuralProsodyPredictor.OutputSize, ActivationType.Sigmoid),
    ]);

    private static byte[] Serialize(MlpModel model)
    {
        using var stream = new MemoryStream();
        model.Save(stream);
        return stream.ToArray();
    }

    private static (BrainCatalog Catalog, StubHandler Handler) Create(
        Func<HttpRequestMessage, HttpResponseMessage> responder)
    {
        var handler = new StubHandler(responder);
        var client = new HttpClient(handler) { BaseAddress = new Uri(BaseAddress) };
        return (new BrainCatalog(client), handler);
    }

    private static HttpResponseMessage Json(HttpStatusCode status, string json) => new(status)
    {
        Content = new StringContent(json, Encoding.UTF8, "application/json"),
    };

    [Fact]
    public async Task ListBrains_EmptyManifest_ReturnsEmptyListWithoutWarning()
    {
        var (catalog, _) = Create(_ => Json(HttpStatusCode.OK, "[]"));

        var result = await catalog.ListBrainsAsync();

        Assert.Empty(result.Files);
        Assert.Null(result.Warning);
    }

    [Fact]
    public async Task ListBrains_MissingManifest_IsTheNormalEmptyState()
    {
        var (catalog, _) = Create(_ => Json(HttpStatusCode.NotFound, "{}"));

        var result = await catalog.ListBrainsAsync();

        Assert.Empty(result.Files);
        Assert.Null(result.Warning);
    }

    [Fact]
    public async Task ListBrains_ValidManifest_ReturnsSanitizedNames()
    {
        var (catalog, _) = Create(_ => Json(HttpStatusCode.OK, """["r2.dsbrain", "../evil.dsbrain"]"""));

        var result = await catalog.ListBrainsAsync();

        Assert.Equal(["r2.dsbrain"], result.Files);
        Assert.NotNull(result.Warning);
    }

    [Fact]
    public async Task ListBrains_MalformedJson_FallsBackWithWarning()
    {
        var (catalog, _) = Create(_ => Json(HttpStatusCode.OK, "not json"));

        var result = await catalog.ListBrainsAsync();

        Assert.Empty(result.Files);
        Assert.NotNull(result.Warning);
        Assert.Contains("JSON", result.Warning, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task ListBrains_NetworkFailure_FallsBackWithWarning()
    {
        var (catalog, _) = Create(_ => throw new HttpRequestException("offline"));

        var result = await catalog.ListBrainsAsync();

        Assert.Empty(result.Files);
        Assert.NotNull(result.Warning);
    }

    [Fact]
    public async Task ListBrains_UsesAppRelativeUrl()
    {
        var (catalog, handler) = Create(_ => Json(HttpStatusCode.OK, "[]"));

        await catalog.ListBrainsAsync();

        Assert.Single(handler.Requests);
        Assert.Equal(
            new Uri(BaseAddress + "brains/manifest.json"),
            handler.Requests[0]);
    }

    [Fact]
    public async Task LoadBrain_DownloadsAndValidatesTheModel()
    {
        var model = CreateValidBrain();
        var bytes = Serialize(model);
        var (catalog, handler) = Create(
            _ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent(bytes) });

        var loaded = await catalog.LoadBrainAsync("r2.dsbrain");

        Assert.Equal(model.InputSize, loaded.InputSize);
        Assert.Equal(model.OutputSize, loaded.OutputSize);
        Assert.Equal(new Uri(BaseAddress + "brains/r2.dsbrain"), handler.Requests[0]);
    }

    [Fact]
    public async Task LoadBrain_MissingFile_Throws()
    {
        var (catalog, _) = Create(_ => new HttpResponseMessage(HttpStatusCode.NotFound));

        await Assert.ThrowsAsync<HttpRequestException>(() => catalog.LoadBrainAsync("r2.dsbrain"));
    }

    [Fact]
    public async Task LoadBrain_InvalidContainer_ThrowsInvalidData()
    {
        var (catalog, _) = Create(
            _ => new HttpResponseMessage(HttpStatusCode.OK) { Content = new ByteArrayContent([1, 2, 3]) });

        await Assert.ThrowsAsync<InvalidDataException>(() => catalog.LoadBrainAsync("r2.dsbrain"));
    }

    [Theory]
    [InlineData("../secret.dsbrain")]
    [InlineData("brains/r2.dsbrain")]
    [InlineData("http://evil.test/r2.dsbrain")]
    [InlineData("r2.txt")]
    public async Task LoadBrain_RejectsUnsafeName_WithoutIssuingARequest(string fileName)
    {
        var (catalog, handler) = Create(_ => new HttpResponseMessage(HttpStatusCode.OK));

        await Assert.ThrowsAsync<ArgumentException>(() => catalog.LoadBrainAsync(fileName));
        Assert.Empty(handler.Requests);
    }

    [Fact]
    public async Task LoadBrain_DeclaredOversize_Throws()
    {
        var (catalog, _) = Create(_ =>
        {
            var content = new ByteArrayContent(new byte[16]);
            content.Headers.ContentLength = BrainCatalog.MaxBrainBytes + 1;
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
        });

        await Assert.ThrowsAsync<InvalidOperationException>(() => catalog.LoadBrainAsync("big.dsbrain"));
    }

    [Fact]
    public void LoadBrainBytes_ValidContainer_LoadsTheModel()
    {
        var model = CreateValidBrain();

        var loaded = BrainCatalog.LoadBrainBytes(Serialize(model));

        Assert.Equal(model.InputSize, loaded.InputSize);
        Assert.Equal(model.OutputSize, loaded.OutputSize);
    }

    [Fact]
    public void LoadBrainBytes_InvalidContainer_Throws()
    {
        Assert.Throws<InvalidDataException>(() => BrainCatalog.LoadBrainBytes([1, 2, 3]));
    }

    [Fact]
    public void LoadBrainBytes_OversizedBuffer_Throws()
    {
        var oversized = new byte[BrainCatalog.MaxBrainBytes + 1];

        Assert.Throws<InvalidOperationException>(() => BrainCatalog.LoadBrainBytes(oversized));
    }

    private sealed class StubHandler(Func<HttpRequestMessage, HttpResponseMessage> responder)
        : HttpMessageHandler
    {
        public List<Uri> Requests { get; } = [];

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request,
            CancellationToken cancellationToken)
        {
            Requests.Add(request.RequestUri!);
            return Task.FromResult(responder(request));
        }
    }
}