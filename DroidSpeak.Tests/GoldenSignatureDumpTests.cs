using System.Text;
using DroidSpeak;
using DroidSpeak.DSP;

namespace DroidSpeak.Tests;

/// <summary>
/// Tooling for an intentional DSP change: regenerates the acoustic signatures pinned in
/// <see cref="PcmSnapshotTests"/> and <see cref="DroidVoiceProfileTests"/>. It is skipped by
/// default so nobody can silently re-sign the goldens from a single platform; clear the Skip
/// reason, run <c>dotnet test --filter "FullyQualifiedName~GoldenSignatureDumpTests"</c>, and paste
/// the printed blocks over the existing dictionary entries.
/// </summary>
public class GoldenSignatureDumpTests
{
    [Fact(Skip = "Only clear this Skip when the DSP changes intentionally; paste the output into PcmSnapshotTests.")]
    public void DumpGoldenSignatures()
    {
        var builder = new StringBuilder();

        foreach (var (textId, text) in PcmSnapshotTests.Texts)
        {
            foreach (var profileName in new[] { "R2", "Grumpy", "Bass", "Tiny" })
            {
                foreach (var seed in new[] { 0, 12345 })
                {
                    var synthesizer = new DroidSynthesizer(
                        seed: seed,
                        profile: PcmSnapshotTests.Profile(profileName));
                    var samples = synthesizer.Synthesize(text);
                    var key = $"{textId}|{profileName}|{seed}";
                    builder.AppendLine(
                        $$"""        ["{{key}}"] = "{{PcmFingerprint.Format(samples, synthesizer.SampleRate)}}", """);
                }
            }
        }

        var reduced = new DroidSynthesizer(sampleRate: 22050, seed: 0, profile: DroidVoiceProfile.R2);
        builder.AppendLine(
            $$"""        ["hello|R2|0|22050"] = "{{PcmFingerprint.Format(reduced.Synthesize("Hello, world!"), reduced.SampleRate)}}", """);

        // PcmSnapshotTests.V04Legacy is deliberately NOT regenerated here: those signatures must
        // keep coming from the v0.4.0 tag sources, otherwise a broken v0.5 neutrality gate would
        // simply re-sign its own regression.

        // Surface the signatures through the test failure message: easy to copy, and the Skip above
        // means this only ever runs on purpose.
        Assert.Fail(builder.ToString());
    }
}
