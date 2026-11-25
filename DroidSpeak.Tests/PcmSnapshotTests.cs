using System.Runtime.InteropServices;
using System.Security.Cryptography;
using DroidSpeak.DSP;

namespace DroidSpeak.Tests;

/// <summary>
/// Cross-version PCM snapshots captured from v0.3.0 before the v0.4.0 prosody/DSP split.
/// Every refactoring commit must keep these hashes byte-identical.
/// </summary>
public class PcmSnapshotTests
{
    private static readonly Dictionary<string, string> Golden = new(StringComparer.Ordinal)
    {
        ["hello|R2|0"] = "7085652B91F8112380611C539E83DF100FAB073608F4A43A70FB70F5F29EB53F",
        ["hello|R2|12345"] = "748BA0BE72322B46763217065A891CBF5805352401EB1AE386EFF305800419C4",
        ["hello|Grumpy|0"] = "105806C479E57D7BD90BD83474FF20434994473840C2E4846E8D112EFDBD91EE",
        ["hello|Grumpy|12345"] = "CE7D46837901E372F7C18E777460CEE45A420CF7A0161AC03EA1817077B45E54",
        ["hello|Bass|0"] = "6FA5D1140B8E486A163559BF8A044A6DD218A83240018378616D4C28C436CF2F",
        ["hello|Bass|12345"] = "2FC3BB9C1CAD75DFBB90798081B27599387249E446D8C47ED5879D1051BFFEE8",
        ["hello|Tiny|0"] = "54FB096C9817B404C4E377D6CC758B72919F9E248B083985FBC08974809F28FC",
        ["hello|Tiny|12345"] = "B27A4983FB5D9FDD5FF876547B123F9007401214C231DC96409942D7490EA55F",
        ["hyphen|R2|0"] = "6D660F7E82945153BDCCF1B309BD31B1656D9D0E3EC5F0B639D024AF4F924D2A",
        ["hyphen|R2|12345"] = "EF1618366D684269427BF3A92EAA4D43E3B7B39898039AC0DEEFCAF2CB213016",
        ["hyphen|Grumpy|0"] = "2216CBC8D48B64C48B97518FF6FE562F96C3E24D6762F5CBAB88A1FFCC71B8AD",
        ["hyphen|Grumpy|12345"] = "667372435A1E36FB2DFD3309FB07064F360155C8AF8ABAE81A75AAEC7420DFC0",
        ["hyphen|Bass|0"] = "5BAB00510F02A5A5774DA0D777A5547BC194407A6A9D91D5A3BB9AE85A00A5A5",
        ["hyphen|Bass|12345"] = "93DB8F6E04FA614B7BA612B4076C261AAD4FE8D80AD425EBB3C0138E1DE3234D",
        ["hyphen|Tiny|0"] = "4B895BABFA689ECD488B6E192FBCA38145B135F55AECD20291DB54C4B48128C6",
        ["hyphen|Tiny|12345"] = "D098C88EBD9476ECD966548B4C74DFD416DBBFF444936169028832B7A2214F19",
        ["ellipsis|R2|0"] = "87DFD0743686719F203D01A1343E1B39F78B9D1F509E471F4D44652058B88707",
        ["ellipsis|R2|12345"] = "EA09C5DBE6C14815367DBE2B3D84A8751BC006CAE3B3D5C83E003E008A360F03",
        ["ellipsis|Grumpy|0"] = "E1430647B3F77ADFDD3E2B46032DC570472EAF723D19955523FCE9A32432C581",
        ["ellipsis|Grumpy|12345"] = "671FC8B9721A183DC9AD8E74BDF34D5190E7CB371EC6F734F929260F5B7C6089",
        ["ellipsis|Bass|0"] = "9A50098AD51B981A2199690B18016315465E3CD0A0F299AB0C16763D31BFC09D",
        ["ellipsis|Bass|12345"] = "A4476D681DE0E17DFC76995E739EE8B77453EBFC24F236D54C2BB130304A6346",
        ["ellipsis|Tiny|0"] = "229AC70A6BF824FD307D26E41F4D0D0371AB20646DB67B29610BB376876EB948",
        ["ellipsis|Tiny|12345"] = "C49D2361CCF044E5EAC4036F4C9F09F72BD7A7F440B022D9F8B3F833E968B645",
        ["shout|R2|0"] = "22D0F0B01B0086161E7230204F7E07A3DB78475932042B970041E602E499216C",
        ["shout|R2|12345"] = "35D891C3690709BA9B3240F856C6CF5342E46DF217820EC756041F29BD3C5FCE",
        ["shout|Grumpy|0"] = "4DE3C91FAC4AA49E26C5EBE311141A5C6C286B9F97C6762C81F85820C7B9AA06",
        ["shout|Grumpy|12345"] = "3308AAF4786CB7568CDA12B3A6C13DBF9C35E20E67B1CC8A4649D29F2DE428B8",
        ["shout|Bass|0"] = "5FD56CE4D09B9E7C4552EBCE6F471E31BF2BE3B45A7CBAD1549ADF8D83D7ABA3",
        ["shout|Bass|12345"] = "6080598B74BB81B8805364A3EAB3CA09ECED181B4DA1798A0B7AFBEA9B4D9414",
        ["shout|Tiny|0"] = "1174D19FEAEF9A1FE2ABB902777956FC059991134EC1EAFCCF5FE8E7B103B7E9",
        ["shout|Tiny|12345"] = "75F90AFBC74E04BE05214D475FCB97D57599C673862D5BFCDADD4C8B9FC63BB4",
        ["cyrillic|R2|0"] = "C7E55B842F33A5BF093AC0A43693F4FA6F0929A8834DCDA5E0C81ADAE031BCE7",
        ["cyrillic|R2|12345"] = "D6C6A124DC5E9541BF8F69DC8378101A23AD5ADB9905C9ACED59F851781E1CB4",
        ["cyrillic|Grumpy|0"] = "D5280F8D4351DA636A980995F7B425082EEF336236CB0EF408EC8E82C16E0FBE",
        ["cyrillic|Grumpy|12345"] = "85C7F6F3085F78D44FAFBD13E20BBEE980088B7FFA7509CD0DB28F72C99EE4D2",
        ["cyrillic|Bass|0"] = "1141F2F68FB09DE883D8BA3732B8F2CA8D9BF46134A3F50081645F7E22A6346D",
        ["cyrillic|Bass|12345"] = "1822E0C69600E16AD47765FF517BAA6440C4F482D652685564834264EC677AA9",
        ["cyrillic|Tiny|0"] = "7A1317E1B4CC0AA3AFD903D0CBB6AD27031389128CB0C7847DAF03DA2183627B",
        ["cyrillic|Tiny|12345"] = "C023577B8D3446550A7796A293FBD7B65FFAF7E31528AE5FEF14EC29962DE347",
        ["pangram|R2|0"] = "BCE20BAA5108CB8B4F6D8474E5E0356AA747E8F87FC568810AB8F07B75D90E4F",
        ["pangram|R2|12345"] = "2AECCA42817A0901A72BCBE48CAC4210157A5C6D9A52BDBEBA9FD377BEF8DE90",
        ["pangram|Grumpy|0"] = "E4134540EB6014A83DDF1098BBC3E802A59B23426DEB87F4730CA4F9E0D5CDE2",
        ["pangram|Grumpy|12345"] = "6878F6B397F19CE2616634B9E0E6E5FABA3DE4E0E5AA7AB31257A62A9142085F",
        ["pangram|Bass|0"] = "B045B5EA999A24F9DA2F14379186C3D38B47116699587B86D3356306FDD7E229",
        ["pangram|Bass|12345"] = "658B126CC4D4AB9E699802D815FB831A73A4C3285ADE67C2CB71F3C5E3E28D62",
        ["pangram|Tiny|0"] = "27EAA212CF0FC496731012B77E298AA5D6E083FDED02A3383A3DC50252D1C9EE",
        ["pangram|Tiny|12345"] = "244C404C17444B32ABC29C5E1E4802CB3D699A7CA19F83D8BC4030C2F3B92FFE",
        ["excited|R2|0"] = "D74882206B6A30141D912261CFAD5638960BDD628BFD30B9C8739C5873C71227",
        ["excited|R2|12345"] = "23CB069C3261981DF01BDF743FA62E6282601A3B0138C61971AC574163A0EF59",
        ["excited|Grumpy|0"] = "8520850A8EE0356FF121A5CE3C0D7F39863125E972061BD0BA4962F720FD2001",
        ["excited|Grumpy|12345"] = "8A403A3AD667DF1CF41CDB60100057CBEB795759340EEE76336A29FBE4A4D8AA",
        ["excited|Bass|0"] = "4BF295C5B9C9FF10F63644179879A645C9DB181E50ECA4889E22382AC425AE13",
        ["excited|Bass|12345"] = "6D02701B16035A7E0804CCA308AB7A37FA9AEC045CEED032051DBEEA4C22BD0A",
        ["excited|Tiny|0"] = "97D3D16B936096C831148F86A3322B357A284D007A1F7A1C7120B9B465D1D480",
        ["excited|Tiny|12345"] = "659759C7F7CB6F14C77F4F139010F1F918FFE721BFBC3B93E64D691FB4629BBB",
        ["hello|R2|0|22050"] = "A35F3D77B4C7DA31BF931E7AF32D561F314243CF2E1F04007F1C75A8344D3F34",
    };

    private static readonly Dictionary<string, string> Texts = new(StringComparer.Ordinal)
    {
        ["hello"] = "Hello, world!",
        ["hyphen"] = "R2-D2, are you okay?",
        ["ellipsis"] = "I am... not sure.",
        ["shout"] = "BEEP BOOP BOOP",
        ["cyrillic"] = "Привет, как дела?",
        ["pangram"] = "The quick brown fox jumps over the lazy dog.",
        ["excited"] = "WOW! THIS IS AMAZING! I LOVE IT!",
    };

    public static TheoryData<string, string, int> GoldenCases()
    {
        var data = new TheoryData<string, string, int>();

        foreach (var (textId, _) in Texts)
        {
            foreach (var profileName in new[] { "R2", "Grumpy", "Bass", "Tiny" })
            {
                foreach (var seed in new[] { 0, 12345 })
                {
                    data.Add(textId, profileName, seed);
                }
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(GoldenCases))]
    public void GoldenPcm_IsByteIdentical(string textId, string profileName, int seed)
    {
        var text = Texts[textId];
        var profile = Profile(profileName);
        var synthesizer = new DroidSynthesizer(seed: seed, profile: profile);

        var samples = synthesizer.Synthesize(text);

        Assert.Equal(Golden[$"{textId}|{profileName}|{seed}"], Hash(samples));
    }

    [Theory]
    [MemberData(nameof(GoldenCases))]
    public void SampleCount_MatchesRenderedLength(string textId, string profileName, int seed)
    {
        var text = Texts[textId];
        var profile = Profile(profileName);
        var synthesizer = new DroidSynthesizer(seed: seed, profile: profile);

        var count = synthesizer.GetSampleCount(text);
        var samples = synthesizer.Synthesize(text);

        Assert.Equal(count, samples.Length);
        Assert.Equal(count, synthesizer.GetSampleCount(text));
    }

    [Fact]
    public void GoldenPcm_IsStableAtReducedSampleRate()
    {
        var synthesizer = new DroidSynthesizer(sampleRate: 22050, seed: 0, profile: DroidVoiceProfile.R2);

        var samples = synthesizer.Synthesize("Hello, world!");

        Assert.Equal(Golden["hello|R2|0|22050"], Hash(samples));
    }

    [Theory]
    [MemberData(nameof(GoldenCases))]
    public void SpanOverload_WritesTheSameBytesAsAllocatingOverload(string textId, string profileName, int seed)
    {
        var text = Texts[textId];
        var profile = Profile(profileName);
        var synthesizer = new DroidSynthesizer(seed: seed, profile: profile);

        var expected = synthesizer.Synthesize(text);
        var destination = new float[synthesizer.GetSampleCount(text)];

        var written = synthesizer.Synthesize(text, destination);

        Assert.Equal(expected.Length, written);
        Assert.Equal(Hash(expected), Hash(destination));
    }

    private static DroidVoiceProfile Profile(string name) => name switch
    {
        "R2" => DroidVoiceProfile.R2,
        "Grumpy" => DroidVoiceProfile.Grumpy,
        "Bass" => DroidVoiceProfile.Bass,
        "Tiny" => DroidVoiceProfile.Tiny,
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "Unknown profile."),
    };

    private static string Hash(float[] samples) =>
        Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(samples.AsSpan())));
}
