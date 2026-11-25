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
        ["hello|R2|0"] = "3F13189C18152E4E7BDBACFC8B0460014B011E41BD59006D961FC45402E4EF14",
        ["hello|R2|12345"] = "DFD8C5E2FDD48C8A769AC656B806EA0CA165A19E2436ECB4646555AC103BDDFE",
        ["hello|Grumpy|0"] = "E746BE1E56A56F20FBD705C22945094372D9E805EAC6804112CC5122D94B9E5F",
        ["hello|Grumpy|12345"] = "D94E255BA6DD74CE70DD95D5D8D34D41F6D5F99BB27EB136529306C5248F38DC",
        ["hello|Bass|0"] = "1D417299D322A0CCEB997DCF37EAEE85332668DD96E2DE5054E6ACBB93698415",
        ["hello|Bass|12345"] = "D35859068D6343C2835D7ADBB18FF07526EEEFD4C5713BB03DCC5A8BCA1AB1BC",
        ["hello|Tiny|0"] = "6FDC0A484D1C1144C1FE4816FE3E9CA74B6A5BBF9DB0C56C17E01DBFDAC7D82D",
        ["hello|Tiny|12345"] = "E009958FE2CB018687C87F28E0ACBDF4371BA3F5DA7C6AE387F331EC83D45F41",
        ["hyphen|R2|0"] = "390259E07096A7BABFB0EE5FA1947E3CC16657AE7830B7A50557CB3ED098056A",
        ["hyphen|R2|12345"] = "EC830287770BA00FB9FAD18256A74C6E1D84DFA9C7780B97B5C4A6181326B6D3",
        ["hyphen|Grumpy|0"] = "BF51BF9455D1D9ED7F573C4094D7357596C1E4AF44D594D9C9CB3BB7F536D828",
        ["hyphen|Grumpy|12345"] = "FEA99B42349F84F16D8ECC723C8469E92A1FEEDF68BCE9B0105B364B52C348DB",
        ["hyphen|Bass|0"] = "31D3D5E8A631BB5D68663D9EFD2E776CC34CD9628389C20F09C802E8238CE7B5",
        ["hyphen|Bass|12345"] = "9BEA9035EFBB94E21574228FE752EEE5E85680470123D6B171CE43A43B889984",
        ["hyphen|Tiny|0"] = "F690F290EA1327C5F6B77B66A83655D090578BC6B7910339C0E64E772865FB6E",
        ["hyphen|Tiny|12345"] = "ADBC1F33B3DFEE39CD766AEA957807C600D35115BF5E00102F05073A91ABDCA0",
        ["ellipsis|R2|0"] = "7E7FB266A1E95E1788CF4F12A31BCB49F800D8F6AA91E8D815BB12610CAE21AE",
        ["ellipsis|R2|12345"] = "F1346E79C5466495461BF7FC6A1178D54B44E444CC74F2EBF4AB71E776B62A8C",
        ["ellipsis|Grumpy|0"] = "2D52D8D7393BDB9CDD0C2263C48B9C72D0A98FA263C1C9610A959B46C2592C3F",
        ["ellipsis|Grumpy|12345"] = "31B6902BEC00A717D570F398663ACF8F4F26D2E1CF73C390BB26D55107E4235B",
        ["ellipsis|Bass|0"] = "EEE2BEFA71B3AFC07499439EAC63DCAA191B3E150DB1F2E921CD6CE3FFF81C2C",
        ["ellipsis|Bass|12345"] = "68880BB4F1D5FC686DE12856AF1D9834AA311BD14B20DA446B800AABF2D82813",
        ["ellipsis|Tiny|0"] = "ACA8EDF951971C6AC1BB76EFF1093F3E1C60ED6C8814A6A0AD27C2D80AF54083",
        ["ellipsis|Tiny|12345"] = "5F7C7E6DD7ED51B466794B33361DE4300DE88377AE077AF2681D2D9340A4C74F",
        ["shout|R2|0"] = "FE9DCD300ABAB0140FDCA7835D3ABADD4C016D66BBDD06579987D7470DF8AC50",
        ["shout|R2|12345"] = "058B9894A88E946F6AC7043F09EC52E5F4EAD3E148824CC53BEF5B244E4D312C",
        ["shout|Grumpy|0"] = "9E370E7EE2F5269D8BCA904DF09B678D785FB12CD4C1E9ED954F7AD0001C362F",
        ["shout|Grumpy|12345"] = "69C2359CF30080DD2778B41C906D34203A03AC95CA5BBA315D47F05ACFE4DA57",
        ["shout|Bass|0"] = "786ECE50D02C3138A6B5AB62AB68F3461EB7C103482A299A824901EC1BA6A589",
        ["shout|Bass|12345"] = "FC747BEC645EC057E66AB65AA01F4522583BC1EC150CB774EE9DC1B789BAB608",
        ["shout|Tiny|0"] = "04413B18AC949081542C59B5E2566E24539ABD0192C33616203023FBDC2C4526",
        ["shout|Tiny|12345"] = "B940AE43C843FBB4D2D8731319A98B85459881D493D813EF1FEBCA28A51B8BB6",
        ["cyrillic|R2|0"] = "660EBB09199AE37818B77C6E94DABF2966891FDC7B51538CFD707D5C3F0CDD3E",
        ["cyrillic|R2|12345"] = "B15BA923C1C10DD7C5C521E60AC209CC9DC931205731A0C7599F50906305E685",
        ["cyrillic|Grumpy|0"] = "D826EA75F66B7545266541372ADFEB5EF157C576F4E1AD40BFD0C20BC764AD04",
        ["cyrillic|Grumpy|12345"] = "1CDC0E707ADC673A928FE0D408319D8EAC38C49ABD507C0389D7A764C9CA7FC2",
        ["cyrillic|Bass|0"] = "03EC746D70DD8189367389E00702A1E6DD512C604C0839E20ABF68F21D14525E",
        ["cyrillic|Bass|12345"] = "280811FAE5F7F2AD1DB5545B0BB6F70B033201138210895D0A37E0330FB01D5A",
        ["cyrillic|Tiny|0"] = "94CD4CECED3E8FCF4EC0E26B1C66F95C78DAA4F1FD829FBE622C32FE00B5F20C",
        ["cyrillic|Tiny|12345"] = "C767C6B9BB3EC3168645BD90617B06B041B82B7961C85B06793155B6379C935D",
        ["pangram|R2|0"] = "35661F824E7681A4F0615180E1EC19AB2906FCB698037E8EE1F861F697917F76",
        ["pangram|R2|12345"] = "CF23B9FCF83977A3A3652153B3ABB02078873D5E111E6378640134E2AD94A7EE",
        ["pangram|Grumpy|0"] = "2C08E080C7DC58AB07E0AA9AA29135F43E6B08F668B6E38FE22CFC0BA9F48059",
        ["pangram|Grumpy|12345"] = "1157D1B5E818D36C5C75FB255D80355DFA16E6DC5CE6A2784459411EA961EF89",
        ["pangram|Bass|0"] = "9D13BF6D07C923A54D0C260808314CB0DDA804635D33ED23B912A3DBC3F9F6EC",
        ["pangram|Bass|12345"] = "905A17BDFB7A787ABB81BD0B2C8F2DF91283E280D484276361D794A1E4E42924",
        ["pangram|Tiny|0"] = "61886D73856F71FD1B1C87D5BE474CC3CC3D5BDBF90FCF45B927FCB985714B7F",
        ["pangram|Tiny|12345"] = "EEFDF1EB48F15D9A8BE10D13AC1AAB02A62C52DE7107DDD380D4B5C373567569",
        ["excited|R2|0"] = "EE15EF35B59C0C5AFEBEA955540BF679D176E329AE8DEA441021E7AC36E14DB1",
        ["excited|R2|12345"] = "B48A15DB4CFFABA930BBEE4CAB1B9A2B5FBA413FA8D15BF39C3F3E29FC8A8D7C",
        ["excited|Grumpy|0"] = "3770AD789CD0E63DB9EFD6EE02A4A81A56C6BED6932D6EA4E934AC84FDD626E0",
        ["excited|Grumpy|12345"] = "71837254A3C3F68B03E7651B67EEF767BAD59401BFDED2DAB7BAF7FCC3F96E82",
        ["excited|Bass|0"] = "D4713D788E2BACD0547DDCDAA6C85EBF435DE8D3F7F5AC1B8A190EE1E79A35B6",
        ["excited|Bass|12345"] = "61CD87318D6936E0659FCBDA498DA3ABDB7D3749C2D29B7FEF11140A4CA8DC3E",
        ["excited|Tiny|0"] = "8BB5343CB3FD55F7A37EB9988D1E059FD8EC32CBB1B3586BF8556D4ED959B3EF",
        ["excited|Tiny|12345"] = "3E1F6FE27E2415977694D001B992E57834F15357CD333693CBB0052B6F1EA33D",
        ["hello|R2|0|22050"] = "193824E30D6C610BF9E9F25AB3CEE59A5CA1515FA8F61779F735C3B5AFE848BD",
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
