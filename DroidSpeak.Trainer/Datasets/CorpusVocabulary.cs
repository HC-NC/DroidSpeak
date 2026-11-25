using System.Text;

namespace DroidSpeak.Trainer.Datasets;

/// <summary>
/// The fixed vocabulary the procedural corpus draws from: grouped word pools plus the canonical
/// utterances that guarantee coverage of every phonetic and emotional case the network must learn.
/// <para>
/// The pools are not thematic decoration. Each group exists to exercise one input feature:
/// <list type="bullet">
/// <item>vowel-bearing words hit every <c>SyllableFeatureEncoder.VowelClass</c> one-hot, including
/// the consonant-only class;</item>
/// <item>cluster words widen the onset and coda blocks;</item>
/// <item>hyphenated compounds and apostrophe contractions vary the hyphen-boundary and word-length
/// features;</item>
/// <item>shouted words flip <c>IsShout</c>;</item>
/// <item>lexicon words are the only entries that move valence and arousal.</item>
/// </list>
/// </para>
/// <para>
/// <see cref="CoverageUtterances"/> is encoded before any random sentence, so even a small corpus
/// covers the phonetic inventory and the whole emotion grid; the random composer then fills the
/// requested count with fresh combinations.
/// </para>
/// </summary>
internal static class CorpusVocabulary
{
    /// <summary>
    /// Sentences emitted first, in this order, by <see cref="DistillationCorpus"/>.
    /// <para>
    /// The first nine entries are the emotion grid: one sentence (or phrase) for each cell of
    /// valence {positive, neutral, negative} crossed with arousal {excited, neutral, calm}. The rest
    /// are phonetic showcases: the Cyrillic and Latin vowel inventories, consonant clusters,
    /// hyphenated compounds, apostrophe contractions, shouted words, and every terminal punctuation
    /// the parser distinguishes.
    /// </para>
    /// </summary>
    public static IReadOnlyList<string> CoverageUtterances { get; } =
    [
        // (valence, arousal) grid: positive / neutral / negative by excited / neutral / calm.
        "Спасибо, друг, хорошо...",                 // positive, calm
        "УРА! отлично!",                            // positive, excited
        "Плохо... сломалось...",                    // negative, calm
        "ТРЕВОГА! danger! alert!",                  // negative, excited
        "друг, готов, кислород...",                 // neutral, calm
        "Кислород! Давление!",                      // neutral, excited
        "Кислород, давление, температура.",         // neutral, neutral
        "Спасибо, друг.",                           // positive, neutral
        "Плохо, сломалось.",                        // negative, neutral

        // Phonetic showcases.
        "Ёлка, эхо, иней, якорь, южный, уран, быстро.",      // every Cyrillic vowel class
        "apples, energy, ink, ocean, umbrella, system, hmm.", // every Latin class + consonant-only
        "Взрыв, страх, структура, контроль, штурм.",         // Cyrillic consonant clusters
        "spring, strength, thrust, cluster, script.",        // Latin consonant clusters
        "hyper-drive, system-error, старт-стоп, R2-D2.",      // hyphenated compounds
        "Don't touch, it's fine... we're ready!",             // apostrophes, comma, ellipsis
        "I can't, won't, don't.",                             // more contractions
        "ВНИМАНИЕ! ОПАСНОСТЬ! MAYDAY!",                       // shouted words
        "Как давление? Растёт, но... СТОП!",                  // question, comma, ellipsis, shout
    ];

    /// <summary>
    /// Words that carry emotion in the parser's lexicon, plus ordinary friendly droid vocabulary.
    /// These matter most: they are the only words whose valence and arousal are non-zero, and a
    /// corpus without them would teach a flat voice.
    /// </summary>
    private static readonly string[] EmotionalWords =
    [
        "привет", "здравствуй", "спасибо", "ура", "отлично", "хорошо", "люблю", "друг", "готов",
        "hello", "hi", "beep", "boop", "ready", "good", "great",
        "тревога", "перегрев", "реактор", "сломалось", "сломался", "ошибка", "плохо", "нет",
        "больно", "опять", "ну", "alert", "danger", "error", "sad", "sorry", "system",
    ];

    /// <summary>
    /// Neutral vocabulary, chosen to vary vowel class, syllable count, word length and onset class.
    /// The Cyrillic entries between <c>ёлка</c> and <c>быстро</c> exist purely for their vowels; the
    /// Latin entries from <c>spring</c> onward are consonant clusters.
    /// </summary>
    private static readonly string[] NounWords =
    [
        "мир", "кабина", "люк", "датчик", "сопротивление", "кислород", "баки", "двигатель",
        "щит", "канал", "система", "показатели", "давление", "температура", "память", "энергия",
        "антенна", "шина", "сервопривод", "стабилизатор", "топливо", "кислота", "искажение",
        "взрыв", "страх", "структура", "контроль", "штурм", "прыжок", "хрусталь", "ствол",
        "брызги", "справка", "ёлка", "эхо", "иней", "якорь", "уран", "роза", "море", "икра",
        "утро", "осень", "южный", "быстро",
        "world", "coolant", "pressure", "reactor", "engines", "panel", "circuit", "hydraulics",
        "sensor", "diagnostics", "power", "cell", "transmission", "oxygen", "shield", "conduit",
        "processor", "capacitor", "thruster", "bulkhead", "relay", "coupling", "calibration",
        "master", "computer", "hyperdrive", "motivator", "sublight", "flux",
        "spring", "strength", "thrust", "cluster", "script", "splinter", "shrink", "apple", "ocean",
    ];

    /// <summary>Connectors that keep generated text from being a bare word list.</summary>
    private static readonly string[] ConnectorWords =
    [
        "и", "но", "или", "что", "где", "почему", "пожалуйста", "снова", "всегда", "сейчас",
        "and", "but", "or", "why", "where", "please", "again", "always", "now", "still",
    ];

    /// <summary>All-caps tokens the parser recognises as shouts, in both languages.</summary>
    private static readonly string[] ShoutWords =
    [
        "SOS", "MAYDAY", "NEGATIVE", "DANGER", "CRITICAL", "WARNING", "STOP", "NOW", "HELP",
        "HALT", "ТРЕВОГА", "СТОП", "СРОЧНО", "ОПАСНОСТЬ", "ВНИМАНИЕ", "ПОМОГИТЕ", "УРА",
    ];

    /// <summary>
    /// English contractions. The parser keeps <c>'</c> inside a word, so these exercise the
    /// apostrophe branch of the tokenizer while splitting normally into syllables.
    /// </summary>
    private static readonly string[] ApostropheWords =
    [
        "don't", "can't", "won't", "it's", "that's", "we're", "they're", "I'm", "isn't",
        "aren't", "let's", "droid's",
    ];

    /// <summary>
    /// Hyphenated compounds whose hyphen falls on a syllable boundary, so the context builder
    /// actually sets <c>IsHyphenBoundary</c> rather than only the parser seeing a hyphen.
    /// </summary>
    private static readonly string[] HyphenWords =
    [
        "R2-D2", "hyper-drive", "system-error", "auto-pilot", "sublight-drive",
        "старт-стоп", "пере-запуск", "энерго-щит",
    ];

    /// <summary>
    /// Punctuation is the main expression channel, so every random row picks from this pool:
    /// statement, question, exclamation, ellipsis, and two comma-heavy statements.
    /// </summary>
    private static readonly string[] Endings = [".", "?", "!", "...", ", и...", ", but..."];

    /// <summary>
    /// Composes one random sentence: a few words, occasionally a comma clause, a shouted or
    /// contracted token, and a hyphen join, finished with a punctuation mark.
    /// </summary>
    /// <param name="random">Deterministic source; the same seed always yields the same sentence.</param>
    public static string ComposeRandom(ref RandomSource random)
    {
        var builder = new StringBuilder(96);
        var words = 3 + random.NextInt(6);
        var hasComma = random.NextBool(0.30f);
        var hasHyphenJoin = random.NextBool(0.12f);
        var commaPlaced = false;

        for (var i = 0; i < words; i++)
        {
            if (i > 0)
            {
                builder.Append(' ');
            }

            var word = PickWord(ref random, i);

            if (random.NextBool(0.10f))
            {
                // Capitalised words are how a caller shouts at the synthesizer, and the parser
                // derives its shout ratio from them, so the corpus needs them too.
                word = word.ToUpperInvariant();
            }

            builder.Append(word);

            if (hasComma && !commaPlaced && i == words / 2)
            {
                builder.Append(',');
                commaPlaced = true;
            }

            if (hasHyphenJoin && i == 1)
            {
                builder.Append('-');
                builder.Append(PickWord(ref random, i + 2));
            }
        }

        builder.Append(Endings[random.NextInt(Endings.Length)]);
        return builder.ToString();
    }

    private static string PickWord(ref RandomSource random, int position)
    {
        // Every sentence opens on an emotional word so valence and arousal never sit at zero for a
        // whole utterance; shout tokens appear often enough for the parser's ratio to vary.
        if (position == 0)
        {
            return EmotionalWords[random.NextInt(EmotionalWords.Length)];
        }

        var roll = random.NextFloat();

        if (roll < 0.08f)
        {
            return ShoutWords[random.NextInt(ShoutWords.Length)];
        }

        if (roll < 0.16f)
        {
            return ApostropheWords[random.NextInt(ApostropheWords.Length)];
        }

        if (roll < 0.26f)
        {
            return HyphenWords[random.NextInt(HyphenWords.Length)];
        }

        if (roll < 0.55f)
        {
            return EmotionalWords[random.NextInt(EmotionalWords.Length)];
        }

        if (roll < 0.68f)
        {
            return ConnectorWords[random.NextInt(ConnectorWords.Length)];
        }

        return NounWords[random.NextInt(NounWords.Length)];
    }
}
