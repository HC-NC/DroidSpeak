namespace DroidSpeak.NLP;

/// <summary>
/// A tiny, deterministic emotion lexicon. It is intentionally small: typography remains the
/// primary source of expression, while common words add a little semantic colour.
/// </summary>
internal static class EmotionLexicon
{
    private static readonly Dictionary<string, EmotionVector> Entries = new(StringComparer.OrdinalIgnoreCase)
    {
        // Positive / friendly.
        ["привет"] = new(0.45f, 0.15f),
        ["здравствуй"] = new(0.40f, 0.10f),
        ["спасибо"] = new(0.50f, 0.10f),
        ["ура"] = new(0.60f, 0.50f),
        ["отлично"] = new(0.50f, 0.20f),
        ["хорошо"] = new(0.40f, 0.10f),
        ["люблю"] = new(0.60f, 0.20f),
        ["друг"] = new(0.40f, 0.00f),
        ["готов"] = new(0.30f, 0.10f),
        ["hello"] = new(0.40f, 0.10f),
        ["hi"] = new(0.40f, 0.15f),
        ["beep"] = new(0.30f, 0.20f),
        ["boop"] = new(0.30f, 0.10f),
        ["ready"] = new(0.35f, 0.15f),
        ["good"] = new(0.40f, 0.10f),
        ["great"] = new(0.50f, 0.30f),

        // Negative / alarmed.
        ["тревога"] = new(-0.40f, 0.60f),
        ["перегрев"] = new(-0.35f, 0.55f),
        ["реактор"] = new(-0.10f, 0.20f),
        ["сломалось"] = new(-0.55f, 0.10f),
        ["сломался"] = new(-0.50f, 0.15f),
        ["ошибка"] = new(-0.40f, 0.30f),
        ["плохо"] = new(-0.45f, 0.05f),
        ["нет"] = new(-0.35f, 0.15f),
        ["больно"] = new(-0.50f, 0.30f),
        ["опять"] = new(-0.20f, 0.00f),
        ["ну"] = new(-0.10f, -0.05f),
        ["alert"] = new(-0.25f, 0.65f),
        ["danger"] = new(-0.50f, 0.60f),
        ["error"] = new(-0.40f, 0.35f),
        ["sad"] = new(-0.55f, -0.20f),
        ["sorry"] = new(-0.40f, -0.10f),
        ["system"] = new(0.00f, 0.10f),
    };

    public static EmotionVector Lookup(string word)
    {
        var normalized = Phonetics.NormalizeWord(word);
        return normalized.Length > 0 && Entries.TryGetValue(normalized, out var emotion)
            ? emotion
            : EmotionVector.Neutral;
    }
}
