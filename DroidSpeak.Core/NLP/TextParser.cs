using System.Text;

namespace DroidSpeak.NLP;

/// <summary>
/// Parses arbitrary text into phrases, words, syllables and emotions.
/// <para>
/// Typography drives the performance:
/// <list type="bullet">
/// <item><c>!</c> raises arousal (louder, faster, wider frequency spread);</item>
/// <item><c>?</c> adds a rising glide over the final two syllables;</item>
/// <item><c>...</c> slows the tempo and adds a falling tone;</item>
/// <item><c>UPPERCASE</c> words add local FM overload (a droid scream).</item>
/// </list>
/// </para>
/// </summary>
public static class TextParser
{
    private const float LexiconWeight = 0.40f;

    /// <summary>Parses <paramref name="text"/> into phrases. Empty or whitespace-only input yields no phrases.</summary>
    public static IReadOnlyList<Phrase> Parse(string text)
    {
        ArgumentNullException.ThrowIfNull(text);

        var phrases = new List<Phrase>();
        if (string.IsNullOrWhiteSpace(text))
        {
            return phrases;
        }

        var buffer = new StringBuilder(text.Length);

        for (var i = 0; i < text.Length; i++)
        {
            var character = text[i];
            switch (character)
            {
                case '!':
                    while (i + 1 < text.Length && text[i + 1] is '!' or '?')
                    {
                        i++;
                    }

                    AddPhrase(phrases, buffer, PhrasePunctuation.Exclamation);
                    break;

                case '?':
                    while (i + 1 < text.Length && text[i + 1] is '?' or '!')
                    {
                        i++;
                    }

                    AddPhrase(phrases, buffer, PhrasePunctuation.Question);
                    break;

                case '.':
                    var dots = 1;
                    while (i + 1 < text.Length && text[i + 1] == '.')
                    {
                        dots++;
                        i++;
                    }

                    AddPhrase(
                        phrases,
                        buffer,
                        dots >= 2 ? PhrasePunctuation.Ellipsis : PhrasePunctuation.Statement);
                    break;

                case '…':
                    AddPhrase(phrases, buffer, PhrasePunctuation.Ellipsis);
                    break;

                case '\r':
                    if (i + 1 < text.Length && text[i + 1] == '\n')
                    {
                        i++;
                    }

                    AddPhrase(phrases, buffer, PhrasePunctuation.Statement);
                    break;

                case '\n':
                    AddPhrase(phrases, buffer, PhrasePunctuation.Statement);
                    break;

                default:
                    buffer.Append(character);
                    break;
            }
        }

        AddPhrase(phrases, buffer, PhrasePunctuation.Statement);
        return phrases;
    }

    private static void AddPhrase(
        List<Phrase> phrases,
        StringBuilder buffer,
        PhrasePunctuation punctuation)
    {
        var rawText = buffer.ToString().Trim();
        buffer.Clear();

        if (rawText.Length == 0)
        {
            return;
        }

        var words = ParseWords(rawText);
        if (words.Count == 0)
        {
            return;
        }

        phrases.Add(BuildPhrase(rawText, words, punctuation));
    }

    private static List<Word> ParseWords(string text)
    {
        var words = new List<Word>();
        var current = new StringBuilder(16);

        for (var i = 0; i <= text.Length; i++)
        {
            var character = i < text.Length ? text[i] : '\0';
            var isWordCharacter = i < text.Length &&
                (char.IsLetterOrDigit(character) || character is '\'' or '’' or '-');

            if (isWordCharacter)
            {
                current.Append(character);
                continue;
            }

            if (i < text.Length && character is ',' or ';' or ':')
            {
                if (current.Length > 0)
                {
                    var wordText = current.ToString();
                    current.Clear();
                    words.Add(CreateWord(wordText, isFollowedByComma: true));
                }
                else if (words.Count > 0 && !words[^1].IsFollowedByComma)
                {
                    // The comma is separated from the previous word by whitespace.
                    var last = words[^1];
                    words[^1] = new Word(last.Text, last.IsShout, isFollowedByComma: true, last.Syllables);
                }

                continue;
            }

            if (current.Length > 0)
            {
                var wordText = current.ToString();
                current.Clear();
                words.Add(CreateWord(wordText, isFollowedByComma: false));
            }
        }

        return words;
    }

    private static Word CreateWord(string text, bool isFollowedByComma)
    {
        var letterCount = 0;
        var allUppercase = true;

        foreach (var character in text)
        {
            if (!char.IsLetter(character))
            {
                continue;
            }

            letterCount++;
            if (!char.IsUpper(character))
            {
                allUppercase = false;
            }
        }

        var isShout = letterCount >= 2 && allUppercase;
        return new Word(text, isShout, isFollowedByComma, SyllableSplitter.Split(text));
    }

    private static Phrase BuildPhrase(
        string text,
        List<Word> words,
        PhrasePunctuation punctuation)
    {
        float valence = 0f;
        float arousal = 0f;

        switch (punctuation)
        {
            case PhrasePunctuation.Exclamation:
                arousal += 0.55f;
                valence += 0.05f;
                break;

            case PhrasePunctuation.Question:
                arousal += 0.10f;
                valence += 0.05f;
                break;

            case PhrasePunctuation.Ellipsis:
                arousal -= 0.35f;
                valence -= 0.30f;
                break;
        }

        var shoutWords = 0;
        foreach (var word in words)
        {
            var emotion = EmotionLexicon.Lookup(word.Text);
            valence += emotion.Valence * LexiconWeight;
            arousal += emotion.Arousal * LexiconWeight;

            if (word.IsShout)
            {
                shoutWords++;
            }
        }

        var shoutRatio = shoutWords / (float)words.Count;
        arousal += 0.50f * shoutRatio;
        valence -= 0.25f * shoutRatio;

        return new Phrase(
            text,
            new EmotionVector(valence, arousal),
            punctuation,
            words,
            shoutRatio);
    }
}
