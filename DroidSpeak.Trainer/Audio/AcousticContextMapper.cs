using DroidSpeak;
using DroidSpeak.Neural;
using DroidSpeak.NLP;
using DroidSpeak.Prosody;

namespace DroidSpeak.Trainer.Audio;

/// <summary>
/// Turns the segmentation of one reference clip into the two flat arrays the neural prosody trainer
/// speaks: a <see cref="SyllableContext"/> and a <see cref="SyllableFrame"/> per acoustic pulse.
/// <para>
/// This is the acoustic counterpart of <c>DistillationCorpus</c>. There is no text to parse, so the
/// "context" is reconstructed from the clip's own structure (words, syllable positions, stress from
/// energy) plus a deliberately small file-name convention, and the "frame" is measured from the
/// samples by <see cref="AcousticFeatureExtractor"/> rather than predicted by the rule engine.
/// Everything that cannot be measured from a single pulse is filled with the neutral midpoint of its
/// target range, so a hand-extracted dataset never teaches the network an extreme value it has no
/// evidence for.
/// </para>
/// <para>
/// The text side supports the tags <c>[happy]</c>, <c>[sad]</c>, <c>[alert]</c>, <c>[angry]</c>,
/// <c>[calm]</c> and <c>[question]</c> anywhere in the path (a file name or a containing folder),
/// the trailing <c>?</c>, <c>!</c> or <c>...</c> for the contour, and the remaining word tokens in
/// the file name are distributed over the detected word groups by index. A clip called
/// <c>alert[alert]!!.wav</c> therefore becomes an alarmed exclamation whose syllables cycle through
/// "alert". This is an acknowledged approximation: the recording itself is the only ground truth,
/// and the tags are a hint, not a transcript.
/// </para>
/// <para>
/// Two fields need cross-domain care. Sample positions are converted from the analysis rate to the
/// render rate, and the duration and final pause are snapped through
/// <see cref="SyllableTargetDecoder"/> so <see cref="SyllableTargetEncoder"/> can invert them
/// exactly. Without that snap a pulse boundary falling between two render samples would make the
/// round trip off by one sample in the duration target, which is noise the network would have to
/// learn around.
/// </para>
/// <para>
/// The mapping is otherwise deterministic. The clip id is a stable hash of the path, the syllable
/// seed is derived from that id and the global seed, and no clock or hash randomization enters the
/// result. The same clip therefore always produces byte-identical contexts and frames, which is what
/// lets the JSON-free <c>.dsdata</c> container be compared across machines and runs.
/// </para>
/// </summary>
internal static class AcousticContextMapper
{
    /// <summary>Inter-word silence above which the preceding word is treated as comma-delimited.</summary>
    public const float PreviousCommaPauseSeconds = 0.08f;

    /// <summary>Peak frame RMS above which a pulse is treated as shouted.</summary>
    public const float ShoutPeakRms = 0.5f;

    /// <summary>Emotion tags understood anywhere in a clip's path.</summary>
    private static readonly string[] EmotionTags = ["happy", "sad", "alert", "angry", "calm"];

    /// <summary>Tag that forces a question contour regardless of trailing punctuation.</summary>
    private const string QuestionTag = "question";

    /// <summary>Fallback word text for clips whose name contains no letters or digits.</summary>
    private const string FallbackWord = "clip";

    /// <summary>Tempo scale used for every extracted phrase.</summary>
    private const float ExtractedTempoScale = 1f;

    /// <summary>Base growl rate handed to a measured frame, in Hz.</summary>
    private const float BaseGrowlFrequency = 30f;

    /// <summary>
    /// Stable 32-bit clip identifier, derived from the path with FNV-1a. Separators are normalised so
    /// a clip hashed on Windows and on a POSIX build agrees when the path is spelled the same way.
    /// </summary>
    /// <param name="path">Path the clip was read from.</param>
    /// <exception cref="ArgumentNullException"><paramref name="path"/> is null.</exception>
    public static uint ClipId(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        return Fnv1a32(path.Replace('\\', '/'));
    }

    /// <summary>
    /// Maps a segmented clip onto its syllable contexts and frames.
    /// </summary>
    /// <param name="audio">Decoded mono clip; its rate is the rate the pulse samples are expressed in.</param>
    /// <param name="segmentation">Pulses and word grouping for <paramref name="audio"/>.</param>
    /// <param name="profile">Voice profile the targets are encoded against; normalized internally.</param>
    /// <param name="path">Clip path, hashed into the per-word hash and the clip id; its tags and file
    /// name drive the reconstructed text side.</param>
    /// <param name="renderRate">Sample rate baked into the contexts and frame timing.</param>
    /// <param name="seed">Global seed mixed into every syllable seed.</param>
    /// <returns>
    /// Two arrays of equal length, in pulse order, or two empty arrays when the clip holds no pulses.
    /// </returns>
    /// <exception cref="ArgumentException"><paramref name="path"/> is empty or whitespace.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// <paramref name="renderRate"/> is outside 8000..192000 Hz, or the audio rate is not positive.
    /// </exception>
    public static (SyllableContext[] Contexts, SyllableFrame[] Frames) Extract(
        WavData audio,
        Segmentation segmentation,
        DroidVoiceProfile profile,
        string path,
        int renderRate,
        uint seed)
    {
        ArgumentNullException.ThrowIfNull(profile);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);

        if (renderRate is < 8000 or > 192000)
        {
            throw new ArgumentOutOfRangeException(
                nameof(renderRate),
                renderRate,
                "The render rate must be between 8000 and 192000 Hz.");
        }

        if (audio.SampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(audio),
                audio.SampleRate,
                "The audio rate must be positive.");
        }

        if (audio.Samples is null)
        {
            throw new ArgumentException("The audio clip has no sample buffer.", nameof(audio));
        }

        profile = profile.Normalized();

        var pulses = segmentation.Pulses ?? [];
        var words = segmentation.Words ?? [];

        if (pulses.Length == 0 || words.Length == 0)
        {
            return ([], []);
        }

        var (taggedEmotion, forcedQuestion) = ReadTags(path);
        var stem = StripTags(Path.GetFileNameWithoutExtension(path));
        var punctuation = ResolvePunctuation(stem, forcedQuestion);
        var tokens = Tokenize(stem);
        if (tokens.Count == 0)
        {
            var letters = LettersOnly(stem);
            tokens = [letters.Length > 0 ? letters : FallbackWord];
        }

        var emotion = taggedEmotion ?? InferEmotion(tokens);

        var clipId = ClipId(path);
        var (wordOfPulse, syllableOfPulse) = MapPulsesToWords(pulses.Length, words);
        var scale = renderRate / (float)audio.SampleRate;

        var contexts = new SyllableContext[pulses.Length];
        var frames = new SyllableFrame[pulses.Length];

        for (var i = 0; i < pulses.Length; i++)
        {
            var pulse = pulses[i];
            var wordIndex = wordOfPulse[i];
            var syllableIndex = syllableOfPulse[i];
            var word = words[wordIndex];
            var isSegmentStart = syllableIndex == 0;

            var wordText = tokens[wordIndex % tokens.Count];
            var wordHash = Phonetics.StableHash(wordText);
            var (onset, vowels) = SplitWord(wordText);
            var hasVoiced = Phonetics.ContainsVoicedConsonant(onset);
            var hasSibilant = Phonetics.ContainsSibilant(onset);

            var wordStart = (int)MathF.Round(word.StartSample * scale);
            var wordEnd = (int)MathF.Round(word.EndSample * scale);

            contexts[i] = new SyllableContext
            {
                SampleRate = renderRate,
                GlobalSeed = seed,
                WordHash = wordHash,
                PhraseIndex = 0,
                WordIndexInPhrase = wordIndex,
                SyllableIndex = syllableIndex,
                WordSyllableCount = Math.Max(1, word.PulseCount),
                PhraseSyllableCount = pulses.Length,
                WordTextLength = Math.Max(1, wordText.Length),
                TempoScale = ExtractedTempoScale,
                EmotionValence = emotion.Valence,
                EmotionArousal = emotion.Arousal,
                Punctuation = punctuation,
                IsFirstSpokenPhrase = wordIndex == 0,
                PreviousWordHadComma = wordIndex > 0 &&
                    WordTrailingPause(words, pulses, wordIndex - 1) > PreviousCommaPauseSeconds,
                IsShout = pulse.PeakRms > ShoutPeakRms || IsShout(wordText),
                IsStressed = IsWordStress(pulses, word, i),
                HasVoicedOnset = hasVoiced,
                HasSibilantOnset = hasSibilant,
                IsHyphenBoundary = false,
                Nucleus = vowels.Count > 0 ? vowels[syllableIndex % vowels.Count] : '\0',
            };

            var frame = AcousticFeatureExtractor.ExtractFrame(
                audio.Samples,
                audio.SampleRate,
                pulse,
                profile,
                isSegmentStart);

            // The extractor measures in analysis samples and owns no cross-domain timing; the mapper
            // rescales positions to the render rate and snaps the duration so the encoder inverts it.
            frame.StartSample = (int)MathF.Round(pulse.StartSample * scale);
            frame.WordStartSample = wordStart;
            frame.WordEndSample = wordEnd;
            frame.StartsLegatoSegment = isSegmentStart;
            frame.HasVoicedOnset = hasVoiced;
            frame.Seed = MixSeed(seed, wordHash, (uint)syllableIndex);
            frame.GrowlFrequency = BaseGrowlFrequency;

            // With pitch curvature enabled a measured contour needs a midpoint. The geometric centre of
            // the two endpoints is the neutral 0.5 target the encoder asks for, so it neither pushes the
            // network toward a glide it did not hear nor fights the curve clamp.
            frame.MidFreq = profile.PitchCurve > 0f
                ? MathF.Sqrt(MathF.Max(1f, frame.StartFreq) * MathF.Max(1f, frame.EndFreq))
                : 0f;

            SnapDuration(ref frame, pulse.DurationSeconds, contexts[i], profile);

            if (isSegmentStart && wordIndex > 0)
            {
                var previous = pulses[word.FirstPulse - 1];
                var pauseSamples = Math.Max(0, (int)MathF.Round(previous.PauseAfterSeconds * renderRate));
                frame.PauseBeforeSamples = pauseSamples;
                frame.PauseBeforeSeconds = pauseSamples / (float)renderRate;
            }

            if (i == pulses.Length - 1)
            {
                SnapPause(ref frame, pulse.PauseAfterSeconds, punctuation, profile, renderRate);
            }

            if (syllableIndex < word.PulseCount - 1)
            {
                // Legato joins inside a word reuse the profile release; only the last pulse of a
                // word has a real tail the extractor can measure.
                frame.ReleaseTime = profile.ReleaseSeconds;
            }

            frames[i] = frame;
        }

        return (contexts, frames);
    }

    /// <summary>
    /// Snaps the frame duration to the whole-sample length the duration decoder would produce for the
    /// pulse's measured length, so the encoder inverts it exactly.
    /// </summary>
    private static void SnapDuration(
        ref SyllableFrame frame,
        float seconds,
        in SyllableContext context,
        DroidVoiceProfile profile)
    {
        var amount = Math.Clamp(
            ((seconds * profile.SpeedMultiplier / context.TempoScale) - SyllableTargetDecoder.MinDurationSeconds) /
            (SyllableTargetDecoder.MaxDurationSeconds - SyllableTargetDecoder.MinDurationSeconds),
            0f,
            1f);

        var decoded = SyllableTargetDecoder.DurationSeconds(amount, context.TempoScale, profile);
        var samples = Math.Max(1, (int)MathF.Round(decoded * context.SampleRate));

        frame.LengthSamples = samples;
        frame.DurationSeconds = samples / (float)context.SampleRate;
    }

    /// <summary>
    /// Snaps the phrase-final pause through the pause decoder. A last pulse with no measured tail
    /// keeps a zero pause instead of inventing the base pause a neutral target would decode to.
    /// </summary>
    private static void SnapPause(
        ref SyllableFrame frame,
        float seconds,
        PhrasePunctuation punctuation,
        DroidVoiceProfile profile,
        int renderRate)
    {
        if (seconds <= 0f)
        {
            frame.PauseAfterSamples = 0;
            frame.PauseAfterSeconds = 0f;
            return;
        }

        var basePause = punctuation == PhrasePunctuation.Ellipsis
            ? SyllableTargetDecoder.EllipsisPauseSeconds
            : SyllableTargetDecoder.PhrasePauseSeconds;

        var amount = Math.Clamp(
            ((seconds * profile.SpeedMultiplier / basePause) - SyllableTargetDecoder.PauseScaleMin) /
            (SyllableTargetDecoder.PauseScaleMax - SyllableTargetDecoder.PauseScaleMin),
            0f,
            1f);

        var decoded = SyllableTargetDecoder.PauseSeconds(amount, punctuation, profile);
        var samples = Math.Max(0, (int)MathF.Round(decoded * renderRate));

        frame.PauseAfterSamples = samples;
        frame.PauseAfterSeconds = samples / (float)renderRate;
    }

    /// <summary>
    /// Maps every pulse to the index of the word that owns it and to its ordinal inside that word. A
    /// pulse the word grouping skipped is attached to word zero so the two output arrays can never
    /// disagree in length.
    /// </summary>
    private static (int[] Word, int[] Syllable) MapPulsesToWords(int pulseCount, AcousticWord[] words)
    {
        var wordOfPulse = new int[pulseCount];
        var syllableOfPulse = new int[pulseCount];

        for (var wordIndex = 0; wordIndex < words.Length; wordIndex++)
        {
            var word = words[wordIndex];
            for (var k = 0; k < word.PulseCount; k++)
            {
                var pulse = word.FirstPulse + k;
                if ((uint)pulse < (uint)pulseCount)
                {
                    wordOfPulse[pulse] = wordIndex;
                    syllableOfPulse[pulse] = k;
                }
            }
        }

        return (wordOfPulse, syllableOfPulse);
    }

    /// <summary>Whether this pulse is the loudest one of its word, i.e. the stressed syllable.</summary>
    private static bool IsWordStress(AcousticPulse[] pulses, AcousticWord word, int pulseIndex)
    {
        var best = -1;
        var bestPeak = float.NegativeInfinity;

        for (var k = 0; k < word.PulseCount; k++)
        {
            var candidate = word.FirstPulse + k;
            if ((uint)candidate >= (uint)pulses.Length)
            {
                continue;
            }

            if (pulses[candidate].PeakRms > bestPeak)
            {
                bestPeak = pulses[candidate].PeakRms;
                best = candidate;
            }
        }

        return best == pulseIndex;
    }

    /// <summary>Trailing silence of the last pulse of a word, in seconds.</summary>
    private static float WordTrailingPause(AcousticWord[] words, AcousticPulse[] pulses, int wordIndex)
    {
        if ((uint)wordIndex >= (uint)words.Length)
        {
            return 0f;
        }

        var word = words[wordIndex];
        var last = word.FirstPulse + word.PulseCount - 1;
        return (uint)last < (uint)pulses.Length ? pulses[last].PauseAfterSeconds : 0f;
    }

    /// <summary>
    /// Reads the bracketed emotion and question tags from anywhere in a path. The first emotion tag in
    /// <see cref="EmotionTags"/> order wins so a path with several tags is still deterministic.
    /// </summary>
    private static (EmotionVector? Emotion, bool Question) ReadTags(string path)
    {
        EmotionVector? emotion = null;

        foreach (var tag in EmotionTags)
        {
            if (emotion is null && ContainsTag(path, tag))
            {
                emotion = TagEmotion(tag);
            }
        }

        return (emotion, ContainsTag(path, QuestionTag));
    }

    /// <summary>Removes every known bracket tag from a stem, so only the spoken words remain.</summary>
    private static string StripTags(string text)
    {
        foreach (var tag in EmotionTags)
        {
            text = RemoveAll(text, $"[{tag}]");
        }

        return RemoveAll(text, $"[{QuestionTag}]");
    }

    private static bool ContainsTag(string text, string tag) =>
        text.Contains($"[{tag}]", StringComparison.OrdinalIgnoreCase);

    private static string RemoveAll(string text, string token)
    {
        while (true)
        {
            var index = text.IndexOf(token, StringComparison.OrdinalIgnoreCase);
            if (index < 0)
            {
                return text;
            }

            text = text.Remove(index, token.Length);
        }
    }

    /// <summary>
    /// The explicit affect anchoring for the supported tags. These are convention, not measurement:
    /// they exist so an emotion tag has a stable meaning the lexicon does not have to carry.
    /// </summary>
    private static EmotionVector TagEmotion(string tag) => tag switch
    {
        "happy" => new(0.55f, 0.35f),
        "sad" => new(-0.55f, -0.20f),
        "alert" => new(-0.25f, 0.65f),
        "angry" => new(-0.60f, 0.55f),
        "calm" => new(0.30f, -0.35f),
        _ => EmotionVector.Neutral,
    };

    /// <summary>Picks the contour from the explicit tag or the trailing punctuation.</summary>
    private static PhrasePunctuation ResolvePunctuation(string text, bool forcedQuestion)
    {
        if (forcedQuestion)
        {
            return PhrasePunctuation.Question;
        }

        var trimmed = text.TrimEnd();
        if (trimmed.EndsWith("?", StringComparison.Ordinal))
        {
            return PhrasePunctuation.Question;
        }

        if (trimmed.EndsWith("!", StringComparison.Ordinal))
        {
            return PhrasePunctuation.Exclamation;
        }

        if (trimmed.EndsWith("...", StringComparison.Ordinal) ||
            trimmed.EndsWith("…", StringComparison.Ordinal))
        {
            return PhrasePunctuation.Ellipsis;
        }

        return PhrasePunctuation.Statement;
    }

    /// <summary>Splits the stem into letter/digit tokens, dropping punctuation and separators.</summary>
    private static List<string> Tokenize(string text)
    {
        var tokens = new List<string>();
        var start = -1;

        for (var i = 0; i <= text.Length; i++)
        {
            var isWord = i < text.Length && IsWordChar(text[i]);

            if (isWord && start < 0)
            {
                start = i;
                continue;
            }

            if (isWord || start < 0)
            {
                continue;
            }

            tokens.Add(text[start..i]);
            start = -1;
        }

        return tokens;
    }

    /// <summary>Keeps letters, digits and the two characters that glue a word together.</summary>
    private static bool IsWordChar(char value) =>
        char.IsLetterOrDigit(value) || value is '\'' or '-';

    /// <summary>Everything alphanumeric in a string, used as the fallback word.</summary>
    private static string LettersOnly(string text)
    {
        var buffer = new char[text.Length];
        var count = 0;

        for (var i = 0; i < text.Length; i++)
        {
            if (char.IsLetterOrDigit(text[i]))
            {
                buffer[count++] = text[i];
            }
        }

        return new string(buffer, 0, count);
    }

    /// <summary>Averages the lexicon vector of every word; falls back to neutral for silence.</summary>
    private static EmotionVector InferEmotion(List<string> tokens)
    {
        if (tokens.Count == 0)
        {
            return EmotionVector.Neutral;
        }

        var valence = 0f;
        var arousal = 0f;

        for (var i = 0; i < tokens.Count; i++)
        {
            var vector = EmotionLexicon.Lookup(tokens[i]);
            valence += vector.Valence;
            arousal += vector.Arousal;
        }

        return new EmotionVector(valence / tokens.Count, arousal / tokens.Count);
    }

    /// <summary>Whether a word is written entirely in uppercase letters.</summary>
    private static bool IsShout(string word)
    {
        var hasLetter = false;
        var hasUpper = false;
        var hasLower = false;

        for (var i = 0; i < word.Length; i++)
        {
            var value = word[i];
            if (!char.IsLetter(value))
            {
                continue;
            }

            hasLetter = true;
            hasUpper |= char.IsUpper(value);
            hasLower |= char.IsLower(value);
        }

        return hasLetter && hasUpper && !hasLower;
    }

    /// <summary>
    /// Splits a word into its leading consonant onset and every vowel it contains. The onset decides
    /// the sibilant / voiced flags; the vowel list is cycled over the word's pulses to give each
    /// syllable a nucleus without re-running the parser.
    /// </summary>
    private static (string Onset, List<char> Vowels) SplitWord(string word)
    {
        var vowels = new List<char>();
        var onsetEnd = 0;
        var inOnset = true;

        for (var i = 0; i < word.Length; i++)
        {
            var value = word[i];
            if (!char.IsLetter(value) && value is not '\'')
            {
                continue;
            }

            if (Phonetics.IsVowel(value))
            {
                inOnset = false;
                vowels.Add(value);
            }
            else if (inOnset)
            {
                onsetEnd = i + 1;
            }
        }

        return (onsetEnd > 0 ? word[..onsetEnd] : string.Empty, vowels);
    }

    /// <summary>FNV-1a over the UTF-16 characters of <paramref name="text"/>.</summary>
    internal static uint Fnv1a32(string text)
    {
        var hash = 2166136261u;

        foreach (var character in text)
        {
            hash ^= character;
            hash *= 16777619u;
        }

        return hash;
    }

    /// <summary>Continues an FNV hash with a 32-bit word.</summary>
    internal static uint FnvContinue(uint hash, uint value)
    {
        hash ^= value & 0xFFu;
        hash *= 16777619u;
        hash ^= (value >> 8) & 0xFFu;
        hash *= 16777619u;
        hash ^= (value >> 16) & 0xFFu;
        hash *= 16777619u;
        hash ^= value >> 24;
        hash *= 16777619u;
        return hash;
    }

    /// <summary>Derives a deterministic per-syllable noise seed from the run, word and syllable.</summary>
    internal static uint MixSeed(uint globalSeed, uint wordHash, uint syllableIndex)
    {
        var value = FnvContinue(globalSeed ^ 0x9E3779B9u, wordHash);
        return FnvContinue(value, syllableIndex);
    }
}
