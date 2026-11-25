using DroidSpeak.DSP;
using DroidSpeak.Neural;
using DroidSpeak.Prosody;

namespace DroidSpeak.Demo;

/// <summary>
/// Mutable state of one interactive demo session: the active voice profile, the active prosody
/// predictor and the model path behind it.
/// <para>
/// <see cref="DroidSynthesizer"/> is immutable, so a session rebuilds it lazily through
/// <see cref="Synth"/> whenever <see cref="CurrentProfile"/> or <see cref="CurrentPredictor"/>
/// changes. That is what lets <c>/voice</c> and <c>/brain</c> swap engines without touching Core.
/// </para>
/// </summary>
internal sealed class VoiceSession
{
    /// <summary>Output sample rate shared by every synthesizer the session builds.</summary>
    public const int SampleRate = DroidSynthesizer.DefaultSampleRate;

    private DroidSynthesizer? _synth;
    private DroidVoiceProfile _profile;
    private string _voiceName;

    /// <summary>
    /// Creates a session.
    /// </summary>
    /// <param name="profile">Initial voice profile; normalized on entry.</param>
    /// <param name="voiceName">Preset or file stem used in the prompt.</param>
    /// <param name="predictor">
    /// Neural predictor loaded at startup, or <c>null</c> for rule-based prosody.
    /// </param>
    /// <param name="brainPath">Model path that produced <paramref name="predictor"/>, if any.</param>
    public VoiceSession(
        DroidVoiceProfile? profile,
        string voiceName,
        NeuralProsodyPredictor? predictor = null,
        string? brainPath = null)
    {
        _profile = (profile ?? DroidVoiceProfile.R2).Normalized();
        _voiceName = string.IsNullOrWhiteSpace(voiceName) ? "r2" : voiceName;
        CurrentPredictor = (IProsodyPredictor?)predictor ?? RuleBasedProsodyPredictor.Shared;
        BrainPath = predictor is null ? null : brainPath;
    }

    /// <summary>The active voice profile. Setting it invalidates the cached synthesizer.</summary>
    public DroidVoiceProfile CurrentProfile
    {
        get => _profile;
        set
        {
            _profile = value.Normalized();
            _synth = null;
        }
    }

    /// <summary>
    /// The active prosody engine; <see cref="RuleBasedProsodyPredictor.Shared"/> unless a model was
    /// loaded.
    /// </summary>
    public IProsodyPredictor CurrentPredictor { get; private set; }

    /// <summary>Full path of the active <c>.dsbrain</c> model, or <c>null</c> in rule-based mode.</summary>
    public string? BrainPath { get; private set; }

    /// <summary>
    /// Preset or profile name without the brain suffix. Used to build
    /// <see cref="VoiceLabel"/> after switching voices.
    /// </summary>
    public string VoiceName
    {
        get => _voiceName;
        set => _voiceName = string.IsNullOrWhiteSpace(value) ? "r2" : value;
    }

    /// <summary>The neural predictor when one is loaded, otherwise <c>null</c>.</summary>
    public NeuralProsodyPredictor? NeuralPredictor => CurrentPredictor as NeuralProsodyPredictor;

    /// <summary>Whether a <c>.dsbrain</c> model is currently driving prosody.</summary>
    public bool IsNeural => NeuralPredictor is not null;

    /// <summary>
    /// Prompt label: the plain voice name in rule-based mode, or
    /// <c>name+brain(model)</c> when a neural model is active, e.g. <c>grumpy+brain(tiny)</c>.
    /// </summary>
    public string VoiceLabel =>
        IsNeural && BrainPath is not null
            ? $"{_voiceName}+brain({Path.GetFileNameWithoutExtension(BrainPath)})"
            : _voiceName;

    /// <summary>
    /// The synthesizer for the current profile and predictor. Rebuilt only after a change; callers
    /// may hold onto the returned instance for the duration of a single command.
    /// </summary>
    public DroidSynthesizer Synth =>
        _synth ??= new DroidSynthesizer(
            SampleRate,
            profile: _profile,
            predictor: CurrentPredictor);

    /// <summary>Switches to neural prosody with an already validated predictor.</summary>
    /// <param name="predictor">Predictor to use; never <c>null</c>.</param>
    /// <param name="path">Path of the model, used for status output and the prompt label.</param>
    public void UseNeural(NeuralProsodyPredictor predictor, string path)
    {
        ArgumentNullException.ThrowIfNull(predictor);
        CurrentPredictor = predictor;
        BrainPath = path;
        _synth = null;
    }

    /// <summary>Switches back to the built-in rule engine.</summary>
    public void UseRuleBased()
    {
        CurrentPredictor = RuleBasedProsodyPredictor.Shared;
        BrainPath = null;
        _synth = null;
    }
}