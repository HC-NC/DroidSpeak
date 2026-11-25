namespace DroidSpeak.DSP;

/// <summary>
/// Continuous oscillator and filter state for one voice. The offline renderer keeps it on the
/// stack for the whole utterance, while <see cref="DroidSpeechStream"/> stores it in a field so
/// chunk boundaries are invisible.
/// </summary>
internal struct VoiceState
{
    /// <summary>Carrier phase in radians.</summary>
    public double CarrierPhase;

    /// <summary>First (harmonic) FM modulator phase in radians.</summary>
    public double ModulatorPhase;

    /// <summary>Second (inharmonic) FM modulator phase in radians.</summary>
    public double ModulatorPhase2;

    /// <summary>Ring-modulation carrier phase in radians.</summary>
    public double RingPhase;

    /// <summary>Smoothed center frequency of the first formant.</summary>
    public float Formant1Frequency;

    /// <summary>Smoothed center frequency of the second formant.</summary>
    public float Formant2Frequency;

    /// <summary>Whether the formant bank already has a valid center frequency.</summary>
    public bool FormantsInitialized;

    /// <summary>First formant band-pass.</summary>
    public BiquadBandPass Formant1;

    /// <summary>Second formant band-pass.</summary>
    public BiquadBandPass Formant2;

    /// <summary>Fixed high formant band-pass (~2.5 kHz) that adds air.</summary>
    public BiquadBandPass Formant3;

    /// <summary>Resets every oscillator phase and filter.</summary>
    public void Reset()
    {
        CarrierPhase = 0d;
        ModulatorPhase = 0d;
        ModulatorPhase2 = 0d;
        RingPhase = 0d;
        ResetFormants();
    }

    /// <summary>Clears the formant bank and its smoothing state.</summary>
    public void ResetFormants()
    {
        Formant1Frequency = 0f;
        Formant2Frequency = 0f;
        FormantsInitialized = false;
        Formant1.Reset();
        Formant2.Reset();
        Formant3.Reset();
    }
}
