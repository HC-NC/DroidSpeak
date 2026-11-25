namespace DroidSpeak.Prosody;

/// <summary>
/// The two kinds of mechanical micro-chatter the droid can emit inside word gaps.
/// </summary>
public enum ChatterKind
{
    /// <summary>A tiny stepped arpeggio, like a relay bank cycling.</summary>
    Arpeggio,

    /// <summary>A mechanical relay click with a sharp downward frequency drop.</summary>
    RelayClick,
}
