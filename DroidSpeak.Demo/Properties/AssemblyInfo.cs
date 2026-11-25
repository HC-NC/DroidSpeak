using System.Runtime.CompilerServices;

// The demo exposes a few internal parsing helpers (DemoOptions, DemoCommandLine.ParseArguments,
// TryLoadBrain) so they can be covered by the test project without making them public API.
[assembly: InternalsVisibleTo("DroidSpeak.Tests")]