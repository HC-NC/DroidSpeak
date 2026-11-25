using DroidSpeak.Trainer;

// Offline trainer for DroidSpeak prosody networks. It exists so a voice can be distilled from the
// rule-based engine into a small .dsbrain file on a developer machine, without shipping any training
// code or any dependency to the synthesizer.
if (args.Length == 0)
{
    PrintUsage();
    return 0;
}

switch (args[0].ToLowerInvariant())
{
    case "selftest":
        // Proves the backpropagation and Adam path works before any corpus is involved.
        return XorProbe.Run() ? 0 : 1;

    case "-h":
    case "--help":
    case "help":
        PrintUsage();
        return 0;

    default:
        Console.Error.WriteLine($"error: unknown command '{args[0]}'.");
        Console.Error.WriteLine();
        PrintUsage();
        return 1;
}

static void PrintUsage()
{
    Console.WriteLine("""
        DroidSpeak.Trainer - offline prosody network training

        usage: dotnet run --project DroidSpeak.Trainer -- <command> [options]

        commands:
          selftest            train a tiny XOR network to prove backprop and Adam work

        further commands: distill (build a corpus and train a .dsbrain) and eval (score one).
        """);
}
