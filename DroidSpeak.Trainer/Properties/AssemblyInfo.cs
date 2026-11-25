using System.Runtime.CompilerServices;

// The trainer's math (TrainableMlp, TrainingRun, DistillationCorpus) is internal because it is a
// developer tool, not part of the library's contract. The test suite exercises it directly rather
// than re-implementing the backpropagation it is meant to verify.
[assembly: InternalsVisibleTo("DroidSpeak.Tests")]
