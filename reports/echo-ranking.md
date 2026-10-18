# Local echoed-probability ranking optimization

One real four-bit prompt took 32.879 seconds under cProfile. Python's full
vocabulary sorting consumed 23.561 seconds; model eval consumed 0.456 seconds.
See echo-ranking/before.txt. This is evaluation response construction overhead,
not evidence that the model generates at that rate.

The nonstreaming echoed-probability path now uses NumPy lexsort to preserve the
same descending (score, token ID) tuple ordering, including ties, and materializes
only the requested top entries. Token likelihoods and the softmax computation
are unchanged. This is a local llama-cpp-python patch on revision
1652066e0af45f2313b339670ef9555e8a54e545, layered with the decoding-buffer patch.
It is not a custom evaluation adapter or an upstream release.

Twelve exact ordering comparisons pass, including tied finite scores and a
262144-entry array. All eight complete synthetic Q4 response records match the
pre-optimization records exactly. Real pilot replay validation is recorded
separately before expanding evaluation. No general model-speed claim is made.

Real Q4 replay completed all 40 candidates in 163 seconds versus 1680 seconds previously. All candidate scores and greedy flags match exactly, with identical document hashes. This roughly tenfold reduction is evaluation overhead improvement, not generation throughput. The original and replay result files remain under the Stage 1 pilot evidence directory.
