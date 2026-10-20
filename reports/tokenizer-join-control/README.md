# Join control boundary correction

Expanded local-Q4 preflight found one mismatch: the year 2007 followed by U+200C in frozen Bangla group 105. The original tokenizer emits the digit groups 2 and 007; the earlier local patch skipped that grouping because it treated U+200C and U+200D as word continuations.

Direct tests with the original tokenizers Split/Regex implementation confirm both join controls are boundaries here. Combining marks and letter suffixes still prevent the split. Removed only the two join-control exceptions from the Tiny-Aya-only branch; no model or scoring changes.

Fifteen standalone C++ cases, including Bengali and Devanagari digits and both join controls, pass; AFMoE preservation passes. The rebuilt shared library then matches all 900 frozen contexts and 3,600 candidate token sequences, including exact candidate decoding. validation.json records the new library and sample-lock hashes.

The previous 150-question results retain their old runtime provenance. The expanded local-only run restarts all 300 per language under a new manifest rather than relabeling or reusing those older scores. This does not claim universal Unicode regex equivalence.
