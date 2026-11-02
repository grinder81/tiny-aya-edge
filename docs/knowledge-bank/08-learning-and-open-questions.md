# Study questions and research notebook

## A progressive study sequence

| Session | Focus | A useful output |
| --- | --- | --- |
| 1 | Follow one token from lookup to next-token probabilities | Redraw the complete graph, including the shared output matrix |
| 2 | Read one parallel decoder block | Explain which values attention and MLP receive and where the residual is added |
| 3 | Work through Q/K/V/O and GQA | Derive the four matrix shapes and the number of cached elements |
| 4 | Work through gate/up/down | Derive the MLP shapes and prove one valid channel transformation |
| 5 | Read actual format assignments | Explain why a Q3_K_M file can contain a six-bit embedding |
| 6 | Audit the eight-site choices | Identify each confound and propose a control without assuming a positive result |
| 7 | Compare measurement endpoints | Distinguish weight error, activation error, KL, correctness and generation quality |
| 8 | Write one intervention decision record | State what evidence would justify spending more GPU time |

These are discussion units, not a fixed calendar or an instruction to resume experiments. Revisit a session whenever a new source or result challenges the explanation.

## Worked reasoning questions

### Why is the model called 3.35B when the embedding alone is over half a billion?

The exact stored total is 3,349,227,520. One shared embedding contributes 536,870,912, the 36 MLP groups contribute 2,434,793,472, attention contributes 377,487,360 and normalization contributes 75,776. Their sum is the exact total. Count the embedding once in storage even though it participates twice in the graph. The atlas recomputes this accounting from every stored tensor.

### Does global attention look into the future?

No. Global here means all permitted prefix positions rather than a fixed local window. Causal masking still excludes future positions. No explicit RoPE in global blocks is a separate fact from which positions the mask permits.

### Why does the middle MLP not directly consume this block's attention output?

The model was built with parallel branches. Both receive the same normalized incoming residual; their results are added afterward. Attention from previous blocks is already present in that incoming state. It would be wrong to conclude that the MLP has no contextual information.

### Can we call one high-scoring tensor a Bangla tensor?

No. A larger damage measurement on Bangla text could reflect shared features, script, domain, tokenization, sample variation or nonlinear downstream amplification. A label about language-selective function would need interventions and controls that distinguish those explanations. Shared multilingual parameters do not partition neatly by language.

### Is a six-bit tensor always worth more than a three-bit tensor?

More precision usually offers a better approximation of that tensor under a given quantizer, but the final output and task effects depend on the model and data. The useful question for allocation is the additional benefit for the additional bytes, subject to the complete-model budget and interactions. Even a nonnegative isolated KL reduction should be measured rather than assumed for all formats and inputs.

### Why not simply rank every matrix by its raw error?

Raw summed error grows with matrix size, ignores activation directions and can miss nonlinear routing or output effects. A ranking based on native output KL is closer to the behavior of interest but costs more and still depends on background precision and sample. A cheap proxy needs validation, and its ranking quality and magnitude accuracy are separate questions.

### Does an unchanged tensor ensure its activations stay unchanged?

No. If an earlier component changes the input trajectory, later unchanged weights receive different activations. “Only one tensor changed” identifies the intervention, not the extent of its downstream consequences. Within the same Tiny Aya block, the parallel branches share the same incoming normalized state; later blocks receive the changed combined residual.

### Why not choose the pair with the biggest observed difference?

Searching for the largest noisy effect can exaggerate apparent differences. The frozen pilot chooses the first eligible repeat-stable pair in a fixed order and tests it on independent sources. That limits one kind of selection, but the tiny screen still does not establish a global optimum or a practical task effect.

### Why is the embedding diagnostic absent from the four confirmation pairs?

It has no same-shape partner in the selected subset. Swapping precision with one small MLP or V matrix would not preserve bytes. The restricted pilot deliberately avoids a more complex many-to-one compensation rule. Studying the embedding's best allocation would need a separately specified feasible trade-off.

### If balancing preserves the exact function, how can it change a compressed result?

The exact scaling cancels before quantization. Quantizing the two transformed matrices independently introduces errors that generally do not cancel. That can change the compressed function even though the uncompressed functions are mathematically identical. Whether it helps must be measured with a stable native quantizer and an unquantized parity check.

### Why did storage fail despite the model being small enough for the GPU?

GPU feasibility and host storage are separate constraints. Research stored multiple full diagnostic models and roughly 185 GB of reference distributions inside a growing WSL disk on C:. Virtual Linux capacity did not guarantee physical host space. The incident was an execution failure, not evidence about model architecture or language sensitivity. The model/recovery work remains paused.

## Questions that require new evidence

| Open question | What would answer it | What would not answer it |
| --- | --- | --- |
| Which omitted Q/K/O matrices matter? | Controlled native interventions with relevant depth coverage | The current V-only selection rationale |
| Does sensitivity repeat across domains? | Source-disjoint confirmation in more than one domain | More token windows from the same talk |
| Are language differences semantic or script-related? | Content, length, orthography and code-mixing controls | A language label on a heatmap |
| Do isolated rankings transfer to a compressed background? | Whole-map measurement and controlled restoration tests | Adding isolated costs alone |
| Which embedding use dominates damage? | Carefully matched side-specific interventions and tied-weight checks | A single shared-matrix score |
| Can a cheap proxy replace native scans? | Ranking/magnitude validation with independent samples and uncertainty | Correlation on a few chosen sites |
| Does a fidelity improvement matter to users? | Task and genuine generation outcomes at matched bytes | Lower average KL alone |
| Does a transformation run efficiently on a phone? | Compatible export and actual controlled device measurements | A desktop Python-hook result |

## Notebook template

Create a dated entry under this knowledge bank only when there is a useful new source, derivation or result. Use this structure:

```text
Question:
Model and implementation revision:
Statement category: structural fact / derivation / observation / hypothesis
Claim:
Source or reproducible calculation:
Conditions and competing explanations:
Intervention and held-fixed variables, if applicable:
Result and uncertainty, if measured:
Effect on the next decision:
What remains unresolved:
```

Keep negative findings and contradictions. If a new result invalidates an earlier explanation, update the explanation and preserve the correction in a short dated note. Avoid accumulating multiple conflicting “latest” summaries.

## Knowledge boundaries to preserve

An inspected source function is stronger evidence for the computation graph than a generic Transformer illustration. A tensor inventory is stronger evidence for actual types and bytes than a format label. A complete measured candidate is stronger evidence for its behavior than an additive prediction. None of these automatically establishes a causal semantic role for a neuron or a reliable deployment benefit.

The learning objective is to make each choice explainable: what the component computes, why the proposed change could help, what it costs, and what observation would show that the explanation was wrong.
