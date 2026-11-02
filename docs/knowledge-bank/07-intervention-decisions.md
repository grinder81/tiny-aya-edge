# Intervention choices and prioritization

## Begin with the change and its intended benefit

An architecture diagram identifies places where an intervention is possible. It does not identify which change is useful. A decision should connect a specific trigger, intervention, measurement, cost and criterion that could disconfirm the idea.

For example: “At the frozen operating model size, English-derived and Bangla-derived evidence may prefer different precision placements. Compare their actual equal-byte models on independent sources.” That is more testable than “protect the language layers.” It also exposes the possibility that the two languages prefer the same placement, or that sampling variation dominates both.

## Mechanisms change different resources

| Mechanism | What changes | Potential benefit | Evidence required before a larger experiment |
| --- | --- | --- | --- |
| Per-tensor precision allocation | Native stored types, same matrix shapes | Spend a fixed byte budget where it preserves more capability | Actual equal-byte exports, stable choices and complete-model outcomes |
| Quantizer calibration | Text-derived statistics guide numerical approximation | Better error placement at a fixed format | Controlled calibration language/amount, fixed exporter and separate evaluation |
| Equivalent MLP balancing | Compensated up/down channel scales | Better quantization ranges without changing the exact unquantized function | Algebra, finite-precision parity, fixed-format/equal-byte improvement |
| Equivalent channel permutation | Consistent gate/up row and down-column reorder | Change grouping seen by the actual quantizer | Quantizer-layout sensitivity, parity and independent outcomes |
| Branch restoration or activation patching | Replace selected computation/state during diagnosis | Locate potentially recoverable damage | Defined background trajectory, aligned tokens, matched controls and rescue endpoint |
| Learned correction or adapter | Add/update a small set of parameters | Approximate a useful repair without retraining everything | Independent training/selection data, full artifact bytes and runtime support |
| Branch/head/layer pruning | Actually remove operations or stored tensors | Smaller model or less executed work | Export graph correctness, recovered quality and real runtime measurements |
| Low-rank factorization | Replace a matrix with two smaller matrices | Reduce stored parameters or arithmetic | Chosen ranks, fitting/recovery cost and backend kernels; changed shapes are substantive |
| Attention replacement | Substitute a recurrent/local/other mixer | Reduce history-dependent state or execution | Local matching, complete-model recovery and compatible efficient runtime |
| KV cache policy | Change retention or cache dtype | Lower context-dependent RAM/latency | Long-context task retention, fresh compatible state and measured memory |
| Tokenizer/vocabulary change | Change segmentation, IDs or output candidates | Potentially fewer decoding steps or vocabulary bytes | Compatible embeddings, training/recovery and multilingual/generation evaluation |

These are alternatives or separately controlled factors. Combining them in the first trial would make a positive or negative result harder to attribute. The existing protocol concerns the first row; the remaining rows are a knowledge map, not new execution authorization.

## Use evidence gates before a numerical priority score

**Compatibility gate.** Can the intervention preserve the intended graph, token IDs, shape interpretation, shared weights and cache semantics? Verify an unchanged implementation path first. For a function-preserving transform, establish unquantized equivalence before assessing quantization.

**Measurement gate.** Can the proposed endpoint distinguish the claimed effect? A smaller weight MSE does not establish better task correctness. A teacher-patching rescue requires the teacher and is not itself a deployable repair. A source-cluster confidence interval does not measure calibration-sample instability.

**Resource gate.** Can we complete a bounded trial with real host storage, RAM/VRAM and runtime headroom? Model bytes, diagnostic copies and saved distributions are separate demands. The storage incident makes host-volume capacity an explicit requirement in addition to virtual-disk free space. No recovery action is authorized by this document.

**Confirmation gate.** Can we define an independent sample and freeze the choice before seeing it? If selection and confirmation share sources or repeated peeking, a stable-looking effect may be selection noise. Reserve the final holdout for its existing once-only role.

Only after these gates should we compare expected benefit, uncertainty reduction, device feasibility and engineering cost. A weighted score can help organize a discussion, but unsupported numbers would create false precision. Record the reasons rather than inventing percentages for “chance of success.”

## What should influence priority

| Consideration | Strong evidence | Weak substitute |
| --- | --- | --- |
| Decision value | Resolves a specific uncertainty that changes the next study | Produces another visualization without changing a decision |
| Language relevance | Repeated language preferences exceed within-language construction variation | One favorable Bangla sample |
| Practical impact | Better task/generation quality at matched shipped bytes and acceptable English change | Lower local MSE or KL alone |
| Cost benefit | Actual artifact and device measurements under a fixed workload | Smaller parameter count or theoretical FLOPs alone |
| Interpretability | Intervention changes one defined mechanism with useful controls | Broad simultaneous modifications |
| Portability | Supported actual formats/graph in the deployment runtime | A Python-only hook with no export plan |
| Reproducibility | Source/sample/hash-bound implementation and independent confirmation | Mutable defaults and unnamed calibration text |

Large parameter share creates a possible storage payoff, not proof of sensitivity. A small high-impact matrix can be worth preserving despite modest byte savings. Conversely, a large matrix may look important by raw summed error simply because it has more elements. Normalize quantities appropriately and keep the actual allocation objective explicit.

## Current decision records

### Precision placement

The existing eight-site pilot asks whether restricted English/Bangla preferences repeat and survive actual complete-map export. A positive screen would justify considering a larger task/generation study; it would not establish final allocation success. No comparative result has been collected. Learning about architecture can reveal reasons to propose a separate amended design before execution resumes, but the existing frozen files must remain available.

### MLP channel balancing

The algebra in [chapter 3](03-mlp-and-residual.md) supplies a concrete alternative with unchanged shapes and a candidate export path. Its first uncertainty is finite-precision equivalence, followed by whether different language-derived scales change real quantization usefully. It should retain identity and English mirrors, control calibration and compare at actual equal bytes. AWQ supplies relevant prior art, so novelty cannot be claimed for the general idea of activation-aware equivalent scaling. [AWQ](https://arxiv.org/abs/2306.00978).

### Behavioral restoration

A restoration experiment can ask whether a damaged model has recoverable error at a particular branch. Restoring BF16 computation on the candidate's current hidden input is different from injecting a teacher hidden state that replaces accumulated upstream error. [Activation patching methodology](https://arxiv.org/abs/2309.16042) is a useful follow-up reading; a mechanism-specific implementation and endpoint must be reviewed before use.

### Larger architecture changes

Pruning, low rank, new mixers and tokenizer surgery can address different device constraints. Their recovery training, changed shapes or runtime support make them separate projects. Tiny Aya's parallel branches, LayerNorm, tied matrix and local-only RoPE are compatibility constraints, not reasons to reject all such ideas. Each needs a derivation and a bounded feasibility question.

## A reusable decision record

For each proposed intervention, write a short record with:

1. The exact component, graph boundary and model revision.
2. The computational hypothesis and competing explanations.
3. What changes and what is held fixed, including background precision and cache state.
4. The primary endpoint and its relationship to the practical deployment objective.
5. Data amount, source separation, language/content matching and independent confirmation.
6. Actual byte trade-offs, runtime support and a measured resource bound.
7. A criterion for promising, inconclusive and implementation failure.
8. The finding that would lower this idea's priority or make us stop.

If we cannot fill these fields, the next useful step is learning, source inspection or a small compatibility check. It need not be another model run. The current user instruction chooses that learning stage.
