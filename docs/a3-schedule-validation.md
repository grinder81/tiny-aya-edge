# A3 same-schedule validation v2 preregistration

Frozen after the failed v1 bounded check and before v2 scored requests on
22 October 2026. Retain v1 raw cold-logit oracle and seven HTTP records. V1
plain-letter four candidates exactly bridged A2. Chat token strings/counts
matched, but continuation errors up to 0.02854 nats and supplied-prompt errors
up to 0.18238 exceeded the original numerical bounds. Generated suffix scores
also differed. These observations do not prove an indexing defect or its cause.

V2 tests the hypothesis that different prefix reuse and forward partitioning
explain those differences. On the same fixed 48-context selection, use the exact
supported wrapper create_completion call and candidate order as HTTP, starting
with a fresh model in each process. Read raw model._scores immediately after
each completion and independently compute full-vocabulary float64 log-softmax
for the exact locked IDs. Never use returned echo probabilities as the reference.
This exercises the runtime generator but keeps the echo and upstream parser
probability reference independent. Assert actual evaluated input IDs exactly.

The unchanged upstream HTTP client then repeats the identical order on a fresh
server. All v1 token/count/BOS/suffix, per-token 0.0002-nat, summed
max(0.001,k*0.0002)-nat, parser 1e-8 and A2 bridge 0.001-nat bounds remain.
No numeric tolerance is relaxed. Compare and preserve cold versus reused-prefix
oracle differences separately; passing v2 cannot establish partition invariance.
No custom adapter/scorer or canonical contract change. Full inference remains
gated on independent review, runtime implications, and a runner PR.
