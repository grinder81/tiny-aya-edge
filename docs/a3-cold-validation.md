# A3 cold-prefill runtime v3 preregistration

Frozen before v3 scored requests on 22 October 2026. The v1 correctness failure
and its original A2 numerical gate remain preserved. The seven-candidate v2
same-schedule explanation matched supplied echoes within 9.3e-7 nats,
continuation sums within 6.8e-8, and suffix scores within 1.9e-7. V2 is incomplete
and does not pass the full validation gate. Fresh-server request permutations
changed plain A by about 0.00968 nats and chat A by about 0.00652 nats versus cold
forwards, demonstrating request-order dependence under the previous runtime.

V3 uses a separate cloned wrapper environment and a narrow runtime patch:
echo=True with logprobs not None resets state before prefill and bypasses saved
cache restoration. This includes logprobs=0; nonscoring generation is unchanged.
The original environment is preserved. The installed-source CPU test covers 12
echo/logprobs/cache combinations and ordering before generate; GPU tests remain
the decisive correctness evidence. Record actual interpreter/prefix/import path.

Repeat the v1 full frozen 48-context /192-candidate independent cold-logit and
unchanged upstream HTTP comparisons in both arms. All original strict numerical,
ID, token-string, BOS, suffix and parser bounds remain. Then send the nine fixed
index-zero/rotation-zero language-by-variant contexts (all four candidates) in
reverse row/candidate order through the warmed server and compare against the
same cold logits. This order check is a separate 36-candidate artifact; it cannot
substitute for a complete primary gate. Report every argmax/margin and identify
tie-sensitive examples without choosing by accuracy. Main requests start on a
fresh server; order checks intentionally retain previous history.

The legacy A2 bridge still records every candidate difference, prediction and
margin, including all absolute differences exceeding its original 0.001-nat
bound. Its failures are explicit legacy-runtime differences, separate from the
new cold-runtime probability correctness gate. No equivalence to historical A2
is assumed; historical scores are never relabeled. This changed runtime has a
fresh identity. Canonical-contract and affected-baseline decisions remain user
gates before the sweep. No custom adapter or scorer is introduced.
