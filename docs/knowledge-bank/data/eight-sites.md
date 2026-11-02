# Exact eight pilot sites

Shapes below use the PyTorch output by input convention. Formats are actual stored tensor types in each reference export.

| GGUF tensor | Human layer | Shape | Parameters | Local Q4 type | Operating Q3 type |
| --- | ---: | --- | ---: | --- | --- |
| `token_embd.weight` | Shared interface | 262144 x 2048 | 536,870,912 | Q6_K | Q6_K |
| `blk.0.attn_v.weight` | 1 | 512 x 2048 | 1,048,576 | Q6_K | Q5_K |
| `blk.3.attn_v.weight` | 4 | 512 x 2048 | 1,048,576 | Q6_K | Q4_K |
| `blk.0.ffn_down.weight` | 1 | 2048 x 11008 | 22,544,384 | Q6_K | Q5_K |
| `blk.17.ffn_down.weight` | 18 | 2048 x 11008 | 22,544,384 | Q4_K | Q4_K |
| `blk.35.ffn_down.weight` | 36 | 2048 x 11008 | 22,544,384 | Q6_K | Q4_K |
| `blk.17.ffn_gate.weight` | 18 | 11008 x 2048 | 22,544,384 | Q4_K | Q3_K |
| `blk.17.ffn_up.weight` | 18 | 11008 x 2048 | 22,544,384 | Q4_K | Q3_K |

These eight stored matrices contain 651,689,984 parameters (19.458% of the model). This fraction is dominated by the embedding and is not representative coverage of behavior.

| Prespecified pair | Predicted bytes per complete orientation | Eligible |
| --- | ---: | --- |
| `blk.0.attn_v.weight` / `blk.3.attn_v.weight` | 1,799,224,832 | True |
| `blk.0.ffn_down.weight` / `blk.17.ffn_down.weight` | 1,788,644,864 | True |
| `blk.17.ffn_down.weight` / `blk.35.ffn_down.weight` | 1,791,462,912 | True |
| `blk.17.ffn_gate.weight` / `blk.17.ffn_up.weight` | 1,797,451,264 | True |
