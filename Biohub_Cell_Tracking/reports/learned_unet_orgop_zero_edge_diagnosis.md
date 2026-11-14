# learned_unet_orgop_v1 zero-edge diagnosis

Same family as draft PR #7. Do **not** merge. Do **not** retune
`learned_unet_orgop_v1` hash `250c5bfacd4a8198`. Do **not** overwrite
`mahdimzafari/biohub-learned-unet-orgop-v1` v1. Frozen `6bba` was not
opened. No 199-video weights.

## Reproduced failure

Ignored orgop graphs `44b6_12dfb391` / `44b6_267148e4`: **62480 / 28672
nodes**, **0 / 0** softmax>0.5 candidate edges, **0 / 0** ILP edges.
Official 44b6 combined remains **0.0 vs infer 0.655751**.

## Checkpoint inspection (ignored artifacts)

Both `learned_unet_v1_best.pt` and `learned_unet_v1_last.pt` load with
**strict=True**, **0 missing / 0 unexpected** keys, **60 transformer
tensors**, config hash `4c38035d6657f2ad`. Tensor-wise **best == last**.
Orgop and infer both fell back to `learned_unet_v1_last.pt`; that is the
same epoch-2 checkpoint as best. Splits contain no `6bba`.

Relative L2 vs seed-0 init (the training seed):

| group | n_params | rel to init |
|---|---|---|
| UNet | 1,497,600 | **6.34** |
| detect_head | 33 | 0.40 |
| transformer | 580,353 | **0.11** |
| pair_mlp | 41,601 | 0.25 |

Training **did** include `edge_loss` (softmax dim=0, GT-node index). The
UNet moved; the transformer stayed near initialization. Random 400×80
pairs from the locked checkpoint: `prob_max=0.012`, **0** pairs > 0.5.

## Infer vs orgop path

Locked infer uses the same weights but **greedy 12 µm** and never reads
transformer logits. Orgop uses organizer defaults (0.99 / softmax 0.5 /
ILP). `load_state_dict` is strict in both. Softmax dimension matches
training (`dim=0` over sources). There is **no** silent partial load and
**no** wrong-checkpoint defect.

Dense-source softmax>0.5 is structurally rare (`n_src=400`, peak logit
3.0 still `prob_max<0.5`). That is the same empty-edge mode as rejected
`learned_unet_link_v1`. It is not fixed by re-running orgop defaults.

## Decision

`no_implementation_defect_transformer_near_init`.

Do **not** create `learned_unet_orgop_fix_v1`. Do **not** lower the 0.5
threshold. Do **not** open `6bba`.

Follow-up canary: `learned_unet_edge_train_canary_v1` hash
`1a3d524e4d922701`.
