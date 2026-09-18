# Local runs (not a Kaggle submission)

Two official train samples were downloaded as partial volumes (selected timepoints only). No notebook was submitted to Kaggle.

| experiment | config | local score | edge Jaccard | node recall | missed | ID switches | broken tracks | runtime s | peak RAM MB |
|---|---|---|---|---|---|---|---|---|---|
| baseline | `configs/baseline.yaml` greedy NN | 0.382 | 0.390 | 0.909 | 4 | 8 | 13 | 10.0 | 335 |
| hungarian | `configs/hungarian.yaml` | 0.387 | 0.395 | 0.909 | 4 | 7 | 12 | 10.1 | 355 |

Metric label: local reimplementation of the published edge Jaccard + node-count adjustment, with a division proxy. These two samples contain **zero GT divisions**, so the division term is undefined and dropped.

Linking using GT centroids as detections recovered every GT edge on the available frames (diagnostic edge Jaccard 1.0). The remaining errors are mostly detection localization and identity switches among dense nuclei, not the absence of a linker.

`estimated_number_of_nodes` is scaled by the fraction of annotated timepoints kept so partial-volume adjustment is not dominated by the full-video node prior.
