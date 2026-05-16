# 2026-05-16 Hit Probability Router

- created_at: `2026-05-16T12:23:24`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_best_anchor: `direct_selector_rank2_selectorsoft.csv = 0.68440`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\hitprob_rank1_anchorblendtop3p4w015.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\hitprob_rank2_anchorblendtop3p2w015.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\hitprob_rank3_anchorblendtop3p1w015.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\hitprob_rank4_anchorblendsoftp4w015.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\hitprob_rank5_anchorblendsoftp2w015.csv']`

## Problem Redefinition

- Official target: predict the +80ms 3D sensor-local coordinate from 11 historical coordinates sampled every 40ms.
- Official metric: R-Hit@1cm, so a prediction is useful only when its 3D distance is <= 0.01m.
- Previous selector models mostly learned the nearest candidate label. This experiment learns each candidate's hit probability directly.
- No test labels or external trajectory data are used. Test features are only passed through models fitted on train data.

References:

- https://dacon.io/competitions/official/236716/overview/description
- https://dacon.io/competitions/official/236716/overview/evaluation
- https://dacon.io/competitions/official/236716/overview/rules

## CV Leaderboard

| strategy | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | route_fraction | anchor_r_hit | delta_hit_vs_anchor |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anchor_blend_top3p4_w0.15 | 0.658000 | 0.021113 | 0.639500 | 0.011646 | 0.007253 | nan | 0.656700 | 0.001300 |
| anchor_blend_top3p2_w0.15 | 0.658000 | 0.021113 | 0.639500 | 0.011646 | 0.007253 | nan | 0.656700 | 0.001300 |
| anchor_blend_top3p1_w0.15 | 0.658000 | 0.021113 | 0.639500 | 0.011646 | 0.007253 | nan | 0.656700 | 0.001300 |
| anchor_blend_softp4_w0.15 | 0.657667 | 0.020569 | 0.639500 | 0.011646 | 0.007246 | nan | 0.656700 | 0.000967 |
| anchor_blend_softp2_w0.15 | 0.657667 | 0.020569 | 0.639500 | 0.011646 | 0.007246 | nan | 0.656700 | 0.000967 |
| anchor_blend_softp1_w0.15 | 0.657667 | 0.020569 | 0.639500 | 0.011646 | 0.007246 | nan | 0.656700 | 0.000967 |
| anchor_blend_softp4_w0.30 | 0.657667 | 0.020569 | 0.639500 | 0.011649 | 0.007253 | nan | 0.656700 | 0.000967 |
| anchor_blend_softp2_w0.30 | 0.657667 | 0.020569 | 0.639500 | 0.011649 | 0.007254 | nan | 0.656700 | 0.000967 |
| anchor_blend_softp1_w0.30 | 0.657667 | 0.020569 | 0.639500 | 0.011649 | 0.007254 | nan | 0.656700 | 0.000967 |
| anchor_blend_top3p4_w0.30 | 0.657500 | 0.020622 | 0.639500 | 0.011649 | 0.007251 | nan | 0.656700 | 0.000800 |
| anchor_blend_top3p2_w0.30 | 0.657500 | 0.020622 | 0.639500 | 0.011649 | 0.007251 | nan | 0.656700 | 0.000800 |
| anchor_blend_top3p1_w0.30 | 0.657500 | 0.020622 | 0.639500 | 0.011649 | 0.007251 | nan | 0.656700 | 0.000800 |
| anchor_route_blend25_p0.65 | 0.657500 | 0.021604 | 0.638500 | 0.011646 | 0.007252 | 0.693333 | 0.656700 | 0.000800 |
| anchor_route_blend25_p0.60 | 0.657500 | 0.021604 | 0.638500 | 0.011646 | 0.007252 | 0.733500 | 0.656700 | 0.000800 |
| anchor_route_blend25_p0.55 | 0.657500 | 0.021604 | 0.638500 | 0.011647 | 0.007252 | 0.767667 | 0.656700 | 0.000800 |
| anchor_route_blend25_p0.50 | 0.657500 | 0.021604 | 0.638500 | 0.011647 | 0.007249 | 0.797167 | 0.656700 | 0.000800 |
| anchor_blend_top3p4_w0.45 | 0.657333 | 0.021168 | 0.639000 | 0.011653 | 0.007243 | nan | 0.656700 | 0.000633 |
| anchor_blend_top3p2_w0.45 | 0.657333 | 0.021168 | 0.639000 | 0.011653 | 0.007243 | nan | 0.656700 | 0.000633 |
| anchor_blend_top3p1_w0.45 | 0.657333 | 0.021168 | 0.639000 | 0.011653 | 0.007243 | nan | 0.656700 | 0.000633 |
| anchor_route_blend50_p0.65 | 0.657000 | 0.021777 | 0.638500 | 0.011650 | 0.007253 | 0.693333 | 0.656700 | 0.000300 |
| anchor_route_blend50_p0.60 | 0.657000 | 0.021777 | 0.638500 | 0.011651 | 0.007252 | 0.733500 | 0.656700 | 0.000300 |
| anchor_blend_softp4_w0.45 | 0.657000 | 0.021113 | 0.638500 | 0.011653 | 0.007263 | nan | 0.656700 | 0.000300 |
| anchor_blend_softp2_w0.45 | 0.657000 | 0.021113 | 0.638500 | 0.011653 | 0.007263 | nan | 0.656700 | 0.000300 |
| anchor_blend_softp1_w0.45 | 0.657000 | 0.021113 | 0.638500 | 0.011653 | 0.007263 | nan | 0.656700 | 0.000300 |
| anchor_route_blend50_p0.55 | 0.656833 | 0.021842 | 0.638500 | 0.011652 | 0.007255 | 0.767667 | 0.656700 | 0.000133 |
| anchor_route_blend50_p0.50 | 0.656833 | 0.021842 | 0.638500 | 0.011652 | 0.007249 | 0.797167 | 0.656700 | 0.000133 |
| current_candidate | 0.656667 | 0.021554 | 0.637500 | 0.011665 | 0.007278 | nan | 0.656700 | -0.000033 |
| hitprob_soft_p2 | 0.656667 | 0.021877 | 0.637500 | 0.011668 | 0.007275 | nan | 0.656700 | -0.000033 |
| hitprob_soft_p1 | 0.656667 | 0.021877 | 0.637500 | 0.011668 | 0.007276 | nan | 0.656700 | -0.000033 |
| hitprob_soft_p4 | 0.656500 | 0.021604 | 0.637500 | 0.011668 | 0.007274 | nan | 0.656700 | -0.000200 |
| anchor_route_replace_p0.65 | 0.656500 | 0.021777 | 0.637000 | 0.011662 | 0.007259 | 0.693333 | 0.656700 | -0.000200 |
| anchor_route_replace_p0.60 | 0.656500 | 0.022096 | 0.637000 | 0.011663 | 0.007247 | 0.733500 | 0.656700 | -0.000200 |
| anchor_route_replace_p0.50 | 0.656333 | 0.021658 | 0.637500 | 0.011666 | 0.007238 | 0.797167 | 0.656700 | -0.000367 |
| anchor_route_replace_p0.55 | 0.656167 | 0.021877 | 0.637000 | 0.011665 | 0.007243 | 0.767667 | 0.656700 | -0.000533 |
| hitprob_top3_p4 | 0.656000 | 0.022456 | 0.635500 | 0.011672 | 0.007237 | nan | 0.656700 | -0.000700 |
| hitprob_top3_p2 | 0.656000 | 0.022456 | 0.635500 | 0.011672 | 0.007237 | nan | 0.656700 | -0.000700 |
| hitprob_top3_p1 | 0.656000 | 0.022456 | 0.635500 | 0.011672 | 0.007238 | nan | 0.656700 | -0.000700 |
| hitprob_hard | 0.655667 | 0.021733 | 0.636000 | 0.011672 | 0.007216 | nan | 0.656700 | -0.001033 |

## Candidate OOF Hit Table

| idx | candidate | forward | side | up | best_distance_label_count | candidate_hit_rate | candidate_mean_distance | candidate_median_distance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | side104_up094 | 1.020000 | 1.040000 | 0.940000 | 64 | 0.655900 | 0.011645 | 0.007130 |
| 0 | base_f102_s100_u100 | 1.020000 | 1.000000 | 1.000000 | 1343 | 0.655700 | 0.011645 | 0.007134 |
| 1 | prev_f102_s104_u096 | 1.020000 | 1.040000 | 0.960000 | 0 | 0.655500 | 0.011647 | 0.007135 |
| 2 | best_f102_s106_u094 | 1.020000 | 1.060000 | 0.940000 | 0 | 0.655500 | 0.011648 | 0.007144 |
| 3 | side108_up092 | 1.020000 | 1.080000 | 0.920000 | 0 | 0.655200 | 0.011650 | 0.007139 |
| 11 | forward103_side108_up092 | 1.030000 | 1.080000 | 0.920000 | 1550 | 0.655200 | 0.011706 | 0.007180 |
| 8 | forward103_side106_up094 | 1.030000 | 1.060000 | 0.940000 | 13 | 0.654900 | 0.011704 | 0.007183 |
| 5 | side106_up096 | 1.020000 | 1.060000 | 0.960000 | 32 | 0.654800 | 0.011651 | 0.007142 |
| 6 | forward103_side104_up096 | 1.030000 | 1.040000 | 0.960000 | 1652 | 0.654800 | 0.011702 | 0.007192 |
| 10 | side110_up090 | 1.020000 | 1.100000 | 0.900000 | 967 | 0.654600 | 0.011653 | 0.007151 |
| 7 | forward101_side104_up096 | 1.010000 | 1.040000 | 0.960000 | 2250 | 0.654100 | 0.011629 | 0.007156 |
| 9 | forward101_side106_up094 | 1.010000 | 1.060000 | 0.940000 | 2129 | 0.653800 | 0.011630 | 0.007164 |

## Outputs

| rank | strategy | submission | cv_mean_r_hit | cv_delta_vs_anchor | route_fraction | vs_public_best_mean_delta | vs_public_best_median_delta | vs_public_best_p95_delta | vs_public_best_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | anchor_blend_top3p4_w0.15 | hitprob_rank1_anchorblendtop3p4w015.csv | 0.658000 | 0.001300 | nan | 0.000045 | 0.000030 | 0.000129 | 0.000312 |
| 2 | anchor_blend_top3p2_w0.15 | hitprob_rank2_anchorblendtop3p2w015.csv | 0.658000 | 0.001300 | nan | 0.000045 | 0.000030 | 0.000129 | 0.000312 |
| 3 | anchor_blend_top3p1_w0.15 | hitprob_rank3_anchorblendtop3p1w015.csv | 0.658000 | 0.001300 | nan | 0.000045 | 0.000030 | 0.000129 | 0.000312 |
| 4 | anchor_blend_softp4_w0.15 | hitprob_rank4_anchorblendsoftp4w015.csv | 0.657667 | 0.000967 | nan | 0.000026 | 0.000016 | 0.000093 | 0.000188 |
| 5 | anchor_blend_softp2_w0.15 | hitprob_rank5_anchorblendsoftp2w015.csv | 0.657667 | 0.000967 | nan | 0.000026 | 0.000016 | 0.000094 | 0.000187 |

## Supplemental Submission Set

Top 1~3 are nearly identical, so three more extrapolated blend candidates were generated to avoid wasting submission slots.

| submission | source | vs_public_best_mean_delta | vs_public_best_p95_delta | note |
| --- | --- | --- | --- | --- |
| `hitprob_extra_top3blend030.csv` | 30% top3 hit-prob blend | 0.000089 | 0.000258 | rank1 direction, stronger move |
| `hitprob_extra_top3blend045.csv` | 45% top3 hit-prob blend | 0.000134 | 0.000386 | aggressive probe |
| `hitprob_extra_softblend030.csv` | 30% soft hit-prob blend | 0.000052 | 0.000187 | softer alternative |

Recommended actual submission order: `hitprob_rank1_anchorblendtop3p4w015.csv`, `hitprob_extra_top3blend030.csv`, `hitprob_rank4_anchorblendsoftp4w015.csv`, `hitprob_extra_softblend030.csv`, `hitprob_extra_top3blend045.csv`.

## Notes

- This is a metric-aligned router: rows are sample-candidate pairs and the binary label is candidate_hit = distance <= 0.01m.
- Boundary samples around 1cm receive extra weight because a tiny movement can flip the leaderboard hit.
- If public still trails 0.68440, the next reset should move away from candidate routing and into sequence/regime representation learning.
