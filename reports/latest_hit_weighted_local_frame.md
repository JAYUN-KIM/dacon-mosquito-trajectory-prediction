# Hit-Weighted Local Frame

- Created at: `2026-05-07T10:44:21`
- Data dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- Feature count: `752`
- Objectives: `['l2']`
- Weight modes: `['uniform', 'base_boundary', 'candidate_boundary', 'near_hit_band', 'hard_miss_band']`
- CV seeds: `[42]`
- Full ensemble seeds: `[42, 777, 2026]`
- Candidate feature families: `['phys_a000', 'phys_a150', 'phys_a275', 'phys_a400', 'phys_v098_a275', 'phys_v102_a275', 'poly_w3_d1', 'poly_w3_d2', 'poly_w4_d1', 'poly_w4_d2', 'poly_w5_d1', 'poly_w5_d2', 'poly_w7_d1', 'poly_w7_d2', 'poly_w11_d1', 'poly_w11_d2', 'poly_w11_d3', 'wdiff_w5_d050', 'wdiff_w5_d075', 'wdiff_w7_d060', 'wdiff_w11_d070']`
- Written submissions: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\hit_weighted_rank1_l2_base_boundary_f0.46_s0.58_u0.70.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\hit_weighted_rank2_l2_base_boundary_f0.46_s0.55_u0.62.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\hit_weighted_rank3_l2_near_hit_band_f0.52_s0.58_u0.70.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\hit_weighted_rank4_l2_candidate_boundary_f0.46_s0.55_u0.62.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\hit_weighted_rank5_l2_near_hit_band_f0.52_s0.58_u0.78.csv']`

## Top 40 Weighted Local-Frame Configs

| objective | weight_mode | forward_shrink | side_shrink | up_shrink | mean_r_hit | std_r_hit | min_r_hit | mean_distance | median_distance | risk_adjusted_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| l2 | base_boundary | 0.460000 | 0.580000 | 0.700000 | 0.670000 | nan | 0.670000 | 0.011677 | 0.007073 | 0.670000 |
| l2 | base_boundary | 0.460000 | 0.550000 | 0.620000 | 0.670000 | nan | 0.670000 | 0.011697 | 0.007131 | 0.670000 |
| l2 | near_hit_band | 0.520000 | 0.580000 | 0.700000 | 0.669500 | nan | 0.669500 | 0.011640 | 0.007050 | 0.669500 |
| l2 | candidate_boundary | 0.460000 | 0.550000 | 0.620000 | 0.669500 | nan | 0.669500 | 0.011758 | 0.007249 | 0.669500 |
| l2 | near_hit_band | 0.520000 | 0.580000 | 0.780000 | 0.669000 | nan | 0.669000 | 0.011627 | 0.007066 | 0.669000 |
| l2 | base_boundary | 0.520000 | 0.580000 | 0.700000 | 0.669000 | nan | 0.669000 | 0.011630 | 0.007039 | 0.669000 |
| l2 | base_boundary | 0.460000 | 0.580000 | 0.780000 | 0.669000 | nan | 0.669000 | 0.011665 | 0.007068 | 0.669000 |
| l2 | near_hit_band | 0.480000 | 0.550000 | 0.700000 | 0.669000 | nan | 0.669000 | 0.011678 | 0.007073 | 0.669000 |
| l2 | base_boundary | 0.460000 | 0.580000 | 0.620000 | 0.669000 | nan | 0.669000 | 0.011693 | 0.007101 | 0.669000 |
| l2 | near_hit_band | 0.460000 | 0.520000 | 0.700000 | 0.669000 | nan | 0.669000 | 0.011701 | 0.007117 | 0.669000 |
| l2 | candidate_boundary | 0.480000 | 0.550000 | 0.700000 | 0.669000 | nan | 0.669000 | 0.011733 | 0.007191 | 0.669000 |
| l2 | candidate_boundary | 0.460000 | 0.550000 | 0.780000 | 0.669000 | nan | 0.669000 | 0.011741 | 0.007203 | 0.669000 |
| l2 | candidate_boundary | 0.460000 | 0.550000 | 0.700000 | 0.669000 | nan | 0.669000 | 0.011748 | 0.007203 | 0.669000 |
| l2 | candidate_boundary | 0.460000 | 0.580000 | 0.620000 | 0.669000 | nan | 0.669000 | 0.011756 | 0.007232 | 0.669000 |
| l2 | candidate_boundary | 0.460000 | 0.520000 | 0.620000 | 0.669000 | nan | 0.669000 | 0.011762 | 0.007250 | 0.669000 |
| l2 | base_boundary | 0.520000 | 0.580000 | 0.780000 | 0.668500 | nan | 0.668500 | 0.011618 | 0.007047 | 0.668500 |
| l2 | near_hit_band | 0.520000 | 0.550000 | 0.700000 | 0.668500 | nan | 0.668500 | 0.011645 | 0.007053 | 0.668500 |
| l2 | base_boundary | 0.480000 | 0.580000 | 0.780000 | 0.668500 | nan | 0.668500 | 0.011648 | 0.007058 | 0.668500 |
| l2 | base_boundary | 0.480000 | 0.580000 | 0.700000 | 0.668500 | nan | 0.668500 | 0.011660 | 0.007073 | 0.668500 |
| l2 | base_boundary | 0.460000 | 0.550000 | 0.780000 | 0.668500 | nan | 0.668500 | 0.011669 | 0.007105 | 0.668500 |
| l2 | near_hit_band | 0.480000 | 0.580000 | 0.700000 | 0.668500 | nan | 0.668500 | 0.011673 | 0.007067 | 0.668500 |
| l2 | base_boundary | 0.480000 | 0.550000 | 0.620000 | 0.668500 | nan | 0.668500 | 0.011680 | 0.007099 | 0.668500 |
| l2 | near_hit_band | 0.460000 | 0.550000 | 0.780000 | 0.668500 | nan | 0.668500 | 0.011684 | 0.007103 | 0.668500 |
| l2 | base_boundary | 0.460000 | 0.520000 | 0.700000 | 0.668500 | nan | 0.668500 | 0.011685 | 0.007104 | 0.668500 |
| l2 | near_hit_band | 0.460000 | 0.550000 | 0.700000 | 0.668500 | nan | 0.668500 | 0.011696 | 0.007111 | 0.668500 |
| l2 | near_hit_band | 0.460000 | 0.550000 | 0.620000 | 0.668500 | nan | 0.668500 | 0.011713 | 0.007134 | 0.668500 |
| l2 | candidate_boundary | 0.480000 | 0.580000 | 0.700000 | 0.668500 | nan | 0.668500 | 0.011730 | 0.007191 | 0.668500 |
| l2 | candidate_boundary | 0.480000 | 0.520000 | 0.700000 | 0.668500 | nan | 0.668500 | 0.011736 | 0.007197 | 0.668500 |
| l2 | candidate_boundary | 0.460000 | 0.580000 | 0.780000 | 0.668500 | nan | 0.668500 | 0.011738 | 0.007173 | 0.668500 |
| l2 | candidate_boundary | 0.480000 | 0.520000 | 0.620000 | 0.668500 | nan | 0.668500 | 0.011747 | 0.007230 | 0.668500 |
| l2 | base_boundary | 0.480000 | 0.550000 | 0.780000 | 0.668000 | nan | 0.668000 | 0.011652 | 0.007068 | 0.668000 |
| l2 | near_hit_band | 0.520000 | 0.580000 | 0.620000 | 0.668000 | nan | 0.668000 | 0.011657 | 0.007075 | 0.668000 |
| l2 | base_boundary | 0.480000 | 0.520000 | 0.700000 | 0.668000 | nan | 0.668000 | 0.011669 | 0.007105 | 0.668000 |
| l2 | near_hit_band | 0.460000 | 0.580000 | 0.780000 | 0.668000 | nan | 0.668000 | 0.011679 | 0.007102 | 0.668000 |
| l2 | base_boundary | 0.460000 | 0.550000 | 0.700000 | 0.668000 | nan | 0.668000 | 0.011681 | 0.007105 | 0.668000 |
| l2 | near_hit_band | 0.460000 | 0.520000 | 0.780000 | 0.668000 | nan | 0.668000 | 0.011689 | 0.007110 | 0.668000 |
| l2 | near_hit_band | 0.460000 | 0.580000 | 0.700000 | 0.668000 | nan | 0.668000 | 0.011691 | 0.007103 | 0.668000 |
| l2 | candidate_boundary | 0.480000 | 0.550000 | 0.780000 | 0.668000 | nan | 0.668000 | 0.011726 | 0.007195 | 0.668000 |
| l2 | candidate_boundary | 0.480000 | 0.550000 | 0.620000 | 0.668000 | nan | 0.668000 | 0.011744 | 0.007234 | 0.668000 |
| l2 | candidate_boundary | 0.460000 | 0.580000 | 0.700000 | 0.668000 | nan | 0.668000 | 0.011745 | 0.007203 | 0.668000 |

## Readout

- This is a broader restart experiment, not a route-blend micro-tune.
- It adds normalized trajectory geometry features and trains with hit-boundary-aware sample weights.
- If this beats the 0.6604 anchor, continue with metric-aware objectives; otherwise move to regime/cluster routing or sequence neural models.
