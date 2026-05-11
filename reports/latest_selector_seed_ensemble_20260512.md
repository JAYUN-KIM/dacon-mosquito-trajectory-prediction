# 2026-05-12 Selector Seed Ensemble

- created_at: `2026-05-12T02:26:34`
- data_dir: `C:\open\dacon-mosquito-trajectory-prediction\data\raw\open (3)`
- public_best_anchor: `direct_selector_rank2_selectorsoft.csv = 0.68440`
- generated_outputs: `['C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\seedens_rank1_seedens3.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\seedens_rank2_seedens3blend20.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\seedens_rank3_seedens3blend35.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\seedens_rank4_seedens5.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\seedens_rank5_seedens5blend20.csv', 'C:\\open\\dacon-mosquito-trajectory-prediction\\submissions\\seedens_rank6_seedens5blend35.csv']`

## Outputs

| rank | submission | strategy | seeds | blend_with_seedens | vs_public_best_mean_delta | vs_public_best_median_delta | vs_public_best_p95_delta | vs_public_best_max_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | seedens_rank1_seedens3.csv | seedens3 | [42, 777, 2026] | 1.000000 | 0.000036 | 0.000025 | 0.000104 | 0.000489 |
| 2 | seedens_rank2_seedens3blend20.csv | seedens3 | [42, 777, 2026] | 0.200000 | 0.000007 | 0.000005 | 0.000021 | 0.000098 |
| 3 | seedens_rank3_seedens3blend35.csv | seedens3 | [42, 777, 2026] | 0.350000 | 0.000013 | 0.000009 | 0.000036 | 0.000171 |
| 4 | seedens_rank4_seedens5.csv | seedens5 | [42, 777, 2026, 88010, 12345] | 1.000000 | 0.000028 | 0.000020 | 0.000079 | 0.000367 |
| 5 | seedens_rank5_seedens5blend20.csv | seedens5 | [42, 777, 2026, 88010, 12345] | 0.200000 | 0.000006 | 0.000004 | 0.000016 | 0.000073 |
| 6 | seedens_rank6_seedens5blend35.csv | seedens5 | [42, 777, 2026, 88010, 12345] | 0.350000 | 0.000010 | 0.000007 | 0.000028 | 0.000129 |

## Public Results

| submission | public_score | note |
| --- | --- | --- |
| seedens_rank1_seedens3.csv | 0.68420 | seed ensemble full soft underperformed the public-best anchor |
| seedens_rank3_seedens3blend35.csv | 0.68440 | 35% blend tied the current public best |

## Notes

- Temperature/top-k hurt public, so this probe keeps the original candidate pool and stabilizes selector probabilities with seed ensembling.
- Blend outputs are safer if the full seed ensemble moves too far from the current public best.
