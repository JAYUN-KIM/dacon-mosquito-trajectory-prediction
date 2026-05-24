# DACON 모기 비행 궤적 예측 AI 경진대회

40ms 간격으로 관측된 과거 11개 3D 좌표를 활용해 모기의 `+80ms` 미래 위치 `(x, y, z)`를 예측하는 프로젝트입니다.  
단순 평균 거리보다 대회 지표인 `R-Hit@1cm`를 직접 끌어올리는 것을 목표로, 물리 기반 baseline에서 출발해 local-frame residual, selective retrieval routing, hit-boundary weighted modeling까지 확장했습니다.

## 프로젝트 개요

- 대회: 모기 비행 궤적 예측 AI 경진대회
- 플랫폼: DACON
- 평가 지표: R-Hit@1cm
- 입력 데이터: 40ms 간격 과거 11개 3D 좌표 `-400ms ~ 0ms`
- 목표: 마지막 관측 시점 기준 `+80ms` 미래 좌표 `(x, y, z)` 예측
- hit 기준: 예측 좌표와 실제 좌표의 3D Euclidean distance가 `0.01m` 이하
- 좌표 단위: meter

## 현재 성과

<!-- AUTO:PROJECT_STATUS:START -->
- 최고 Public LB: **0.69140**
- 최신 최고점 확인일: **2026-05-24**
- 핵심 개선 축: temporal-backcast pseudo-supervision + constant-turn curvature correction + sample-wise curvature gate
- 최신 새 축 검토: 2026-05-24 regime miss policy는 `0.69060`, analog KNN residual은 `0.68860`으로 실패
- 최신 개선: `t52` curvature gate를 유지하고 correction alpha를 `0.105`에서 `0.1025` 근처로 낮추며 `0.69140`까지 갱신
- 상세 실험 기록은 `docs/`, `reports/`, `experiments/` 디렉토리에 분리 보관
<!-- AUTO:PROJECT_STATUS:END -->

## 예측 타겟

| 타겟 | 의미 |
|---|---|
| x | sensor-local forward 방향 미래 위치 |
| y | sensor-local left 방향 미래 위치 |
| z | sensor-local up 방향 미래 위치 |

## 핵심 접근법

1. 물리 기반 baseline
   - constant velocity, constant acceleration, polynomial extrapolation을 먼저 비교했습니다.
   - `R-Hit@1cm` 기준으로 velocity/acceleration 계수를 직접 탐색했습니다.

2. Residual ML
   - 물리 예측값을 anchor로 두고 LightGBM이 residual을 보정하도록 구성했습니다.
   - residual 예측을 그대로 더하지 않고 shrink를 적용해 과보정을 줄였습니다.

3. Local-frame residual
   - 마지막 속도 방향을 forward 축으로 하는 local coordinate frame을 만들었습니다.
   - global `x, y, z` residual 대신 `진행 방향 / 좌우 방향 / 상하 방향` residual을 예측했습니다.
   - 이 전환으로 Public LB가 `0.64120`에서 `0.65900`까지 크게 올랐습니다.

4. Selective retrieval routing
   - train에서 유사한 과거 궤적을 검색해 residual을 가져오는 kNN/retrieval 계열을 실험했습니다.
   - retrieval 단독은 약했지만, confidence가 높은 일부 샘플에만 섞는 route blend가 `0.66040`까지 개선했습니다.

5. Hit-boundary weighted local-frame
   - 평균 거리 최적화가 아니라 `1cm hit 경계` 근처 샘플을 더 중요하게 학습했습니다.
   - normalized trajectory geometry feature를 추가하고, base physics error가 1cm 근방인 샘플에 가중치를 줬습니다.
   - 이 축에서 `0.67100`, 이후 5-seed refine으로 `0.67220`까지 상승했습니다.

6. Direct-step local target
   - residual anchor 보정만으로는 정체가 보여, 마지막 관측 좌표 기준 `+80ms displacement`를 local frame에서 직접 예측했습니다.
   - sample별 motion scale로 target을 정규화한 뒤 추론 시 다시 복원하는 방식을 적용했습니다.
   - 순수 direct-step LGBM boundary 후보가 Public LB `0.67800`으로 새 최고점을 만들었습니다.

7. Selector confidence routing
   - direct-step 후보의 forward/side/up multiplier를 샘플별로 고르는 selector를 구성했습니다.
   - hard routing은 과적합 위험이 있어 confidence가 높은 샘플만 선택적으로 route했습니다.
   - `selector_conf0.55`가 `0.68340`, 이후 conf0.45 방향을 약하게 섞은 후보가 `0.68360`까지 개선했습니다.

8. Probability-weighted selector soft routing
   - threshold로 후보를 하나 고르는 대신, selector probability로 여러 multiplier 후보를 부드럽게 평균했습니다.
   - `conf0.45` pull grid는 `0.68360`에서 포화됐지만, `selector_soft`가 Public LB `0.68440`으로 새 최고점을 만들었습니다.

9. Selector soft 후속 검증
   - temperature, top-k truncation, expanded candidate pool, seed ensemble을 확인했습니다.
   - seed ensemble blend는 `0.68440` 동률을 만들었지만, temperature/top-k와 expanded pool은 public에서 하락했습니다.

10. 새 축 재탐색
   - route gain binary fallback과 analog residual correction을 테스트했습니다.
   - OOF에서는 약한 개선 신호가 있었지만, public에서는 `0.68300 ~ 0.68360`으로 하락해 주력 축에서 내렸습니다.

11. Temporal-backcast pseudo-supervision
   - train 궤적 내부 시점 `c=6,7,8`을 현재로 보고 `c+2`를 pseudo target으로 삼아 학습 데이터를 확장했습니다.
   - 부족한 앞쪽 history는 초기 속도 기반 backcast로 채워 11점 입력 구조를 유지했습니다.
   - temporal-backcast 모델 단독은 `0.68640`, 기존 public-best anchor와 50% blend한 후보는 `0.68780`으로 새 최고점을 만들었습니다.

12. Constant-turn curvature correction
   - 최근 속도 방향의 3D 회전량을 추정해 +80ms 동안 같은 방향으로 휘어진다고 가정했습니다.
   - constant-turn 단독 예측 대신 `(constant_turn - constant_velocity)` correction만 강한 temporal anchor에 작게 더했습니다.
   - `anchor_alpha=0.09` 후보가 Public LB `0.69000`으로 새 최고점을 만들었습니다.

13. Curvature gate
   - 모든 샘플에 같은 curvature correction을 더하지 않고, train OOF에서 correction이 이득인 샘플을 학습했습니다.
   - gate probability가 중간 이상인 샘플에만 `alpha=0.105` correction을 적용하는 후보가 public에서 개선됐습니다.
   - `threshold=0.52`, `alpha=0.105` 후보가 Public LB `0.69120`으로 새 최고점을 만들었습니다.

14. Gate residual correction
   - curvature gate 이후 남은 OOF residual vector를 다시 학습해 아주 작게 보정하는 실험을 진행했습니다.
   - OOF에서는 소폭 개선됐지만 public은 `0.69040`으로 하락해, 현재는 주력 축에서 제외합니다.

15. 2026-05-20 새 축 재탐색
   - mirror-symmetry temporal TTA, multi-curvature action router, MLP sequence pseudo-supervision을 실험했습니다.
   - 세 축 모두 첫 public probe가 `0.69020`으로 하락해, current best 주변을 흐트러뜨리는 과적합성 후보로 판단했습니다.
   - 반면 기존 curvature gate의 `threshold=0.54`, `alpha=0.105` 후보가 `0.69120` 동률을 재현해 gate 축의 안정성을 확인했습니다.

16. 2026-05-21 post-process 새 축 손절
   - local target manifold projection은 OOF proxy에서 좋아 보였지만 public은 `0.68980`으로 크게 하락했습니다.
   - hit-rescue specialist는 champion이 놓칠 것 같은 일부 샘플만 temporal55로 되돌렸지만 `0.69060`에 그쳤습니다.
   - `t52_a105`와 `t54_a105` co-champion blend는 `w50`, `w65`, `w35` 모두 `0.69120`을 유지해 안정권을 재확인했습니다.
   - 다음 돌파는 current best 후처리가 아니라 temporal-backcast급의 새 pseudo-label/supervision 축에서 찾아야 합니다.

17. 2026-05-23 공격적 새 축 재탐색
   - temporal-backcast를 더 넓은 cutoff와 horizon1 velocity pseudo-label로 확장했지만 제출 후보 3개가 모두 `0.69060`으로 하락했습니다.
   - exp-weighted polynomial smoothing은 `0.67160`으로 크게 무너져 smoothing/denoising 계열을 폐기했습니다.
   - jerk/snap rebound 물리축은 `0.68660`으로 smoothing보다 낫지만 champion과 거리가 컸습니다.
   - 다음은 새 좌표 후보를 바로 섞는 것이 아니라 champion miss regime을 먼저 분해하는 검증/selector 설계가 필요합니다.

18. 2026-05-24 champion 미세조정
   - regime miss policy와 analog KNN residual은 각각 `0.69060`, `0.68860`으로 하락해 새 축 후보에서 내렸습니다.
   - `t52` curvature gate 자체는 유지하고 alpha만 조정했을 때, `alpha=0.1075`는 `0.69100`으로 하락했습니다.
   - 반대로 `alpha=0.1025`는 `0.69140`으로 상승해 현재 최고점을 갱신했습니다.
   - 후속 alpha band probe도 `0.69140` 동률을 기록해, 현재는 새 좌표축보다 `t52` gate의 alpha calibration이 가장 믿을 만합니다.

## 주요 인사이트

- 단순 좌표계 residual보다 마지막 속도 방향 기준 local-frame residual이 훨씬 안정적이었습니다.
- forward/side/up 축별 shrink를 다르게 주는 것이 단일 shrink보다 유리했습니다.
- retrieval은 단독 모델로는 약하지만, 일부 high-confidence 샘플 보정 재료로는 효과가 있었습니다.
- 가장 큰 돌파는 feature 수를 무작정 늘린 것이 아니라, `1cm hit 경계`를 직접 겨냥한 sample weighting에서 나왔습니다.
- 2026-05-08 기준으로는 residual 보정 축보다 pure direct-step target 전환이 더 큰 public 개선을 만들었습니다.
- Public과 CV가 완전히 일치하지 않으므로, 새 축은 빠르게 public probe하고 강한 신호가 나온 축만 확장하는 전략이 효과적입니다.
- 2026-05-10 기준 selector/routing은 큰 돌파 축은 아니지만 public에서 재현된 얇은 개선 축입니다.
- velocity smoothing/local frame denoising은 OOF proxy에서 크게 하락해 당분간 폐기합니다.
- 2026-05-11 기준 hard/threshold routing보다 selector 확률 분포를 그대로 쓰는 soft routing이 더 강했습니다.
- 2026-05-12 기준 selector soft 후처리보다 route label 설계와 hit 전환 가능성 예측이 다음 연구 우선순위입니다.
- 2026-05-15 기준 route gain과 analog residual correction은 public에서 약해, 다음은 완전히 다른 새 축을 우선합니다.
- 2026-05-16 기준 궤적 내부 pseudo-supervision을 활용하는 temporal-backcast 축이 가장 강한 새 돌파구입니다.
- temporal-backcast는 단독보다 기존 selector-soft anchor와 `50%` 전후로 섞을 때 public에서 더 강했습니다.
- 2026-05-18 기준 temporal-backcast는 `55%` 근처에서 포화됐고, 그 위에 constant-turn curvature correction을 작게 더하는 방식이 추가 돌파를 만들었습니다.
- curvature correction은 단독 모델이 아니라 강한 anchor 위에 `0.08~0.10` 수준으로 얹을 때 가장 안정적입니다.
- 2026-05-19 기준 curvature correction은 전 샘플 동일 적용보다 sample-wise gate가 더 강했습니다.
- gate threshold는 너무 낮게 넓히면 하락했고, `0.50~0.52` 근처의 중간 확신 샘플만 correction하는 쪽이 public에서 가장 좋았습니다.
- low-confidence 샘플에 약한 correction을 남기는 방식과 residual-on-gate correction은 public에서 하락했습니다.
- 2026-05-20 기준 새 축 후보가 서로 다른 방식이어도 `0.69020` 근처로 반복 하락해, OOF/아이디어 신호만 믿고 current best를 섞는 방식은 위험합니다.
- curvature gate는 `t52_a105`와 `t54_a105`가 모두 `0.69120`으로 재현됐고, `alpha=0.110`은 `0.69080`으로 하락해 alpha 0.105 근처가 안정권입니다.
- 2026-05-21 기준 manifold projection과 hit-rescue hard swap 모두 public에서 하락해, champion 위 post-process 보정만으로는 0.7 돌파가 어렵다고 판단합니다.
- co-champion blend 3종이 모두 `0.69120`으로 동률을 유지해 안정성은 확인했지만, 점수 상한을 뚫지는 못했습니다.
- 다음 연구는 제출 파일을 바로 만드는 것보다 train 내부 oracle hit potential로 새 pseudo-label 후보군의 추가 hit 가능성을 먼저 확인해야 합니다.
- 2026-05-23 기준 temporal curriculum 확장, smoothing, snap 물리축도 모두 실패해 단순한 새 물리식/새 pseudo-label 추가는 막혔다고 봅니다.
- smoothing 계열은 특히 위험했습니다. 최근 관측의 노이즈 제거보다 순간 turn/acceleration 보존이 더 중요한 문제로 보입니다.
- 2026-05-24 기준 regime miss/KNN residual도 public에서 하락했으므로, 단기적으로는 `t52` gate의 alpha를 `0.101~0.103` 근처에서 더 촘촘히 보정하는 것이 가장 현실적입니다.

## Public Score 흐름

| 제출 파일 | Public score | 요약 |
|---|---:|---|
| `physics_param_search_best.csv` | 0.61539 | threshold-aware 물리 baseline |
| `aggressive_lgbm_residual.csv` | 0.63420 | LGBM residual 유효성 확인 |
| `candidate_binary_hit_selector.csv` | 0.62600 | 후보 선택기는 residual보다 약함 |
| `public_push_lgbm_wide_a0275_s0.40_5seed.csv` | 0.64120 | 5-seed residual ensemble |
| `local_frame_lgbm_a0275_s0.55_5seed.csv` | 0.65900 | local-frame residual target |
| `local_axis_rank1_f0.48_s0.55_u0.62.csv` | 0.65940 | 축별 shrink calibration |
| `retr_blend_rank1_confidentrouteblend...csv` | 0.66040 | selective retrieval route blend |
| `hit_weighted_rank1_l2_base_boundary_f0.46_s0.58_u0.70.csv` | 0.67100 | hit-boundary weighted local-frame breakthrough |
| `hit_breakthrough_rank1_basea5s0045_f0.52_s0.58_u0.70_5seed.csv` | 0.67220 | 5-seed hit-weighted breakthrough refine |
| `regime_hit_rank1_globalhitweighted_a0.00_f0.56_s0.58_u0.70.csv` | 0.67300 | regime router 축은 큰 돌파 없이 안전 변형 |
| `direct_step_rank2_cadeltascaledlgbmboundary_f0.52_s0.58_u0.78.csv` | 0.67340 | scale-normalized CA residual LGBM |
| `direct_step_pure_lgbmboundary_f1.02_s1.00_u1.00.csv` | **0.67800** | pure direct-step local target 새 최고점 |
| `direct_refine_rank1_caa6s0055c0105_f1.02_s1.00_u1.00_5seed.csv` | 0.68240 | CA-boundary direct-step 5seed jump |
| `direct_refine_rank2_caa6s0055c0105_f1.02_s1.04_u0.96_5seed.csv` | 0.68260 | side/up multiplier tilt |
| `direct_micro_rank1_fromcaa6_f1.02_s1.06_u0.94.csv` | 0.68300 | multiplier micro-probe |
| `direct_selector_rank1_selectorconf055.csv` | 0.68340 | sample-wise multiplier selector confidence routing |
| `selector_adjust_rank1_extend115.csv` | 0.68340 | selector 방향 15% extension은 anchor와 동률 |
| `selector_adjust_rank2_shrink075.csv` | 0.68320 | selector 이동 축소는 하락 |
| `selector_adjust_rank4_conf45pull015.csv` | 0.68360 | conf0.45 route 방향 약한 보정 |
| `selector_adjust_rank5_extend130.csv` | 0.68340 | selector 방향 30% extension은 추가 개선 없음 |
| `route_refine_rank2_conf45pull200.csv` | 0.68360 | conf0.45 pull grid는 기존 best와 동률 |
| `route_refine_rank1_conf45pull100.csv` | 0.68360 | conf0.45 pull 0.10도 기존 best와 동률 |
| `route_refine_rank3_conf45pull250.csv` | 0.68360 | conf0.45 pull 0.25도 기존 best와 동률 |
| `direct_selector_rank4_selectorconf045.csv` | 0.68420 | full conf0.45 routing, 기존 best 대비 개선 |
| `direct_selector_rank2_selectorsoft.csv` | **0.68440** | 현재 최고점, probability-weighted selector soft routing |
| `softtemp_rank8_top3t100.csv` | 0.68420 | top-3 truncation은 full soft보다 약함 |
| `softtemp_rank1_softt075.csv` | 0.68420 | sharper temperature도 full soft보다 약함 |
| `seedens_rank1_seedens3.csv` | 0.68420 | selector seed ensemble 단독은 하락 |
| `seedens_rank3_seedens3blend35.csv` | **0.68440** | seed ensemble blend는 현재 최고점과 동률 |
| `expanded_selector_rank1_expandedsoftblend015.csv` | 0.68400 | expanded candidate pool은 public에서 약함 |
| `route_gain_top_candidates` | 0.68420 | route gain fallback 후보 2개 모두 best 아래 |
| `analogres_rank1_k64s010.csv` | 0.68300 | analog residual correction OOF 1위였지만 public 약세 |
| `analogres_rank2_k96s015.csv` | 0.68300 | stronger analog correction도 개선 없음 |
| `analogres_rank3_k128s010.csv` | 0.68360 | analog 계열 중 최고지만 best 미달 |
| `hitprob_rank1_anchorblendtop3p4w015.csv` | 0.68420 | 후보별 hit 확률 라우터는 best보다 약함 |
| `hitprob_extra_top3blend030.csv` | 0.68420 | hit-prob 방향을 더 키워도 개선 없음 |
| `temporalbc_rank1_anchorblend35_tbc678w020_f1.02_s1.00_u1.00.csv` | 0.68620 | temporal-backcast 축 유효성 확인 |
| `temporalbc_rank1_anchorblend50_tbc678w020_f1.02_s1.00_u1.00.csv` | 0.68780 | temporal-backcast 50% blend |
| `temporalbc_rank1_tbc678w020_f1.02_s1.00_u1.00.csv` | 0.68640 | temporal-backcast 단독은 강하지만 50% blend보다 약함 |
| `temporalbc_refine_r1f102s100u100_w52.csv` | 0.68800 | 52% temporal blend 추가 개선 |
| `temporalbc_refine_r1f102s100u100_w55.csv` | 0.68880 | temporal-only strength 최적점 근처 |
| `temporalbc_refine_avgr1r2_w52.csv` | 0.68820 | temporal direction ensemble은 rank1보다 약함 |
| `temporalbc_refine_truew555_r1f102s100u100.csv` | 0.68860 | true 55.5% blend는 기존 w55보다 약함 |
| `turncurve_rank1_temporalbest_w1tm0p25s0p5d0p98_a08.csv` | 0.68940 | constant-turn curvature correction 유효성 확인 |
| `turncurve_refine_temporalbest_w1tm0p25s0p5d0p98_a09.csv` | **0.69000** | 현재 최고점, curvature correction alpha 0.09 |
| `turncurve_refine_temporalbest_w1tm0p25s0p5d0p98_a10.csv` | 0.68960 | alpha 0.10은 0.09보다 약함 |
| `curvgate_rank2_gatet50a105.csv` | 0.69100 | curvature correction을 gate 확신 샘플에만 적용 |
| `curvgate_rank3_gatet38a105.csv` | 0.69000 | threshold를 너무 낮게 넓히면 기존 best 수준으로 하락 |
| `curvgate_refine_rank2_gatet52a105.csv` | **0.69120** | 현재 최고점, gate threshold 0.52와 alpha 0.105 |
| `curvgate_refine_rank6_gatet50a105low025.csv` | 0.69040 | low-confidence 샘플에 약한 correction을 남긴 변형은 하락 |
| `gate_residual_exp_sh085_cap0022.csv` | 0.69040 | residual-on-gate 새 축은 OOF 대비 public 미재현 |
| `mirror_tta_rank1_mirrortta_temporal_w55_gate_t52_a105_bestblend35.csv` | 0.69020 | mirror-symmetry temporal TTA 새 축은 public 미재현 |
| `multicurv_action_rank2_currentblend25actiontop3p2.csv` | 0.69020 | multi-curvature action router도 current best를 흐트러뜨림 |
| `mlpseq_rank2_blend08base.csv` | 0.69020 | MLP sequence pseudo-supervision blend도 public 하락 |
| `curvgate_rank4_gatet54a105.csv` | **0.69120** | t52와 동률, threshold 0.54도 안정권임을 확인 |
| `curvgate_refine_rank8_gatet52a110.csv` | 0.69080 | alpha 0.110은 과보정 |
| `manifoldproj_rank2_k256_b060_cap0003.csv` | 0.68980 | local target manifold projection은 OOF 대비 public 과적합 |
| `hitrescue_rank1_temporal55_top075.csv` | 0.69060 | hard-swap rescue specialist도 champion 대비 하락 |
| `cochamp_blend_t52_t54_w50.csv` | **0.69120** | t52/t54 co-champion 50:50 blend 동률 |
| `cochamp_blend_t52_t54_w65.csv` | **0.69120** | t52-heavy co-champion blend 동률 |
| `cochamp_blend_t52_t54_w35.csv` | **0.69120** | t54-heavy co-champion blend 동률 |
| `tempcurr_rank1_tcc5678w012v6789w006champblend15f102s100u100.csv` | 0.69060 | temporal curriculum 확장 후보도 champion 대비 하락 |
| `tempcurr_rank4_tcc678w022v89w004champblend15f102s100u100.csv` | 0.69060 | 보수적 temporal curriculum 후보도 하락 |
| `tempcurr_rank5_tcc5678w012v6789w006cochampblend20f102s100u100.csv` | 0.69060 | co-champion 기반 temporal curriculum도 하락 |
| `fastnew_rank2_snapsnapv102a035jm0p2d096blend18.csv` | 0.68660 | jerk/snap rebound 물리축은 약함 |
| `fastnew_rank1_smoothewpolyw11d2r55blend18.csv` | 0.67160 | exp-weighted polynomial smoothing은 강한 하락 |
| `regimemiss_rank1_c64_min55_net0003_p165.csv` | 0.69060 | champion miss regime selector는 public에서 하락 |
| `analogknn_rank1_k64_p10_s012_cap00025_r100.csv` | 0.68860 | 유사 궤적 KNN residual 전이는 약함 |
| `champmicro_rank1_gatet520a1075.csv` | 0.69100 | t52 gate alpha를 키우면 과보정 |
| `champmicro_rank3_gatet520a1025.csv` | **0.69140** | t52 gate alpha를 낮추며 새 최고점 갱신 |
| `champalpha_rank1_t52a1015.csv` | **0.69140** | alpha-down 주변 후속 probe도 최고점 동률 |

## 대표 실험 코드

| 파일 | 역할 |
|---|---|
| `scripts/run_physics_baselines.py` | 기본 물리 baseline 평가 및 제출 생성 |
| `scripts/search_physics_params.py` | velocity/acceleration 계수 grid search |
| `scripts/run_local_frame_residual.py` | local-frame residual target 실험 |
| `scripts/run_local_frame_axis_shrink.py` | forward/side/up 축별 shrink 탐색 |
| `scripts/run_local_frame_fine_axis_search.py` | local-axis shrink 주변 정밀 탐색 |
| `scripts/run_trajectory_retrieval.py` | 유사 궤적 retrieval/kNN 후보 생성 |
| `scripts/run_retrieval_blend_router.py` | local-frame anchor와 retrieval 선택적 blend |
| `scripts/run_retrieval_route_refine.py` | retrieval route blend 세부 탐색 |
| `scripts/run_hit_weighted_local_frame.py` | hit-boundary weighted local-frame breakthrough |
| `scripts/run_hit_weighted_breakthrough_refine.py` | 0.671 breakthrough 5-seed 안정화 |
| `scripts/run_regime_hit_weighted_router.py` | motion regime별 hit-weighted router 실험 |
| `scripts/run_direct_step_geometry.py` | direct-step target, scale-normalized residual, CatBoost/LGBM 비교 |
| `scripts/run_direct_step_refine_20260509.py` | 0.678 direct-step branch의 weight/multiplier/5seed 확장 |
| `scripts/make_direct_step_multiplier_probe_20260509.py` | 0.6826 주변 multiplier micro-probe 후보 생성 |
| `scripts/run_direct_multiplier_selector_20260510.py` | direct-step multiplier 후보를 샘플별로 고르는 selector/routing |
| `scripts/make_selector_adjustment_candidates_20260510.py` | public에서 오른 selector 방향 주변 조정 후보 생성 |
| `scripts/make_direct_velocity_smoothing_probe_20260510.py` | velocity smoothing/local frame denoising probe |
| `scripts/run_selector_route_refine_20260511.py` | conf0.45 pull grid와 continuous multiplier regression 후보 생성 |
| `scripts/run_selector_soft_temperature_20260512.py` | selector soft temperature/top-k truncation 후보 생성 |
| `scripts/run_expanded_selector_pool_20260512.py` | expanded multiplier candidate pool selector 실험 |
| `scripts/run_selector_seed_ensemble_20260512.py` | selector probability seed ensemble 후보 생성 |
| `scripts/run_route_gain_model_20260515.py` | selector soft 실패 위험 샘플 fallback 후보 생성 |
| `scripts/run_analog_residual_correction_20260515.py` | 유사 궤적 OOF 잔차 보정 후보 생성 |
| `scripts/run_hit_probability_router_20260516.py` | 후보별 1cm hit probability 라우터 실험 |
| `scripts/run_temporal_backcast_augmentation_20260516.py` | 궤적 내부 pseudo-supervision temporal-backcast 후보 생성 |
| `scripts/make_temporal_backcast_refine_candidates_20260516.py` | temporal-backcast 50% 주변 blend/refine 후보 생성 |
| `scripts/run_constant_turn_curvature_20260518.py` | constant-turn curvature correction 후보 생성 |
| `scripts/run_curvature_gate_20260519.py` | curvature correction 적용 샘플을 고르는 gate 후보 생성 |
| `scripts/make_curvature_gate_refine_candidates_20260519.py` | public 신호 기반 curvature gate threshold/alpha refine |
| `scripts/make_gate_residual_experimental_20260519.py` | curvature gate 이후 residual-on-gate 실험 후보 생성 |
| `scripts/run_mirror_tta_temporal_gate_20260520.py` | mirror-symmetry temporal TTA와 curvature gate blend 실험 |
| `scripts/run_multi_curvature_action_router_20260520.py` | 여러 curvature action 후보별 hit probability router 실험 |
| `scripts/run_mlp_sequence_pseudo_blend_20260520.py` | MLP sequence pseudo-supervision blend 실험 |
| `scripts/run_local_target_manifold_projection_20260521.py` | champion local displacement를 train target-local manifold로 약하게 투영 |
| `scripts/run_hit_rescue_specialist_20260521.py` | champion miss 가능 샘플만 hard swap하는 rescue specialist |
| `scripts/run_temporal_curriculum_fast_20260522.py` | temporal-backcast cutoff/velocity pseudo-label 확장 후보 생성 |
| `scripts/run_aggressive_new_axes_20260523.py` | smoothing/action distillation 공격 축 진단 |
| `scripts/run_fast_two_new_axes_20260523.py` | exp-weighted smoothing과 jerk/snap rebound 물리축 후보 생성 |
| `scripts/run_regime_miss_policy_20260524.py` | champion miss regime별 후보 교체 policy 실험 |
| `scripts/run_analog_knn_residual_20260524.py` | 유사 궤적 기반 KNN residual 전이 실험 |
| `scripts/run_consensus_hit_mode_20260524.py` | 후보 좌표 cloud의 weighted consensus mode 실험 |
| `scripts/make_champion_micro_tuning_20260524.py` | t52/t54 champion 주변 threshold/alpha 미세조정 |
| `scripts/make_champion_alpha_refine_20260524.py` | public feedback 기반 t52 alpha band 후보 생성 |
| `scripts/validate_submission.py` | 제출 파일 shape/null/finite/id 검증 |
| `scripts/publish_to_github.py` | 코드/리포트 범위만 GitHub commit/push |

## 프로젝트 구조

```text
dacon-mosquito-trajectory-prediction/
├── data/                  # 원본/가공 데이터, gitignore
├── docs/                  # 인수인계 및 일별 실험 요약
├── experiments/           # public score와 실험 로그
├── notebooks/             # EDA 및 모델링 노트북
├── reports/               # 최신 실험 리포트
├── scripts/               # 실험/후보 생성/검증 스크립트
├── src/                   # 공통 모듈
├── submissions/           # 제출 파일, gitignore
└── README.md
```

## 데이터 배치

DACON에서 받은 `open.zip` 압축 해제 후 아래 형태를 기대합니다.

```text
data/raw/
├── train/
│   ├── TRAIN_00001.csv
│   └── ...
├── test/
│   ├── TEST_00001.csv
│   └── ...
├── train_labels.csv
└── sample_submission.csv
```

압축 해제 결과가 `data/raw/open (3)/...`처럼 한 단계 더 들어가 있어도 주요 스크립트가 자동 탐지합니다.

## 재현 흐름

```powershell
cd C:\open\dacon-mosquito-trajectory-prediction

# 최신 breakthrough 계열 후보 생성
python scripts/run_hit_weighted_breakthrough_refine.py

# direct-step geometry 계열 후보 생성
python scripts/run_direct_step_geometry.py

# 제출 파일 검증 예시
python scripts/validate_submission.py submissions/direct_step_pure_lgbmboundary_f1.02_s1.00_u1.00.csv

# GitHub 업로드
python scripts/publish_to_github.py --message "Document 2026-05-08 direct-step breakthrough"
```

## 상세 기록

- [2026-05-06 local-frame 실험 정리](docs/experiment_summary_2026-05-06.md)
- [2026-05-07 hit-weighted breakthrough 정리](docs/experiment_summary_2026-05-07.md)
- [2026-05-08 direct-step target 전환 실험 정리](docs/experiment_summary_2026-05-08.md)
- [2026-05-09 direct-step refine 정리](docs/experiment_summary_2026-05-09.md)
- [2026-05-10 selector routing 실험 정리](docs/experiment_summary_2026-05-10.md)
- [2026-05-11 selector soft routing 실험 정리](docs/experiment_summary_2026-05-11.md)
- [2026-05-12 selector soft 후속 연구 정리](docs/experiment_summary_2026-05-12.md)
- [2026-05-15 새 축 재탐색 정리](docs/experiment_summary_2026-05-15.md)
- [2026-05-16 temporal-backcast breakthrough 정리](docs/experiment_summary_2026-05-16.md)
- [2026-05-18 constant-turn curvature breakthrough 정리](docs/experiment_summary_2026-05-18.md)
- [2026-05-19 curvature gate breakthrough 정리](docs/experiment_summary_2026-05-19.md)
- [2026-05-20 새 축 재탐색과 gate 재현성 정리](docs/experiment_summary_2026-05-20.md)
- [2026-05-21 post-process 새 축 손절과 co-champion 안정성 확인](docs/experiment_summary_2026-05-21.md)
- [2026-05-23 temporal curriculum과 공격적 물리 새 축 실패 정리](docs/experiment_summary_2026-05-23.md)
- [2026-05-24 champion alpha calibration 정리](docs/experiment_summary_2026-05-24.md)
- [public score 기록](experiments/public_scores.csv)
- [hit-weighted breakthrough refine 리포트](reports/latest_hit_weighted_breakthrough_refine.md)
- [retrieval blend/router 리포트](reports/latest_retrieval_blend_router.md)
- [direct-step geometry 리포트](reports/latest_direct_step_geometry.md)
- [direct-step refine 리포트](reports/latest_direct_step_refine_20260509.md)
- [direct-step multiplier probe 리포트](reports/latest_direct_multiplier_probe_20260509.md)
- [direct multiplier selector 리포트](reports/latest_direct_multiplier_selector_20260510.md)
- [selector adjustment 리포트](reports/latest_selector_adjustments_20260510.md)
- [velocity smoothing probe 리포트](reports/latest_direct_velocity_smoothing_probe_20260510.md)
- [selector route refine 리포트](reports/latest_selector_route_refine_20260511.md)
- [selector soft temperature 리포트](reports/latest_selector_soft_temperature_20260512.md)
- [expanded selector pool 리포트](reports/latest_expanded_selector_pool_20260512.md)
- [selector seed ensemble 리포트](reports/latest_selector_seed_ensemble_20260512.md)
- [route gain model 리포트](reports/latest_route_gain_model_20260515.md)
- [analog residual correction 리포트](reports/latest_analog_residual_correction_20260515.md)
- [hit probability router 리포트](reports/latest_hit_probability_router_20260516.md)
- [temporal-backcast augmentation 리포트](reports/latest_temporal_backcast_augmentation_20260516.md)
- [temporal-backcast refine 리포트](reports/latest_temporal_backcast_refine_20260516.md)
- [constant-turn curvature 리포트](reports/latest_constant_turn_curvature_20260518.md)
- [curvature gate 리포트](reports/latest_curvature_gate_20260519.md)
- [curvature gate refine 리포트](reports/latest_curvature_gate_refine_20260519.md)
- [gate residual experimental 리포트](reports/latest_gate_residual_experimental_20260519.md)
- [mirror-symmetry temporal TTA 리포트](reports/latest_mirror_tta_temporal_gate_20260520.md)
- [multi-curvature action router 리포트](reports/latest_multi_curvature_action_router_20260520.md)
- [MLP sequence pseudo blend 리포트](reports/latest_mlp_sequence_pseudo_blend_20260520.md)
- [local target manifold projection 리포트](reports/latest_local_target_manifold_projection_20260521.md)
- [hit-rescue specialist 리포트](reports/latest_hit_rescue_specialist_20260521.md)
- [fast temporal curriculum 리포트](reports/latest_temporal_curriculum_fast_20260522.md)
- [aggressive new axes 리포트](reports/latest_aggressive_new_axes_20260523.md)
- [fast two new axes 리포트](reports/latest_fast_two_new_axes_20260523.md)
- [regime miss policy 리포트](reports/latest_regime_miss_policy_20260524.md)
- [analog KNN residual 리포트](reports/latest_analog_knn_residual_20260524.md)
- [consensus hit mode 리포트](reports/latest_consensus_hit_mode_20260524.md)
- [champion micro tuning 리포트](reports/latest_champion_micro_tuning_20260524.md)
- [champion alpha refine 리포트](reports/latest_champion_alpha_refine_20260524.md)

## 비고

원본 데이터와 제출 파일은 용량 및 대회 규정 관리를 위해 GitHub에 포함하지 않습니다.  
이 저장소는 실험 코드, 핵심 결과, 재현 가능한 연구 기록을 중심으로 정리합니다.
