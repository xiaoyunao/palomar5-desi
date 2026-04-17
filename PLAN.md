# PLAN

## Current objective

按 `PAL5_CODEX_HANDOFF.md` 的统一思路，先完成 Palomar 5 项目的 Phase 0
预处理基线，产出一个干净、Bonaca-compatible 的 Pal 5 预处理星表和基础 QC，
然后再进入 member selection、distance gradient 和 morphology。

## Scientific goal

本项目当前聚焦两类核心问题：

1. leading tail 在 Bonaca+20 检测到的 low-surface-brightness fan 之后，
   是否还向更南延伸到更低表面亮度区域。
2. 整条星流的形态是否能更好测量，包括：
   - track
   - width
   - density
   - fan / gap 的真实性和几何形态

当前明确不把 spur 定量测量放进第一阶段。

## Agreed strategy

- 主骨架尽量沿用 Bonaca+20。
- 只在真正必要的地方升级。
- 先把 preprocessing 和 baseline morphology 做稳。
- 不立即上全局 2D generative model。
- 不在 preprocessing 阶段直接做物理解读。

## Fixed constraints from handoff

以下约束当前视为已定，不应擅自改动：

1. preprocessing 阶段只保留 `TYPE == PSF`，不要回到 `PSF + REX`。
2. 最终只保留单一 `g0 < 25` 样本，不再输出 24/25 双 catalog。
3. `PSFDEPTH_G` 必须先转为 5-sigma point-source depth in mag 再参与 faint-end guard。
4. dereddening 优先使用 catalog 自带 `MW_TRANSMISSION_G/R/Z`，没有时再 fallback 到 `dustmaps.sfd`。
5. preprocessing 阶段不要做：
   - z-band stellar locus member selection
   - isochrone box member selection
   - distance gradient fitting
   - background template construction
   - 1D density modeling
   - fan / gap 物理解读
6. 第一阶段不要做 spur 分析。
7. 第一阶段不要上全局 2D morphology / generative model。
8. 在 background / completeness 没稳之前，不要对 fan 或 gap 下结论。

## Pipeline roadmap

### Phase 0: Bonaca-compatible preprocessing baseline

目标：建立可复现的 clean stellar catalog 和 QC 基线。

内容：

1. RA-Dec 预裁天区。
2. cluster core hole mask。
3. 严格点源清洗。
4. allmask / bright-star 清洗。
5. grz finite 检查。
6. dereddening。
7. ICRS -> Pal 5 `(phi1, phi2)` 坐标变换。
8. Pal 5 window 裁切。
9. `g0 < 25` + faint-end depth guard。
10. 生成基础 QC 图。

Phase 0 预期输出：

- `pal5_preprocessed_glt25.fits`
- `pal5_preprocessed_summary.json`
- `diagnostics_pal5_preproc/`
- `reports_preproc/preprocess_cutflow.txt`
- `tmp_pal5_preproc/`

### Phase 1: Minimal necessary upgrades

目标：在不改大框架的前提下纳入关键系统学。

内容：

1. distance gradient 改为全流、低自由度、平滑联合拟合。
2. background 改为经验 background template，不再使用裸 linear background。
3. completeness / effective area 开始显式记录。
4. strict vs deep 样本并行，比较稳健性。

### Phase 2: Key structure adjudication

目标：裁决 southern leading tail 的性质，以及 gap 是否真实。

内容：

1. targeted 2D model comparison。
2. injection / recovery。
3. 联合 corrected density、width 和 residual 进行判断。

## Method decisions already made

### Distance gradient

不采用：

- 每个小 `phi1` bin 独立扫 distance modulus。

采用：

- 基于初始 morphology 的全流联合拟合；
- 固定 age 和 metallicity；
- 使用低自由度平滑函数表示 `DM(phi1)`：
  - 两臂分段线性
  - 二次函数
  - 或 3-knot spline

### Background

不采用：

- Bonaca+20 式每个 `phi1` bin 的裸 linear background，直接套到当前更深且覆盖不均匀的数据。

采用：

- 经验 background template：
  - CMD sidebands
  - off-MSTO control sample
  - 与主样本共享 photometric / morphology / depth 条件

原因：

- 当前背景是 `true background x selection function`；
- 若直接套 linear background，会污染 density、width，甚至误判 fan / gap。

## Fan / gap decision criteria

### Fan

应同时考虑：

- track 连续性
- width 显著增大
- peak surface density 下降
- corrected linear density 不一定同步塌掉

### Real gap

应满足：

- integrated / corrected linear density 显著下降
- 对背景和 sample 选择不敏感
- 不与 local depth / coverage / mask pattern 对齐
- 后续最好通过 injection / recovery 验证

### Detached structure

常见信号：

- 单高斯 transverse model 局部明显失效
- 2D residual 呈现有组织的单侧 excess
- 不能仅靠 widening 解释

## Benchmark against Bonaca+20

后续 morphology 阶段可用以下 benchmark 做 sanity check：

- leading tail 在 Bonaca+20 中显著到约 `phi1 ~ +7 deg`
- trailing tail 到约 `phi1 ~ -15 deg`
- leading tail 存在 fan
- prominent gaps 约在 `phi1 ~ -7 deg` 和 `phi1 ~ +3 deg`
- leading 远端宽度可到 `sigma ~ 0.4 deg`
- trailing 通常更窄

## Current repository/runtime split

- `~/Desktop/Palomar 5`：代码修改、备份、git 提交与推送的唯一仓库。
- `~/Desktop/Pal5`：真实运行目录，放大文件、运行产物和本地数据。

## Current status

- background 拟合主线当前已暂停；后续主线重新锚定到 `step4c + step3b(control+MAP)`。
- 当前主线输出目录改为：
  - `/Users/island/Desktop/Pal5/mainline_step4c_rerun_20260417`
  - 其中：
    - `step4c_outputs/`
    - `step4c_step3b_outputs_control/`
- `pal5_step3b_selection_aware_1d_model.py` 已做一个保守的主线修复：
  - arm quadratic 在拟合 `track_poly` 时做轻量 outlier clipping
  - `qc_step3b_density_phi12_local.png` 用平滑后的 `track_poly` 画主 track
- 这样可以避免像旧 `step4c_step3b_outputs_control` 里 `phi1=7.75` 那个 raw local-fit 点把右侧 track 在 local density QC 图里拉坏。
- 仓库已恢复基础结构。
- `pal5_preprocess_step1.py` 已重建并通过语法检查。
- 运行目录中已有一版旧预处理产物，但当前仓库还没有记录一次正式 Phase 0 rerun。
- step 2 strict member-selection 脚本与说明已纳入仓库。
- step 2 已在 `/Users/island/Desktop/Pal5` 用 `astro` 环境成功运行一次，得到 `444,232` 个 strict members。
- step 3 Bonaca-style 1D spatial model 脚本与说明已纳入仓库。
- step 3 已完成两次运行：
  - 默认版 `step3_outputs/`: `n_success = 14 / 41`
  - `--pass2-phi2-halfwidth 1.5` 对照版 `step3_outputs_hw15/`: `n_success = 15 / 41`
- 当前 step 3 还不足以视为稳定 baseline，需要先对图像和失败模式做人工判断。
- step 3b selection-aware 脚本与说明已纳入仓库并完成两组文档规定的 MAP 运行：
  - `step3b_outputs_control/`: `n_success = 39 / 41`, `n_success_excluding_cluster = 37`
  - `step3b_outputs_control_depth/`: `n_success = 27 / 41`, `n_success_excluding_cluster = 25`
- 当前最有前景的 baseline 是 `eta-mode = control` 的 step 3b 结果；`control_times_depth` 暂时不如它稳定。
- `emcee` 已补装到 `astro` 环境，并完成 `step3b_outputs_control_emcee/`：
  - `n_success = 39 / 41`
  - `n_success_excluding_cluster = 37`
  - 与 MAP 版相比，成功 bin 数不变，但 width 整体略宽。
- step 3c 已完成 Bonaca-style comparison / figure-making：
  - `step3c_outputs/` 已生成 figure、metrics、summary、report
  - 当前正式 baseline 应采用 `control + MAP`
  - `control + emcee` 保留为 posterior sanity check
- 新会话恢复确认：
  - 仓库当前 `main` 相对 `origin/main` 为 `ahead 1`
  - 存在未跟踪的 step 4 草稿文件：
    - `PAL5_STEP4_CODEX_INSTRUCTIONS.md`
    - `pal5_step4_refined_dm_selection.py`
  - 项目实际停在“step 4 已起草但尚未运行、尚未提交”的状态
- step 4 refined-DM selection 已完成，并已串联 step 3b / step 3c：
  - `step4_outputs/` 已生成 refined members、DM anchors、DM track 和 QC 图
  - `step4_step3b_control_map/`: `n_success = 40 / 41`, `n_success_excluding_cluster = 38`
  - `step4_step3b_control_emcee/`: `n_success = 38 / 41`, `n_success_excluding_cluster = 36`
  - `step4_step3c_outputs/` 已生成 figure、metrics、summary、report
- refined-DM 版本相对旧 baseline 的主要变化：
  - `|phi1| < 8` integrated stars: `2872 -> 2912`
  - near-cluster width: `0.118 -> 0.123`
  - leading width in `[5, 8]`: `0.287 -> 0.293`，改善极小
  - trailing outer width near `phi1 ~ -10.25`: `0.554 -> 0.486`
  - `|phi1| < 5` trailing/leading: `1.75 -> 1.84`，没有改善到 Bonaca `~1.5`
- 当前结论：
  - refined coarse-anchor `DM(phi1)` 值得保留为一次重要对照，因为它改善了 integrated counts 和 outer trailing width
  - 但 leading fan 偏窄的问题并未因此解决，因此下一阶段瓶颈更可能在 completeness / background / 更平滑的全流 distance model，而不是 step 2 的 two-arm DM baseline 本身
- step 4c RR-Lyrae 弱先验版本现已明确分成两套基线约定：
  - frozen formal baseline v1：继续保留 `step3b_outputs_control/` 与其对应 `step3c_outputs/`
  - working baseline v2：固定为 `step4c_outputs/pal5_step4c_rrlprior_members.fits` + `step4c_step3b_outputs_control/` + `step4c_step3c_vs_step3b_baseline/`
- step 4c working baseline v2 的已核对指标：
  - refined selected members: `456,496`
  - `DM(phi1=-15) = 16.803`
  - `DM(phi1=+8) = 16.540`
  - `|phi1| < 8` integrated stars: `2959.7`
  - `trailing/leading_abs8 = 1.499`
  - `trailing/leading_abs5 = 1.639`
  - leading `[5, 8]` max width: `0.301 deg`
  - trailing `[-15, -5]` max width: `0.535 deg`
- 后续新的 background 实验默认输入：
  - `step4c_outputs/pal5_step4c_rrlprior_members.fits`
  - plotting/QC 只允许改诊断显示，不改 step4c selection 或其科学结果
- step4c plotting-only patch 已实跑完成：
  - `qc_step4c_segment_cmds_fullcmd.png` 已确认显示 full parent on-stream CMD 背景，而不是仅显示打分窗口样本
  - log-scale selected density 图已重生成为后续 background 诊断默认 QC 图
- step4c CMD plotting fix 现以 v3 为准：
  - 保持原始 `on-stream - 0.5 x off-stream` excess Hess 风格
  - 不再叠加 contour / grayscale full-CMD 风格
  - Hess 的显示 parent sample 改为 full z-locus parent，而不是 `20 < g0 < 23` strict-mag-limited sample
  - 当前推荐图为 `step4c_outputs/qc_step4c_segment_cmds_stylefixed_fullhess.png`
- mock-track 可视化链路现已接通：
  - `pal5_visualize_suite.py` 已纳入仓库
  - `pal5_mock_track_fit_refactor.py` 可用 `--make-plots` 在 best-fit 产品落盘后自动出图
  - 默认读取 `run-dir/observed_track_used.fits`，因此能直接吃当前 mock-fit 输出目录
- step5b hybrid empirical-background experiment 已完成，但当前判断为 **not adoptable**：
  - `n_success = 0 / 41`
  - `track_poly_trailing = null`
  - `track_poly_leading = null`
  - `|phi1| < 8 integrated stars = 0`
  - 41 个 bin 全部报 `Desired error not necessarily achieved due to precision loss.`
- 当前解释：
  - step5b 没有重现 step5 的 `sigma_max` blow-up，但 fixed-total-count + empirical-template + weak-log-warp 的 MAP 优化在当前参数化下整体不可识别，导致所有 bin 都被 success gate 淘汰。
- 后续若继续 background 线，优先方向应是：
  - 检查局部 fit 的 success 判据是否过严
  - 检查 BFGS / Hessian 近奇异导致的“precision loss”是否把本可用 bin 全部误杀
  - 只在明确 root cause 后再考虑 step5c，而不是直接增大背景自由度
- step5d weakly-curved sideband background 已完成默认参数试跑，当前结论是 **不优于 step5c**：
  - `n_success = 36 / 41`
  - `n_success_excluding_cluster = 34`
  - `|phi1| < 8 integrated stars = 4468`
  - `trailing/leading_abs8 = 1.436`
  - `trailing/leading_abs5 = 1.594`
  - near-cluster width `= 0.130 deg`
  - leading `[5, 8]` max width `= 0.450 deg`
  - trailing `[-15, -5]` max width `= 0.550 deg`
- 当前解释：
  - step5d 保持了 sideband-anchored family 的稳定性，没有回到 step5b 那种不可识别或 step5 那种撞边失稳。
  - 但弱曲率二次背景在当前正则强度下没有压低 counts，反而让 integrated counts 比 step5c 还高，并且把 outer trailing width 拉回接近 step4c baseline。
- 当前工作排序：
  - `step4c + step3b(control+MAP)` 仍是 frozen working baseline v2
  - 当前继续推进主线时，优先使用 `/Users/island/Desktop/Pal5/mainline_step4c_rerun_20260417`
  - `step5c` 仍是当前最值得保留的 background-upgrade 对照
  - `step5d` 只有在更强正则或更窄 sideband 参数下出现显著改观时才值得继续
- mock-stream 主线已开始从旧 notebook 迁移到独立脚本：
  - 仓库内新增 `pal5_mock_track_fit_refactor.py`
  - 新脚本不再从观测 catalog 重新提取 track，而是直接吃已有 track 表
  - 当前仓库里的实际默认输入应优先使用：
    - `/Users/island/Desktop/Pal5/mainline_step4c_rerun_20260417/step4c_step3b_outputs_control/pal5_step3b_profiles.csv`
  - 因为当前运行目录里并不存在单独的 `phi1/phi2/phi2_err` step4c track 表，所以脚本已兼容 `step3b` profile 列名别名：
    - `phi1_center -> phi1`
    - `mu -> phi2`
    - `mu_err -> phi2_err`
    - `sigma -> width`
    - `sigma_err -> width_err`
- step 4b MSTO-weighted refined-DM selection 已完成，并已串联 step 3b / step 3c：
  - `step4b_outputs/` 已生成 refined members、MSTO-weighted DM anchors、DM track 和 QC 图
  - `step4b_step3b_outputs_control/`: `n_success = 39 / 41`, `n_success_excluding_cluster = 37`
  - `step4b_step3b_outputs_control_emcee/`: `n_success = 40 / 41`, `n_success_excluding_cluster = 38`
  - `step4b_step3c_outputs/` 已生成 figure、metrics、summary、report
- step 4b 相对 step 4 的主要变化：
  - refined member count: `460,278 -> 451,842`
  - anchors success: `15 / 15 -> 12 / 15`
  - `DM(phi1=-15)` 更远，`DM(phi1=+8)` 更近，整体更像 trailing-far / leading-near
  - `MAP` leading width `[5, 8]`: `0.293 -> 0.313`
  - `MAP` trailing outer width `[-15, -5]`: `0.486 -> 0.465`
  - `MAP` integrated stars within `|phi1| < 8`: `2912 -> 2843`
  - `MAP` trailing/leading within `|phi1| < 5`: `1.84 -> 1.81`
- 当前判断：
  - step 4b 说明 MSTO-weighted anchor score 确实能把 morphology 往 Bonaca 的 leading-fan 方向推
  - 但它没有把 integrated counts 和 asymmetry 同步修好，因此不能单靠这一招解决当前差异
  - `step4b + emcee` 可把 leading width 推到 `0.375`，接近 Bonaca `~0.4`，但仍只适合作为 posterior sanity check，而不是新的 formal baseline
- step 5 empirical background + effective-area 模型已做第一次 MAP 试跑：
  - 技术上跑通，但科学结果明显退化
  - `step5_outputs_control_map/`: `n_success = 41 / 41`, `n_success_excluding_cluster = 39`
  - 但 `track_poly_trailing = null`, `track_poly_leading = null`
  - `max_width_leading = 1.2`, `max_width_trailing = 1.2`
  - `integrated_total_abs8 = 3734`
  - 多个外侧 bin 的 `sigma` 顶到上限，说明 stream/background 分解失稳
- 当前结论：
  - step 5 的科学方向仍然合理，但当前实现暂时不可用，不应进入 baseline 候选
  - 这轮运行更像是一次模型调试定位：empirical background + area 的 identifiability 还不够稳
- step 5a off-stream anchored empirical-background 模型已完成第一次 MAP 试跑：
  - `step5a_outputs_control_map/`: `n_success = 41 / 41`, `n_success_excluding_cluster = 39`
  - `track_poly_trailing` / `track_poly_leading` 均恢复正常，不再是 `null`
  - `leading_width_max_5to8 = 0.311`
  - `trailing_width_max_m15to5 = 0.358`
  - `trailing/leading_abs5 = 1.14`
  - 但 `integrated_total_abs8 = 4839`，明显高于 Bonaca-like `~3000`
- 当前结论：
  - step 5a 明显比原 step 5 稳定得多，是 empirical-background 路线上首个“可调试”的版本
  - 它在 asymmetry 和 outer trailing width 上给出了强烈改善信号
  - 但 integrated counts 过高，说明当前 off-stream anchored normalization 仍然有系统偏差
  - 因此 step 5a 现在是“值得继续调试的候选”，但仍不能替代 formal baseline
- step 4c RRL-prior refined-DM selection 已完成，并已串联 step 3b / step 3c：
  - `step4c_outputs/` 已生成 RRL-enriched cache、combined anchors、DM track、report 和 QC 图
  - `step4c_step3b_outputs_control/`: `n_success = 40 / 41`, `n_success_excluding_cluster = 38`
  - `step4c_step3c_vs_step3b_baseline/` 已生成与冻结 formal baseline 的 Bonaca-style对照
- step 4c 相对冻结 formal baseline (`step3b control + MAP`) 的主要变化：
  - `|phi1| < 8` integrated stars: `2872 -> 2960`
  - `trailing/leading_abs8`: `1.68 -> 1.50`
  - `trailing/leading_abs5`: `1.75 -> 1.64`
  - `leading_width_max_5to8`: `0.287 -> 0.301`
  - `trailing_width_max_m15to5`: `0.554 -> 0.535`
  - near-cluster width: `0.118 -> 0.120`
- 当前结论：
  - step 4c 是目前 selection 线中最平衡的一版：counts 仍在 Bonaca-like 范围内，同时 asymmetry、leading width 和 outer trailing width 都朝正确方向改善
  - 因此 `step4c + step3b(control+MAP)` 应作为当前 **working baseline v2 候选**
  - 冻结的 formal baseline 仍保持 `step3b control + MAP`，直到明确决定切换

## Immediate next steps

1. 现在的 selection 线讨论入口应升级为三套对照：
   - 重点比较：
     - `step4c_step3c_vs_step3b_baseline/pal5_step3c_summary.json`
     - `step4_step3c_outputs/pal5_step3c_summary.json`
     - `step4b_step3c_outputs/pal5_step3c_summary.json`
     - `step4_outputs/plots_step4/qc_step4_dm_track.png`
     - `step4b_outputs/plots_step4b/qc_step4b_dm_track.png`
     - `step4c_outputs/qc_step4c_dm_track.png`
     - `step4_outputs/plots_step4/qc_step4_segment_cmds.png`
     - `step4b_outputs/plots_step4b/qc_step4b_segment_cmds.png`
     - `step4c_outputs/qc_step4c_segment_cmds.png`
2. 其中：
   - step 4 更接近 Bonaca 的 integrated counts
   - step 4b 更接近 Bonaca 的 leading width / outer trailing width
   - step 4c 目前是在 counts、asymmetry、leading width、outer trailing width 之间最平衡的一版
3. 当前默认 formal baseline 仍固定为：
   - `control + MAP` = formal baseline
   - `control + emcee` = posterior sanity check
4. refined-DM 系列对照目前可分为：
   - step 4: coarse-anchor refined DM, counts-oriented improvement
   - step 4b: MSTO-weighted refined DM, morphology-oriented improvement
5. mock-stream 下一步不再回旧 notebook：
   - 先用 `pal5_mock_track_fit_refactor.py` 对当前 mainline step3b track 做一次 track-only first pass
   - 第一轮参数保持：
     - free: `log10_mhalo, r_s, q_z, pm_ra_cosdec, pm_dec, distance`
     - fixed: `q_y = 1`, `prog_mass = 2e4 Msun`
   - 若第一轮稳定，再决定是否引入 width term、bar、或额外 halo/progenitor 自由度
   - step 4c: MSTO + RRL weak-prior refined DM, current working-baseline-v2 candidate
5. 下一阶段优先讨论：
   - smoother global `DM(phi1)` model
   - completeness / effective-area bookkeeping
   - empirical background template
6. 若继续 step 5 路线，优先不是直接跑 `emcee`，而是先调试：
   - stream/background mixture 的可辨识性
   - area template 与 empirical background 的归一化
   - `sigma` 与 stream fraction 的先验/参数化
7. 若继续 step 5a 路线，优先调试：
   - off-stream normalization window (`off_inner`, `off_outer`, `bg_exclude_half`)
   - stream `n_stream` 与 integrated-count bookkeeping 的一致性
   - near-cluster region 是否因固定背景过强而被过度扣除
8. 不建议继续只围绕 step 4b 的局部 anchor-scoring 参数做更多手调，除非新的 QC 明确指出某个窗口设置仍在系统性拉偏。
