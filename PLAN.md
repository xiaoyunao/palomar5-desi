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

## Immediate next steps

1. 以 `step4_outputs/` 和 `step4b_outputs/` 两套 refined-DM 对照一起作为下一轮科学讨论入口：
   - 重点比较：
     - `step4_step3c_outputs/pal5_step3c_summary.json`
     - `step4b_step3c_outputs/pal5_step3c_summary.json`
     - `step4_outputs/plots_step4/qc_step4_dm_track.png`
     - `step4b_outputs/plots_step4b/qc_step4b_dm_track.png`
     - `step4_outputs/plots_step4/qc_step4_segment_cmds.png`
     - `step4b_outputs/plots_step4b/qc_step4b_segment_cmds.png`
2. 其中：
   - step 4 更接近 Bonaca 的 integrated counts
   - step 4b 更接近 Bonaca 的 leading width / outer trailing width
3. 当前默认 formal baseline 仍固定为：
   - `control + MAP` = formal baseline
   - `control + emcee` = posterior sanity check
4. refined-DM 系列对照目前可分为：
   - step 4: coarse-anchor refined DM, counts-oriented improvement
   - step 4b: MSTO-weighted refined DM, morphology-oriented improvement
5. 下一阶段优先讨论：
   - smoother global `DM(phi1)` model
   - completeness / effective-area bookkeeping
   - empirical background template
6. 不建议继续只围绕 step 4b 的局部 anchor-scoring 参数做更多手调，除非新的 QC 明确指出某个窗口设置仍在系统性拉偏。
