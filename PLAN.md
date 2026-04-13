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

## Immediate next steps

1. 优先人工检查 `step3b_outputs_control/`：
   - `qc_step3b_control_density_phi12.png`
   - `qc_step3b_density_phi12_local.png`
   - `qc_step3b_track.png`
   - `qc_step3b_width.png`
   - `qc_step3b_linear_density.png`
   - `qc_step3b_example_local_fits.png`
2. 与 `step3b_outputs_control_depth/` 对照，确认 `control` 模式是否在形态和稳定性上都更优。
3. 若 `control` 模式图像也合理，则把它作为当前 step 3 baseline，并只在那之后考虑：
   - `--sampler emcee` 的 posterior refinement
4. 在 step 3b baseline 确认后，再讨论 smoother `DM(phi1)` 与进一步 background/completeness 升级。
