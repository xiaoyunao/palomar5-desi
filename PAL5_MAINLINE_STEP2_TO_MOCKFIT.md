# Pal 5 主线流程（从 step2 到 MCMC 与可视化）

这个文件只保留当前建议的主线，不再把调试分支、旧 notebook 轨迹重提取、step5/5a 替代背景实验混进来。

## 主线目标

从已经完成的预处理样本出发：

```text
final_g25_preproc.fits
    ↓
step2  严格成员选择（Bonaca-style baseline）
    ↓
step3b control + MAP（先在 step2 strict sample 上建立 selection-aware ridge）
    ↓
step4c RR Lyrae weak-prior DM(phi1) refinement
    ↓
step3b control + MAP（在 step4c refined members 上重跑，形成新的主线轨迹表）
    ↓
pal5_mock_track_fit_refactor.py（直接吃 step3b profile table）
    ↓
pal5_visualize_suite.py
```

## 为什么这条是主线

1. **从 step2 开始**：你已经明确说了不需要再回到 preprocessing / 各种旧调试链。
2. **step4c + step3b**：这是当前 repo README 里已经写明的 photometric working baseline。
3. **不再重新提取 observed track**：mockfit 直接吃 `step3b_profiles.csv`，这是当前 README 已经明确写出来的用法。
4. **可视化独立但可串联**：mockfit 可以在 best-fit 写完后自动调用 visualization suite。

## 需要的核心输入

### 代码目录（repo root）

至少需要这些文件：

- `pal5_step2_member_selection.py`
- `pal5_step3b_selection_aware_1d_model.py`
- `pal5_step4c_rrlprior_dm_selection.py`
- `pal5_mock_track_fit_refactor.py`
- `pal5_visualize_suite.py`
- `pal5_rrl_price_whelan_2019_subset.csv`

### 数据/运行目录（runtime directory）

至少需要这些输入：

- `final_g25_preproc.fits`
- `pal5.dat`
- 可选：`step3_outputs_hw15/pal5_step3_pass1_prior_track.txt`

## 每一步的作用和关键输出

### Step 2

脚本：`pal5_step2_member_selection.py`

输入：
- `final_g25_preproc.fits`
- `pal5.dat`

输出：
- `step2_outputs/pal5_step2_strict_members.fits`
- `step2_outputs/pal5_step2_summary.json`

这一步是保守的 Bonaca-style strict sample，不是最终 deep sample。

---

### Step 3b（baseline）

脚本：`pal5_step3b_selection_aware_1d_model.py`

输入：
- `step2_outputs/pal5_step2_strict_members.fits`
- `final_g25_preproc.fits`
- `step2_outputs/pal5_step2_summary.json`
- `pal5.dat`
- 可选：`step3_outputs_hw15/pal5_step3_pass1_prior_track.txt`

输出：
- `step3b_outputs_control/pal5_step3b_profiles.csv`
- `step3b_outputs_control/pal5_step3b_summary.json`
- `step3b_outputs_control/pal5_step3b_mu_prior.txt`

这一步的任务不是最终定稿，而是先得到一个 selection-aware 的稳定 ridge / morphology baseline。

---

### Step 4c

脚本：`pal5_step4c_rrlprior_dm_selection.py`

输入：
- `final_g25_preproc.fits`
- `step2_outputs/pal5_step2_summary.json`
- `pal5.dat`
- `step3b_outputs_control/pal5_step3b_mu_prior.txt`
- `pal5_rrl_price_whelan_2019_subset.csv`

输出：
- `step4c_outputs/pal5_step4c_rrlprior_members.fits`
- `step4c_outputs/pal5_step4c_summary.json`
- `step4c_outputs/pal5_step4c_dm_track.csv`

这一步的核心是：
- 保留 step4b 的 MSTO-weighted 想法；
- 用少量 RRL 只做 `DM(phi1)` 形状上的 weak prior；
- 不拿 RRL 去硬替换 cluster zero-point。

---

### Step 3b（重跑在 step4c refined members 上）

脚本：`pal5_step3b_selection_aware_1d_model.py`

输入：
- `step4c_outputs/pal5_step4c_rrlprior_members.fits`
- `final_g25_preproc.fits`
- `step2_outputs/pal5_step2_summary.json`
- `pal5.dat`
- `step3b_outputs_control/pal5_step3b_mu_prior.txt`

输出：
- `step4c_step3b_outputs_control/pal5_step3b_profiles.csv`
- `step4c_step3b_outputs_control/pal5_step3b_summary.json`

这一步产出的 `pal5_step3b_profiles.csv` 就是后面 mockfit 直接读取的主线 observed-track 表。

---

### Mockfit

脚本：`pal5_mock_track_fit_refactor.py`

输入：
- `step4c_step3b_outputs_control/pal5_step3b_profiles.csv`

输出：
- `mockfit_mainline_step4c_trackonly/best_fit_params.csv`
- `mockfit_mainline_step4c_trackonly/best_fit_model_track.fits`
- `mockfit_mainline_step4c_trackonly/best_fit_mock_stream_particles.fits`
- `mockfit_mainline_step4c_trackonly/observed_track_used.fits`
- MCMC chain / summary / corner 等

默认推荐先做 **track-only fit**，不强行把 width term 打开。

---

### Visualization

脚本：`pal5_visualize_suite.py`

最方便的方式是让 mockfit 用 `--make-plots` 直接串联。

默认输出：
- `mockfit_mainline_step4c_trackonly/pal5_plots/`

主线默认保留：
- stream / track overlay
- model vs observed track + residual
- corner / walkers / log-prob
- orbit figures

主线默认不再生成：
- `11` 到 `14` 的 RV / bar-vs-no-bar 图
- `15` 的 literature `q_z` comparison

这样当前主线可视化默认停在 `10_orbit_distance_grid`。

## 一条龙 driver

文件：`pal5_mainline_step2_to_mockfit.sh`

这个 driver 已经把上面这些步骤都串起来了，而且默认采用：

- `step3b` 用 `control + MAP`
- `step4c` 用 `RR Lyrae weak prior`
- `mockfit` 直接吃 `step4c_step3b_outputs_control/pal5_step3b_profiles.csv`
- `mockfit` 完成后自动调用 `pal5_visualize_suite.py`

### 最常见的调用方式

```bash
chmod +x pal5_mainline_step2_to_mockfit.sh

CODE_DIR=/path/to/palomar5-desi \
DATA_DIR=/path/to/Pal5 \
PYTHON_PIPELINE=/path/to/python \
PYTHON_MOCKFIT=/path/to/python \
./pal5_mainline_step2_to_mockfit.sh
```

## 这个 driver 里保留的可调参数

### 是否断点续跑

```bash
RESUME=1
```

如果关键输出已经存在，就跳过该 stage。

### mockfit 采样参数

```bash
MCMC_NCORES=1
MCMC_MP_START_METHOD=spawn
MCMC_NWALKERS=16
MCMC_BURNIN=5
MCMC_STEPS=5
MOCKFIT_N_STREAM_STEPS=3000
MOCKFIT_MIN_VALID_FRACTION=0.30
```

这里把 stream length 拉回到更接近旧 notebook MCMC 配置的量级。
之前为了压低 walltime 临时用过 `1500`，会让 model-track 的有效节点范围明显缩短，不适合当前主线默认值。
并行接口现在保留为可选项，但在当前 macOS 本机测试里还不够稳定，因此默认值仍保守保持 `MCMC_NCORES=1`。

### 是否启用 width term

```bash
USE_WIDTH_TERM=0
```

默认关闭，先保持 track-only 主线。

### 是否自动画图

```bash
RUN_PLOTS=1
```

### 是否跳过重型 RV 网格图

```bash
PLOT_SKIP_RV=1
PLOT_SKIP_RV_GRIDS=1
PLOT_SKIP_LITERATURE=1
```

默认跳过全部 RV 与 literature 图，主线只保留 `01` 到 `10`。

## 当前最推荐的跑法

对你现在这个项目阶段，最推荐的是：

1. 先把 `step2 -> step3b -> step4c -> step3b` 跑顺；
2. 用这条主线的 `pal5_step3b_profiles.csv` 去做 mockfit；
3. 当前 poster / 结果展示先以：
   - DM 图
   - observed stream + extracted track
   - best-fit mock vs observed
   - orbit / track residual 图
   为主；
4. corner 图保留，但不要当 poster 的主角。

## 不再纳入这条主线的东西

下面这些先都不作为主线：

- 旧 notebook 中重新从 filtered catalog 直接提 observed track
- 旧 notebook 中 particle-to-track 的 likelihood
- step5 / step5a 的背景实验分支
- 各种临时 plotting patch 当成核心流程的一部分

它们可以留作对照或诊断，但不再放进“从头到尾”的默认主线。
