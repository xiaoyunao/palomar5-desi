# WORKLOG

## 2026-04-12

- Task: 按 handoff 重建 Palomar 5 代码仓库，并把项目当前的科学目标、约束和阶段计划重新落到仓库文档里。
- Files changed: `README.md`, `WORKLOG.md`, `PLAN.md`, `.gitignore`, `pal5_preprocess_step1.py`
- Commands run:
  - `git status --short --branch`
  - `git branch --show-current`
  - `git fetch --all --prune`
  - `git log --oneline --decorate --graph -n 15 --all`
  - `ls -la`
  - `sed -n '1,520p' /Users/island/Desktop/PAL5_CODEX_HANDOFF.md`
  - `python -m py_compile pal5_preprocess_step1.py`
  - `git add .gitignore README.md WORKLOG.md PLAN.md pal5_preprocess_step1.py`
  - `git commit -m "Restore Pal 5 preprocessing repository baseline"`
- Key findings:
  - 代码仓库 `/Users/island/Desktop/Palomar 5` 在最新提交后只剩 `.git`，需要从 handoff 重建。
  - 真实运行目录是 `/Users/island/Desktop/Pal5`，只用于实际运行和保存大产物；代码和 git 操作必须留在当前仓库。
  - `PAL5_CODEX_HANDOFF.md` 已明确当前第一阶段目标不是直接做 morphology，而是先完成 clean stellar catalog、Pal 5 坐标变换和 QC 的 preprocessing baseline。
  - 当前已确定的科学边界包括：
    - 先不做 spur
    - 不做全局 2D generative model
    - preprocessing 阶段不做 member selection / DM 拟合 / background template / fan-gap interpretation
  - 当前已确定的实现约束包括：
    - 点源定义使用 `TYPE == PSF`
    - 只保留单一 `g0 < 25` catalog
    - `PSFDEPTH_G` 需先转为 5-sigma depth magnitude
    - dereddening 优先使用 `MW_TRANSMISSION_G/R/Z`，否则 fallback 到 `dustmaps.sfd`
- Validation result:
  - `pal5_preprocess_step1.py` 已通过 `python -m py_compile`
  - 仓库基线已提交为 `4131b3d Restore Pal 5 preprocessing repository baseline`
- Remaining issues:
  - 还没有从当前仓库驱动一次正式的 preprocessing rerun
  - 运行目录中的旧输出尚未整理成正式 run note
  - `PLAN.md` 和 `WORKLOG.md` 初版过于简略，已在本次更新中改为 handoff-aware 版本
- Next step:
  - 继续把脚本与运行目录衔接好
  - 在 `/Users/island/Desktop/Pal5` 环境里正式重跑 preprocessing
  - 把 cutflow、summary、QC 检查结果回写到仓库文档
