# WORKLOG

## 2026-04-11

- Task: initialize local Pal 5 rebuild workspace and recover remote context
- Files changed: `README.md`, `WORKLOG.md`, `PLAN.md`
- Commands run: `git status --short --branch`; `git branch --show-current`; `git fetch --all --prune`; `git log --oneline --decorate --graph -n 15 --all`; `ls -la`; multiple `ssh yunao@perlmutter.nersc.gov ...` inspections under `/pscratch/sd/y/yunao`
- Key findings:
  - local directory was empty and not yet a git repository
  - relevant remote directories are `/pscratch/sd/y/yunao/palomar5` and `/pscratch/sd/y/yunao/Palomar_5_new`
  - `Palomar_5_new` contains the newer real-stream detection front end split into pre-selection, isochrone handling, and CMD/spatial probability estimation
  - login node access is currently valid
- Validation result: remote access confirmed; code structure identified
- Remaining issues:
  - notebook logic still needs to be rewritten into scripts
  - full-scale processing must be split into login-safe and compute-node stages
  - background model should be improved for non-uniform depth and source density
- Next step: implement a script pipeline for the real-stream detection stage and add targeted methodological improvements
