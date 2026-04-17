# Pal 5 visualization suite usage

主程序：`pal5_visualize_suite.py`

## 和 mock-fit 主程序直接串联

现在可以直接在 `pal5_mock_track_fit_refactor.py` 后面接可视化，不需要再手动补
`--track-file`。mock-fit 会先写出 `observed_track_used.fits`，然后把它传给
可视化程序。

最小示例：

```bash
python pal5_mock_track_fit_refactor.py \
  --track /Users/island/Desktop/Pal5/mainline_step4c_rerun_20260417/step4c_step3b_outputs_control/pal5_step3b_profiles.csv \
  --outdir /Users/island/Desktop/Pal5/mockfit_mainline_step4c_trackonly \
  --ncores 6 \
  --nwalkers 48 \
  --burnin 200 \
  --steps 600 \
  --make-plots \
  --plot-skip-rv-grids
```

如果要叠加背景星表：

```bash
python pal5_mock_track_fit_refactor.py \
  --track /Users/island/Desktop/Pal5/mainline_step4c_rerun_20260417/step4c_step3b_outputs_control/pal5_step3b_profiles.csv \
  --outdir /Users/island/Desktop/Pal5/mockfit_mainline_step4c_trackonly \
  --make-plots \
  --plot-star-file /path/to/filtered_data.fits \
  --plot-star-ra-col RA \
  --plot-star-dec-col DEC
```

## 最常用的运行方式

假设你已经有：

- step4c 输出的观测 track：`pal5_track_from_step4c.fits`
- refactor 主程序输出目录：`pal5_mockfit_run/`
- 可选的背景星表：`filtered_data.fits`

### 1. 画全部图（最全）
```bash
python pal5_visualize_suite.py \
  --run-dir pal5_mockfit_run \
  --track-file pal5_track_from_step4c.fits \
  --star-file filtered_data.fits \
  --star-ra-col RA \
  --star-dec-col DEC \
  --star-distance-col distance \
  --star-max-distance 0.2
```

### 2. 不画最耗时的 RV distance-grid
```bash
python pal5_visualize_suite.py \
  --track-file pal5_track_from_step4c.fits \
  --run-dir pal5_mockfit_run \
  --star-file filtered_data.fits \
  --skip-rv-grids
```

### 3. 只做基础图（track + MCMC + orbit）
```bash
python pal5_visualize_suite.py \
  --track-file pal5_track_from_step4c.fits \
  --run-dir pal5_mockfit_run \
  --skip-rv
```

## 默认会输出的图

保存到 `pal5_plots/`：

1. `01_stream_overlay_pal5`  
   Pal 5 坐标下的背景星图 + mock 粒子 + 观测 track + model track

2. `02_stream_overlay_tracks_only`  
   去掉背景 hexbin，只保留 stream / track

3. `03_track_comparison_residual`  
   上面是 observed vs model track，下面是 residual

4. `04_width_and_counts`  
   width 对比（如果观测表里有 width），以及沿流 counts

5. `05_bonaca_style_summary`  
   一个合并 summary figure

6. `06_mcmc_corner`  
   corner plot

7. `07_mcmc_chains`  
   MCMC walker chains

8. `08_mcmc_logprob`  
   log-probability

9. `09_bestfit_orbit`  
   最优参数轨道的 R-Z 和 X-Y 投影

10. `10_orbit_distance_grid`  
    distance-grid 轨道图

11. `11_rv_aitoff_bar_vs_nobar`  
    全天 Aitoff RV 图

12. `12_rv_radec_bar_vs_nobar`  
    RA-Dec 平面 RV 图

13. `13_rv_distance_grid_nobar`  
    no-bar 的 distance-grid RV 图

14. `14_rv_distance_grid_bar_vs_nobar`  
    bar / no-bar 对照的 distance-grid RV 图

15. `15_qz_literature_comparison`  
    你原 notebook 里的 qz 文献比较图

## 输入文件要求

### observed track
最少需要：
- `phi1`
- `phi2`

推荐同时有：
- `phi2_err`
- `width`
- `width_err`

### run-dir
需要这些文件（由 refactor 主程序产出）：
- `best_fit_params.csv`
- `best_fit_model_track.fits`
- `best_fit_mock_stream_particles.fits`
- 推荐同时存在 `observed_track_used.fits`

如果 `run-dir/observed_track_used.fits` 存在，`pal5_visualize_suite.py` 可以省略
`--track-file`。

如果存在以下文件，还会自动画 MCMC 诊断图：
- `mcmc_samples.csv`
- `chain.npy`
- `log_prob.npy`

## 可调参数

- `--pattern-speed`：bar 的 pattern speed，默认 42
- `--orbit-step-myr` / `--orbit-nsteps`：轨道积分设置
- `--rv-step-myr` / `--rv-nsteps`：RV mock stream 设置
- `--orbit-dmin --orbit-dmax --orbit-ndist`：orbit distance-grid
- `--rv-dmin --rv-dmax --rv-ndist`：RV no-bar distance-grid
- `--rv-npairs`：bar vs no-bar distance-grid 的距离个数
