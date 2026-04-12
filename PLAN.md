# PLAN

## Current objective

Rebuild the Palomar 5 repository as the canonical code/backup/git workspace and
restore the preprocessing baseline described in the handoff.

## Milestones

1. Restore the repository structure and project memory files.
2. Recreate the standard preprocessing script `pal5_preprocess_glt25.py`.
3. Commit the baseline repository state.
4. Use this repository to drive the next actual runtime execution in `/Users/island/Desktop/Pal5`.
5. After preprocessing is confirmed, move to strict member-selection baseline.

## Outstanding issues

- The repository was emptied by the latest commit and had to be rebuilt.
- Runtime products and actual data are outside the repository.
- The real preprocessing run still needs to be executed in the runtime environment.

## Validation criteria

- `README.md`, `WORKLOG.md`, and `PLAN.md` exist and reflect the current workflow.
- `pal5_preprocess_glt25.py` passes syntax validation.
- The repository is ready for a clean milestone commit.

## Next recommended steps

1. Review and commit the restored baseline files.
2. If needed, parameterize script I/O paths for cleaner handoff between repo and runtime directory.
3. Run the preprocessing job from the runtime environment and archive the run note in this repo.
