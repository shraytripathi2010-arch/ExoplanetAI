# Environment notes — platform workarounds that are not obvious

Things that cost real debugging time on this project and would otherwise have
to be rediscovered. Everything here was verified on the machine it describes,
not recalled. Each entry states the symptom first, because the symptom is what
a future session will actually have in hand.

Host this was written against: macOS (Darwin 25.x), Apple Silicon, system
Python 3.11 from `/Library/Frameworks/Python.framework`, **no Homebrew
installed**.

---

## 1. LightGBM / XGBoost need OpenMP, and this machine has no Homebrew

**Symptom.** `import lightgbm` or `import xgboost` fails with a loader error
naming `@rpath/libomp.dylib` — the library exists on disk nowhere the dynamic
loader looks.

**Why the obvious fix fails.** Setting `DYLD_LIBRARY_PATH` works when you test
it interactively in a shell, and then silently does nothing in the actual run.
**macOS System Integrity Protection strips every `DYLD_*` variable when
exec'ing a protected binary.** This project's long jobs go through
`/usr/bin/caffeinate`, which is protected, so the variable was gone by the time
Python started. Verified directly at the time: the child process read
`DYLD seen by child: None`. Any wrapper under `/usr/bin` or `/bin` does this —
it is not specific to `caffeinate`.

**The fix that works.** Both dylibs declare their dependency as
`@rpath/libomp.dylib` and ship with rpaths baked in at build time. Check them:

```bash
otool -l "$(python3 -c 'import lightgbm,os;print(os.path.dirname(lightgbm.__file__))')/lib/lib_lightgbm.dylib" | grep -A2 LC_RPATH
```

On this machine that prints `/opt/homebrew/opt/libomp/lib` and
`/opt/local/lib/libomp` — the Homebrew and MacPorts locations. Neither exists
without those package managers, but **`/opt` is writable without sudo**, so the
directory can simply be created and populated. PyTorch ships its own copy of
libomp, which is the one used here:

```bash
mkdir -p /opt/homebrew/opt/libomp/lib
cp "$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')/lib/libomp.dylib" /opt/homebrew/opt/libomp/lib/
```

Verified state on this machine: `/opt/homebrew/opt/libomp/lib/libomp.dylib`,
856,096 bytes, md5 `768c82fd9ff13987a264bd6315765e32` — byte-identical to
torch's copy. It is a plain copy, not a symlink, so it survives a torch
reinstall. This satisfies the rpath permanently; no environment variable is
involved and nothing can strip it.

**This is a local-machine workaround, not a project dependency.** It lives
outside the repo by necessity. On a fresh machine, prefer `brew install libomp`
if Homebrew is available — that produces the same path legitimately. The recipe
above is the fallback when it is not.

## 2. The ctypes preload that fixed §1 then became the bug

**Symptom.** The run dies at **exit code 139 (SIGSEGV)** after printing its
header and before any result. Silent, with no traceback — a segfault is not a
Python exception, so `try/except` never sees it and the log just stops.

**Cause.** Before §1 was solved, the workaround was to `ctypes.CDLL()` a libomp
up front. Once the rpath was satisfied, that preload loaded a *second* OpenMP
runtime into the same process. **Two OpenMP runtimes in one process is
undefined behaviour and reliably segfaults here.** (The preload never actually
worked as a fix on its own either: `ctypes.CDLL` does not satisfy an
`@rpath/` dependency for a library loaded later by the dynamic loader.)

**The fix.** `preload_libomp()` in `code/experiments/bakeoff_followup.py`
**probes first** — attempts the import, and only falls back to preloading if
that fails. Keep that shape. If you write a new script that preloads
unconditionally, you will reintroduce a silent segfault.

**Lesson worth keeping separately from the specific bug:** a job that exits
non-zero with no traceback should be checked for 139 before anything else is
theorised. `echo $?` immediately after the run.

## 3. Nested parallelism thrashes instead of scaling

**Symptom.** A `RandomizedSearchCV(n_jobs=-1)` over a multithreaded
gradient-booster ran at **36.9% CPU across 8 cores** — slower than
single-threaded, with no error.

**Fix.** Pin the inner library to one thread and let the outer search own the
parallelism:

```bash
OMP_NUM_THREADS=1 python3 your_script.py
```

That took the same job to **306.7% CPU**. Applies to LightGBM, XGBoost,
CatBoost and sklearn's HistGradientBoosting alike.

### 3a. …but 8 workers is past the memory limit, not the CPU limit

`OMP_NUM_THREADS=1` fixes CPU contention; it does nothing about RAM. This
machine has 8 physical cores, so 8 worker processes looks correct and is not.

**Symptom.** A `ProcessPoolExecutor(max_workers=8)` running CatBoost fits went
to **load average 203** while the workers sat at **15–25% CPU each** — high
load with idle CPUs, which is paging, not computing. `sysctl vm.swapusage`
confirmed **5.9 GB of 7 GB swap used**. The last four resamples of that run
took longer than the first eight had, despite having the machine to
themselves.

**Rule of thumb.** 6 workers has run clean repeatedly here (the 28-arm sweep);
8 tipped it into swap. Prefer 6 for anything holding a copy of the training
matrix per worker, and check `sysctl vm.swapusage` before blaming the code
when a parallel job crawls.

## 4. TabPFN cannot be installed here

`TabPFNLicenseError` on import: it requires a **one-time interactive license
acceptance** before it will download model weights, and these sessions have no
interactive terminal. Not worked around — recorded as genuinely unavailable so
nobody re-attempts it. (The dataset, 4,386 × 24, would have fit inside its
~10k-row / ~500-feature envelope with no subsampling, so this is a real gap in
coverage rather than a moot point.)

## 5. `timeout` does not exist on macOS

**Symptom, and why it is dangerous:** a command wrapped in `timeout ...` fails,
but the pipeline's exit status comes from a later `echo`, so the whole thing
reports **exit 0 and the real work never ran**. This silently skipped an
install on this project once.

Use `gtimeout` (coreutils) if present, or restructure so the command's own exit
status is what gets checked. Do not assume a `0` from a compound command means
the interesting part succeeded.

## 6. Long-running processes hold pre-fix code

This project has been bitten by this **four times**. Python caches modules at
import; the Flask app under launchd keeps whatever `web/*.py` looked like when
it started. Editing a file and observing the old behaviour is not a mystery, it
is this.

**Check before concluding a fix did not work:**

```bash
ps -p "$(pgrep -f 'app.py')" -o lstart=
```

Compare that to the file's mtime (`stat -f "%Sm %N" web/app.py`). If the file
is newer, restart:

```bash
launchctl kickstart -k gui/$(id -u)/com.exoplanetai.app
```

Then confirm: `curl -s localhost:5050/health` should return 200 with a
`seconds_since_last_tick` in the single digits. Exit status −15 in
`launchctl list` is SIGTERM (a clean stop); −9 is SIGKILL.

## 7. Port 5000 is not this app

macOS AirTunes/AirPlay Receiver binds port 5000 and will answer requests,
producing confusing responses that look like a broken app. **This app listens
on 5050.**

## 8. `oktopus` / PRF fitting is unavailable

`lightkurve.prf` warns that `tpfmodel` is unavailable without `oktopus`, which
needs an older `autograd`. This is expected and harmless. The centroid vetting
in this project deliberately uses a **difference-image centroid** (numpy +
`scipy.ndimage.center_of_mass`), which needs none of it.

## 9. `roc_auc_score` is 25 ms/call and dominates any bootstrap

**Symptom.** A paired bootstrap over the 1,098-star test set was projected at
**47 minutes** for a single sweep. Profiling put `sklearn.metrics.roc_auc_score`
at **25 ms per call** on n=1098 — almost all of it input validation, not
arithmetic. At 2,000 iterations x 2 arms x 56 comparisons that is the entire
runtime. This is why earlier validation runs took 35–75 minutes and why they
were sized at 8 resamples.

**Fix.** AUC is a rank statistic, so compute it directly via the rank-sum
identity, with *averaged* ranks so ties are handled exactly:

```python
def fast_auc(y, p):
    n = p.shape[0]
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    newgrp = np.empty(n, bool); newgrp[0] = True
    np.not_equal(sp[1:], sp[:-1], out=newgrp[1:])
    gid = np.cumsum(newgrp) - 1
    avg = (np.bincount(gid, weights=np.arange(1, n + 1, dtype=np.float64))
           / np.bincount(gid))
    r = np.empty(n, np.float64); r[order] = avg[gid]
    n1 = int(y.sum())
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * (n - n1))
```

**~18x faster, and exact** — verified to 1e-12 against `roc_auc_score` over 400
random cases including tie-heavy ones. Ties are not hypothetical here: isotonic
calibration emits long plateaus of identical probabilities, so ordinal ranks
without tie-averaging would give wrong answers for exactly those arms.

**Where it lives: `code/fast_auc.py`**, importable by both the experiment
scripts and `web/` (which already puts `code/` on `sys.path`). It exports two
names:

- `fast_auc(y, p)` — the bare statistic, for hot loops.
- `roc_auc_score(y, p, **kwargs)` — a **drop-in** for sklearn's that fast-paths
  the binary / 1-D / no-keyword / finite case and **delegates to sklearn for
  everything else** (multiclass, `sample_weight`, `max_fpr`, NaNs). Swapping the
  import therefore cannot change a result: it either returns the identical
  value or calls the original function.

Run `python3 code/fast_auc.py` to re-verify exactness, the delegation paths,
and the speedup on this machine.

Applied across 34 scripts plus `web/retrain_pipeline.py`. The production
`_paired_bootstrap_auc_diff` was checked old-vs-new on real probability vectors
(including a heavy-tie isotonic arm): mean and both CI bounds identical to
1.1e-16, 15.3x faster (31.4 s -> 2.0 s).
