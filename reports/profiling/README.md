# Profiling Reports

Phase 2 profiling artifacts are generated locally with:

```bash
make profile-train
make profile-predict
```

The scripts write raw `cProfile` files and readable `*_cprofile.txt` summaries
under this directory. The raw `.prof` files are ignored because they are
machine/run-specific. The text summaries are kept as lightweight evidence for
the Phase 2 profiling deliverable.
