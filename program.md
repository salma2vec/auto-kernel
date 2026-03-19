# Program

Use this loop for each experiment:

1. Run `prepare.py` once per environment change.
2. Run `bench.py` to establish a correctness-checked baseline.
3. Run `profile.py` to classify the bottleneck.
4. Make one focused change to `kernel.py`.
5. Re-run `bench.py`.
6. Keep the change only if correctness holds and the results improve.
7. Record the outcome in `MEMORY.md`, `memory/<kernel>.md`, and `results.tsv`.

