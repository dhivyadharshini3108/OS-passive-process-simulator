# Adaptive Behaviour-Aware CPU Scheduling Simulator

A little side project I built to poke at a question that always bugged me in OS class: *what if the scheduler actually knew what kind of process it was dealing with?*

Linux's scheduler makes decisions using priority and niceness, but it doesn't really "know" whether a process is something you're actively typing into, a background daemon quietly doing its thing, or a heavy batch job crunching numbers. This tool doesn't touch the real scheduler at all (that would be a whole different, much scarier project) — instead it **watches** every running process on your machine, guesses what "type" it is using a simple machine learning model, and shows you how far off the OS's actual scheduling decisions are from what the model thinks would be smarter.

Think of it as a passive x-ray of your system's scheduling behavior.

## What it actually does

1. **Reads `/proc`** — for every PID on the system, it pulls stats out of `/proc/[pid]/stat`, `statm`, `status`, and `cmdline`. No special permissions or kernel hooks, just plain file reads.
2. **Builds a feature vector** per process — memory footprint, code size, page faults, context switches, etc.
3. **Classifies each process** as `interactive`, `daemon`, or `batch` using a hand-rolled k-NN classifier trained on a small seed dataset, with a heuristic fallback (keyword matching on the command name) for anything the model isn't confident about.
4. **Predicts a CPU "burst" time** — basically, how long the process is likely to want the CPU for its next chunk of work — and from that recommends a scheduling time slice (the "STS" class, inspired by ideas from an old scheduling paper I referenced while building this).
5. **Compares reality vs. recommendation** — it looks at the process's actual `nice`/priority values and computes a "mismatch score" showing how much the real OS scheduling diverges from what the model would suggest.
6. **Shows you the results** — either as a live-refreshing colored terminal table, or as a proper HTML dashboard with charts (process type breakdown, priority comparisons, etc.), or as raw JSON if you want to feed it somewhere else.

It also quietly logs what it observes back into a history CSV, so its predictions get a little more grounded in real data on your machine over time.

## Why I built it

Mostly curiosity, honestly. I wanted to see if a lightweight model could reasonably guess process "personality" just from memory/fault/context-switch statistics — no syscall tracing, no kernel patches, nothing invasive. Turns out you can get a surprisingly reasonable classification just from a handful of numbers already exposed in `/proc`. It also gave me an excuse to actually build and visualize something instead of just reading about scheduling theory.

## Building & running it

You'll need a C++17 compiler (g++) and you'll need to be on Linux, since this leans entirely on `/proc`.

```bash
make                 # builds ./passive_monitor
./passive_monitor    # single snapshot, top 15 processes
```

A few flags that are actually useful:

```bash
./passive_monitor --limit 20              # show more processes
./passive_monitor --html report.html      # spit out a nice HTML report
./passive_monitor --loop 3                # live-refresh every 3 seconds
./passive_monitor --output snapshot.json  # dump raw JSON
./passive_monitor --help                  # see everything
```

Or just run `make demo` to get a one-shot JSON + HTML report out of the box.

## A few honest caveats

- **It's a simulator, not a real scheduler.** It never touches actual process scheduling — it only observes and recommends. Nothing here changes how your OS actually runs processes.
- **The classifier is intentionally simple.** k-NN with a handful of seed samples plus keyword heuristics isn't going to rival anything production-grade — it's meant to be understandable and hackable, not state-of-the-art.
- **First-run numbers can look a little off.** CPU% needs two samples to compute a delta, so the very first reading takes an extra beat to stabilize.
- There's a stray `samp.c` and `test_visibility.sh` in here from when I was testing that the monitor could actually "see" a busy-looping process — not part of the core tool, just scratch files I never cleaned up.

## Repo layout

```
passive_monitor.cpp    # everything lives here — reading /proc, classifying, predicting, rendering
Makefile                # build + demo + clean targets
test_visibility.sh      # quick manual test: spins up a busy-loop process and checks the monitor sees it
samp.c                  # tiny infinite-loop test binary used for the above
```

## Possible next steps

If I keep poking at this, ideas I'd want to try:
- Train the k-NN on a bigger, more realistic dataset instead of the small seed list baked into the code
- Add per-core CPU breakdown instead of just an aggregate normalized percentage
- Maybe actually experiment with feeding these recommendations into a `cgroups`-based scheduling nudge, just to see what happens
