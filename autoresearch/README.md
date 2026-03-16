# Autoresearch inspired function with agent debate

A Claude Code skill that extends [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) concept beyond ML training into any domain with a measurable metric.

## Background

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) demonstrated that an AI agent can autonomously run ML experiments overnight -- modifying `train.py`, running 5-minute training loops, evaluating validation loss (bits-per-byte), and iterating. The agent produces ~12 experiments per hour, waking you up to a log of attempts and (often) improved models.

This skill generalizes that idea:
- **Any domain** -- ML training, web performance, algorithm optimization, compiler tuning, etc.
- **Soul Debate** -- two adversarial personas (The Architect and The Oracle) debate each experiment before it runs, reducing wasted cycles
- **Configurable via `research.toml`** -- metric name, direction, extraction command, target/frozen files
- **Progress visualization** -- auto-generated `progress.png` chart after every experiment

## Getting Started

### 1. Install the skill

Copy `autoresearch.md` into your Claude Code commands directory:

```bash
cp autoresearch.md ~/.claude/commands/autoresearch.md
```

### 2. Initialize a project

```
/autoresearch init
```

The skill will ask about your research domain and scaffold:
- `research.toml` -- project configuration (metric, files, commands)
- `program.md` -- operating instructions
- `soul_architect.md`, `soul_oracle.md` -- soul personas
- `spiritualguidance.md` -- living memory of soul consultations
- `results.tsv` -- experiment log
- `chart.py` -- progress visualization

### 3. Run the loop

```
/autoresearch run
```

The agent will cycle indefinitely: consult souls, form a hypothesis, edit target files, run the experiment, evaluate, keep or discard, update the chart, and repeat.

### 4. Other commands

```
/autoresearch status    # summarize results + show chart
/autoresearch chart     # regenerate progress.png
/autoresearch add-soul <name> <temperament>  # add a custom soul persona
```

## How It Differs from Karpathy's Original

| | Karpathy's autoresearch | This skill |
|---|---|---|
| Domain | ML training (GPT on text) | Any measurable domain |
| Agent | Generic agent instructions | Soul Debate system with adversarial personas |
| Config | Hardcoded to `train.py` / val_bpb | `research.toml` with configurable metric, files, commands |
| Visualization | Manual | Auto-generated `progress.png` after every cycle |
| Memory | None | `spiritualguidance.md` with Current Canon promotion |
