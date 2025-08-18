## Fix: Evaluation Function Coverage Inconsistency

### Problem
The current `evaluate()` function produces inconsistent results based on `batch_size × block_size` due to variable validation coverage. Models with different batch sizes aren't fairly comparable.

### Root Cause
- `iter_full_split()` creates non-overlapping windows of size `batch_size × block_size + 1`
- Number of evaluation windows varies: `floor((len(val_ids) - span) / span) + 1`
- Same denominator (`len(val_text)`) but different numerators create batch-size-dependent metrics

### Example Impact
- `if batch_size * block_size / len(val_text) < 2`: (1 window) → artificially low loss
- `if batch_size * block_size / len(val_text) > 2`: (2 windows) → higher loss for same model

### Solution
Added `evaluate_consistent_coverage()` function that:
- Record token number
- Processes all available validation tokens
- Returns per-token loss for fair comparison
- Maintains backward compatibility (original function unchanged)

### Testing
- Verified consistent results across different batch sizes
- Confirmed same model produces same evaluation score
- Original evaluation preserved for assessment compatibility

This ensures fair model comparison and eliminates batch-size-dependent evaluation artifacts.