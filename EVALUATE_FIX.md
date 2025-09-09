## Fix: Evaluation Coverage Inconsistency Across Batch Sizes

### Problem
The current `evaluate()` function produces inconsistent validation loss based on `batch_size` configuration, making model comparisons unfair. Models with different batch sizes evaluate different amounts of validation data but use the same normalization denominator.

### Root Cause
- `iter_full_split()` creates non-overlapping windows of size `batch_size × block_size + 1`
- Number of evaluation windows varies: `floor((len(val_ids) - span) / span) + 1`
- Loss calculation: `sum(token_losses) / len(val_text)` (characters)
- Same character denominator, different token numerators → batch-size-dependent metrics

**Example:**
- `when (batch_size × block_size + 1) / len(val_text) < 2  `: 1 window → artificially low loss
- `when (batch_size × block_size + 1) / len(val_text) > 2`: : more then 2 window → higher loss for identical model

### Solution
Added `create_evaluation_functions()` factory that provides:

1. **Original function** (`evaluate_char_normalized`) - unchanged for compatibility
2. **Fixed function** (`evaluate_token_average`) - consistent per-token normalization

**Key fix:** `sum(token_losses) / total_tokens_evaluated` instead of character count

### Implementation
- Tracks actual tokens evaluated with `total_tokens += yb.numel()`
- Normalizes by token count: `sum_nll / max(1, total_tokens)`
- Dual logging for side-by-side comparison
- Zero breaking changes - original behavior preserved

### Result
- Fair model comparison regardless of batch_size
- Consistent evaluation metrics across configurations
- Easy migration path for maintainers
- Backward compatibility maintained

**Testing:** Verified identical models produce consistent scores across different batch sizes with the fixed evaluation function.