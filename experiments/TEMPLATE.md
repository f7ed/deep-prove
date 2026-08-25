# Experiment NNN: Title

## Metadata

| Item | Value |
|---|---|
| Status | Planned / In progress / Archived |
| Branch | `exp/<short-topic>` |
| Base commit | `<commit>` |
| Archive commit | `<commit when complete>` |
| Date and timezone | `<date>` |

## Research question

State the narrow question and the decision this experiment should support.

## Scope

List what may change and what is deliberately excluded.

## Baseline definition

Record the exact workload, timing boundaries, commands, backend, quantization, thread count, machine, random seed, and repetitions.

## Existing implementation

Trace the concrete code and data flow before modifying it. For stateful inference, record the stored representation, layout, scale, lifetime, and downstream consumer.

## Change

Describe the smallest implementation change and preserved invariants.

## Correctness

| Boundary | Criterion | Result |
|---|---|---:|
| State | Bit-exact or tolerance | ... |
| Downstream computation | Bit-exact or tolerance | ... |
| Final output | Bit-exact or tolerance | ... |

## Results

Report the targeted phase and declared end-to-end baseline separately.

| Workload | Original | Modified | Change | Interpretation |
|---|---:|---:|---:|---|
| ... | ... | ... | ... | ... |

## Analysis

Explain the result using measured time shares, operation counts, workload structure, and observed variance.

## Takeaways

Separate ML-side findings, cryptographic implications, representation or state decisions, and deferred work.

## Archive

- Final report: `<path>`
- Raw A/B data: `<paths>`
- Derived summary: `<path>`
- Correctness evidence: `<path>`
- Integration or proof smoke: `<path>`
- Source commit: `<commit>`
- Excluded local artifacts: `<paths and reason>`

## Next experiment

State the next question without silently expanding the scope of this experiment.
