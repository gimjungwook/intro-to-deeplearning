# Iteration Log

각 줄: `cycle | hypothesis | expected effect | code change | restart at`

| C | hypothesis | expected effect | code change | restart |
|---|------------|------------------|-------------|---------|
| C0-smoke | pipeline end-to-end works on tiny subset before committing GPU-hours | loss decreases, attention peaks, beam>greedy | n/a (smoke) | run_id 20260527-2242, 200 imgs, 8 ep — BLEU-4 0.129 (data-starved, expected) |
| C0-full  | full 8000-img Flickr8k + 30 epochs + early stop will clear BLEU-4 ≥ 0.18 | BLEU-4 ≈ 0.18–0.22 range for Show-Attend-Tell on Flickr8k | n/a (baseline) | run_id 20260527-2252, 4800/1600/1600 split, vocab 2248, early stop @ ep10 (best ep05 val 3.286) — **ACCEPT** (beam3 BLEU-4=0.2195, noun_peak 23/24, beam3-greedy +0.0146, no NaN) |

## ACCEPT — single-cycle convergence

C0-full cleared all four ACCEPT-gate conditions on the first full attempt; no C1-C5 retraining was required. Hypotheses kept on reserve for follow-up work (not run for this report) are recorded below for completeness.

### Reserve hypotheses (unused — kept for future iteration)

- **H1 lr cosine schedule** — replace flat Adam 4e-4 with cosine to 1e-5 over 30 epochs, expected BLEU-4 +0.005~0.01
- **H2 attention dropout** — add 0.1 dropout on attention scores to reduce peak collapse, expected attention diversity +
- **H3 beam length penalty sweep** — current default 0.7, sweep 0.5/0.7/1.0, expected beam-5 BLEU-4 +0.003
- **H4 vocab cutoff 3** — current 5, lower to 3 to reduce `<unk>` rate in references, expected BLEU-4 +0.005
- **H5 image augmentation** — random horizontal flip in encoder.py, expected generalization +
- **H6 Transformer decoder** — replace LSTMCell with 2-layer Transformer decoder + cross-attn, expected BLEU-4 +0.02~0.04 (scope: outside C0-C5 baseline budget)

