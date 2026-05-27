---
layout: page
title: "Term Project — Intro to Deep Learning"
---

# Term Project — Intro to Deep Learning

**Course:** Intro to Deep Learning (Spring 2026)
**Instructor:** Heeyoul "Henry" Choi
**Student:** Jungwook Kim (22000168) — Handong Global University
**Title:** *Reproducing **Show, Attend and Tell**: Soft-Attention Image Captioning on Flickr8k*

---

## Final result (accepted run `20260527-2252`)

| Beam | BLEU-1 | BLEU-2 | BLEU-3 | **BLEU-4** |
|----:|-------:|-------:|-------:|-----------:|
| 1 (greedy) | 0.5995 | 0.4268 | 0.2926 | 0.2049 |
| **3** | 0.6283 | 0.4517 | 0.3159 | **0.2195** |
| 5 | 0.6356 | 0.4583 | 0.3228 | 0.2266 |

- **Attention noun-peak QC:** 23 / 24
- **ACCEPT gate:** all four conditions cleared on the first full cycle (no C1-C5 retraining)
- **Full report:** [`reports/final-report.md`](https://github.com/gimjungwook/intro-to-deeplearning/blob/main/reports/final-report.md)
- **Iteration log:** [`ITER_LOG.md`](https://github.com/gimjungwook/intro-to-deeplearning/blob/main/ITER_LOG.md)
- **Run directory:** [`reports/runs/20260527-2252/`](https://github.com/gimjungwook/intro-to-deeplearning/tree/main/reports/runs/20260527-2252)

---

## Proposal summary

People effortlessly describe a photograph in natural language, but teaching a machine to do the same requires solving three problems at once: extracting visual structure, generating fluent text, and aligning the two modalities word by word. Image captioning sits at this intersection and remains a foundational testbed for multimodal learning.

This project reproduces *Show, Attend and Tell* (Xu et al., 2015), the encoder–decoder architecture that introduced visual soft attention. A pre-trained ResNet-101 encoder converts each image into a 14×14 grid of feature vectors, and an LSTM decoder generates one word at a time. At every step, an attention module computes a weighted sum over the grid, letting the decoder focus on different regions for different words.

Experiments run on Flickr8k. The encoder is frozen; only the attention module, decoder, and word embeddings are trained. Beam search is used at inference. The report will quantify performance with BLEU-1 through BLEU-4 and analyze attention heatmaps qualitatively, highlighting alignment between generated words and image regions. A beam-size ablation is included.

This work integrates CNN feature extraction (Ch. 9), LSTM sequence modeling (Ch. 10), and the attention mechanism foundational to modern multimodal LLMs such as GPT-4V and Gemini Vision.
