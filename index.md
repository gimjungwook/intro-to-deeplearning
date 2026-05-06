---
title: "Term Project Proposal — Intro to Deep Learning"
---

# Term Project Proposal — Intro to Deep Learning

**Course:** Intro to Deep Learning (Spring 2026)
**Instructor:** Heeyoul "Henry" Choi

---

## Student(s)

**Jungwook Kim** (22000168) — Handong Global University

## Title

**Reproducing *Show, Attend and Tell*: Soft-Attention Image Captioning on Flickr8k**

## Summary

People effortlessly describe a photograph in natural language, but teaching a machine to do the same requires solving three problems at once: extracting visual structure, generating fluent text, and aligning the two modalities word by word. Image captioning sits at this intersection and remains a foundational testbed for multimodal learning.

This project reproduces *Show, Attend and Tell* (Xu et al., 2015), the encoder–decoder architecture that introduced visual soft attention. A pre-trained ResNet-101 encoder converts each image into a 14×14 grid of feature vectors, and an LSTM decoder generates one word at a time. At every step, an attention module computes a weighted sum over the grid, letting the decoder focus on different regions for different words.

Experiments run on Flickr8k. The encoder is frozen; only the attention module, decoder, and word embeddings are trained. Beam search is used at inference. The report will quantify performance with BLEU-1 through BLEU-4 and analyze attention heatmaps qualitatively, highlighting alignment between generated words and image regions. A beam-size ablation is included.

This work integrates CNN feature extraction (Ch. 9), LSTM sequence modeling (Ch. 10), and the attention mechanism foundational to modern multimodal LLMs such as GPT-4V and Gemini Vision.
