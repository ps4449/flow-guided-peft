---
marp: true
theme: default
size: 16:9
paginate: true
footer: 'Information Flow and LoRA Placement in LLaVA-1.5-7B  |  AI6130 Large Language Models'
style: |
  section {
    background: #ffffff;
    color: #1f2933;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 25px;
    line-height: 1.38;
    padding: 58px 72px 56px 72px;
    place-content: safe start start;
  }
  section::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 12px;
    background: #0b3c52;
  }
  section > footer { left: 72px; bottom: 18px; color: #6e7781; font-size: 16px; }
  section::after { color: #6e7781; font-size: 17px; right: 72px; bottom: 18px; }
  h1 {
    color: #0b3c52;
    font-size: 36px;
    font-weight: 700;
    line-height: 1.18;
    margin: 0 0 22px 0;
    padding-bottom: 12px;
    border-bottom: 3px solid #c2410c;
  }
  h2 { color: #0b3c52; font-size: 25px; font-weight: 700; margin: 14px 0 6px 0; }
  strong { color: #c2410c; }
  em { color: #0077a0; font-style: normal; font-weight: 600; }
  ul, ol { margin: 6px 0; padding-left: 26px; }
  li { margin: 6px 0; }
  li::marker { color: #c2410c; }
  p { margin: 8px 0; }
  img { display: block; margin: 0 auto; background: transparent; }
  blockquote {
    border-left: 6px solid #c2410c;
    background: #f7f8f9;
    color: #1f2933;
    font-size: 29px;
    margin: 24px 0 0 0;
    padding: 18px 26px;
  }
  blockquote::before, blockquote::after { content: none; }
  table {
    display: table;
    width: 100%;
    border-collapse: collapse;
    font-size: 22px;
    margin: 8px 0 12px 0;
  }
  table thead th {
    background: #0b3c52;
    color: #ffffff;
    font-weight: 600;
    text-align: left;
    border: none;
    padding: 7px 12px;
  }
  table tbody tr { background: transparent !important; }
  table tbody td {
    background: transparent;
    border: none;
    border-bottom: 1px solid #d8dee4;
    padding: 6px 12px;
  }
  section.cover, section.divider { background: #0b3c52; color: #ffffff; place-content: safe center start; }
  section.cover::before, section.divider::before { display: none; }
  section.cover h1, section.divider h1 { color: #ffffff; font-size: 46px; border-bottom: 4px solid #e8703a; }
  section.cover p { color: #d5e3ea; font-size: 25px; margin: 6px 0; }
  section.cover strong { color: #ffffff; }
  section.divider p { color: #d5e3ea; font-size: 31px; }
  section.stats table { margin: 18px 0 26px 0; table-layout: fixed; }
  section.stats table thead th {
    background: #f7f8f9;
    color: #c2410c;
    font-size: 54px;
    font-weight: 700;
    border-top: 5px solid #c2410c;
    padding: 18px 20px 2px 20px;
  }
  section.stats table tbody td { background: #f7f8f9; color: #57606a; font-size: 20px; border: none; padding: 2px 20px 16px 20px; vertical-align: top; }
  section.outline ol { font-size: 30px; margin-top: 22px; }
  section.outline li { margin: 16px 0; padding-left: 8px; }
  section.outline li::marker { font-weight: 700; }
  section.two { display: grid; grid-template-columns: 1fr 1fr; column-gap: 48px; align-content: start; }
  section.two h1 { grid-column: 1 / 3; }
  section.two h2:nth-of-type(1), section.two ul:nth-of-type(1) { grid-column: 1; }
  section.two h2:nth-of-type(2), section.two ul:nth-of-type(2) { grid-column: 2; }
  section.two h2:nth-of-type(1), section.two h2:nth-of-type(2) { grid-row: 2; }
  section.two ul:nth-of-type(1), section.two ul:nth-of-type(2) { grid-row: 3; }
  section.two h2:nth-of-type(3) { grid-column: 1 / 3; grid-row: 4; margin-top: 18px; }
  section.two ul:nth-of-type(3) { grid-column: 1 / 3; grid-row: 5; }
---

<!-- _class: cover -->
<!-- _paginate: false -->
<!-- _footer: '' -->

# Does Visual Information Flow Indicate<br>Where to Fine-Tune a Multimodal Model?

An evaluation of LoRA placement in LLaVA-1.5-7B

**Nop Polboon, Priyanka Srinivas, Ananya Tirumala, Swarangi Satpute**

AI6130: Large Language Models, Nanyang Technological University

---

<!-- _class: outline -->

# Outline

1. Motivation and research question
2. Measuring information flow by attention knockout
3. Experimental setup and preliminary results
4. Results on layer placement and adapter choice
5. Flow selection compared with simple depth rules
6. Conclusions and future work

---

# Motivation

- LoRA adapts a model by training a small low-rank update beside frozen weights
- Standard practice applies it uniformly to all 32 language-model layers
- Zhang et al. (2025) mapped where image information reaches the question inside the model
- That map has been used to explain models, but not to choose where to tune them

![w:1136](slides_assets/motivation.svg)

> **Research question:** can the information-flow map identify which layers should receive an adapter?

---

# Method: Attention Knockout

![w:1000](slides_assets/knockout_method.svg)

- Compare the probability of the correct answer with and without the connection
- Block a 9-layer window at each of the 32 positions; score each layer by the sum of both drops
- Measured once on 989 ScienceQA *training* questions, so no test data is used for selection

---

# The Flow Map Ranks Layers 0–15 Highest

![w:1000](slides_assets/knockout_map.svg)

Blocking layers one at a time, **layer 0 alone accounts for −8.1%**. *Layers 1–4 rank highly mainly because their windows include layer 0.*

---

# Experimental Setup

| Component | Configuration |
|---|---|
| Model | LLaVA-1.5-7B; LoRA on query and value projections, rank 8 |
| ScienceQA (image subset) | 6,218 training and 2,017 test questions; new to the model |
| GQA | 20,000 training and 12,578 test questions; part of LLaVA's training data |
| Evaluation | Mostly three seeds; paired tests on identical test questions |

## Implementation corrections made before the reported results

| Issue | Effect of the correction |
|---|---|
| Batches of four retained only the first sample | ScienceQA uniform LoRA, seed 0: 77.64 → **80.86** |
| GQA training answers were lowercase | GQA AdaLoRA, seed 0: 61.44 → **62.56** |

---

# Preliminary Result: GQA Is Insensitive to Rank Allocation

| Allocation of the same 4.19M parameters | Accuracy (%) | Gain over zero-shot |
|---|---:|---:|
| Zero-shot baseline | 61.91 | – |
| Uniform | 62.73 ± 0.19 | +0.82 |
| More rank in flow layers | 62.62 ± 0.22 | +0.71 |
| More rank in other layers | 62.75 ± 0.04 | +0.84 |

- All three allocations are statistically tied, with little gain over the baseline
- LLaVA already saw GQA during training, which may limit what is left to learn
- *Only rank allocation was tested here; layers were not removed on GQA*

The main study therefore uses **ScienceQA**, which offers substantially more room for improvement.

---

<!-- _class: stats -->

# Flow-Pruned LoRA Matches Full LoRA at Half the Parameters

| 81.36% | 2.10M | 8.4 MB |
|---|---|---|
| Flow-pruned accuracy, against 81.21% for full LoRA and 69.16% zero-shot | Trainable parameters on layers 0–15, down from 4.19M | Saved adapter size, down from 16.8 MB |

- Paired comparison: 64 vs 55 questions, p = 0.46, so no detectable difference
- The two methods are within 0.25 points on every seed
- *Training time barely changes (57 vs 61 minutes), because gradients still pass through all layers*

---

# Placement Matters at an Equal Parameter Budget

![w:1020](slides_assets/placement_16.svg)

Flow-pruned LoRA leads each 2.10M alternative on every seed, and **the pooled paired differences are all significant** (p < 0.001).

---

<!-- _class: divider -->
<!-- _footer: '' -->
<!-- _paginate: false -->

# However, Layers 0–15 Are Exactly<br>the First Half of the Model

Is the effect due to information flow, or simply to depth?

---

# A Simple Early-Layer Rule Performs Comparably

![w:940](slides_assets/placement_8.svg)

- Combined flow vs fixed early block: 176 vs 150 questions, p = 0.17, **not significant**
- A gradient-size rule selects the same 16 layers as flow, without any knockout
- *Accuracy falls largely with depth, so flow has not been shown to outperform a depth rule*

---

# Accuracy Declines Gradually as the Adapter Shrinks

![w:1000](slides_assets/size_accuracy.svg)

Share of the full-LoRA gain retained: **101% with 16 layers**, 89% with 8 layers, 73% with 4 layers (single seed).

---

# Adapter Choice Adds More Than Further Placement Changes

![w:1000](slides_assets/variants.svg)

- **PiSSA improves on plain LoRA by 1.70 points** on layers 0–15 (270 vs 167 questions)
- *The lead over all-layer PiSSA (0.69 points) is suggestive only: the order reverses on seed 1*

---

# Inverse Placement Loses Most on Image-Heavy Categories

![w:930](slides_assets/categories.svg)

*The link to the flow map is partial:* state capitals form the largest layer-0-dependent group (74 of 194 questions), yet accuracy barely changes.

---

# Additional Experiments Showed No Accuracy Gain

| Experiment | Dataset, seeds | Accuracy (%) | Compared with |
|---|---|---:|---:|
| More rank in flow layers | ScienceQA, 3 | 81.37 | 81.21 (p = 0.51) |
| Learnable gates on the flow | ScienceQA, 1 | 81.16 | 80.86 (p = 0.33) |
| Loss rewarding use of the flow | GQA, 1 | 62.84 | 62.80 (p = 0.82) |
| The same loss with random crops | GQA, 1 | 62.04 | 62.80 (worse) |
| LoRA on text tokens only | ScienceQA, 1 | 81.06 | 81.11 |
| LoRA on image tokens only | ScienceQA, 1 | 80.71 | 81.11 |
| AdaLoRA | ScienceQA, 3 | 71.36 | 81.21 |
| AdaLoRA without regularisation | ScienceQA, 3 | 76.42 | 81.21 |

AdaLoRA's remaining gap of about 5 points is *not yet explained*.

---

<!-- _class: two -->

# Conclusions

## Supported by the evidence

- **Half the adapter can match full LoRA** on ScienceQA
- **Early layers matter far more than late layers** (+8.53 points at equal budget)
- PiSSA adds about 1.7 points on the same layers

## Not established

- *That flow selects layers better than a simple depth rule*
- That layer 0 alone is sufficient, as it was never trained in isolation
- That results extend to other models, other tasks, or pruning on GQA

## Future work

- Train layer 0 alone, and compare layers 5–8 with layers 9–12 to separate flow from depth
- Evaluate pruned placements on GQA, whose flow map has strong middle-layer activity

---

<!-- _class: divider -->
<!-- _footer: '' -->
<!-- _paginate: false -->

# Thank You

Questions and discussion are welcome.
