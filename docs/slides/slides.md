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

# Analyzing the Effect of Parameter-Efficient Finetuning<br>using Visual Information Flow Signals<br>for Multimodal Language Models

Where should LoRA go in LLaVA-1.5-7B?

**Priyanka Srinivas, Ananya Tirumala, Nop Polboon, Swarangi Satpute**

AI6130: Large Language Models, Nanyang Technological University

---

<!-- _class: outline -->

# Outline

1. Motivation and research question
2. Measuring information flow by attention knockout
3. Main result: half the adapter, same accuracy
4. Why these layers: placement, pathways and questions
5. Other adapters and GQA
6. Conclusions and limitations

---

# Motivation

- LoRA adapts a model by training a small low-rank update beside frozen weights
- Standard practice applies it uniformly to all 32 language-model layers
- Information-flow studies (Zhang et al., 2025; Yin et al., 2025) map where image information reaches the question
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

The image enters the question in the first layers; the question carries it to the answer through the middle layers. After layer 18, neither pathway matters.

---

# Layer 0 Carries the Image into the Question

![w:960](slides_assets/layer0.svg)

- Layer 0 holds **about 90%** of the early image-to-question flow
- The middle-layer flow is spread out, which is why the 9-layer window is used to rank layers
- 20% of questions lose more than 10%: mostly *maps and geography*

---

# Experimental Setup

| Component | Configuration |
|---|---|
| Model | LLaVA-1.5-7B; LoRA on query and value projections, rank 8 |
| ScienceQA (image subset) | 6,218 training and 2,017 test questions, new to the model |
| GQA | 20,000 training and 12,578 test questions, part of LLaVA's training data |
| Evaluation | Three seeds; paired McNemar tests on identical test questions |

## Implementation corrections made before the reported results

| Issue | Effect of the correction |
|---|---|
| Batches of four retained only the first sample | ScienceQA uniform LoRA, seed 0: 77.64 → **80.86** |
| GQA training answers were lowercase | GQA AdaLoRA, seed 0: 61.44 → **62.56** |

---

<!-- _class: stats -->

# Flow-Pruned LoRA Matches Full LoRA at Half the Parameters

| 81.36% | 2.10M | 8.4 MB |
|---|---|---|
| Flow-pruned accuracy, against 81.21% for full LoRA and 69.16% zero-shot | Trainable parameters on layers 0–15, down from 4.19M | Saved adapter size, down from 16.8 MB |

- Within 0.25 points on every seed; paired difference not significant (64 vs 55 questions, p = 0.46)
- Same training loss (0.193 vs 0.192): the 16 layers left out add nothing to the fit
- *The saving is in adapter size, not compute: 57 vs 61 minutes of training*

---

# Placement Matters at an Equal Parameter Budget

![w:1020](slides_assets/placement_16.svg)

- **Rank 4 everywhere** loses 0.84 points: the adapter cannot simply be shrunk everywhere
- **Random half** misses layers 0, 2 and 3, where the image enters the question
- **Opposite half** gains only 3.7 points, and its training loss stays about 50% higher

Flow-pruned leads on every seed; all paired differences are significant (p < 0.001).

---

# The Image-to-Question Layers Cannot Be Left Out

![w:1020](slides_assets/placement_8.svg)

- Eight layers, two ways of using the map: the combined score keeps layers 0–4 and 9–11
- **Question → Last only** drops layers 0–4: −1.0 point (256 vs 194 questions, p = 0.004)
- **Middle block 12–19** loses 3.8 points (405 vs 178, p = 2×10⁻²¹), most in the hardest grades

---

# Accuracy Declines Gradually as the Adapter Shrinks

![w:1000](slides_assets/size_accuracy.svg)

Share of the full-LoRA gain retained: **101% with 16 layers**, 89% with 8 layers, 73% with 4 layers (single seed). Four flow layers beat 16 inverse layers by 5.6 points with a quarter of the parameters.

---

# Without the Flow Layers, Image Questions Fail

![w:930](slides_assets/categories.svg)

- Maps, particle diagrams and climate charts drop by **20–37 points** with inverse placement
- Questions answered mostly from the words (state capitals, economics) barely move
- 130 test questions are right with flow-pruned and wrong with inverse on **all three seeds**; only 18 go the other way

---

# Examples Inverse Placement Always Gets Wrong

![w:1080](slides_assets/examples.png)

Right with flow-pruned LoRA and wrong with inverse LoRA on all three seeds: a highlighted country, the driest month on a climate chart, and particle concentration. These are the same map and diagram questions that depend on layer 0 in the knockout.

---

# The Same Layers Work with Stronger Adapters

![w:1000](slides_assets/variants.svg)

- **PiSSA on the flow layers: 83.06%**, 0.69 above PiSSA on all layers (213 vs 171, p = 0.04), with an eight times smaller spread across seeds
- PiSSA improves on plain LoRA in the same layers by 1.7 points (270 vs 167, p = 10⁻⁶)
- LoRA+ points the same way (seed 0); DoRA and a 4× learning rate match plain LoRA

---

# On GQA, the Model Already Knows the Task

| Same 4.19M parameters, rank moved between layers | Accuracy (%) |
|---|---:|
| Zero-shot | 61.91 |
| Uniform ranks | 62.73 ± 0.19 |
| More rank in flow layers | 62.62 ± 0.22 |
| More rank elsewhere | 62.75 ± 0.04 |
| AdaLoRA | 62.49 ± 0.06 |

- Every rank allocation gains under 1 point, against 12 on ScienceQA; none of the differences is significant
- The GQA map shows the same early image-to-question flow, so the map is not wrong here
- *GQA is part of LLaVA's training data, so there is little left for any adapter to learn*

---

# Further Experiments

| Experiment | Dataset, seeds | Accuracy (%) | Compared with |
|---|---|---:|---:|
| More rank in flow layers, all 32 kept | ScienceQA, 3 | 81.37 | 81.21 (p = 0.51) |
| Less rank in flow layers, all 32 kept | ScienceQA, 3 | 79.21 | 81.21 (worse every seed) |
| LoRA on text tokens only | ScienceQA, 1 | 81.06 | 81.11 |
| LoRA on image tokens only | ScienceQA, 1 | 80.71 | 81.11 |
| Learnable gates on the flow | ScienceQA, 1 | 81.16 | 80.86 (p = 0.33) |
| Loss rewarding use of the flow | GQA, 1 | 62.84 | 62.80 (p = 0.82) |
| AdaLoRA | ScienceQA, 3 | 71.36 | 81.21 |

Text-only and image-only LoRA both match all tokens: what LoRA tunes in these layers is the *mixing* of image and question.

---

<!-- _class: two -->

# Conclusions

## Findings

- **Knockout locates** where LLaVA mixes the image into the question: layer 0, then the middle layers
- **LoRA on those 16 layers matches full LoRA** with half the parameters
- Equal-budget alternatives are worse on every seed

## Limitations

- The saving is in adapter size, not training compute
- One model and one new task; several controls are single-seed
- The map is built from questions the model already answers

## Recipe and future work

- Run the knockout once on about 1,000 training questions, keep the top half of the layers, train any LoRA variant there
- Next: larger models, and new tasks such as charts, documents or medical images

---

<!-- _class: divider -->
<!-- _footer: '' -->
<!-- _paginate: false -->

# Thank You

Questions and discussion are welcome.
