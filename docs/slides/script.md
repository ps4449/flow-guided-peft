# Presentation script

About 15 minutes at a steady pace. Each heading matches a slide in [slides.md](slides.md).

---

## Slide 1: Title (20 seconds)

Good morning, everyone. Our project asks whether visual information flow can show where a multimodal language model should be fine-tuned. This is joint work by Nop, Priyanka, Ananya and myself, and we carried it out on LLaVA 1.5.

---

## Slide 2: Outline (20 seconds)

We will begin with the motivation and how we measure information flow. We'll then show our main result, explain why these particular layers matter, look at other adapters and at GQA, and finish with our conclusions.

---

## Slide 3: Motivation (45 seconds)

LoRA adapts a model by training a small low-rank update beside the frozen weights. Standard practice, shown in the top row, is to place an adapter on every layer. In our model, that means all 32 layers of the language model. This is simple, but it does not ask where adaptation is actually needed.

Meanwhile, recent interpretability work has mapped where image information reaches the question inside these models. That map has been used to explain models, but not to decide where to tune them.

This leads to our research question: can the information-flow map identify which layers should receive an adapter?

---

## Slide 4: Method: Attention Knockout (50 seconds)

We build the map using attention knockout. We take a question the model answers correctly and record the probability of the correct answer. We then run the model again with some attention connections removed, and measure how far that probability falls.

As the diagram shows, we block two pathways. In the first, the question cannot read the image tokens. In the second, the answer position cannot read the question.

We block a window of nine layers at each position, and score each layer by the sum of the two drops. We measured this once, on 989 ScienceQA training questions, so no test data is involved in choosing the layers.

---

## Slide 5: The Flow Map Ranks Layers 0–15 Highest (45 seconds)

Here is the resulting map. The orange line is image to question. It costs about twelve percent of the answer probability while the window covers the first layers, and about one percent after that. The blue line, question to last, matters through the middle layers. After about layer 18, neither pathway matters.

So the image enters the question early, and the question carries it to the answer in the middle. The sixteen highest-scoring layers, shaded here, are layers 0 to 15.

---

## Slide 6: Layer 0 Carries the Image into the Question (45 seconds)

A nine-layer window cannot tell us which of the first layers matters, so we repeated the image-to-question knockout one layer at a time.

Layer 0 alone costs about 8 percent, which is about 90 percent of the early flow. No other single layer costs more than one percent. The middle-layer flow is spread out: no single middle layer matters on its own, but blocking nine together does. That is why we rank layers with the nine-layer window.

The effect comes from specific questions. One in five questions loses more than 10 percent, and these are mostly map and geography questions. We will come back to these.

---

## Slide 7: Experimental Setup (45 seconds)

We add rank 8 LoRA to the query and value projections of the language model. Our main dataset is the image subset of ScienceQA, which is new to the model. Our second dataset is GQA, which was part of LLaVA's training data. Every main comparison uses three seeds, with paired tests on identical test questions.

I should also mention two corrections, because they materially changed our results. First, a batching bug meant that only the first sample in each batch of four was used. Correcting it raised uniform LoRA on ScienceQA from about 77.6 to about 80.9 percent. Second, GQA training answers were lowercase, whereas LLaVA answers with a capital letter. Correcting that added about one point.

---

## Slide 8: Flow-Pruned LoRA Matches Full LoRA at Half the Parameters (50 seconds)

This is our main result. Placing LoRA only on layers 0 to 15 achieves 81.36 percent, compared with 81.21 for LoRA on all 32 layers and 69.16 for the untuned model. It does so with half the trainable parameters, and half the saved adapter size.

The two are within a quarter of a point on every seed. On questions where exactly one method is correct, the count is 64 against 55, with a p-value of 0.46, so we call this a match. The training loss is also the same, so the 16 layers the map leaves out add nothing to how well the model fits the task.

The saving is in adapter size, not in compute. Training takes 57 minutes against 61, because gradients still pass through the whole network.

---

## Slide 9: Placement Matters at an Equal Parameter Budget (55 seconds)

One could argue that rank 8 simply exceeds what this task needs, so that any half would work. To test this, we compared different ways of spending the same 2.1 million parameters. Each strip shows which layers receive an adapter.

Spreading the budget as rank 4 across all layers gives 80.52 percent, below flow-pruned and also below full rank-8 LoRA. So the adapter cannot simply be shrunk everywhere; it has to be shrunk where the map says it is not needed. One fixed random set gives 79.82. It shares nine of the sixteen flow layers, but misses layers 0, 2 and 3, where the image enters the question. The opposite half, layers 16 to 31, gives only 72.83, and its training loss stays about 50 percent higher, so these layers cannot even fit the training data well.

Flow-pruned LoRA leads on every seed, and all three differences are significant.

---

## Slide 10: The Image-to-Question Layers Cannot Be Left Out (50 seconds)

Our layer score combines two pathways. To see which one matters for placement, we chose eight layers in three ways.

The combined score keeps layers 0 to 4, where the image enters the question, plus layers 9 to 11, and reaches 79.87 percent. Choosing by question-to-last alone moves all eight adapters into the middle and drops layers 0 to 4. That costs one point, and the difference is significant. A middle block, layers 12 to 19, costs nearly four points, with the largest losses in the hardest grades.

Both pathways contribute, but the layers where the image enters the question are the ones that cannot be left out.

---

## Slide 11: Accuracy Declines Gradually as the Adapter Shrinks (40 seconds)

This chart shows accuracy against adapter size. The orange line follows the flow-selected layers. With sixteen layers, we recover the full gain over the baseline. With eight layers, we keep 89 percent, and with four layers, 73 percent, although the four-layer result is from a single seed.

Even four flow layers beat sixteen inverse layers by more than five points, with a quarter of the parameters.

---

## Slide 12: Without the Flow Layers, Image Questions Fail (50 seconds)

We then asked which questions suffer when the flow layers receive no adapter. Each row compares flow-pruned LoRA in orange with inverse placement in blue.

The largest gaps are on questions that need the picture: maps, particle diagrams and climate charts drop by 20 to 37 points. Oceania geography falls from about 63 percent to about 26. Questions that can be answered mostly from the words, such as state capitals or basic economics, barely move.

These failures are consistent. 130 test questions are answered correctly by flow-pruned LoRA on all three seeds and wrongly by inverse LoRA on all three. Only 18 go the other way.

---

## Slide 13: Examples Inverse Placement Always Gets Wrong (35 seconds)

Here are three of those 130 questions. On a map asking which country is highlighted, inverse LoRA answers New Zealand instead of Palau on every seed. On a climate chart, it misreads the driest month. On a particle diagram, it picks the solution with the lower concentration.

These are the same map and diagram questions that depend on layer 0 in the knockout. The map and the fine-tuning results point to the same questions.

---

## Slide 14: The Same Layers Work with Stronger Adapters (45 seconds)

The placement does not depend on the adapter type, so we trained other adapters on the same sixteen layers.

PiSSA, which starts from the top singular vectors of each weight, reaches 83.06 percent on the flow layers. That is 0.69 points above PiSSA on all layers, a significant difference, with an eight times smaller spread across seeds. It is also 1.7 points above plain LoRA on the same layers.

LoRA+ points the same way on a single seed. DoRA matches plain LoRA, and so does plain LoRA with four times the learning rate, so the PiSSA gain is not just a larger step size.

---

## Slide 15: On GQA, the Model Already Knows the Task (40 seconds)

GQA is part of LLaVA's own training data. Here we kept all layers and moved rank between them. Uniform ranks, more rank in the flow layers, more rank elsewhere, and AdaLoRA all land within one point of the untuned model, and none of the differences is significant.

The GQA knockout map has the same early image-to-question flow, so the map is not wrong here. There is simply little left for any adapter to learn. The map tells us where to adapt for a task the model has not yet learned.

---

## Slide 16: Further Experiments (40 seconds)

We also tested several further ideas. Keeping all 32 layers and giving the flow layers more rank does not help, but giving them less rank costs two points, on every seed.

Restricting LoRA to the text tokens or to the image tokens matches LoRA on all tokens. Either side of the image-to-question exchange is enough, which supports our reading of the map: LoRA tunes the mixing of image and question, not one modality on its own.

Learnable gates on the flow and a loss rewarding use of the flow gave no gain, and AdaLoRA performed poorly on ScienceQA. These details are in the appendix of our report.

---

## Slide 17: Conclusions (50 seconds)

To conclude. Attention knockout shows that LLaVA mixes the image into the question in its first layer and passes it on through the middle layers. Placing LoRA only on the 16 layers this map ranks highest matches full LoRA with half the parameters, on every seed. The same budget spread over all layers, a random half, and the opposite half are all worse, and without the flow layers the model loses exactly the questions that need the picture.

There are limits. The saving is in adapter size, not training time. We studied one model and one new task, and several controls have one seed.

The recipe is simple: run the knockout once on about a thousand training questions, keep the top half of the layers, and train any LoRA variant there. Next, we would test larger models and new tasks such as charts, documents or medical images.

---

## Slide 18: Thank You (10 seconds)

Thank you for your attention. We would be happy to take any questions.

---

# If someone asks

**Why call it a match if the test was simply not significant?**
A non-significant test alone does not prove equality. However, a bootstrap over questions places the difference between about −0.2 and +0.5 points, so any real difference is small.

**Are pooled p-values fair, given that the same questions repeat across seeds?**
They are somewhat optimistic for that reason. The large gaps survive a bootstrap by question: inverse +8.5 (7.1 to 10.0), random +1.5 (1.0 to 2.1), and rank 4 +0.8 (0.4 to 1.3). The smallest is PiSSA on 16 versus 32 layers: +0.69 (0.07 to 1.34).

**How strong is the random control?**
It is limited. It is one fixed set of layers trained with three seeds, so the seeds vary the training but not the choice of layers.

**Layers 0–15 are the first half of the model. Is this just depth?**
Our study is about what the flow map reveals, not about competing selection rules. We ran a fixed early block as a sanity check, and it is in the appendix of the report. What the map adds is the reason: layer 0 is where the image enters the question, and the questions that fail without it are the image-heavy ones.

**How stable is the layer choice?**
The best two eight-layer regions share 15 of the 16 layers. A window of 3 instead of 9 changes the ranking but still shares 15 of the top 16. Over 5,000 bootstrap resamples of the 989 questions, the question-to-last top 8 reappears in more than 99 percent of them.

**What about work that tunes only the middle layers?**
Palacios et al. restrict full fine-tuning, not LoRA, to layers 6 to 16 of LLaVA-1.5-7B. We did not run that window. Our middle block, layers 12 to 19, is a later window, and under LoRA it loses 3.8 points against the combined flow choice, which keeps layers 0 to 4.

**Does PiSSA keep the size saving?**
It keeps half the trainable parameters. The code saves trained PiSSA as an equivalent rank-16 LoRA, so the 16-layer file is 16.8 MB, the same as full plain LoRA, and half the size of PiSSA on all layers (33.6 MB).

**Was AdaLoRA trained the same way as the other methods?**
No. It used micro-batches of 1 with 16 accumulation steps, rather than 4 and 4, and the regularisation-off setting was chosen after looking at seed-0 test results. So that row should be read as exploratory. Uniform LoRA in the same training loop reached 81.21 on seed 0.

**Where does the ScienceQA baseline come from?**
The 69.16 is recorded in [RESULTS.md](../../RESULTS.md), but its raw test predictions are not stored in the repository.

**How did the batching bug go unnoticed?**
Without `modalities` passed for every sample, the installed LLaVA code kept only the first sample of each micro-batch, and the runs still completed without an error. It is documented in [MULTIMODAL_BATCH_FIX.md](../MULTIMODAL_BATCH_FIX.md).
