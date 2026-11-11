# Presentation script

About 14 minutes at a steady pace. Each heading matches a slide in `slides.md`.

---

## Slide 1: Title (20 seconds)

Good morning, everyone. Our project asks whether visual information flow can indicate where a multimodal language model should be fine-tuned. This is joint work by Nop, Priyanka, Swarangi and myself, and we carried it out on LLaVA 1.5.

---

## Slide 2: Outline (20 seconds)

I will begin with the motivation and how we measure information flow. I will then describe our setup and main results, and spend some time on an important question: whether our effect comes from flow or simply from depth. I will finish with our conclusions and future work.

---

## Slide 3: Motivation (45 seconds)

LoRA adapts a model by training a small low-rank update beside the frozen weights. Standard practice, shown in the top row, is to place an adapter on every layer. In our model, that means all 32 layers of the language model. This is simple, but it does not ask where adaptation is actually needed.

Meanwhile, recent interpretability work has mapped where image information reaches the question inside these models. That map has been used to explain models, but not to decide where to tune them.

This leads to our research question: can the information-flow map identify which layers should receive an adapter?

---

## Slide 4: Method: Attention Knockout (50 seconds)

We build the map using attention knockout. We take a question the model answers correctly and record the probability of the correct answer. We then run the model again with some attention connections removed, and measure how far that probability falls.

As the diagram shows, we block two pathways. In the first, the question cannot read the image tokens. In the second, the answer position cannot read the question.

We block a window of nine layers at each position, and score each layer by the sum of the two drops. We measured this once, on 1000 ScienceQA training questions, so no test data is involved in choosing the layers.

---

## Slide 5: The Flow Map Ranks Layers 0–15 Highest (50 seconds)

Here is the resulting map. The orange line is image to question. It costs about twelve percent of the answer probability while the window covers the first layers, and about one percent after that. The blue line, question to last, matters through the middle layers. After about layer 18, neither pathway matters.

The sixteen highest-scoring layers, shaded here, are exactly layers 0 to 15.

We also blocked one layer at a time, which was informative. Layer 0 alone shows a drop of about 8 percent, so layers 1 to 4 rank highly mainly because their windows include layer 0. This point becomes relevant later.

---

## Slide 6: Experimental Setup (50 seconds)

We add rank 8 LoRA to the query and value projections of the language model. Our main dataset is the image subset of ScienceQA, which is new to the model. Our second dataset is GQA, which was part of LLaVA's training data. Most comparisons use three seeds, with paired tests on identical test questions.

I should also mention two corrections, because they materially changed our results. First, a batching bug meant that only the first sample in each batch of four was used for the loss. Correcting it raised uniform LoRA on ScienceQA from about 77.6 to about 80.9 percent. Second, GQA training answers were lowercase, whereas LLaVA answers with a capital letter. Correcting that added about one point.

---

## Slide 7: Preliminary Result: GQA Is Insensitive to Rank Allocation (40 seconds)

We began with GQA, keeping the same total budget and moving rank between layers. Uniform allocation, more rank in flow layers, and more rank elsewhere all reach about 62.7 percent, roughly 0.8 points above the baseline. They are statistically tied.

LLaVA already saw GQA during training, which may limit what remains to be learned. Note that we only moved rank here. We did not remove LoRA on any layers completely.

We therefore carried out the main study on ScienceQA, which offers substantially more room for improvement.

---

## Slide 8: Flow-Pruned LoRA Matches Full LoRA at Half the Parameters (50 seconds)

This is our first main result. Placing LoRA only on layers 0 to 15 achieves 81.36 percent, compared with 81.21 for LoRA on all 32 layers and 69.16 for the untuned model. It does so with half the trainable parameters, and half the saved adapter size.

The difference is not detectable. On questions where exactly one method is correct, the count is 64 against 55, with a p-value of 0.46. We therefore describe this as a match, rather than an improvement.

The saving is in adapter size, not in compute. Training time is 57 minutes against 61, because gradients still pass through the whole network.

---

## Slide 9: Placement Matters at an Equal Parameter Budget (50 seconds)

We can argue that rank 8 simply exceeds what this task requires, so that any half would work. To test this, we compared different ways of spending the same 2.1 million parameters. Each strip shows which layers receive an adapter.

Spreading the budget as rank 4 across all layers gives 80.52 percent. One fixed random set gives 79.82. The opposite half, layers 16 to 31, gives only 72.83, and its training loss also stays much higher.

Flow-pruned LoRA leads on every seed, and all three differences are significant. Placement therefore clearly matters.

---

## Slide 10: However, Layers 0–15 Are Exactly the First Half (20 seconds)

However, there is an important qualification, and it is the key point of this project.

Layers 0 to 15 are exactly the first half of the model. So is this effect due to information flow, or simply to depth?

[Pause briefly before moving on.]

---

## Slide 11: A Simple Early-Layer Rule Performs Comparably (65 seconds)

To examine this, we compared four ways of choosing eight layers.

Combined flow selection reaches 79.87 percent. A fixed block of the first eight layers, which requires no measurement at all, reaches 79.44. That difference is not significant, with a p-value of 0.17.

Looking further down, accuracy falls largely as the chosen layers move deeper. In addition, a simple gradient-size rule selects the same sixteen layers as flow, without any knockout.

Layers 9 to 16 also omit layer 0 entirely, yet retain about 80 percent of the gain, so layer 0 is not strictly required.

Our conclusion is therefore a careful one. The map offers a plausible explanation of why early layers may matter, but we have not shown that it selects layers better than a simple depth rule.

---

## Slide 12: Accuracy Declines Gradually as the Adapter Shrinks (40 seconds)

This chart shows accuracy against adapter size. The orange line follows the flow-selected layers. With sixteen layers, we recover the full gain over the baseline. With eight layers, we retain 89 percent, and with four layers, 73 percent, although the four-layer result is from a single seed.

The grey points show the other placements at the same 2.1 million parameters, all of which fall below the orange line.

---

## Slide 13: Adapter Choice Adds More Than Further Placement Changes (45 seconds)

We also changed the adapter itself. DoRA adds nothing, and neither does simply raising the learning rate. PiSSA, which initialises from the top singular vectors of each weight, reaches 83.06 percent with very little variation across seeds. That is 1.7 points above plain LoRA, and this result holds up well.

LoRA+ gave our best single run, at 83.59 percent, but it has only one seed.

PiSSA on layers 0 to 15 averages 0.69 points above PiSSA on all layers. We regard this as suggestive only, because the order reverses on seed 1.

---

## Slide 14: Inverse Placement Loses Most on Image-Heavy Categories (50 seconds)

We then asked which questions suffer when the early layers receive no adapter. Each row compares flow-pruned LoRA in orange with inverse placement in blue.

The largest gaps appear in image-heavy categories such as maps, particle diagrams and climate charts. Oceania geography falls from about 63 percent to about 26. Basic economics, which can largely be answered from the text, hardly changes.

However, the link to the flow map is only partial. State capitals are the largest group of questions that depend on layer 0, at 74 of 194, yet their accuracy barely changes. Magnets, which also uses diagrams, barely changes either.

---

## Slide 15: Additional Experiments Showed No Accuracy Gain (40 seconds)

We also tested several further ideas, none of which improved accuracy.

Giving flow layers more rank made no significant difference. Learnable gates on the flow, and a loss rewarding use of the flow, had no meaningful effect. The random-crop version of that loss made results worse. Restricting LoRA to text tokens or image tokens gave similar accuracy on a single seed.

AdaLoRA performed poorly on ScienceQA. Removing its regularisation recovers about five points, but a gap of about five points remains unexplained.

---

## Slide 16: Conclusions (55 seconds)

To conclude, three findings are well supported. On ScienceQA, half the adapter can match full LoRA. Early layers matter far more than late layers. And PiSSA adds real accuracy on those layers.

Three things have not been established. We have not shown that flow selects layers better than a simple depth rule. We have not trained layer 0 in isolation. And this work covers one model and one new task, without pruning on GQA.

Future work follows directly. We would train layer 0 alone, and compare layers 5 to 8 with layers 9 to 12, to help separate flow from depth. We would also evaluate pruned placements on GQA, whose flow map shows strong middle-layer activity.

---

## Slide 17: Thank You (10 seconds)

Thank you for your attention. I would be happy to take any questions.

---

# If someone asks

**Why call it a match if the test was simply not significant?**
A non-significant test alone does not prove equality. However, a bootstrap over questions places the difference between about −0.2 and +0.5 points, so any real difference is small.

**Are pooled p-values fair, given that the same questions repeat across seeds?**
They are somewhat optimistic for that reason. The large gaps survive a bootstrap by question: inverse +8.5 (7.1 to 10.0), random +1.5 (1.0 to 2.1), and rank 4 +0.8 (0.4 to 1.3). The fragile result is PiSSA on 16 versus 32 layers: +0.69 (0.07 to 1.34).

**How strong is the random control?**
It is limited. It is one fixed set of layers trained with three seeds, so the seeds vary the training but not the choice of layers.

**Why run the knockout at all, if gradient size selects the same layers?**
On this data, it is not needed to choose the layers. Its value is that it offers an explanation of what those layers do. We have not shown that it is better at selection.

**How stable is the layer choice?**
Reasonably, but not perfectly. Resampling the 989 questions reproduces the exact top 16 about 56% of the time and the exact top 8 about 51% of the time. For the top 16, layers 5, 6 and 16 are the ones that change.

**Do Palacios et al. not say that the middle layers matter most?**
They identify layers 6 to 16 in LLaVA-7B, but they study full fine-tuning rather than LoRA. We did not run that exact window, so we cannot claim to contradict them. Our 12–19 block is a different choice of layers.

**Does PiSSA keep the size saving?**
Compared with PiSSA on all layers, yes. However, the code saves trained PiSSA as rank 16, so the 16-layer file should be about the same size as full plain LoRA. We infer this from the code, as those weights are not in the repository.

**Was AdaLoRA trained the same way as the other methods?**
No. It used micro-batches of 1 with 16 accumulation steps, rather than 4 and 4, and the regularisation-off setting was chosen after looking at seed-0 test results. So that row should be read as exploratory. Uniform LoRA in the same training loop reached 81.21 on seed 0.

**Where does the ScienceQA baseline come from?**
The 69.16 is recorded in `RESULTS.md`, but its raw test predictions are not stored in the repository.

**How did the batching bug go unnoticed?**
Without `modalities` passed for every sample, the installed LLaVA code kept only the first sample of each micro-batch, and the runs still completed without an error. It is documented in `docs/MULTIMODAL_BATCH_FIX.md`.
