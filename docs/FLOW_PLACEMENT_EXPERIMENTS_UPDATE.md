# Flow-guided PEFT: quick team update
*LLaVA-1.5-7B · ScienceQA · 29 October 2026*

## Why we did these follow-up experiments

Our main result was promising for **efficiency**: applying rank-8 LoRA to **16 flow-selected layers** reached **81.36% ± 0.31%** ScienceQA accuracy (three seeds) with **2.10M adapter parameters**, compared with **81.21% ± 0.41%** for full uniform LoRA using **4.19M**. However, flow selected exactly **layers 0–15**, and gradient-based selection found the same layers. We therefore asked: **does information flow help us choose *where* to fine-tune, beyond simply using early layers?**

## The experiments, in order

**1. Region-guided placement — does selecting whole regions change the result?**  
Zhang measures information transfer over *windows of layers*, not isolated layers. We tested whether selecting two complete eight-layer regions would be more meaningful than selecting 16 individual high-scoring layers. Exhaustive search chose **0–7 and 9–16**. This shared **15/16 layers** with the existing **0–15** selection and retained **99.35%** of its flow score. **Decision:** skip training an almost identical configuration.

**2. Smaller knockout windows — are our layer scores too coarse?**  
The original nine-layer window produced a sharp score drop when layer 0 left the window. On the *same 200 calibration questions*, we compared windows of **9 versus 3 layers**. The rankings changed (Spearman correlation **0.686**), but the window-3 top-16 still shared **15/16 layers** with the historical selection and did not meet our stability criterion. The accompanying graph shows why window size matters: the sharp early peak shifts when the window changes. **Decision:** do not train a weakly distinct selection.

**3. Separate pathways — are we mixing different kinds of information flow?**  
Our original score combines **Image→Question** (visual information entering question representations) and **Question→Last** (integrated information moving toward the answer). Using **989 calibration examples**, we ranked layers separately. Combined-flow top-8 chose **0–4, 9–11**; Question→Last chose **9–16**, with only **3/8** layers in common. The latter selection reappeared in **99.84%** of 5,000 bootstrap samples. **Decision:** test this genuinely different, stable allocation.

**4. Matched eight-layer training — does the stable pathway selection improve accuracy?**  
We trained/evaluated equal-budget, **seed-0** configurations (rank-8 LoRA, **1.05M parameters**, same 2,017 test questions):

| LoRA placement | Layers | ScienceQA accuracy |
| --- | --- | ---: |
| Combined flow | 0–4, 9–11 | **79.57%** |
| Question→Last | 9–16 | 77.94% |
| Fixed middle | 12–19 | 75.46% |

Combined flow performed better than both alternatives for these fixed checkpoints. Paired-question tests supported these differences, but **selection stability did not guarantee fine-tuning performance**.

**5. Final early-layer control — does flow beat a simple position rule?**  
Since combined flow uses five early layers, we tested the simple alternative **0–7**, with exactly the same parameter budget and settings. **Fixed early and combined flow tied at 1,605/2,017 = 79.57%**. Each uniquely answered **42** questions correctly; the paired difference was **0.00 percentage points** (95% paired-question interval **−0.89 to +0.89**). **Decision:** stop adding placement experiments; the claimed unique benefit of flow is not demonstrated.

## Bottom line

**Our defensible finding is efficient sparse LoRA placement**, not that flow is a universally superior selector. The 16-layer method preserves performance close to full uniform LoRA with about half the adapter parameters, but at eight layers a simple early-layer rule matches combined flow in seed 0. These results are useful positive *and* negative evidence. We will now consolidate the report and presentation rather than invent further selection methods.

*Here “pruned” means omitting LoRA adapters, not removing base transformer layers. Paired-question significance concerns fixed checkpoints; seed-0 findings do not establish reliability across training seeds.*
