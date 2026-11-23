"""Draw the slide figures as SVG from the project's raw results (standard library only).

Usage, from the repo root:  python docs/slides/slides_assets/make_figures.py
"""
import csv
import json
import os
import statistics as st
from collections import defaultdict
from xml.sax.saxutils import escape

OUT = "docs/slides/slides_assets"
ORANGE, TEAL = "#c2410c", "#0077a0"
INK, MUTED, GRID, NEUTRAL, BAND = "#1f2933", "#57606a", "#d8dee4", "#8c959f", "#f1f3f5"
FONT = '"Helvetica Neue", Helvetica, Arial, sans-serif'
ZERO_SHOT = 69.16  # 1,395 / 2,017, recorded in RESULTS.md
FLOW_DIR = "output/information_flow/llava_v1_5_7b/ScienceQA/train_correct"


class Svg:
    def __init__(self, width, height):
        self.width, self.height, self.items = width, height, []

    def text(self, x, y, s, size=18, fill=INK, anchor="start", weight=400):
        self.items.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
                          f'text-anchor="{anchor}" font-weight="{weight}">{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=GRID, width=1, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{stroke}" stroke-width="{width}"{d}/>')

    def rect(self, x, y, w, h, fill, rx=0, opacity=1, stroke=None):
        s = f' stroke="{stroke}" stroke-width="1.5"' if stroke else ""
        self.items.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
                          f'fill="{fill}" opacity="{opacity}"{s}/>')

    def dot(self, x, y, r=7, fill=INK, hollow=False):
        if hollow:
            self.items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r - 1}" fill="#ffffff" stroke="{fill}" stroke-width="2.5"/>')
        else:
            self.items.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="#ffffff" stroke-width="2"/>')

    def path(self, d, stroke, width=2.5, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.items.append(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{width}" '
                          f'stroke-linecap="round" stroke-linejoin="round"{extra}/>')

    def save(self, name):
        body = "\n".join(self.items)
        with open(os.path.join(OUT, name), "w") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
                    f'viewBox="0 0 {self.width} {self.height}" font-family=\'{FONT}\'>\n{body}\n</svg>\n')


def accuracy(path_template, seeds=(0, 1, 2)):
    """Mean and sample SD of accuracy, recomputed from exact correct counts."""
    values = []
    for seed in seeds:
        with open(os.path.join("output/peft", path_template.format(seed), "metrics.json")) as f:
            m = json.load(f)
        values.append(100 * m["correct"] / m["total"])
    return st.mean(values), (st.stdev(values) if len(values) > 1 else 0.0)


E = "/seed{}/eval_scienceqa"
A = {
    "full": accuracy("uniform/scienceqa_modalities_fixed" + E),
    "r4": accuracy("uniform_rank4/scienceqa_20261030/seed{}/eval"),
    "flow16": accuracy("flow_pruned16_scienceqa/scienceqa_train_llava" + E),
    "random": accuracy("random_pruned16_scienceqa/scienceqa_train_llava" + E),
    "inverse": accuracy("inverse_pruned16_scienceqa/scienceqa_train_llava" + E),
    "flow8": accuracy("flow_pruned8_scienceqa/scienceqa_train_llava" + E),
    "early8": accuracy("early_pruned8_scienceqa/scienceqa_train_llava" + E),
    "ql8": accuracy("question_last_pruned8_scienceqa/scienceqa_train_llava" + E),
    "mid8": accuracy("middle_pruned8_scienceqa/scienceqa_train_llava" + E),
    "flow4": accuracy("flow_pruned4_scienceqa/scienceqa_train_llava" + E, (0,)),
    "pissa_flow": accuracy("flow_pruned16_scienceqa/scienceqa_train_llava_pissa" + E),
    "pissa_all": accuracy("uniform/scienceqa_train_llava_pissa" + E),
    "plus_flow": accuracy("flow_pruned16_scienceqa/scienceqa_train_llava_loraplus" + E, (0,)),
    "plus_all": accuracy("uniform/scienceqa_train_llava_loraplus" + E, (0,)),
    "dora": accuracy("flow_pruned16_scienceqa/scienceqa_train_llava_dora" + E, (0,)),
    "lr": accuracy("flow_pruned16_scienceqa/lr8e-4_scienceqa_train_llava" + E, (0,)),
}


def fmt(key):
    mean, sd = A[key]
    return f"{mean:.2f} ± {sd:.2f}" if sd else f"{mean:.2f}"


def knockout_method():
    s = Svg(1136, 330)
    boxes = [("System prompt", 40, 220, NEUTRAL), ("Image tokens (576)", 280, 300, ORANGE),
             ("Question", 600, 220, TEAL), ("ASSISTANT:", 840, 256, INK)]
    y, h = 140, 58
    for label, x, w, colour in boxes:
        s.rect(x, y, w, h, "#ffffff", rx=6, stroke=colour)
        s.text(x + w / 2, y + 37, label, size=21, anchor="middle")
    s.text(968, y - 12, "last position: the answer forms here", size=16, fill=MUTED, anchor="middle")
    # attention reads backwards: each token attends to earlier tokens
    s.path("M 710 140 C 710 56, 430 56, 430 136", ORANGE, width=3)
    s.path("M 422 124 L 430 138 L 440 125", ORANGE, width=3)
    s.text(570, 26, "Image → Question", size=22, anchor="middle", weight=700)
    s.text(570, 50, "the question cannot read the image", size=17, fill=MUTED, anchor="middle")
    s.path("M 968 198 C 968 284, 710 284, 710 202", TEAL, width=3)
    s.path("M 700 214 L 710 200 L 720 214", TEAL, width=3)
    s.text(839, 298, "Question → Last", size=22, anchor="middle", weight=700)
    s.text(839, 322, "the answer position cannot read the question", size=17, fill=MUTED, anchor="middle")
    for cx, cy, colour in [(570, 76, ORANGE), (839, 263, TEAL)]:  # the X marks where the connection is cut
        s.rect(cx - 15, cy - 15, 30, 30, "#ffffff", rx=15)
        s.path(f"M {cx - 9} {cy - 9} L {cx + 9} {cy + 9} M {cx + 9} {cy - 9} L {cx - 9} {cy + 9}", colour, width=4)
    s.save("knockout_method.svg")


def knockout_curve(path):
    values = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            values[int(row["layer"])].append(float(row["relative diff first"]))
    return [st.mean(values[layer]) for layer in range(32)]


def knockout_map():
    iq = knockout_curve(f"{FLOW_DIR}/Image___Question/SQA_train_correct_ScienceQA_window9_Image___Question.csv")
    ql = knockout_curve(f"{FLOW_DIR}/Question___Last/SQA_train_correct_ScienceQA_window9_Question___Last.csv")
    s = Svg(1136, 440)
    left, right, top, bottom = 90, 1110, 50, 385
    x = lambda layer: left + (right - left) * layer / 31
    y = lambda v: top + (bottom - top) * (2 - v) / 16  # +2 at the top, -14 at the bottom
    s.rect(x(0), top, x(15.5) - x(0), bottom - top, BAND)
    s.text(x(7.75), bottom - 12, "16 layers selected (0–15)", size=17, fill=MUTED, anchor="middle")
    for v in (0, -4, -8, -12):
        s.line(left, y(v), right, y(v), stroke=INK if v == 0 else GRID)
        s.text(left - 10, y(v) + 6, f"{v}", size=17, fill=MUTED, anchor="end")
    for layer in (0, 5, 10, 15, 20, 25, 31):
        s.text(x(layer), bottom + 26, str(layer), size=17, fill=MUTED, anchor="middle")
    s.text((left + right) / 2, bottom + 52, "Window centre (layer)", size=17, fill=MUTED, anchor="middle")
    s.items.append(f'<text transform="translate(22 {(top + bottom) / 2}) rotate(-90)" font-size="17" fill="{MUTED}" '
                   f'text-anchor="middle">Change in answer probability (%)</text>')
    for series, colour in ((iq, ORANGE), (ql, TEAL)):
        s.path("M " + " L ".join(f"{x(i):.1f} {y(v):.1f}" for i, v in enumerate(series)), colour)
    s.text(x(5.4), y(-9.2), "Image → Question", size=19, weight=700)
    s.text(x(5.4), y(-9.2) + 23, "about −12% while the window covers layer 0", size=17, fill=MUTED)
    s.text(x(16.6), y(-2.9), "Question → Last", size=19, weight=700)
    s.text(x(16.6), y(-2.9) + 23, "matters through the middle layers", size=17, fill=MUTED)
    s.text(x(23), y(1.0), "after layer 18, neither pathway matters", size=17, fill=MUTED, anchor="middle")
    # legend
    s.rect(left, 16, 24, 4, ORANGE, rx=2)
    s.text(left + 32, 24, "Image → Question", size=16, fill=MUTED)
    s.rect(left + 200, 16, 24, 4, TEAL, rx=2)
    s.text(left + 232, 24, "Question → Last", size=16, fill=MUTED)
    s.save("knockout_map.svg")


def layer0():
    one = knockout_curve(f"{FLOW_DIR}/Image___Question/SQA_train_correct_ScienceQA_window1_Image___Question.csv")
    nine = knockout_curve(f"{FLOW_DIR}/Image___Question/SQA_train_correct_ScienceQA_window9_Image___Question.csv")
    s = Svg(1136, 420)
    left, right, top, bottom = 90, 1110, 50, 365
    step = (right - left) / 32
    x = lambda layer: left + step * (layer + 0.5)
    y = lambda v: top + (bottom - top) * (2 - v) / 16
    for v in (0, -4, -8, -12):
        s.line(left, y(v), right, y(v), stroke=INK if v == 0 else GRID)
        s.text(left - 10, y(v) + 6, f"{v}", size=17, fill=MUTED, anchor="end")
    for layer, v in enumerate(one):
        top_y, bar_y = (y(0), y(v)) if v < 0 else (y(v), y(0))
        s.rect(x(layer) - step * 0.35, top_y, step * 0.7, bar_y - top_y, ORANGE, rx=2)
    s.path("M " + " L ".join(f"{x(i):.1f} {y(v):.1f}" for i, v in enumerate(nine)), INK, width=2, dash="6 5")
    for layer in (0, 5, 10, 15, 20, 25, 31):
        s.text(x(layer), bottom + 26, str(layer), size=17, fill=MUTED, anchor="middle")
    s.text((left + right) / 2, bottom + 52, "Layer", size=17, fill=MUTED, anchor="middle")
    s.items.append(f'<text transform="translate(22 {(top + bottom) / 2}) rotate(-90)" font-size="17" fill="{MUTED}" '
                   f'text-anchor="middle">Change in answer probability (%)</text>')
    s.text(x(6), y(-7), f"Layer 0 alone: {one[0]:.1f}%", size=19, weight=700)
    s.text(x(6), y(-7) + 23, "no other single layer costs more than 1%", size=17, fill=MUTED)
    s.rect(left, 16, 24, 14, ORANGE, rx=2)
    s.text(left + 32, 28, "One layer blocked", size=16, fill=MUTED)
    s.line(left + 220, 23, left + 250, 23, stroke=INK, width=2, dash="6 5")
    s.text(left + 258, 28, "Nine layers blocked (the map used for selection)", size=16, fill=MUTED)
    s.save("layer0.svg")


def placement(rows, name, note):
    """Rows: (label, sublabel, layers, colour, cell opacity, accuracy key)."""
    row_h = 64
    s = Svg(1136, 66 + row_h * len(rows) + 40)
    strip_x, cell, bar_x, bar_max = 262, 15, 790, 170
    s.text(strip_x, 26, "Layers with an adapter", size=16, fill=MUTED)
    for layer in (0, 8, 16, 24, 31):
        s.text(strip_x + layer * cell + 6, 52, str(layer), size=15, fill=MUTED, anchor="middle")
    s.text(bar_x, 26, "Gain over zero-shot, with accuracy %", size=16, fill=MUTED)
    for i, (label, sub, layers, colour, opacity, key) in enumerate(rows):
        y = 62 + row_h * i
        s.text(0, y + 22, label, size=20, weight=700 if colour != NEUTRAL else 400)
        s.text(0, y + 45, sub, size=15, fill=MUTED)
        for layer in range(32):
            on = layer in layers
            s.rect(strip_x + layer * cell, y + 8, cell - 2, 32, colour if on else BAND, rx=2, opacity=opacity if on else 1)
        mean = A[key][0]
        s.rect(bar_x, y + 15, bar_max * (mean - ZERO_SHOT) / 14, 18, colour if colour != NEUTRAL else NEUTRAL, rx=4)
        s.text(bar_x + bar_max + 10, y + 31, fmt(key), size=19, weight=700 if colour != NEUTRAL else 400)
        if i:
            s.line(0, y, 1136, y)
    s.text(0, 66 + row_h * len(rows) + 22, note, size=15, fill=MUTED)
    s.save(name)


def size_accuracy():
    s = Svg(1136, 450)
    left, right, top, bottom = 80, 1000, 26, 392
    x = lambda p: left + (right - left) * p / 4.4
    y = lambda v: top + (bottom - top) * (83 - v) / 15
    for v in (70, 74, 78, 82):
        s.line(left, y(v), right, y(v))
        s.text(left - 10, y(v) + 6, str(v), size=17, fill=MUTED, anchor="end")
    for p in (0, 1, 2, 3, 4):
        s.text(x(p), bottom + 26, f"{p}M", size=17, fill=MUTED, anchor="middle")
    s.text((left + right) / 2, bottom + 52, "Trainable adapter parameters", size=17, fill=MUTED, anchor="middle")
    s.items.append(f'<text transform="translate(20 {(top + bottom) / 2}) rotate(-90)" font-size="17" fill="{MUTED}" '
                   f'text-anchor="middle">ScienceQA accuracy (%)</text>')
    s.line(left, y(ZERO_SHOT), right, y(ZERO_SHOT), stroke=NEUTRAL, dash="6 5")
    s.text(right, y(ZERO_SHOT) - 8, f"Zero-shot {ZERO_SHOT}", size=17, fill=MUTED, anchor="end")
    s.line(left, y(A["full"][0]), right, y(A["full"][0]), stroke=NEUTRAL, dash="6 5")
    flow = [(0.524, "flow4", True), (1.049, "flow8", False), (2.097, "flow16", False)]
    s.path("M " + " L ".join(f"{x(p):.1f} {y(A[k][0]):.1f}" for p, k, _ in flow), ORANGE)
    for p, k, hollow in flow:
        s.dot(x(p), y(A[k][0]), fill=ORANGE, hollow=hollow)
    s.text(x(0.524) + 14, y(A["flow4"][0]) + 22, f"4 flow layers  {A['flow4'][0]:.2f}", size=17, weight=700)
    s.text(x(1.049) + 8, y(A["flow8"][0]) + 26, f"8 flow layers  {A['flow8'][0]:.2f}", size=17, weight=700)
    s.text(x(2.097) - 12, y(A["flow16"][0]) - 12, f"16 flow layers  {A['flow16'][0]:.2f}", size=17, weight=700, anchor="end")
    for p, k, label, dy in [(2.097, "r4", "Uniform rank 4", 2), (2.097, "random", "Random 16", 6), (2.097, "inverse", "Inverse 16", 6)]:
        s.dot(x(p), y(A[k][0]), fill=NEUTRAL)
        s.text(x(p) + 14, y(A[k][0]) + dy, f"{label}  {A[k][0]:.2f}", size=17)
    s.dot(x(4.194), y(A["full"][0]), fill=TEAL)
    s.text(x(4.194) - 6, y(A["full"][0]) - 14, f"Full LoRA  {A['full'][0]:.2f}", size=17, weight=700, anchor="end")
    # legend
    lx = 700
    s.dot(lx, y(77.2), fill=ORANGE)
    s.text(lx + 16, y(77.2) + 6, "Flow-selected layers", size=16, fill=MUTED)
    s.dot(lx, y(76.2), fill=NEUTRAL)
    s.text(lx + 16, y(76.2) + 6, "Other placements at 2.10M", size=16, fill=MUTED)
    s.dot(lx, y(75.2), fill=TEAL)
    s.text(lx + 16, y(75.2) + 6, "All 32 layers, rank 8", size=16, fill=MUTED)
    s.dot(lx, y(74.2), fill=ORANGE, hollow=True)
    s.text(lx + 16, y(74.2) + 6, "Single seed; others are three-seed means", size=16, fill=MUTED)
    s.save("size_accuracy.svg")


def variants():
    rows = [("LoRA+", "plus_flow", "plus_all", "1 seed"), ("PiSSA", "pissa_flow", "pissa_all", "3 seeds"),
            ("LoRA, 4× learning rate", "lr", None, "1 seed"), ("LoRA", "flow16", "full", "3 seeds"), ("DoRA", "dora", None, "1 seed")]
    row_h = 62
    s = Svg(1136, 90 + row_h * len(rows) + 44)
    left, right, top = 300, 1110, 70
    bottom = top + row_h * len(rows)
    x = lambda v: left + (right - left) * (v - 80.5) / 3.5
    for v in (81, 82, 83, 84):
        s.line(x(v), top, x(v), bottom)
        s.text(x(v), bottom + 26, str(v), size=17, fill=MUTED, anchor="middle")
    s.text((left + right) / 2, bottom + 52, "ScienceQA accuracy (%), whiskers show ±1 SD over seeds", size=17, fill=MUTED, anchor="middle")
    s.dot(left, 24, fill=ORANGE)
    s.text(left + 14, 30, "Layers 0–15 (2.10M)", size=17, fill=MUTED)
    s.dot(left + 210, 24, fill=TEAL)
    s.text(left + 224, 30, "All 32 layers (4.19M)", size=17, fill=MUTED)
    s.dot(left + 430, 24, fill=ORANGE, hollow=True)
    s.dot(left + 450, 24, fill=TEAL, hollow=True)
    s.text(left + 464, 30, "Hollow: single seed", size=17, fill=MUTED)
    for i, (label, flow_key, all_key, seeds) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        s.text(0, y - 2, label, size=20, weight=700 if label == "PiSSA" else 400)
        s.text(0, y + 20, seeds, size=15, fill=MUTED)
        s.line(left, y, right, y, stroke=BAND)
        for key, colour in ((all_key, TEAL), (flow_key, ORANGE)):
            if key is None:
                continue
            mean, sd = A[key]
            if sd:
                s.line(x(mean - sd), y, x(mean + sd), y, stroke=colour, width=2.5)
            s.dot(x(mean), y, fill=colour, hollow=not sd)
            anchor = "middle"
            if all_key and abs(A[flow_key][0] - A[all_key][0]) < 0.3:  # keep close labels apart
                anchor = "end" if mean < max(A[flow_key][0], A[all_key][0]) else "start"
            s.text(x(mean), y - 15, f"{mean:.2f}", size=16, anchor=anchor)
    s.save("variants.svg")


def motivation():
    s = Svg(1136, 150)
    strip_x, cell = 340, 24
    rows = [("Standard practice", "an adapter on every layer", TEAL), ("Our question", "which layers actually need one?", ORANGE)]
    for i, (label, sub, colour) in enumerate(rows):
        y = 8 + 72 * i
        s.text(0, y + 21, label, size=21, weight=700)
        s.text(0, y + 46, sub, size=17, fill=MUTED)
        for layer in range(32):
            if i == 0:
                s.rect(strip_x + layer * cell, y + 4, cell - 3, 44, colour, rx=3)
            else:
                s.rect(strip_x + layer * cell, y + 4, cell - 3, 44, "#ffffff", rx=3, stroke=colour)
                s.text(strip_x + layer * cell + (cell - 3) / 2, y + 33, "?", size=17, fill=MUTED, anchor="middle")
    s.save("motivation.svg")


def category_accuracy(template, questions):
    correct = defaultdict(list)
    for seed in range(3):
        with open(os.path.join("output/peft", template.format(seed), "predictions.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                ok = str(r["prediction"]).strip().lower().rstrip(".") == str(r["answer"]).strip().lower().rstrip(".")
                correct[questions[r["question_id"]]].append(ok)
    return {c: 100 * st.mean(v) for c, v in correct.items()}


def categories():
    questions = {}
    with open("datasets/scienceqa/test_questions.jsonl") as f:
        for line in f:
            r = json.loads(line)
            questions[r["question_id"]] = r["category"]
    flow = category_accuracy("flow_pruned16_scienceqa/scienceqa_train_llava" + E, questions)
    inv = category_accuracy("inverse_pruned16_scienceqa/scienceqa_train_llava" + E, questions)
    rows = [("Oceania: geography", "Oceania geography"), ("Solutions", "Solutions (particle diagrams)"),
            ("Maps", "Maps"), ("Weather and climate", "Weather and climate"), ("Materials", "Materials"),
            ("Magnets", "Magnets"), ("Basic economic principles", "Basic economics"), ("State capitals", "State capitals")]
    row_h = 45
    s = Svg(1136, 56 + row_h * len(rows) + 54)
    left, right, top = 330, 1090, 52
    bottom = top + row_h * len(rows)
    x = lambda v: left + (right - left) * v / 100
    for v in (0, 25, 50, 75, 100):
        s.line(x(v), top, x(v), bottom)
        s.text(x(v), bottom + 26, str(v), size=17, fill=MUTED, anchor="middle")
    s.text((left + right) / 2, bottom + 50, "Accuracy (%), mean of three seeds", size=17, fill=MUTED, anchor="middle")
    s.dot(left, 20, fill=ORANGE)
    s.text(left + 14, 26, "Flow-pruned, layers 0–15", size=17, fill=MUTED)
    s.dot(left + 270, 20, fill=TEAL)
    s.text(left + 284, 26, "Inverse, layers 16–31", size=17, fill=MUTED)
    for i, (key, label) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        s.text(0, y + 7, label, size=19)
        a, b = flow[key], inv[key]
        s.line(x(a), y, x(b), y, stroke=NEUTRAL, width=3)
        s.dot(x(b), y, fill=TEAL)
        s.dot(x(a), y, fill=ORANGE)
        gap = abs(a - b)
        if gap >= 8:
            s.text(x(b) - 13, y + 6, f"{b:.1f}", size=16, anchor="end")
            s.text(x(a) + 13, y + 6, f"{a:.1f}", size=16)
        else:
            s.text(min(x(a), x(b)) - 13, y + 6, f"{b:.1f} vs {a:.1f}", size=16, anchor="end")
    s.save("categories.svg")


if __name__ == "__main__":
    motivation()
    knockout_method()
    knockout_map()
    layer0()
    full, half = set(range(16)), set(range(16, 32))
    every = set(range(32))
    placement([("Full LoRA", "rank 8, 4.19M", every, TEAL, 1, "full"),
               ("Flow-pruned", "rank 8, 2.10M", full, ORANGE, 1, "flow16"),
               ("Uniform rank 4", "rank 4, 2.10M", every, NEUTRAL, 0.45, "r4"),
               ("Random", "rank 8, 2.10M, one set", {1, 4, 6, 8, 9, 11, 12, 13, 15, 16, 18, 24, 25, 28, 30, 31}, NEUTRAL, 1, "random"),
               ("Inverse", "rank 8, 2.10M", half, NEUTRAL, 1, "inverse")],
              "placement_16.svg", "Three-seed means ± sample SD. Bars start at zero-shot (69.16%).")
    placement([("Combined flow", "both pathways", {0, 1, 2, 3, 4, 9, 10, 11}, ORANGE, 1, "flow8"),
               ("Question → Last", "one pathway only", set(range(9, 17)), NEUTRAL, 1, "ql8"),
               ("Middle block", "layers 12–19", set(range(12, 20)), NEUTRAL, 1, "mid8")],
              "placement_8.svg", "All placements use 1.05M parameters. Three-seed means ± sample SD. Bars start at zero-shot (69.16%).")
    size_accuracy()
    variants()
    categories()
    print({k: fmt(k) for k in A})
