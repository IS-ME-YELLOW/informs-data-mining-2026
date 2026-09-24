"""Create an editable monochrome methodology diagram and its PNG preview."""

from pathlib import Path
from xml.sax.saxutils import escape
import cairosvg
from PIL import Image

OUT = Path(__file__).resolve().parent
parts = []


def text(x, y, value, size=18, weight="normal", anchor="middle"):
    parts.append(
        f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}">{escape(value)}</text>'
    )


def rect(x, y, w, h, dashed=False, stroke=1.5):
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    parts.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="white" stroke="black" stroke-width="{stroke}"{dash}/>'
    )


def path(points, arrow=True, dashed=False, width=1.6):
    d = "M " + " L ".join(f"{x},{y}" for x, y in points)
    end = ' marker-end="url(#arrow)"' if arrow else ""
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    parts.append(f'<path d="{d}" fill="none" stroke="black" stroke-width="{width}"{end}{dash}/>')


def circle(x, y, r=7, fill="white", stroke=1.4):
    parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="black" stroke-width="{stroke}"/>')


def box(x, y, w, h, lines, size=18, leading=25):
    rect(x, y, w, h)
    baseline = y + h / 2 - (len(lines) - 1) * leading / 2 + size * 0.34
    for j, value in enumerate(lines):
        text(x + w / 2, baseline + j * leading, value, size)


parts.append('''<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1040" viewBox="0 0 1800 1040">
<title>Methodology: component forecasting, horizon-specific ensembles, and spatial residual correction</title>
<desc>Black-and-white structure draft. Two parallel tree-model sources form a horizon-specific base forecast. A graph attention model predicts a residual correction, while a skip connection preserves the base forecast. Three lower panels explain the component structure, target-time alignment and tail gating, and spatial residual correction.</desc>
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="black"/></marker></defs>
<rect width="1800" height="1040" fill="white"/>
<g font-family="DejaVu Sans, Arial, sans-serif" fill="black">''')

# Overview: inference path, with parallel tree sources and a base-forecast skip.
text(30, 38, "FORECASTING FRAMEWORK", 23, "bold", "start")
box(30, 145, 215, 145, ["Shared inputs", "72h outage history", "216h weather", "County attributes"], 18, 28)
path([(245, 218), (272, 218), (272, 153), (300, 153)])
path([(272, 218), (272, 286), (300, 286)])
rect(300, 120, 250, 85)
text(425, 145, "① Enhanced", 19)
text(425, 169, "component models", 19)
text(425, 193, "+ selected neighbor features", 13)
box(300, 253, 250, 66, ["Long-horizon", "prediction sources"], 19, 25)
path([(550, 153), (615, 153)])
path([(550, 286), (615, 286)])

rect(615, 117, 350, 215)
text(790, 149, "② Align and combine", 20, "bold")
text(790, 191, "1h / 6h: aligned T₁ / T₆", 18)
text(790, 232, "24h: H[(T₂₄ + S₂₄) / 2]", 18)
text(790, 273, "48h: S₄₈", 18)
text(790, 310, "Horizon-specific base forecast", 16)

path([(965, 199), (1045, 199)])
path([(1005, 199), (1005, 77), (1415, 77), (1415, 181)])
text(1210, 65, "Base forecast (skip connection)", 17)
box(1045, 162, 210, 74, ["③ GAT", "residual model"], 19, 26)
box(1030, 280, 240, 64, ["Node features", "+ geographic graph"], 17, 24)
path([(1150, 280), (1150, 236)])
path([(1255, 199), (1397, 199)])
text(1326, 180, "αₕ r̂ₕ", 21)
circle(1415, 199, 18)
text(1415, 207, "+", 25)
path([(1433, 199), (1480, 199)])
box(1480, 151, 290, 96, ["Postprocess H", "Final OSI forecasts", "1h · 6h · 24h · 48h"], 19, 26)
text(1480, 303, "H: clip to [0, 0.65];", 15.5, anchor="start")
text(1480, 327, "set values < 0.001 to zero", 15.5, anchor="start")
path([(30, 385), (1770, 385)], arrow=False, width=1)

# Mechanism 1: local component structures, not a serial P-to-D pipeline.
rect(30, 425, 520, 585)
text(50, 463, "① Component structure", 22, "bold", "start")
text(50, 501, "LightGBM models for P, N, D, R", 17, anchor="start")

rect(55, 529, 470, 142)
text(290, 560, "Initial-state reconstruction: P₁, P₆", 18, "bold")
box(73, 590, 95, 47, ["p = P₇₁"], 18)
path([(168, 613), (196, 613)])
text(353, 602, "Predict normalized branches a and b", 16)
text(353, 637, "P̂ = p â + (1 − p) b̂", 22)

rect(55, 694, 470, 114)
text(290, 725, "Known-history reconstruction: D₁", 18, "bold")
text(290, 761, "Known contribution K + predicted U", 19)
text(290, 791, "Use when K > 0 (target hours 73–76)", 16)

box(55, 832, 470, 62, ["Other component sources:", "direct regression"], 18, 25)
text(290, 939, "Recombine predicted components", 18, "bold")
text(290, 973, "OSI = max(0, .40P + .35N + .25D − .10R)", 17)

# Mechanism 2: same target, then a separate long-horizon ensemble.
rect(570, 425, 650, 585)
text(590, 463, "② Time alignment and tail gating", 22, "bold", "start")
text(590, 507, "Short horizons: align the target hour", 18, "bold", "start")
box(598, 536, 220, 43, ["Origin 119 + 1h"], 18)
box(598, 604, 220, 43, ["Origin 114 + 6h"], 18)
box(899, 552, 290, 78, ["Same target hour 120", "Average component forecasts"], 17, 26)
path([(818, 557), (858, 557), (858, 577), (899, 577)])
path([(818, 626), (871, 626), (871, 608), (899, 608)])
text(895, 677, "Other eligible horizons join the same average.", 16)
text(895, 712, "Aligned components → OSI → T₁, T₆", 19)
path([(590, 737), (1200, 737)], arrow=False, width=1)
text(590, 773, "Long horizons: 24h and 48h", 18, "bold", "start")
box(590, 795, 258, 109, ["LightGBM / XGBoost / CatBoost", "Direct OSI + direct components", "6 processed forecasts"], 14.5, 26)
path([(848, 850), (889, 850)])
box(889, 795, 306, 109, ["Tail gate", "B > θ: reference B", "Otherwise: six-source mean", "Output S₂₄, S₄₈"], 16.5, 24)
text(590, 936, "B: original LightGBM component forecast", 15.5, anchor="start")
text(590, 963, "θ: inner positive-prediction 95th percentile", 15.5, anchor="start")
text(590, 989, "T₂₄: enhanced components, without time alignment", 15.5, anchor="start")

# Mechanism 3: schematic graph, correction, and a separate training-only note.
rect(1240, 425, 530, 585)
text(1260, 463, "③ Spatial residual correction", 22, "bold", "start")
text(1332, 513, "County graph", 16)
nodes = [(1290, 551), (1345, 546), (1384, 585), (1323, 591), (1280, 623), (1369, 635)]
for a, b in [(0, 1), (0, 3), (0, 4), (1, 2), (1, 3), (2, 3), (2, 5), (3, 4), (3, 5)]:
    path([nodes[a], nodes[b]], arrow=False, width=1.3)
for x, y in nodes:
    circle(x, y, 7)
box(1415, 540, 326, 94, ["Base forecast", "+ node features", "+ geographic context"], 18, 25)
path([(1323, 642), (1323, 688)])
path([(1578, 634), (1578, 688)])
box(1270, 688, 470, 64, ["GAT residual model", "One model per horizon"], 19, 26)
path([(1505, 752), (1505, 796)])
box(1364, 796, 282, 52, ["α × predicted residual"], 19)
text(1505, 884, "Final OSI = H(base + α × residual)", 19)
rect(1270, 927, 470, 66, dashed=True)
text(1505, 952, "Training only: r = y − cross-fitted base", 16)
text(1505, 978, "Choose α using inner county-level CV", 16)

parts.append("</g></svg>")
svg = "\n".join(parts)
OUT.mkdir(parents=True, exist_ok=True)
svg_path = OUT / "methodology_bw_v1.svg"
png_path = OUT / "methodology_bw_v1.png"
svg_path.write_text(svg, encoding="utf-8")
cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(png_path), output_width=2700, output_height=1560)
with Image.open(png_path) as preview:
    preview.convert("L").save(png_path)
print(svg_path)
print(png_path)
