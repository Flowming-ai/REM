#!/usr/bin/env python3
"""Fig. 1 of the Assignment 3 report: the evaluation framework with plain-language labels.

Same drawing as Fig. 1 of the Assignment 2 report, with the labels in the terms of the Assignment 3 report
("reference" instead of "clean", "realised" instead of "actual"). Every label is plain English (no abbreviations or
tool names), so the figure can be read without the text. Every element is drawn as SVG text or geometry; no
raster image is embedded. The bottom panel quotes equation numbers (Eqs. (1)-(4), (6)), which are the same in both
reports; update them if the report's numbering changes. Run from the project root:
    uv run --project experiments --with cairosvg python "assignment 3/code/fig1_model_framework.py"
"""
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET
import cairosvg
import cairocffi as cairo
from cairosvg.surface import PDFSurface


class CompatiblePDFSurface(PDFSurface):
    """Keep the vector asset within the manuscript compiler's PDF 1.5 level."""

    def _create_surface(self, width, height):
        surface, width, height = super()._create_surface(width, height)
        surface.restrict_to_version(cairo.PDF_VERSION_1_5)
        return surface, width, height

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
W, H = 1900, 826
INK = "#181818"
GRAY = "#f3f3f3"
ORANGE = "#fbd5bd"
BLUE = "#c8e5f5"
ORANGE_DARK = "#ad582c"
BLUE_DARK = "#366b94"
FONT = "Times New Roman, Liberation Serif, Noto Serif, serif"
parts = []


def add(s):
    parts.append(s)


def text(x, y, value, size=30, anchor="middle", fill=INK, weight="normal", italic=False):
    style = ' font-style="italic"' if italic else ''
    value = escape(value).replace('t_d', 't<tspan dy="8" font-size="22.5">d</tspan>')
    add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'text-anchor="{anchor}" fill="{fill}" font-weight="{weight}"{style}>{value}</text>')


def rect(x, y, w, h, fill="white", stroke=INK, width=1.8, dash=False):
    d = ' stroke-dasharray="8 6"' if dash else ''
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"{d}/>')


def path(points, stroke=INK, width=2.2, arrow=False, dash=False):
    ps = " ".join(f'{x},{y}' for x, y in points)
    ar = ' marker-end="url(#arrow)"' if arrow else ''
    ds = ' stroke-dasharray="7 5"' if dash else ''
    add(f'<polyline points="{ps}" fill="none" stroke="{stroke}" stroke-width="{width}" '
        f'stroke-linejoin="round" stroke-linecap="round"{ar}{ds}/>')


def circle(x, y, r, fill="white", stroke=INK, width=1.8):
    add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>')


def lines(x, y, values, leading=34, size=30, **kwargs):
    for i, value in enumerate(values):
        text(x, y + i * leading, value, size=size, **kwargs)


def block(x, y, w, h, labels, fill="white", size=30):
    rect(x, y, w, h, fill=fill)
    n = len(labels)
    start = y + h / 2 - (n - 1) * 17 + 10
    lines(x + w / 2, start, labels, size=size)


def module(x, y, w, h, title):
    rect(x, y, w, h, dash=True)
    text(x + w / 2, y + 35, title, size=34, weight="bold")


def battery(x, y, w, h, fill, level=0.6):
    rect(x, y, w, h, fill="white", width=2)
    rect(x + w, y + h * 0.31, w * 0.075, h * 0.38, fill=INK, width=0)
    rect(x + 5, y + 5, (w - 10) * level, h - 10, fill=fill, stroke="none", width=0)
    path([(x + w * .70, y + h * .5), (x + w * .90, y + h * .5)], width=2)
    path([(x + w * .80, y + h * .3), (x + w * .80, y + h * .7)], width=2)


def pv(x, y):
    add(f'<polygon points="{x+12},{y} {x+75},{y} {x+88},{y+45} {x},{y+45}" fill="{BLUE}" stroke="{INK}" stroke-width="2"/>')
    for f in (.33, .66):
        path([(x+12+63*f, y), (x+88*f, y+45)], width=1.3)
    for f in (.33, .66):
        path([(x+12*(1-f), y+45*f), (x+75+13*f, y+45*f)], width=1.3)
    path([(x+39, y+45), (x+39, y+55), (x+18, y+55), (x+65, y+55)], width=2)
    circle(x+75, y-13, 7, fill="#f4cc67", width=1)
    for dx, dy in ((0,-13),(13,0),(10,-10),(-10,-10)):
        path([(x+75+dx*.72,y-13+dy*.72),(x+75+dx,y-13+dy)],width=1.3)


def grid(x, y):
    path([(x+30,y),(x+4,y+68),(x+55,y+68),(x+30,y)],width=2)
    for yy, span in ((18,18),(35,27),(50,34)):
        path([(x+30-span,y+yy),(x+30+span,y+yy)],width=2)
    path([(x+22,y+22),(x+47,y+49),(x+11,y+49),(x+55,y+68)],width=1.4)
    path([(x+38,y+22),(x+12,y+49),(x+49,y+49),(x+4,y+68)],width=1.4)


def tiny_curve(x, y, w, h, color=INK, stairs=False):
    path([(x,y),(x,y+h),(x+w,y+h)],width=1.3)
    if stairs:
        pts=[(x+5,y+h*.82),(x+w*.23,y+h*.82),(x+w*.23,y+h*.55),(x+w*.5,y+h*.55),
             (x+w*.5,y+h*.25),(x+w*.73,y+h*.25),(x+w*.73,y+h*.68),(x+w-.5,y+h*.68)]
    else:
        pts=[(x+4,y+h*.82),(x+w*.22,y+h*.72),(x+w*.4,y+h*.45),(x+w*.62,y+h*.3),
             (x+w*.81,y+h*.47),(x+w,y+h*.85)]
    path(pts,stroke=color,width=3)


def main():
    add(f'<svg xmlns="http://www.w3.org/2000/svg" width="7.16in" height="{7.16*H/W:.6f}in" viewBox="0 0 {W} {H}">')
    add('<title>Paired evaluation of recovery costs after poisoned energy-agent decisions</title>')
    add('<desc>Inputs feed the paired attacked and reference runs. Their detection states feed two parallel branches: constrained recovery-cost evaluation, and rolling recovery-controller comparison. Predicted cost differences and realised outcomes meet at validation. The attribution scalar does not feed any controller. A bottom panel states the optimisation model (objective and constraints) used in steps 1 to 3.</desc>')
    add('<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#181818"/></marker></defs>')
    rect(0,0,W,H,stroke="none",width=0)

    module(10,50,190,605,"Inputs")
    module(220,50,530,605,"1. Run each scenario twice")
    module(800,50,665,280,"2. Extra cost at detection")
    module(800,360,665,295,"3. Recover step by step, compare")
    module(1495,50,395,605,"Results")

    battery(66,116,70,36,BLUE,.6)
    text(105,186,"Battery",size=27)
    pv(63,222)
    lines(105,305,["Solar power","and demand"],leading=28,size=25)
    grid(74,348)
    text(105,446,"Power grid",size=27)
    tiny_curve(40,460,130,48,BLUE_DARK,stairs=True)
    text(105,540,"Electricity",size=25)
    text(105,566,"prices",size=25)
    lines(105,605,["Starting charge","Attack settings"],leading=27,size=24)
    # Input arrows indicate common scenario data, not sensed control feedback.
    path([(200,307),(238,307)],arrow=True)
    path([(200,486),(238,486)],arrow=True)

    block(238,126,496,62,["Same building, weather and prices"],fill=GRAY)
    text(430,232,"With attack",fill=ORANGE_DARK)
    text(430,410,"Without attack",fill=BLUE_DARK)
    for y, fill, labels in ((264,ORANGE,(["False","forecast"],["Controller"],["Battery"])),
                            (443,BLUE,(["True","forecast"],["Same","controller"],["Battery"]))):
        block(238,y,112,86,labels[0],fill,size=26)
        block(364,y,126,86,labels[1],fill,size=24)
        block(504,y,100,86,labels[2],fill,size=26)
        yc=y+43
        path([(350,yc),(364,yc)],arrow=True)
        path([(490,yc),(504,yc)],arrow=True)
        path([(604,yc),(641,yc)],arrow=True)
    rect(624,200,110,353,fill=GRAY)
    lines(679,224,["Battery","level at","detection"],leading=26,size=24)
    battery(646,293,60,29,ORANGE,.30)
    text(679,348,"attacked",size=24)
    battery(646,472,60,29,BLUE,.75)
    text(679,527,"reference",size=24)
    # Repaint short incoming arrows over the state-card boundary.
    path([(604,307),(640,307)],arrow=True)
    path([(604,486),(640,486)],arrow=True)
    block(238,578,496,45,["Log of every action and battery level"],fill=GRAY,size=27)

    # One state-pair fork. Both branches receive the paired initial conditions;
    # neither the attribution estimate nor a reference offset enters control.
    path([(734,405),(775,405)])
    circle(775,405,4,fill=INK,width=0)
    path([(775,405),(775,235),(811,235)])
    path([(775,405),(775,508),(835,508)])

    text(1132,127,"Same forecasts, limits and deadline")
    path([(811,235),(811,190),(820,190)],arrow=True)
    path([(811,235),(811,284),(820,284)],arrow=True)
    block(820,154,210,72,["Cheapest recovery","from attacked level"],ORANGE,size=23)
    block(820,248,210,72,["Cheapest recovery","from reference level"],BLUE,size=23)
    path([(1030,190),(1080,190),(1080,218),(1100,218)],arrow=True)
    path([(1030,284),(1080,284),(1080,269),(1100,269)],arrow=True)
    block(1100,187,230,110,["Debt ΔJ:","difference of","the two costs"],GRAY,size=26)
    text(1214,322,"only if both runs can still recover",size=23)
    path([(1330,244),(1516,244)],arrow=True)
    text(1408,224,"Prediction",size=26)

    # Physical feedback is conditional for the six-method comparison:
    # software-only replay does not observe the executed battery state.
    path([(833,508),(835,508),(835,461),(855,461)],arrow=True)
    path([(835,508),(835,584),(855,584)],arrow=True)
    block(855,429,175,64,["6 recovery","methods"],ORANGE,size=27)
    block(855,552,175,64,["Controller","(no attack)"],BLUE,size=27)
    block(1080,429,115,64,["Battery","step"],ORANGE)
    block(1080,552,115,64,["Battery","step"],BLUE)
    for y, cy in ((429,461),(552,584)):
        path([(1030,cy),(1080,cy)],arrow=True)
        loop_y = y + (104 if y==429 else 98)
        path([(1137,y+64),(1137,loop_y),(942,loop_y),(942,y+64)],arrow=True,dash=(y==429))
        text(1040,loop_y-8,"reading, if used" if y==429 else "battery reading",size=24)
        path([(1195,cy),(1220,cy)])
    path([(1220,461),(1220,584)])
    path([(1220,523),(1250,523)],arrow=True)
    block(1250,451,195,146,["Realised cost","and demand","served"],GRAY)
    path([(1445,523),(1516,523)],arrow=True)

    # Outputs are schematic glyphs, never measurements or fabricated curves.
    text(1692,136,"Extra recovery cost",size=32)
    tiny_curve(1550,168, 90,64,ORANGE_DARK,stairs=True)
    path([(1554,220),(1571,214),(1590,201),(1610,189),(1638,177)],stroke=BLUE_DARK,width=3)
    for row in range(3):
        for col in range(3):
            rect(1680+col*26,170+row*24,26,24,
                 fill=(BLUE if col+row<3 else ORANGE),width=1.2)
    circle(1814,204,31,width=2)
    path([(1814,184),(1814,204),(1830,211)],width=2.5)
    lines(1692,272,["Debt ΔJ, whether recovery","is possible, recovery time"],leading=27,size=24)

    rect(1517,342,350,112,fill=GRAY)
    text(1692,373,"Check the prediction",size=30)
    path([(1540,390),(1540,436),(1615,436)],width=1.3)
    path([(1542,434),(1611,393)],stroke="#777777",width=1.3,dash=True)
    for px,py in ((1549,428),(1560,425),(1574,416),(1586,410),(1594,400),(1606,399)):
        circle(px,py,2.8,fill=BLUE_DARK,stroke=BLUE_DARK,width=1)
    lines(1740,412,["predicted vs.","realised extra cost"],leading=31,size=27)
    # Attribution and actual executions enter comparison independently.
    path([(1480,244),(1506,244),(1506,400),(1516,400)],arrow=True)
    path([(1480,523),(1480,433),(1516,433)],arrow=True)
    circle(1480,244,3.5,fill=INK,width=0)
    circle(1480,523,3.5,fill=INK,width=0)

    text(1692,512,"Realised outcomes",size=32)
    text(1692,553,"Cost • demand served • time")
    tiny_curve(1550,576,116,51,ORANGE_DARK,stairs=True)
    path([(1554,620),(1580,611),(1580,598),(1604,598),(1604,591),(1633,591),(1633,585),(1664,585)],stroke=BLUE_DARK,width=3)
    # Vector service/records glyph; its marks are illustrative icons only.
    for i in range(3):
        rect(1730,574+i*19,14,14,fill=GRAY,width=1.2)
        path([(1733,582+i*19),(1737,586+i*19),(1743,577+i*19)],width=1.5)
        path([(1756,581+i*19),(1824,581+i*19)],width=1.5)

    # Optimisation model shared by the controllers (steps 1 and 3) and the cheapest recovery (step 2).
    rect(220,682,1245,132,dash=True)
    text(325,728,"Optimisation",size=30,weight="bold")
    text(325,762,"model",size=30,weight="bold")
    text(325,797,"used in steps 1–3",size=24)
    rect(432,692,370,112,fill=GRAY)
    text(617,728,"Objective",size=26,weight="bold")
    lines(617,760,["lowest cost of electricity","bought from the grid, Eq. (4)"],leading=29,size=24)
    rect(818,692,632,112,fill=GRAY)
    text(1134,718,"Constraints",size=26,weight="bold")
    lines(1134,745,["stored energy, Eq. (1); power balance, Eq. (2);",
                    "battery, grid and solar limits, Eq. (3);",
                    "after detection: target and deadline, Eq. (6)"],leading=26,size=24)
    path([(480,682),(480,658)],arrow=True)
    path([(1350,682),(1350,658)],arrow=True)

    add('</svg>')
    OUT.mkdir(exist_ok=True)
    svg=OUT/'fig1_model_framework.svg'
    svg.write_text('\n'.join(parts)+'\n',encoding='utf-8')
    root=ET.parse(svg).getroot()
    assert not root.findall('.//{http://www.w3.org/2000/svg}image'), 'Raster element in SVG'
    pdf=OUT/'fig1_model_framework.pdf'
    png=OUT/'fig1_model_framework.png'
    CompatiblePDFSurface.convert(url=str(svg),write_to=str(pdf))
    cairosvg.svg2png(url=str(svg),write_to=str(png),output_width=2280,output_height=round(2280*H/W))
    print(f'Wrote {svg.name}, {pdf.name}, {png.name}: {W} × {H}; 7.16 × {7.16*H/W:.3f} in.')


if __name__ == '__main__':
    main()
