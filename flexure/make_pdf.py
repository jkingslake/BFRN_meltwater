"""
Build derivation.pdf from derivation.md (needs pandoc and a LaTeX install).

    python make_pdf.py        ->  derivation.pdf

The Markdown stays the single source.  This script only adapts it for print:
figure captions, a table of contents, numbered sections, no GitHub-only
links, and a boxed summary.
"""
import datetime
import os
import re
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "derivation.md")
OUT = os.path.join(HERE, "derivation.pdf")

# caption and width for every figure (the Markdown alt texts are short labels)
FIGS = {
    "diagram_setup": ("A vertical slice through the problem. A meltwater lake of depth $d(x)$ "
                      "sits on a floating ice shelf of thickness $H$, which deflects downwards "
                      "by $w(x)$. Vertical scale exaggerated.", "100%"),
    "diagram_buoyancy": ("Pushing a floating column down by $w$ displaces extra sea water and "
                         "produces an extra upthrust $\\rho_{sw} g\\, w$. Buoyancy therefore acts "
                         "like a bed of springs of stiffness $k = \\rho_{sw} g$.", "92%"),
    "diagram_free_body": ("Forces and moments on a slice of the shelf of width $dx$.", "90%"),
    "diagram_bending": ("Cross-sections stay plane as the beam bends, so fibre strain and stress "
                        "vary linearly with height $z$. Integrating $\\sigma z$ across the "
                        "thickness gives $M = D\\kappa$.", "100%"),
    "diagram_maxwell": ("A Maxwell element (a spring and a dashpot in series) and its two "
                        "classic responses: creep under a fixed moment, and relaxation at fixed "
                        "curvature.", "100%"),
    "diagram_limits": ("The two limits of the model: an elastic response at $t = 0^+$ and local "
                       "isostasy as $t \\to \\infty$.", "100%"),
    "step_response": None,  # uses the italic caption that follows it in the Markdown
    "relaxation_spectrum": ("Left: relaxation time against wavelength. Right: each wavelength "
                            "jumps elastically and then relaxes towards isostasy at its own "
                            "rate.", "100%"),
    "fill_drain_timeseries": ("A lake fills for 10 days, is held for 15 days and drains in 6 "
                              "hours. Top: lake depth. Middle: deflection at the lake centre. "
                              "Bottom: bending stress at the upper surface.", "82%"),
    "fill_drain_profiles": ("Deflection and upper-surface stress at key moments of the "
                            "fill-drain cycle.", "80%"),
    "grounding_line": None,
    "diagram_grid": ("The grid, its ghost nodes and the chain of operators "
                     "$w \\to \\kappa \\to M \\to$ force. Kinematic boundary conditions live in "
                     "$L$ and dynamic ones in $L^*$.", "100%"),
    "diagram_timestep": ("One time step of the algorithm.", "88%"),
    "verification": ("Verification. Left: elastic strip load against Hetényi's exact "
                     "solution. Centre: viscoelastic relaxation against the exact Fourier-mode "
                     "solution (grey). Right: second-order convergence in $\\Delta t$ and "
                     "$\\Delta x$.", "100%"),
}
WIDTH_WITH_ITALIC_CAPTION = {"step_response": "82%", "grounding_line": "82%"}

PREAMBLE = r"""
\usepackage[sc]{mathpazo}
\linespread{1.06}
\usepackage{microtype}
\usepackage{float}
\floatplacement{figure}{H}
\usepackage[font=small,labelfont={bf,color=accent},margin=0.6cm,skip=6pt]{caption}
\usepackage{xcolor}
\definecolor{accent}{HTML}{4A3AA7}
\definecolor{inkmuted}{HTML}{52514E}
\definecolor{boxbg}{HTML}{F4F2FC}
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries}{\color{accent}\thesection}{0.7em}{}
\titleformat{\subsection}{\large\bfseries}{\color{accent}\thesubsection}{0.6em}{}
\titlespacing*{\section}{0pt}{2.2ex plus 1ex}{1.2ex}
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\color{inkmuted}Ice-shelf flexure from meltwater loading}
\fancyhead[R]{\small\color{inkmuted}\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\small\color{inkmuted}\thepage}\renewcommand{\headrulewidth}{0pt}}
\usepackage[most]{tcolorbox}
\newtcolorbox{summarybox}{colback=boxbg,colframe=accent,boxrule=0.6pt,arc=2pt,
  left=8pt,right=8pt,top=6pt,bottom=6pt}
\usepackage{booktabs}
\renewcommand{\arraystretch}{1.25}
\setlength{\emergencystretch}{3em}
\usepackage{tocloft}
\setlength{\cftbeforesecskip}{3pt}
\renewcommand{\cftsecfont}{\normalfont}
\renewcommand{\cftsecpagefont}{\normalfont}
\renewcommand{\cftsecleader}{\cftdotfill{\cftdotsep}}
\usepackage{titling}
\setlength{\droptitle}{-5em}
\pretitle{\begin{center}\LARGE\bfseries}
\posttitle{\end{center}\vspace{-0.6em}}
\predate{\begin{center}\small\color{inkmuted}}
\postdate{\end{center}\vspace{-0.5em}}
\usepackage{etoolbox}
\AtBeginEnvironment{longtable}{\small}
"""

LUA = r"""
function Div(el)
  if el.classes:includes('summary') then
    return { pandoc.RawBlock('latex', '\\begin{summarybox}'), el,
             pandoc.RawBlock('latex', '\\end{summarybox}') }
  end
end
"""


def adapt(md: str) -> str:
    lines = md.splitlines()
    out, skip = [], False
    for ln in lines:
        if ln.startswith("# "):            # title comes from metadata
            continue
        if ln.strip() == "## Contents":     # LaTeX makes its own table of contents
            skip = True
            continue
        if skip:
            if ln.strip() == "---":
                skip = False
            continue
        if ln.strip() == "---":             # horizontal rules -> nothing
            continue
        out.append(ln)
    md = "\n".join(out)

    # section numbers come from LaTeX: "## 7. Two limits" -> "## Two limits"
    md = re.sub(r"^(#{2,3}) \d+\. ", r"\1 ", md, flags=re.M)

    # box the opening summary (from the bold lead-in to the end of that paragraph)
    md = re.sub(r"(\*\*The whole model in two lines\.\*\*.*?local isostasy\*\*, \$w = q/k\$\.)",
                r"::: summary\n\1\n:::\n\n```{=latex}\n\\vspace{0.6em}{\\hypersetup{linkcolor=black}"
                r"\\setcounter{tocdepth}{1}\\tableofcontents}\\clearpage\n```\n",
                md, flags=re.S)

    # links to files in the repository -> plain code text
    md = re.sub(r"\[`([^`]+)`\]\((?!http)([^)]+)\)",
                lambda m: f"`flexure/{m.group(2)}`".replace("flexure/../", ""), md)

    # unicode that the fonts lack
    md = (md.replace("✓", r"$\checkmark$").replace("m⁻³", "m$^{-3}$")
            .replace("m⁻¹", "m$^{-1}$").replace("′", "$'$").replace("½", r"$\tfrac12$"))

    # figures: proper captions, and fold the italic caption paragraph into the figure
    def fig(m):
        path = m.group(2)
        key = os.path.splitext(os.path.basename(path))[0]
        italic = m.group(3)
        if FIGS.get(key) is None and italic:
            cap, width = italic.strip()[1:-1], WIDTH_WITH_ITALIC_CAPTION.get(key, "85%")
            return f"![{cap}]({path}){{width={width}}}\n"
        cap, width = FIGS.get(key) or (m.group(1), "85%")
        return f"![{cap}]({path}){{width={width}}}\n" + (f"\n{italic}" if italic else "")

    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)\n(?:\n(\*[^\n]+(?:\n[^\n]+)*\*)\n)?", fig, md)
    return md


def main():
    md = adapt(open(SRC, encoding="utf8").read())
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "derivation.md")
        open(src, "w", encoding="utf8").write(md)
        hdr = os.path.join(tmp, "preamble.tex")
        open(hdr, "w").write(PREAMBLE)
        lua = os.path.join(tmp, "summary.lua")
        open(lua, "w").write(LUA)
        cmd = ["pandoc", src, "-o", OUT, "--pdf-engine=pdflatex",
               "--resource-path", HERE, "--shift-heading-level-by=-1", "--number-sections",
               "-V", "toc-depth=1", "-H", hdr, "--lua-filter", lua,
               "--highlight-style=tango",
               "-V", "documentclass=article", "-V", "papersize=a4", "-V", "fontsize=11pt",
               "-V", "geometry:margin=2.4cm", "-V", "colorlinks=true",
               "-V", "linkcolor=accent", "-V", "urlcolor=accent", "-V", "toccolor=black",
               "-M", "title=Bending an ice shelf with meltwater",
               "-M", "subtitle=A 2-D viscoelastic beam model: derivation and numerics",
               "-M", "date=" + datetime.date.today().strftime("%-d %B %Y")]
        subprocess.run(cmd, check=True, cwd=HERE)
    print("wrote", os.path.relpath(OUT, HERE))


if __name__ == "__main__":
    main()
