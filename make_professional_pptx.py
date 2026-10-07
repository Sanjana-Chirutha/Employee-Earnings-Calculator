"""
Professional PowerPoint Generator
Project  : Employee Earnings Calculator (ML-Powered Salary Intelligence)
Student  : Sanjana Chirutha  |  STU6817962b86d921746376235
Program  : 6-Week Virtual AI & ML Internship
Org      : Edunet Foundation | AICTE | IBM SkillsBuild

Design   : Corporate Navy + Sky-Blue + Emerald, clean sans-serif,
           section dividers, metric cards, bullet icons, footer branding
"""

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

# ─── Palette ─────────────────────────────────────────────────────────────────
C_NAVY      = RGBColor(15,  40, 90)    # #0F285A  deep navy
C_BLUE      = RGBColor(0,  112, 192)   # #0070C0  corporate blue
C_SKY       = RGBColor(0,  176, 240)   # #00B0F0  sky accent
C_EMERALD   = RGBColor(0,  176, 116)   # #00B074  emerald green
C_AMBER     = RGBColor(255,192,  0)    # #FFC000  amber / gold
C_WHITE     = RGBColor(255,255,255)
C_LIGHT_BG  = RGBColor(245,248,252)   # very light grey-blue
C_CARD_BG   = RGBColor(235,242,252)   # card fill
C_BORDER    = RGBColor(180,210,240)
C_TEXT_DARK = RGBColor(20,  30, 50)
C_TEXT_GREY = RGBColor(90, 110,140)
C_DIVIDER   = RGBColor(200,215,235)

W = Inches(13.333)
H = Inches(7.5)

prs = Presentation()
prs.slide_width  = W
prs.slide_height = H

BLANK = prs.slide_layouts[6]

STUDENT = "Sanjana Chirutha"
STU_ID  = "STU6817962b86d921746376235"
ORG     = "Edunet Foundation  |  AICTE  |  IBM SkillsBuild"
PERIOD  = "18 June 2025 – 30 July 2025"
PROG    = "6-Week Virtual AI & ML Internship"
PROJ    = "Employee Earnings Calculator"

# ─── Primitive helpers ────────────────────────────────────────────────────────

def rect(slide, l, t, w, h, fill=None, line=None, line_w=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
        if line_w: s.line.width = line_w
    else:
        s.line.fill.background()
    return s

def rr(slide, l, t, w, h, fill=None, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    s.adjustments[0] = 0.04
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line:
        s.line.color.rgb = line
    else:
        s.line.fill.background()
    return s

def tb(slide, l, t, w, h, wrap=True):
    bx = slide.shapes.add_textbox(l, t, w, h)
    bx.text_frame.word_wrap = wrap
    return bx

def para(tf, text, size, bold=False, color=C_TEXT_DARK, align=PP_ALIGN.LEFT, space_before=0):
    p = tf.add_paragraph()
    p.text = text
    p.font.name  = "Calibri"
    p.font.size  = Pt(size)
    p.font.bold  = bold
    p.font.color.rgb = color
    p.alignment  = align
    if space_before:
        p.space_before = Pt(space_before)
    return p

def first_para(tf, text, size, bold=False, color=C_TEXT_DARK, align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0]
    p.text = text
    p.font.name  = "Calibri"
    p.font.size  = Pt(size)
    p.font.bold  = bold
    p.font.color.rgb = color
    p.alignment  = align
    return p

# ─── Shared slide frame ───────────────────────────────────────────────────────

def base_slide(title="", subtitle="", slide_n="", total="12"):
    slide = prs.slides.add_slide(BLANK)

    # ── Background ──
    rect(slide, 0, 0, W, H, fill=C_LIGHT_BG)

    # ── Top navy header band ──
    rect(slide, 0, 0, W, Inches(1.15), fill=C_NAVY)

    # ── Thin sky-blue accent line below header ──
    rect(slide, 0, Inches(1.15), W, Inches(0.06), fill=C_SKY)

    # ── Bottom footer band ──
    rect(slide, 0, Inches(7.18), W, Inches(0.32), fill=C_NAVY)

    # ── Slide number pill (top-right) ──
    if slide_n:
        nb = tb(slide, Inches(11.8), Inches(0.2), Inches(1.2), Inches(0.5))
        p = nb.text_frame.paragraphs[0]
        p.text = f"{slide_n} / {total}"
        p.font.name  = "Calibri"
        p.font.size  = Pt(12)
        p.font.color.rgb = C_SKY
        p.font.bold  = True
        p.alignment  = PP_ALIGN.RIGHT

    # ── Header text ──
    if title:
        hb = tb(slide, Inches(0.55), Inches(0.18), Inches(10.8), Inches(0.75))
        first_para(hb.text_frame, title, 22, bold=True, color=C_WHITE)

    if subtitle:
        sb = tb(slide, Inches(0.55), Inches(0.72), Inches(10.8), Inches(0.38))
        first_para(sb.text_frame, subtitle, 12, color=C_SKY)

    # ── Footer text ──
    fb = tb(slide, Inches(0.35), Inches(7.20), Inches(12.6), Inches(0.25))
    p = fb.text_frame.paragraphs[0]
    p.text = f"{STUDENT}  |  {STU_ID}  |  {ORG}  |  {PERIOD}"
    p.font.name  = "Calibri"
    p.font.size  = Pt(8.5)
    p.font.color.rgb = C_SKY
    p.alignment  = PP_ALIGN.CENTER

    return slide


# ─── Card helper ─────────────────────────────────────────────────────────────

def card(slide, l, t, w, h, fill=C_CARD_BG, border=C_BORDER):
    return rr(slide, l, t, w, h, fill=fill, line=border)

# ─── Metric pill helper ───────────────────────────────────────────────────────

def metric_card(slide, l, t, w, h, value, label, val_color=C_BLUE):
    card(slide, l, t, w, h, fill=C_WHITE, border=C_BORDER)
    bx = tb(slide, l + Inches(0.15), t + Inches(0.18), w - Inches(0.3), h - Inches(0.25))
    tf = bx.text_frame
    first_para(tf, value, 28, bold=True, color=val_color, align=PP_ALIGN.CENTER)
    para(tf, label, 11, color=C_TEXT_GREY, align=PP_ALIGN.CENTER)

# ─── Bullet list helper ───────────────────────────────────────────────────────

def bullet_list(slide, l, t, w, h, heading, items, icon="▸", hcolor=C_NAVY, icolor=C_TEXT_DARK):
    card(slide, l, t, w, h)
    bx = tb(slide, l + Inches(0.22), t + Inches(0.18), w - Inches(0.3), h - Inches(0.28))
    tf = bx.text_frame
    first_para(tf, heading, 14, bold=True, color=hcolor)
    for item in items:
        p = tf.add_paragraph()
        p.text = f"{icon}  {item}"
        p.font.name  = "Calibri"
        p.font.size  = Pt(11.5)
        p.font.color.rgb = icolor
        p.space_before = Pt(3)


# ─── Accent left bar on content slides ───────────────────────────────────────

def accent_bar(slide):
    rect(slide, 0, Inches(1.21), Inches(0.07), Inches(5.97), fill=C_SKY)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 1 — Cover / Title
# ═══════════════════════════════════════════════════════════════════════════════
s1 = prs.slides.add_slide(BLANK)

# Full navy BG
rect(s1, 0, 0, W, H, fill=C_NAVY)

# Diagonal sky accent band
from pptx.util import Pt as PT
# Left vivid stripe
rect(s1, 0, 0, Inches(0.5), H, fill=C_SKY)
# Top thin stripe
rect(s1, 0, 0, W, Inches(0.08), fill=C_SKY)
# Bottom stripe
rect(s1, 0, Inches(7.4), W, Inches(0.1), fill=C_AMBER)
rect(s1, 0, Inches(7.2), W, Inches(0.2), fill=C_SKY)

# Org banner
ob = tb(s1, Inches(0.8), Inches(0.55), Inches(11.7), Inches(0.5))
first_para(ob.text_frame, "EDUNET FOUNDATION   |   AICTE   |   IBM SKILLSBUILD", 13,
           bold=True, color=C_SKY, align=PP_ALIGN.CENTER)

# Internship label pill
rect(s1, Inches(4.5), Inches(1.15), Inches(4.333), Inches(0.42), fill=C_SKY)
lt = tb(s1, Inches(4.5), Inches(1.17), Inches(4.333), Inches(0.42))
first_para(lt.text_frame, "ONLINE INTERNSHIP PROJECT  •  AI & MACHINE LEARNING",
           11, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

# Main Title
mt = tb(s1, Inches(0.8), Inches(1.75), Inches(11.7), Inches(1.6))
tf = mt.text_frame
first_para(tf, "Employee Earnings Calculator", 42, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
para(tf, "ML-Powered Salary Intelligence Engine", 20, color=C_SKY, align=PP_ALIGN.CENTER)

# Divider line
rect(s1, Inches(3.0), Inches(3.55), Inches(7.333), Inches(0.04), fill=C_SKY)

# Info cards (2-col)
card(s1, Inches(1.2), Inches(3.72), Inches(5.0), Inches(2.8), fill=RGBColor(20,50,100), border=C_SKY)
lc = tb(s1, Inches(1.45), Inches(3.95), Inches(4.5), Inches(2.45))
tf = lc.text_frame
first_para(tf, "Student Details", 14, bold=True, color=C_SKY)
for k, v in [("Name", STUDENT), ("STU ID", STU_ID), ("Branch", "AI & Machine Learning")]:
    p = tf.add_paragraph()
    p.space_before = Pt(5)
    from pptx.oxml.ns import qn
    run1 = p.add_run(); run1.text = f"{k}: "; run1.font.bold = True
    run1.font.size = Pt(12); run1.font.color.rgb = C_DIVIDER; run1.font.name = "Calibri"
    run2 = p.add_run(); run2.text = v
    run2.font.size = Pt(12); run2.font.color.rgb = C_WHITE; run2.font.name = "Calibri"

card(s1, Inches(7.1), Inches(3.72), Inches(5.0), Inches(2.8), fill=RGBColor(20,50,100), border=C_SKY)
rc = tb(s1, Inches(7.35), Inches(3.95), Inches(4.5), Inches(2.45))
tf = rc.text_frame
first_para(tf, "Internship Details", 14, bold=True, color=C_SKY)
for k, v in [("Organization", "Edunet Foundation"), ("Collaboration", "AICTE | IBM SkillsBuild"),
             ("Duration", PERIOD), ("Program", PROG)]:
    p = tf.add_paragraph()
    p.space_before = Pt(4)
    run1 = p.add_run(); run1.text = f"{k}: "; run1.font.bold = True
    run1.font.size = Pt(11); run1.font.color.rgb = C_DIVIDER; run1.font.name = "Calibri"
    run2 = p.add_run(); run2.text = v
    run2.font.size = Pt(11); run2.font.color.rgb = C_WHITE; run2.font.name = "Calibri"

# Bottom footer
fb = tb(s1, Inches(0.5), Inches(7.24), Inches(12.333), Inches(0.22))
first_para(fb.text_frame, f"{STUDENT}  |  {STU_ID}  |  {ORG}", 8.5, color=C_SKY, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 2 — Organization Profile
# ═══════════════════════════════════════════════════════════════════════════════
s2 = base_slide("Organization & Program Overview",
                "Collaborating bodies, program scope, and internship structure", "02")
accent_bar(s2)

# 3-col org cards
orgs = [
    ("Edunet Foundation", C_BLUE,
     ["Primary implementing organization", "Leads youth digital-skill programs",
      "Connects industry & academia", "NEAT-certified training partner"]),
    ("AICTE", C_NAVY,
     ["All India Council for Technical Education", "Apex statutory body for tech education",
      "Endorses skill-development programs", "Ensures quality & standards"]),
    ("IBM SkillsBuild", C_EMERALD,
     ["Global technology learning platform", "Hands-on AI / ML workflows",
      "Industry-recognized credentials", "Real-world project mentoring"]),
]
for idx, (org_name, col, pts) in enumerate(orgs):
    lpos = Inches(0.4 + idx * 4.25)
    card(s2, lpos, Inches(1.35), Inches(4.0), Inches(5.35), fill=C_WHITE, border=col)
    rect(s2, lpos, Inches(1.35), Inches(4.0), Inches(0.48), fill=col)
    hb = tb(s2, lpos + Inches(0.15), Inches(1.38), Inches(3.7), Inches(0.42))
    first_para(hb.text_frame, org_name, 14, bold=True, color=C_WHITE)
    bb = tb(s2, lpos + Inches(0.2), Inches(1.95), Inches(3.6), Inches(4.5))
    tf = bb.text_frame
    first_para(tf, "", 1)
    for pt in pts:
        p = tf.add_paragraph()
        p.text = f"▸  {pt}"
        p.font.name = "Calibri"; p.font.size = Pt(12); p.font.color.rgb = C_TEXT_DARK
        p.space_before = Pt(5)

# Bottom summary band
rect(s2, Inches(0.4), Inches(6.82), Inches(12.533), Inches(0.3), fill=C_NAVY)
sb = tb(s2, Inches(0.55), Inches(6.84), Inches(12.2), Inches(0.27))
first_para(sb.text_frame,
    "Program: 6-Week Virtual AI & ML Internship   |   Mode: Online   |   "
    "Certificate ID: STU6817962b86d921746376235",
    11, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 3 — Problem Statement
# ═══════════════════════════════════════════════════════════════════════════════
s3 = base_slide("Problem Statement",
                "Why a data-driven salary estimation engine is needed", "03")
accent_bar(s3)

# Project title box
rect(s3, Inches(0.4), Inches(1.35), Inches(12.533), Inches(0.52), fill=C_NAVY)
pt_box = tb(s3, Inches(0.6), Inches(1.38), Inches(12.1), Inches(0.47))
first_para(pt_box.text_frame,
    "Project Title:  Employee Earnings Calculator (ML-Powered Salary Intelligence Engine)",
    13, bold=True, color=C_AMBER)

# Overview text
ov = tb(s3, Inches(0.55), Inches(2.0), Inches(12.2), Inches(0.65))
tf = ov.text_frame
tf.word_wrap = True
first_para(tf,
    "Salary estimation in corporate environments is largely manual, subjective, and inconsistent. "
    "HR teams rely on experience-based guesswork rather than data-driven models, leading to "
    "compensation inequity, hiring inefficiency, and talent attrition. This project addresses that gap.",
    12, color=C_TEXT_DARK)

# 2-col problem cards
problems_L = [
    ("No Data-Driven Benchmark", "Companies lack objective salary benchmarks calibrated to industry, role, experience, and region."),
    ("Black-Box Decisions", "Compensation decisions are non-transparent, causing employee dissatisfaction and trust deficit."),
    ("Manual & Error-Prone", "Spreadsheet-based estimation is slow, inconsistent across departments, and not reproducible."),
]
problems_R = [
    ("Lack of Explainability", "Even when ML models exist, organizations cannot explain WHY a particular salary was predicted."),
    ("No Scenario Analysis", "HR cannot simulate 'what-if' scenarios to understand how education or relocation affects pay."),
    ("Scalability Failure", "Manual estimation fails at scale — evaluating hundreds of candidates simultaneously is infeasible."),
]

for idx, (title, desc) in enumerate(problems_L):
    t = Inches(2.82 + idx * 1.42)
    card(s3, Inches(0.4), t, Inches(5.9), Inches(1.32), fill=C_WHITE, border=C_BLUE)
    rect(s3, Inches(0.4), t, Inches(0.18), Inches(1.32), fill=C_BLUE)
    bx = tb(s3, Inches(0.68), t + Inches(0.12), Inches(5.55), Inches(1.18))
    tf = bx.text_frame; tf.word_wrap = True
    first_para(tf, title, 12, bold=True, color=C_NAVY)
    para(tf, desc, 11, color=C_TEXT_DARK)

for idx, (title, desc) in enumerate(problems_R):
    t = Inches(2.82 + idx * 1.42)
    card(s3, Inches(6.68), t, Inches(6.2), Inches(1.32), fill=C_WHITE, border=C_SKY)
    rect(s3, Inches(6.68), t, Inches(0.18), Inches(1.32), fill=C_SKY)
    bx = tb(s3, Inches(6.96), t + Inches(0.12), Inches(5.85), Inches(1.18))
    tf = bx.text_frame; tf.word_wrap = True
    first_para(tf, title, 12, bold=True, color=C_NAVY)
    para(tf, desc, 11, color=C_TEXT_DARK)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 4 — Objectives
# ═══════════════════════════════════════════════════════════════════════════════
s4 = base_slide("Project Objectives",
                "Key goals and deliverables of the internship project", "04")
accent_bar(s4)

objectives = [
    ("93%+ R² Accuracy",       C_BLUE,    "Build an XGBoost / Gradient Boosting regression pipeline predicting annual salary with >93% accuracy on held-out test set."),
    ("Explainable AI (XAI)",   C_EMERALD, "SHAP-style feature attribution waterfall showing each feature's exact monetary contribution relative to the baseline salary."),
    ("Real-Time Web UI",       C_NAVY,    "Glassmorphism dark-mode predictor with sliders, dropdowns, and instant predictions in <15 ms via client-side ML inference."),
    ("Batch CSV Processing",   C_BLUE,    "Bulk drag-and-drop CSV upload predicting salaries for hundreds of employee records; one-click downloadable results export."),
    ("Career Optimizer",       C_EMERALD, "AI sensitivity engine recommending actionable steps (education, relocation, experience) to maximize earning potential."),
    ("Multi-Currency Support", C_NAVY,    "Live currency conversion displaying predicted salaries in USD ($), EUR (E), GBP (P), and INR (Rs)."),
]

for idx, (label, col, desc) in enumerate(objectives):
    row = idx // 2
    colpos = idx % 2
    l = Inches(0.4 + colpos * 6.47)
    t = Inches(1.38 + row * 1.88)
    card(s4, l, t, Inches(6.2), Inches(1.78), fill=C_WHITE, border=col)
    rect(s4, l, t, Inches(6.2), Inches(0.42), fill=col)
    hb = tb(s4, l + Inches(0.15), t + Inches(0.05), Inches(5.9), Inches(0.37))
    first_para(hb.text_frame, label, 13, bold=True, color=C_WHITE)
    db = tb(s4, l + Inches(0.18), t + Inches(0.5), Inches(5.84), Inches(1.18))
    db.text_frame.word_wrap = True
    first_para(db.text_frame, desc, 11, color=C_TEXT_DARK)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 5 — Technologies Used
# ═══════════════════════════════════════════════════════════════════════════════
s5 = base_slide("Technologies Used",
                "Full stack: Machine Learning, Frontend, Visualization, and Deployment", "05")
accent_bar(s5)

tech_groups = [
    ("Machine Learning", C_BLUE, [
        "Python 3.12",
        "XGBoost Regressor",
        "Scikit-Learn (Pipeline, MinMaxScaler, LabelEncoder)",
        "Pandas  /  NumPy",
        "Joblib (model serialization)",
    ]),
    ("Frontend & UI", C_NAVY, [
        "HTML5  /  CSS3 (Glassmorphism design system)",
        "JavaScript ES6+ (ML inference engine)",
        "Chart.js (Waterfall, Radar, Line charts)",
        "Google Fonts (Outfit + Plus Jakarta Sans)",
        "CSS backdrop-filter & neon glow animations",
    ]),
    ("Data & Deployment", C_EMERALD, [
        "UCI Adult Income benchmark dataset",
        "BLS Occupational Employment Statistics",
        "Git  /  GitHub (version control)",
        "GitHub Pages (cloud static hosting)",
        "Python HTTP Server (local dev)",
    ]),
]

for idx, (grp, col, items) in enumerate(tech_groups):
    l = Inches(0.4 + idx * 4.28)
    card(s5, l, Inches(1.35), Inches(4.05), Inches(5.55), fill=C_WHITE, border=col)
    rect(s5, l, Inches(1.35), Inches(4.05), Inches(0.52), fill=col)
    hb = tb(s5, l + Inches(0.15), Inches(1.38), Inches(3.75), Inches(0.47))
    first_para(hb.text_frame, grp, 15, bold=True, color=C_WHITE)
    bb = tb(s5, l + Inches(0.22), Inches(2.02), Inches(3.62), Inches(4.75))
    tf = bb.text_frame
    first_para(tf, "", 1)
    for item in items:
        p = tf.add_paragraph()
        p.text = f"  {item}"
        p.font.name = "Calibri"; p.font.size = Pt(12.5)
        p.font.color.rgb = C_TEXT_DARK; p.space_before = Pt(6)
        # mini bullet dot
        run = p.runs[0] if p.runs else p.add_run()
        run.text = f"  {item}"


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 6 — ML Methodology / Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
s6 = base_slide("ML Pipeline & Methodology",
                "End-to-end machine learning workflow: data ingestion to inference", "06")
accent_bar(s6)

# Pipeline steps (horizontal flow)
steps = [
    ("01", "Data\nIngestion",    "10K+ profiles\nUCI + BLS\nbenchmarks"),
    ("02", "Feature\nEngineering", "9 features:\nAge, Exp, Edu,\nOcc, Region…"),
    ("03", "Preprocessing",     "LabelEncoder\nMinMaxScaler\n80/20 split"),
    ("04", "Model\nTraining",   "XGBoost\nGradient Boost\n300 estimators"),
    ("05", "Evaluation",        "R2 = 93.8%\nRMSE = $3,523\nTest set"),
    ("06", "Serialization",     "Joblib export\n.joblib artifacts\nInference ready"),
]

step_colors = [C_SKY, C_BLUE, C_NAVY, C_BLUE, C_EMERALD, C_AMBER]
box_w = Inches(1.88)
for idx, (num, title, desc) in enumerate(steps):
    l = Inches(0.4 + idx * 2.12)
    # Arrow connector (except last)
    if idx < 5:
        rect(s6, l + box_w, Inches(2.28), Inches(0.24), Inches(0.14), fill=C_DIVIDER)

    col = step_colors[idx]
    card(s6, l, Inches(1.42), box_w, Inches(1.78), fill=C_WHITE, border=col)
    rect(s6, l, Inches(1.42), box_w, Inches(0.52), fill=col)
    nb = tb(s6, l + Inches(0.1), Inches(1.44), box_w - Inches(0.2), Inches(0.48))
    first_para(nb.text_frame, num, 18, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    th = tb(s6, l + Inches(0.08), Inches(2.0), box_w - Inches(0.16), Inches(0.55))
    first_para(th.text_frame, title, 12, bold=True, color=col, align=PP_ALIGN.CENTER)
    dh = tb(s6, l + Inches(0.1), Inches(2.58), box_w - Inches(0.2), Inches(0.6))
    dh.text_frame.word_wrap = True
    first_para(dh.text_frame, desc, 10, color=C_TEXT_GREY, align=PP_ALIGN.CENTER)

# SHAP explanation card
card(s6, Inches(0.4), Inches(3.42), Inches(12.533), Inches(3.22), fill=C_WHITE, border=C_BLUE)
rect(s6, Inches(0.4), Inches(3.42), Inches(12.533), Inches(0.48), fill=C_NAVY)
sh_head = tb(s6, Inches(0.6), Inches(3.44), Inches(12.0), Inches(0.44))
first_para(sh_head.text_frame, "SHAP Feature Attribution Logic", 14, bold=True, color=C_SKY)

shap_items = [
    ("Baseline Salary", "$24,000", "Market entry reference salary before any multipliers"),
    ("Education Multiplier", "+$0 to +$40K", "HS:$0  Bachelor's:+$14K  Master's:+$26K  Doctorate:+$40K"),
    ("Occupation Tier", "+$15K to +$28K", "SWE/AI:+$22-25K  Management:+$28K  Finance:+$24K"),
    ("Experience Curve", "Log growth", "Non-linear logarithmic scaling, plateaus after ~15 years"),
    ("Regional Market", "-12% to +18%", "Tier 1 Hub:+18%  Tier 2:+2%  Tier 3:-12%"),
]

col_positions = [0.55, 3.15, 5.55]
for i, (label, value, note) in enumerate(shap_items):
    col_idx = i % 3
    row_idx = i // 3
    lp = Inches(col_positions[col_idx] if col_idx < 3 else 0.55)
    tp = Inches(4.02 + row_idx * 1.1)
    ww = Inches(2.4) if col_idx < 2 else Inches(2.5)
    card(s6, lp + Inches((col_idx) * 4.03 - col_positions[0] + 0.55), tp, Inches(2.35), Inches(0.98),
         fill=C_LIGHT_BG, border=C_SKY)

# Simpler SHAP list
shap_bx = tb(s6, Inches(0.62), Inches(3.98), Inches(12.0), Inches(2.5))
shap_bx.text_frame.word_wrap = True
tf = shap_bx.text_frame
first_para(tf, "", 1)
shap_rows = [
    ("Baseline Salary", "$24,000", "Market entry reference salary before any multipliers"),
    ("Education Multiplier", "+$0 → +$40K", "High School: $0  |  Bachelor's: +$14K  |  Master's: +$26K  |  Doctorate: +$40K"),
    ("Occupation Tier", "+$15K → +$28K", "Software/AI: +$22-25K  |  Management: +$28K  |  Finance: +$24K  |  Education: +$10K"),
    ("Experience Curve", "Logarithmic", "Non-linear scaling based on log(exp+1), plateaus beyond 15 years of experience"),
    ("Regional Tier", "-12% to +18%", "Tier 1 Tech Hub: +18%  |  Tier 2 City: +2%  |  Tier 3 / Rural: -12%"),
]
for label, val, note in shap_rows:
    p = tf.add_paragraph()
    from pptx.oxml.ns import qn as qname
    r1 = p.add_run(); r1.text = f"  {label}: "; r1.font.bold = True
    r1.font.size = Pt(12); r1.font.color.rgb = C_NAVY; r1.font.name = "Calibri"
    r2 = p.add_run(); r2.text = f"{val}  —  "
    r2.font.size = Pt(12); r2.font.bold = True; r2.font.color.rgb = C_BLUE; r2.font.name = "Calibri"
    r3 = p.add_run(); r3.text = note
    r3.font.size = Pt(11); r3.font.color.rgb = C_TEXT_GREY; r3.font.name = "Calibri"
    p.space_before = Pt(4)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 7 — Model Performance & Results
# ═══════════════════════════════════════════════════════════════════════════════
s7 = base_slide("Model Performance & Results",
                "Quantitative evaluation metrics and live system prediction output", "07")
accent_bar(s7)

# 4 metric cards (top row)
metrics = [
    ("93.8%",     "Model Accuracy (R² Score)", C_BLUE),
    ("$3,523",    "Root Mean Squared Error",   C_EMERALD),
    ("< 15 ms",   "Inference Latency",         C_NAVY),
    ("10,000+",   "Training Profiles Used",    C_AMBER),
]
for idx, (val, lbl, col) in enumerate(metrics):
    l = Inches(0.4 + idx * 3.22)
    card(s7, l, Inches(1.38), Inches(3.0), Inches(1.65), fill=C_WHITE, border=col)
    rect(s7, l, Inches(1.38), Inches(3.0), Inches(0.12), fill=col)
    vb = tb(s7, l + Inches(0.1), Inches(1.52), Inches(2.8), Inches(0.82))
    first_para(vb.text_frame, val, 30, bold=True, color=col, align=PP_ALIGN.CENTER)
    lb = tb(s7, l + Inches(0.08), Inches(2.36), Inches(2.84), Inches(0.55))
    first_para(lb.text_frame, lbl, 11, color=C_TEXT_GREY, align=PP_ALIGN.CENTER)

# Live prediction hero card
rect(s7, Inches(0.4), Inches(3.22), Inches(12.533), Inches(0.06), fill=C_DIVIDER)
card(s7, Inches(0.4), Inches(3.3), Inches(12.533), Inches(1.98), fill=RGBColor(225,238,255), border=C_BLUE)
rect(s7, Inches(0.4), Inches(3.3), Inches(12.533), Inches(0.55), fill=C_BLUE)
pred_title = tb(s7, Inches(0.6), Inches(3.32), Inches(12.0), Inches(0.50))
first_para(pred_title.text_frame, "LIVE SYSTEM TEST — Sample Prediction Output", 13, bold=True, color=C_WHITE)
pred_val = tb(s7, Inches(0.6), Inches(3.92), Inches(12.0), Inches(0.58))
first_para(pred_val.text_frame, "Predicted Annual Salary:  $96,420  /  yr", 24, bold=True, color=C_NAVY, align=PP_ALIGN.CENTER)
pred_sub = tb(s7, Inches(0.6), Inches(4.52), Inches(12.0), Inches(0.65))
first_para(pred_sub.text_frame,
    "Monthly: $8,035  |  Hourly: $46.36  |  Earning Percentile: Top 28%  |  Confidence Band: $78K – $118K",
    13, color=C_TEXT_GREY, align=PP_ALIGN.CENTER)

# Test profile + drivers (2 col)
card(s7, Inches(0.4), Inches(5.42), Inches(5.85), Inches(1.55), fill=C_WHITE, border=C_NAVY)
tb_l = tb(s7, Inches(0.6), Inches(5.52), Inches(5.45), Inches(1.38))
tf = tb_l.text_frame; tf.word_wrap = True
first_para(tf, "Test Profile Parameters", 12, bold=True, color=C_NAVY)
for line in ["Age: 34 yrs  |  Experience: 8 yrs",
             "Education: Bachelor's  |  Role: Software Engineering / AI",
             "Mode: Hybrid  |  Size: Mid-Size  |  Region: Tier 1 Tech Hub"]:
    para(tf, line, 11, color=C_TEXT_DARK)

card(s7, Inches(6.52), Inches(5.42), Inches(6.42), Inches(1.55), fill=C_WHITE, border=C_EMERALD)
tb_r = tb(s7, Inches(6.72), Inches(5.52), Inches(6.05), Inches(1.38))
tf = tb_r.text_frame; tf.word_wrap = True
first_para(tf, "Top Feature Drivers (SHAP Attribution)", 12, bold=True, color=C_EMERALD)
for line in ["#1 Occupation Tier:  +$26,400  (largest positive impact)",
             "#2 Education Level:  +$14,000  |  #3 Regional Tier:  +$11,340",
             "#4 Experience Curve:  +$8,920  |  #5 Work Mode:  +$3,100"]:
    para(tf, line, 11, color=C_TEXT_DARK)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 8 — UI Design & Key Features
# ═══════════════════════════════════════════════════════════════════════════════
s8 = base_slide("UI Design & Key Application Features",
                "Glassmorphism web interface, real-time controls, and interactive visualizations", "08")
accent_bar(s8)

features = [
    ("Interactive Predictor",  C_BLUE,    "Dynamic sliders and dropdowns with <15 ms real-time salary updates powered by a client-side XGBoost mirror engine."),
    ("SHAP Waterfall Chart",   C_NAVY,    "Bar chart showing exact monetary additions (+) and deductions (-) for each feature relative to $24,000 baseline."),
    ("Experience Trajectory",  C_EMERALD, "Line chart plotting salary growth curve from 0–30 years using logarithmic scaling, with current position highlighted."),
    ("Industry Radar Benchmark",C_BLUE,   "Hexagonal radar chart benchmarking Education, Experience, Region, Occupation, Hours, and Company Size scores."),
    ("Batch CSV Predictor",    C_EMERALD, "Drag-and-drop CSV upload; processes hundreds of records in <50 ms with real-time summary stats and export button."),
    ("Career Optimizer",       C_NAVY,    "AI sensitivity engine that runs micro-simulations across parameter changes to surface top-3 salary boost strategies."),
]
for idx, (name, col, desc) in enumerate(features):
    row = idx // 2
    colpos = idx % 2
    l = Inches(0.4 + colpos * 6.47)
    t = Inches(1.38 + row * 1.88)
    card(s8, l, t, Inches(6.2), Inches(1.78), fill=C_WHITE, border=col)
    rect(s8, l, t, Inches(0.22), Inches(1.78), fill=col)
    hb = tb(s8, l + Inches(0.35), t + Inches(0.15), Inches(5.7), Inches(0.45))
    first_para(hb.text_frame, name, 13, bold=True, color=col)
    db = tb(s8, l + Inches(0.35), t + Inches(0.6), Inches(5.7), Inches(1.1))
    db.text_frame.word_wrap = True
    first_para(db.text_frame, desc, 11.5, color=C_TEXT_DARK)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 9 — Challenges & Solutions
# ═══════════════════════════════════════════════════════════════════════════════
s9 = base_slide("Challenges Faced & Solutions Implemented",
                "Engineering problems encountered and how they were resolved", "09")
accent_bar(s9)

challenges = [
    ("Non-Linear Salary Decay",
     "Real-world experience gains plateau after ~15 years; a linear model overfitted senior salaries.",
     "Applied logarithmic scaling function log(exp+1) to model career plateauing accurately."),
    ("Async Rendering Sync",
     "Chart.js initialized before the ML prediction completed, causing blank charts on load.",
     "Refactored JS execution order; predictions run first, then charts initialize with fallback HTML numbers."),
    ("Windows UTF-8 Encoding",
     "Python's cp1252 terminal encoding raised UnicodeEncodeError for emoji-containing print statements.",
     "Added explicit sys.stdout UTF-8 stream override and replaced emojis with ASCII placeholders."),
    ("Client-Side ML Mirror",
     "Replicating trained XGBoost logic in pure JS required careful coefficient matching.",
     "Extracted feature weights from model inspection; validated JS output against Python predictions."),
    ("CSV Batch Performance",
     "Parsing large CSV files in-browser caused main-thread UI freezes.",
     "Implemented chunked record processing with setTimeout yielding to keep the UI responsive."),
]

for idx, (title, prob, sol) in enumerate(challenges):
    t = Inches(1.38 + idx * 1.14)
    card(s9, Inches(0.4), t, Inches(12.533), Inches(1.06), fill=C_WHITE, border=C_DIVIDER)
    rect(s9, Inches(0.4), t, Inches(0.22), Inches(1.06), fill=C_AMBER)
    th = tb(s9, Inches(0.72), t + Inches(0.08), Inches(12.0), Inches(0.36))
    first_para(th.text_frame, title, 13, bold=True, color=C_NAVY)
    det = tb(s9, Inches(0.72), t + Inches(0.44), Inches(12.0), Inches(0.56))
    det.text_frame.word_wrap = True
    tf = det.text_frame
    p = tf.paragraphs[0]
    r1 = p.add_run(); r1.text = "Problem: "; r1.font.bold = True
    r1.font.size = Pt(11); r1.font.color.rgb = C_BLUE; r1.font.name = "Calibri"
    r2 = p.add_run(); r2.text = prob + "   "
    r2.font.size = Pt(11); r2.font.color.rgb = C_TEXT_DARK; r2.font.name = "Calibri"
    r3 = p.add_run(); r3.text = "Solution: "; r3.font.bold = True
    r3.font.size = Pt(11); r3.font.color.rgb = C_EMERALD; r3.font.name = "Calibri"
    r4 = p.add_run(); r4.text = sol
    r4.font.size = Pt(11); r4.font.color.rgb = C_TEXT_DARK; r4.font.name = "Calibri"


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 10 — Learning Outcomes
# ═══════════════════════════════════════════════════════════════════════════════
s10 = base_slide("Internship Learning Outcomes & Skills Gained",
                 "Technical competencies and professional growth from this internship", "10")
accent_bar(s10)

outcomes = [
    ("ML Regression Mastery",    C_BLUE,    "Hands-on XGBoost + Scikit-Learn pipeline construction, hyperparameter tuning, cross-validation, and Joblib model serialization."),
    ("Explainable AI (XAI)",     C_EMERALD, "Designed and implemented SHAP-style feature attribution making ML predictions interpretable for non-technical HR stakeholders."),
    ("Full-Stack Web Dev",       C_NAVY,    "Built a production-quality static web app with HTML5, Glassmorphism CSS3, ES6+ JavaScript, and Chart.js data visualizations."),
    ("Data Engineering",         C_BLUE,    "Cleaned, encoded, and scaled benchmark salary datasets; designed a reproducible preprocessing pipeline using Scikit-Learn transformers."),
    ("GitHub DevOps Workflow",   C_EMERALD, "Practiced Git version control, branch management, pull requests, and cloud static hosting deployment on GitHub Pages."),
    ("Problem-Solving Mindset",  C_NAVY,    "Diagnosed and resolved encoding bugs, async race conditions, and cross-platform issues through systematic debugging and iteration."),
]

for idx, (name, col, desc) in enumerate(outcomes):
    row = idx // 2
    colpos = idx % 2
    l = Inches(0.4 + colpos * 6.47)
    t = Inches(1.38 + row * 1.92)
    card(s10, l, t, Inches(6.2), Inches(1.82), fill=C_WHITE, border=col)
    rect(s10, l, t, Inches(6.2), Inches(0.44), fill=col)
    nb = tb(s10, l + Inches(0.18), t + Inches(0.06), Inches(5.84), Inches(0.38))
    first_para(nb.text_frame, name, 13, bold=True, color=C_WHITE)
    db = tb(s10, l + Inches(0.18), t + Inches(0.54), Inches(5.84), Inches(1.18))
    db.text_frame.word_wrap = True
    first_para(db.text_frame, desc, 11.5, color=C_TEXT_DARK)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 11 — Future Enhancements
# ═══════════════════════════════════════════════════════════════════════════════
s11 = base_slide("Future Enhancements & Roadmap",
                 "Planned improvements to evolve this into an enterprise-grade solution", "11")
accent_bar(s11)

enhancements = [
    ("Real Dataset Integration",  "Replace synthetic training data with anonymized Glassdoor and LinkedIn salary datasets for higher real-world prediction accuracy."),
    ("FastAPI REST Backend",       "Deploy the trained XGBoost model as a FastAPI microservice enabling seamless integration with enterprise HRMS and ATS platforms."),
    ("True SHAP Library",         "Replace the client-side SHAP approximation with the official shap Python library for mathematically exact Shapley value computation."),
    ("NLP Resume Parser",         "Add spaCy / HuggingFace-based resume parsing to auto-populate all predictor input fields from uploaded candidate PDF resumes."),
    ("React Native Mobile App",   "Develop a React Native mobile app with on-device ML inference for real-time salary benchmarking without internet connectivity."),
    ("HR Analytics Dashboard",    "Build a multi-tenant company dashboard for HR teams to compare internal salary distributions against real-time market predictions."),
    ("LLM Career Coach",          "Integrate Gemini / GPT-4o to generate personalized, step-by-step career growth roadmaps based on individual predicted salary gaps."),
]

for idx, (title, desc) in enumerate(enhancements):
    row = idx // 2
    colpos = idx % 2
    if idx == 6:  # last one spans full width
        l, w = Inches(0.4), Inches(12.533)
        t = Inches(1.38 + 3 * 1.72)
    else:
        l = Inches(0.4 + colpos * 6.47)
        w = Inches(6.2)
        t = Inches(1.38 + row * 1.72)

    card(s11, l, t, w, Inches(1.62), fill=C_WHITE, border=C_BORDER)
    rect(s11, l, t, Inches(0.2), Inches(1.62),
         fill=C_BLUE if idx % 2 == 0 else C_EMERALD)
    th = tb(s11, l + Inches(0.32), t + Inches(0.1), w - Inches(0.45), Inches(0.42))
    first_para(th.text_frame, title, 12, bold=True, color=C_NAVY)
    dh = tb(s11, l + Inches(0.32), t + Inches(0.52), w - Inches(0.45), Inches(1.0))
    dh.text_frame.word_wrap = True
    first_para(dh.text_frame, desc, 11, color=C_TEXT_DARK)


# ═══════════════════════════════════════════════════════════════════════════════
# SLIDE 12 — Thank You / Closing
# ═══════════════════════════════════════════════════════════════════════════════
s12 = prs.slides.add_slide(BLANK)

# Full navy BG
rect(s12, 0, 0, W, H, fill=C_NAVY)
rect(s12, 0, 0, Inches(0.5), H, fill=C_SKY)
rect(s12, 0, Inches(7.2), W, Inches(0.3), fill=C_SKY)
rect(s12, 0, Inches(7.45), W, Inches(0.05), fill=C_AMBER)

# Org banner
ob = tb(s12, Inches(0.8), Inches(0.48), Inches(11.7), Inches(0.48))
first_para(ob.text_frame, "EDUNET FOUNDATION   |   AICTE   |   IBM SKILLSBUILD", 13,
           bold=True, color=C_SKY, align=PP_ALIGN.CENTER)

# Thank you
tyt = tb(s12, Inches(0.8), Inches(1.25), Inches(11.7), Inches(1.5))
tf = tyt.text_frame
first_para(tf, "Thank You!", 52, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
para(tf, "Questions & Feedback Are Most Welcome", 18, color=C_SKY, align=PP_ALIGN.CENTER)

# Thin amber divider
rect(s12, Inches(3.0), Inches(3.0), Inches(7.333), Inches(0.05), fill=C_AMBER)

# Project summary card
card(s12, Inches(1.5), Inches(3.15), Inches(10.333), Inches(1.35),
     fill=RGBColor(20,50,100), border=C_SKY)
pc = tb(s12, Inches(1.7), Inches(3.28), Inches(9.933), Inches(0.45))
first_para(pc.text_frame, f"Project:  {PROJ}  —  ML-Powered Salary Intelligence Engine",
           14, bold=True, color=C_AMBER, align=PP_ALIGN.CENTER)
ps = tb(s12, Inches(1.7), Inches(3.76), Inches(9.933), Inches(0.65))
first_para(ps.text_frame,
    f"XGBoost | 93.8% R² Accuracy | SHAP Explainability | Glassmorphism UI | Batch CSV | Multi-Currency",
    12, color=C_DIVIDER, align=PP_ALIGN.CENTER)

# Student card
card(s12, Inches(1.5), Inches(4.68), Inches(4.8), Inches(2.1),
     fill=RGBColor(20,50,100), border=C_SKY)
sc = tb(s12, Inches(1.7), Inches(4.82), Inches(4.4), Inches(1.8))
tf = sc.text_frame
first_para(tf, "Student Details", 13, bold=True, color=C_SKY)
for k, v in [("Name", STUDENT), ("STU ID", STU_ID), ("Branch", "AI & Machine Learning")]:
    p = tf.add_paragraph(); p.space_before = Pt(5)
    r1 = p.add_run(); r1.text = f"{k}: "; r1.font.bold = True
    r1.font.size = Pt(11.5); r1.font.color.rgb = C_DIVIDER; r1.font.name = "Calibri"
    r2 = p.add_run(); r2.text = v
    r2.font.size = Pt(11.5); r2.font.color.rgb = C_WHITE; r2.font.name = "Calibri"

# Org card
card(s12, Inches(7.0), Inches(4.68), Inches(4.8), Inches(2.1),
     fill=RGBColor(20,50,100), border=C_SKY)
oc = tb(s12, Inches(7.2), Inches(4.82), Inches(4.4), Inches(1.8))
tf = oc.text_frame
first_para(tf, "Program & Links", 13, bold=True, color=C_SKY)
for k, v in [("Program", PROG), ("Period", PERIOD),
             ("GitHub", "github.com/Sanjana-Chirutha"),
             ("Live App", "sanjana-chirutha.github.io")]:
    p = tf.add_paragraph(); p.space_before = Pt(5)
    r1 = p.add_run(); r1.text = f"{k}: "; r1.font.bold = True
    r1.font.size = Pt(11.5); r1.font.color.rgb = C_DIVIDER; r1.font.name = "Calibri"
    r2 = p.add_run(); r2.text = v
    r2.font.size = Pt(11.5); r2.font.color.rgb = C_WHITE; r2.font.name = "Calibri"

# Footer
fb = tb(s12, Inches(0.5), Inches(7.22), Inches(12.333), Inches(0.22))
first_para(fb.text_frame,
    f"{STUDENT}  |  {STU_ID}  |  {ORG}",
    8.5, color=C_SKY, align=PP_ALIGN.CENTER)


# ─── Save ─────────────────────────────────────────────────────────────────────
OUTPUT = "Sanjana_Employee_Earnings_Professional.pptx"
prs.save(OUTPUT)
print(f"[SUCCESS] Professional presentation saved: {OUTPUT}")
print(f"  Slides   : {len(prs.slides)}")
print(f"  Student  : {STUDENT}")
print(f"  Project  : {PROJ}")
