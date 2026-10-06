"""
Script to generate PowerPoint Presentation (.pptx) matching exact theme of AICTE & Edunet Foundation Certificate
Organisations: Edunet Foundation | AICTE | IBM SkillsBuild
Student: Sanjana Chirutha (STU ID: STU6817962b86d921746376235)
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # Color Palette matching AICTE/Edunet Certificate PDF
    COLOR_BG = RGBColor(248, 250, 252)       # Light slate #f8fafc
    COLOR_CARD = RGBColor(241, 245, 249)     # Soft blue-gray #f1f5f9
    COLOR_PRIMARY = RGBColor(30, 58, 138)    # Deep Navy Blue #1e3a8a
    COLOR_SECONDARY = RGBColor(2, 132, 199)  # Sky Blue #0284c7
    COLOR_ACCENT = RGBColor(22, 163, 74)     # Emerald Green #16a34a
    COLOR_TEXT = RGBColor(15, 23, 42)        # Dark Slate #0f172a
    COLOR_MUTED = RGBColor(71, 85, 105)      # Slate Muted #475569

    blank_layout = prs.slide_layouts[6]

    def add_base_slide(title_text="", slide_num_str=""):
        slide = prs.slides.add_slide(blank_layout)
        
        # Background
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = COLOR_BG
        bg.line.fill.background()

        # Left Margin Accent Bar (Matching Certificate Left Decorative Bar)
        margin_bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.4), prs.slide_height)
        margin_bar.fill.solid()
        margin_bar.fill.fore_color.rgb = COLOR_SECONDARY
        margin_bar.line.fill.background()

        # Bottom Bar
        bbar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.4), Inches(7.2), Inches(12.933), Inches(0.3))
        bbar.fill.solid()
        bbar.fill.fore_color.rgb = COLOR_PRIMARY
        bbar.line.fill.background()

        if title_text:
            header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(10.5), Inches(0.8))
            tf = header_box.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = title_text
            p.font.name = "Helvetica"
            p.font.size = Pt(26)
            p.font.bold = True
            p.font.color.rgb = COLOR_PRIMARY

            if slide_num_str:
                num_box = slide.shapes.add_textbox(Inches(10.5), Inches(0.4), Inches(2.0), Inches(0.5))
                np = num_box.text_frame.paragraphs[0]
                np.text = slide_num_str
                np.alignment = PP_ALIGN.RIGHT
                np.font.size = Pt(13)
                np.font.bold = True
                np.font.color.rgb = COLOR_SECONDARY

        # Footer Text
        footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.22), Inches(11.7), Inches(0.25))
        fp = footer_box.text_frame.paragraphs[0]
        fp.text = "Sanjana Chirutha (STU ID: STU6817962b86d921746376235)  |  Edunet Foundation  |  AICTE  |  IBM SkillsBuild"
        fp.font.size = Pt(10)
        fp.font.color.rgb = RGBColor(255, 255, 255)

        return slide

    def add_card(slide, left, top, width, height, bg_color=COLOR_CARD):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = bg_color
        card.line.color.rgb = RGBColor(203, 213, 225)
        return card

    # =========================================================================
    # Slide 1: Title & Certificate Cover Slide
    # =========================================================================
    s1 = add_base_slide()

    # Header Org Banner
    ob = s1.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.8))
    ot = ob.text_frame
    op = ot.paragraphs[0]
    op.text = "edunet foundation   |   AICTE   |   IBM SkillsBuild"
    op.font.size = Pt(14)
    op.font.bold = True
    op.font.color.rgb = COLOR_SECONDARY
    op.alignment = PP_ALIGN.CENTER

    tbox = s1.shapes.add_textbox(Inches(0.8), Inches(1.2), Inches(11.7), Inches(2.0))
    tf1 = tbox.text_frame
    p1 = tf1.paragraphs[0]
    p1.text = "Certificate of Completion & Internship Project"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_PRIMARY
    p1.alignment = PP_ALIGN.CENTER

    p2 = tf1.add_paragraph()
    p2.text = "Employee Earnings Calculator (ML-Powered Salary Intelligence)"
    p2.font.size = Pt(20)
    p2.font.color.rgb = COLOR_MUTED
    p2.alignment = PP_ALIGN.CENTER

    # Cards
    c1 = add_card(s1, Inches(1.2), Inches(3.4), Inches(5.2), Inches(3.2))
    tb_c1 = s1.shapes.add_textbox(Inches(1.4), Inches(3.6), Inches(4.8), Inches(2.8))
    t1 = tb_c1.text_frame
    t1.word_wrap = True

    def add_kv(tf, k, v):
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        run1 = p.add_run()
        run1.text = f"{k}: "
        run1.font.bold = True
        run1.font.size = Pt(13)
        run1.font.color.rgb = COLOR_TEXT
        
        run2 = p.add_run()
        run2.text = v
        run2.font.size = Pt(13)
        run2.font.color.rgb = COLOR_PRIMARY

    add_kv(t1, "Student Name", "Sanjana Chirutha")
    add_kv(t1, "Student STU ID", "STU6817962b86d921746376235")
    add_kv(t1, "Program", "6 Weeks Virtual Internship on AI & ML")
    add_kv(t1, "Internship Period", "18/06/2025 – 30/07/2025")

    c2 = add_card(s1, Inches(6.8), Inches(3.4), Inches(5.2), Inches(3.2))
    tb_c2 = s1.shapes.add_textbox(Inches(7.0), Inches(3.6), Inches(4.8), Inches(2.8))
    t2 = tb_c2.text_frame
    t2.word_wrap = True
    add_kv(t2, "Implementing Org", "Edunet Foundation")
    add_kv(t2, "Collaborator", "AICTE")
    add_kv(t2, "Technology Partner", "IBM SkillsBuild")
    add_kv(t2, "Signatory Chairman", "Nagesh Singh (Edunet Foundation)")

    # =========================================================================
    # Slide 2: Organization Overview
    # =========================================================================
    s2 = add_base_slide("Organization & Program Overview", "Slide 02/13")
    
    c_s2_1 = add_card(s2, Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.2))
    tb_s2_1 = s2.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(5.2), Inches(4.8))
    t_s2_1 = tb_s2_1.text_frame
    t_s2_1.word_wrap = True
    
    p = t_s2_1.paragraphs[0]
    p.text = "Collaborating Bodies"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullet_points_1 = [
        "Edunet Foundation: Primary implementing organization leading youth digital skill development.",
        "AICTE (All India Council for Technical Education): Statutory body enabling national technical education standards.",
        "IBM SkillsBuild: Industry technology partner providing practical AI & ML learning workflows."
    ]
    for bp in bullet_points_1:
        p = t_s2_1.add_paragraph()
        p.text = f"• {bp}"
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT

    c_s2_2 = add_card(s2, Inches(6.8), Inches(1.5), Inches(5.7), Inches(5.2))
    tb_s2_2 = s2.shapes.add_textbox(Inches(7.0), Inches(1.7), Inches(5.3), Inches(4.8))
    t_s2_2 = tb_s2_2.text_frame
    t_s2_2.word_wrap = True

    p = t_s2_2.paragraphs[0]
    p.text = "Project Objectives"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullet_points_2 = [
        "Problem: Corporate salary estimation is manual, unstandardized, and black-box.",
        "Goal: Build a multi-variate ML regression pipeline predicting annual employee salaries with >95% accuracy.",
        "Deliverables: Full ML model serialized with Joblib, interactive Glassmorphism UI, SHAP explainability waterfall, multi-currency support ($ USD, € EUR, £ GBP, ₹ INR), and batch CSV predictions."
    ]
    for bp in bullet_points_2:
        p = t_s2_2.add_paragraph()
        p.text = f"• {bp}"
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_TEXT

    # =========================================================================
    # Slide 3: Key Features & Core Modules
    # =========================================================================
    s3 = add_base_slide("Key Features & Core Modules", "Slide 03/13")
    
    features = [
        ("Interactive Predictor", "Dynamic slider and dropdown controls with instant prediction updates (<15 ms latency)."),
        ("SHAP Attribution", "Visual breakdown showing exact monetary additions (+) or deductions (-) relative to baseline."),
        ("Career Optimizer", "AI sensitivity recommendations engine providing actionable advice to boost earning potential."),
        ("Batch CSV Processing", "Bulk upload employee records via drag-and-drop CSV parser with downloadable results."),
        ("Multi-Currency", "Live currency conversion supporting USD ($), EUR (€), GBP (£), and INR (₹).")
    ]
    
    top_pos = 1.5
    for title, desc in features:
        card = add_card(s3, Inches(0.8), Inches(top_pos), Inches(11.7), Inches(0.95))
        tb = s3.shapes.add_textbox(Inches(1.0), Inches(top_pos + 0.1), Inches(11.3), Inches(0.75))
        tf = tb.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        
        r1 = p.add_run()
        r1.text = f"{title}: "
        r1.font.bold = True
        r1.font.size = Pt(15)
        r1.font.color.rgb = COLOR_PRIMARY
        
        r2 = p.add_run()
        r2.text = desc
        r2.font.size = Pt(13)
        r2.font.color.rgb = COLOR_TEXT
        
        top_pos += 1.08

    # =========================================================================
    # Slide 4: ML Pipeline & Preprocessing
    # =========================================================================
    s4 = add_base_slide("Machine Learning Pipeline & Preprocessing", "Slide 04/13")
    
    steps = [
        ("01. Ingestion", "10,000+ Cleaned Profiles from UCI & BLS Benchmarks"),
        ("02. Encoding", "LabelEncoder for Categoricals (Education, Occupation, Region)"),
        ("03. Scaling", "MinMaxScaler for Numericals (Age, Experience, Hours)"),
        ("04. Training", "Gradient Boosting & XGBoost Regressor"),
        ("05. Export", "Joblib Serialization (.joblib Artifacts)")
    ]

    left_pos = 0.8
    for stitle, sdesc in steps:
        card = add_card(s4, Inches(left_pos), Inches(1.6), Inches(2.1), Inches(5.0))
        tb = s4.shapes.add_textbox(Inches(left_pos + 0.1), Inches(1.8), Inches(1.9), Inches(4.6))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = stitle
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        
        p2 = tf.add_paragraph()
        p2.text = sdesc
        p2.font.size = Pt(12)
        p2.font.color.rgb = COLOR_MUTED
        
        left_pos += 2.4

    # =========================================================================
    # Slide 5: Model Performance & Results
    # =========================================================================
    s5 = add_base_slide("Model Performance & Quantitative Results", "Slide 05/13")

    metrics = [
        ("98.00%", "Model Accuracy (R² Score)"),
        ("$3,523.39", "Root Mean Squared Error (RMSE)"),
        ("< 15 ms", "Inference Latency"),
        ("10,000", "Cleaned Training Profiles")
    ]

    positions = [(0.8, 1.6), (6.8, 1.6), (0.8, 4.2), (6.8, 4.2)]
    for idx, (mval, mlbl) in enumerate(metrics):
        l, t = positions[idx]
        card = add_card(s5, Inches(l), Inches(t), Inches(5.7), Inches(2.2))
        tb = s5.shapes.add_textbox(Inches(l + 0.2), Inches(t + 0.3), Inches(5.3), Inches(1.6))
        tf = tb.text_frame
        
        p1 = tf.paragraphs[0]
        p1.text = mval
        p1.font.size = Pt(36)
        p1.font.bold = True
        p1.font.color.rgb = COLOR_ACCENT
        p1.alignment = PP_ALIGN.CENTER

        p2 = tf.add_paragraph()
        p2.text = mlbl
        p2.font.size = Pt(15)
        p2.font.color.rgb = COLOR_TEXT
        p2.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # Slide 6: Live System Test Results (Screenshot Data)
    # =========================================================================
    s6 = add_base_slide("Live System Test Results & Output Showcase", "Slide 06/13")

    # Main Hero Box
    hcard = add_card(s6, Inches(0.8), Inches(1.5), Inches(11.7), Inches(1.8), bg_color=RGBColor(224, 242, 254))
    htb = s6.shapes.add_textbox(Inches(1.0), Inches(1.7), Inches(11.3), Inches(1.4))
    htf = htb.text_frame
    htf.word_wrap = True

    p = htf.paragraphs[0]
    p.text = "Candidate Annual Salary Prediction:  ₹ 13,555,746 / yr"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY
    p.alignment = PP_ALIGN.CENTER

    p_sub = htf.add_paragraph()
    p_sub.text = "Monthly Take-Home: ₹ 1,129,646 / mo   |   Hourly Rate: ₹ 6,517.19 / hr   |   Earning Percentile: Top 21%"
    p_sub.font.size = Pt(15)
    p_sub.font.color.rgb = COLOR_TEXT
    p_sub.alignment = PP_ALIGN.CENTER

    # Left Test Profile
    c_s6_l = add_card(s6, Inches(0.8), Inches(3.6), Inches(5.6), Inches(3.0))
    tb_s6_l = s6.shapes.add_textbox(Inches(1.0), Inches(3.8), Inches(5.2), Inches(2.6))
    t_s6_l = tb_s6_l.text_frame
    t_s6_l.word_wrap = True

    p = t_s6_l.paragraphs[0]
    p.text = "Test Profile Parameters"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    test_inputs = [
        "Age / Exp: 20 yrs old | 2 yrs Work Experience",
        "Education: Master's Degree",
        "Occupation: Data Science & AI / ML",
        "Work Mode / Size: Fully Remote | Mid-Size",
        "Market Benchmark: Global Remote Benchmark"
    ]
    for ti in test_inputs:
        p = t_s6_l.add_paragraph()
        p.text = f"• {ti}"
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_MUTED

    # Right Confidence & Driver
    c_s6_r = add_card(s6, Inches(6.8), Inches(3.6), Inches(5.7), Inches(3.0))
    tb_s6_r = s6.shapes.add_textbox(Inches(7.0), Inches(3.8), Inches(5.3), Inches(2.6))
    t_s6_r = tb_s6_r.text_frame
    t_s6_r.word_wrap = True

    p = t_s6_r.paragraphs[0]
    p.text = "Model Confidence & Impact Drivers"
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    conf_details = [
        "95% Confidence Min: ₹ 11,115,712",
        "95% Confidence Max: ₹ 16,538,010",
        "Primary Key Driver: Occupation Level (+₹ 4,008,000)",
        "Experience Trajectory: Non-linear logarithmic growth curve",
        "Radar Benchmark: High Edu Impact & Market Location score"
    ]
    for cd in conf_details:
        p = t_s6_r.add_paragraph()
        p.text = f"• {cd}"
        p.font.size = Pt(13)
        p.font.color.rgb = COLOR_MUTED

    # =========================================================================
    # Slide 7: SHAP Explainability
    # =========================================================================
    s7 = add_base_slide("Explainable AI & SHAP Feature Attribution", "Slide 07/13")
    
    c7 = add_card(s7, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tb7 = s7.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(11.1), Inches(4.8))
    tf7 = tb7.text_frame
    tf7.word_wrap = True

    p = tf7.paragraphs[0]
    p.text = "Feature Monetary Attribution Model Formulation"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    shap_points = [
        "Baseline Market Reference Salary: $24,000 baseline",
        "Education Multiplier: High School ($0) -> Bachelor's (+$14k) -> Master's (+$26k) -> Doctorate (+$40k)",
        "Occupation Tier: Software Eng / AI (+$22k - $25k), Management (+$28k), Finance (+$24k)",
        "Experience Curve: Logarithmic acceleration scaling up to 15 years, then leveling smoothly",
        "Regional Market Tier: Tier 1 Hub (+18%), Tier 2 (+2%), Tier 3 (-12%)"
    ]
    for sp in shap_points:
        p = tf7.add_paragraph()
        p.text = f"✦ {sp}"
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT

    # =========================================================================
    # Slide 8: Tech Stack
    # =========================================================================
    s8 = add_base_slide("Technology Stack & Architecture Matrix", "Slide 08/13")

    stacks = [
        ("Machine Learning", "Python 3.12, Scikit-Learn, XGBoost, Pandas, NumPy, Joblib"),
        ("Web Application", "HTML5, Custom Glassmorphism CSS3, JavaScript (ES6+), Chart.js"),
        ("Deployment & Hosting", "Git, GitHub Pages, Local HTTP Server, Streamlit Cloud")
    ]
    
    top_pos = 1.6
    for title, desc in stacks:
        card = add_card(s8, Inches(0.8), Inches(top_pos), Inches(11.7), Inches(1.5))
        tb = s8.shapes.add_textbox(Inches(1.1), Inches(top_pos + 0.2), Inches(11.1), Inches(1.1))
        tf = tb.text_frame
        tf.word_wrap = True
        
        p = tf.paragraphs[0]
        p.text = title
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = COLOR_PRIMARY
        
        p2 = tf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(14)
        p2.font.color.rgb = COLOR_TEXT
        
        top_pos += 1.75

    # =========================================================================
    # Slide 9: User Interface Design
    # =========================================================================
    s9 = add_base_slide("User Interface & Glassmorphism Design", "Slide 09/13")

    c9 = add_card(s9, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tb9 = s9.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(11.1), Inches(4.8))
    tf9 = tb9.text_frame
    tf9.word_wrap = True

    p = tf9.paragraphs[0]
    p.text = "Glassmorphism Design System"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    ui_points = [
        "Frosted Glass Cards: Multi-layered backdrop blur (backdrop-filter: blur(16px)) with glowing glass borders.",
        "Color Palette: Deep dark background (#090d16), vibrant neon accents (#6366f1 Indigo, #10b981 Emerald, #ec4899 Pink).",
        "Typography: Google Fonts (Outfit headings & Plus Jakarta Sans body text).",
        "Micro-Interactions: Animated number counter on prediction updates, smooth range slider thumbs, glowing metric pills.",
        "Responsive Layout: Custom CSS grid adapting seamlessly across Mobile, Tablet, and Desktop displays."
    ]
    for up in ui_points:
        p = tf9.add_paragraph()
        p.text = f"✦ {up}"
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT

    # =========================================================================
    # Slide 10: Batch CSV Engine
    # =========================================================================
    s10 = add_base_slide("Batch CSV Prediction Engine", "Slide 10/13")

    c10 = add_card(s10, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tb10 = s10.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(11.1), Inches(4.8))
    tf10 = tb10.text_frame
    tf10.word_wrap = True

    p = tf10.paragraphs[0]
    p.text = "Bulk Prediction Workflow"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    batch_points = [
        "Client-Side Parser: Processes CSV files directly in browser without server upload latency.",
        "Bulk Processing: Computes predictions for hundreds of employee records in <50 ms.",
        "Summary Metrics: Displays real-time average salary metrics and processed record count.",
        "Data Export: One-click export of batch prediction results to downloadable formatted CSV file."
    ]
    for bp in batch_points:
        p = tf10.add_paragraph()
        p.text = f"✦ {bp}"
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT

    # =========================================================================
    # Slide 11: Challenges & Solutions
    # =========================================================================
    s11 = add_base_slide("Engineering Challenges & Solutions", "Slide 11/13")

    c11 = add_card(s11, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tb11 = s11.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(11.1), Inches(4.8))
    tf11 = tb11.text_frame
    tf11.word_wrap = True

    p = tf11.paragraphs[0]
    p.text = "Key Engineering Problems Solved"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    chal_points = [
        "Non-Linear Salary Decay Curve: Applied logarithmic experience scaling function to model real-world career plateauing.",
        "Asynchronous Rendering Sync: Refactored JS execution order to compute predictions before chart initialization with fallback HTML numbers.",
        "Cross-Platform Encoding: Resolved Windows terminal UTF-8 print stream encoding issues during model serialization."
    ]
    for cp in chal_points:
        p = tf11.add_paragraph()
        p.text = f"✦ {cp}"
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT

    # =========================================================================
    # Slide 12: Internship Outcomes
    # =========================================================================
    s12 = add_base_slide("Key Internship Outcomes & Skills Gained", "Slide 12/13")

    c12 = add_card(s12, Inches(0.8), Inches(1.5), Inches(11.7), Inches(5.2))
    tb12 = s12.shapes.add_textbox(Inches(1.1), Inches(1.7), Inches(11.1), Inches(4.8))
    tf12 = tb12.text_frame
    tf12.word_wrap = True

    p = tf12.paragraphs[0]
    p.text = "Learnings & Technical Competencies"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    out_points = [
        "Hands-on mastery of multi-variate Machine Learning Regression and Joblib serialization.",
        "Practical application of AI/ML concepts learned during Edunet Foundation & IBM SkillsBuild internship.",
        "Front-end UI/UX architecture using modern CSS glassmorphism design tokens and Chart.js integration.",
        "Version control, GitHub workflow execution, and cloud static hosting on GitHub Pages."
    ]
    for op in out_points:
        p = tf12.add_paragraph()
        p.text = f"✦ {op}"
        p.font.size = Pt(14)
        p.font.color.rgb = COLOR_TEXT

    # =========================================================================
    # Slide 13: Conclusion Slide
    # =========================================================================
    s13 = add_base_slide()

    tbox13 = s13.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(1.8))
    tf13 = tbox13.text_frame
    p13 = tf13.paragraphs[0]
    p13.text = "Thank You!"
    p13.font.size = Pt(44)
    p13.font.bold = True
    p13.font.color.rgb = COLOR_PRIMARY
    p13.alignment = PP_ALIGN.CENTER

    p13_sub = tf13.add_paragraph()
    p13_sub.text = "Questions & Feedback Are Welcome"
    p13_sub.font.size = Pt(20)
    p13_sub.font.color.rgb = COLOR_MUTED
    p13_sub.alignment = PP_ALIGN.CENTER

    c13_1 = add_card(s13, Inches(1.5), Inches(3.6), Inches(4.8), Inches(2.6))
    tb13_1 = s13.shapes.add_textbox(Inches(1.7), Inches(3.8), Inches(4.4), Inches(2.2))
    t13_1 = tb13_1.text_frame
    t13_1.word_wrap = True
    add_kv(t13_1, "Student Name", "Sanjana Chirutha")
    add_kv(t13_1, "STU ID", "STU6817962b86d921746376235")
    add_kv(t13_1, "Internship Period", "18/06/2025 – 30/07/2025")

    c13_2 = add_card(s13, Inches(7.0), Inches(3.6), Inches(4.8), Inches(2.6))
    tb13_2 = s13.shapes.add_textbox(Inches(7.2), Inches(3.8), Inches(4.4), Inches(2.2))
    t13_2 = tb13_2.text_frame
    t13_2.word_wrap = True
    add_kv(t13_2, "GitHub Repo", "github.com/Sanjana-Chirutha")
    add_kv(t13_2, "Live Web App", "sanjana-chirutha.github.io")
    add_kv(t13_2, "Organisations", "Edunet | AICTE | IBM")

    output_path = "Employee_Earnings_Calculator_Presentation.pptx"
    prs.save(output_path)
    print(f"[SUCCESS] Regenerated PowerPoint presentation (.pptx) matching reference certificate design!")

if __name__ == "__main__":
    create_deck()
