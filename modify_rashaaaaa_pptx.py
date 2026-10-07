"""
Modify ONLY the text content of Rashaaaaa.pptx, preserving all design elements.
Maps Rasha's internship content → Sanjana's Employee Earnings Calculator project.
"""

from pptx import Presentation
from pptx.oxml.ns import qn
from lxml import etree
import copy

INPUT  = "Rashaaaaa.pptx"
OUTPUT = "Sanjana_Employee_Earnings_Presentation.pptx"

prs = Presentation(INPUT)
slides = list(prs.slides)


# ── Low-level helpers ────────────────────────────────────────────────────────

def set_para_text(para, new_text):
    """
    Replace all runs in a paragraph with a single run carrying new_text.
    Preserves the run's rPr (font properties) from the first existing run.
    Works even when there are no runs (direct <a:t> in <a:p>).
    """
    p_elem = para._p
    # Collect existing run elements
    existing_runs = p_elem.findall(qn('a:r'))

    if existing_runs:
        # Keep the first run's rPr, set its text, remove the rest
        first_r = existing_runs[0]
        t_elem = first_r.find(qn('a:t'))
        if t_elem is None:
            t_elem = etree.SubElement(first_r, qn('a:t'))
        t_elem.text = new_text
        for r in existing_runs[1:]:
            p_elem.remove(r)
    else:
        # Build a minimal run
        r_elem = etree.SubElement(p_elem, qn('a:r'))
        t_elem = etree.SubElement(r_elem, qn('a:t'))
        t_elem.text = new_text


def set_tf(shape, texts):
    """
    Set the text of a shape's text frame to the given list of strings.
    One string per paragraph. Extra existing paragraphs are cleared;
    new paragraphs are cloned from the last existing one.
    """
    tf = shape.text_frame
    txBody = tf._txBody
    existing_paras = list(tf.paragraphs)

    # Make sure we have enough paragraph elements
    while len(existing_paras) < len(texts):
        # Clone the last paragraph XML
        last_p = existing_paras[-1]._p
        new_p = copy.deepcopy(last_p)
        txBody.append(new_p)
        existing_paras = list(tf.paragraphs)

    # Set text for each needed paragraph
    for i, text in enumerate(texts):
        set_para_text(existing_paras[i], text)

    # Clear leftover paragraphs (keep at least one)
    for i in range(len(texts), len(existing_paras)):
        set_para_text(existing_paras[i], "")


def replace_text(shape, new_text):
    """Replace shape text frame with a single paragraph."""
    set_tf(shape, [new_text])


def get_textbox(slide, name):
    for s in slide.shapes:
        if s.name == name and s.has_text_frame:
            return s
    return None


# ============================================================================
# Slide 1 – Cover / Internship Evaluation
# ============================================================================
s1 = slides[0]
# TextBox 7  → Name
replace_text(get_textbox(s1, "TextBox 7"), "Sanjana Chirutha")
# TextBox 9  → Roll No  → STU ID
replace_text(get_textbox(s1, "TextBox 9"), "STU6817962b86d921746376235")
# TextBox 11 → Branch
replace_text(get_textbox(s1, "TextBox 11"), "AI & Machine Learning")

# ============================================================================
# Slide 2 – Organization Profile
# ============================================================================
s2 = slides[1]
replace_text(get_textbox(s2, "TextBox 10"), "Edunet Foundation (AICTE | IBM SkillsBuild)")
replace_text(get_textbox(s2, "TextBox 13"), "Virtual / Online Internship")
replace_text(get_textbox(s2, "TextBox 16"), "6 Weeks  (18 June 2025 - 30 July 2025)")
replace_text(get_textbox(s2, "TextBox 19"), "AI & ML Engineer (Project Intern)")
replace_text(get_textbox(s2, "TextBox 22"), "Employee Earnings Calculator (ML Salary Intelligence)")
replace_text(get_textbox(s2, "TextBox 25"), "AICTE - All India Council for Technical Education")
replace_text(get_textbox(s2, "TextBox 28"), "STU6817962b86d921746376235")

# ============================================================================
# Slide 3 – Problem Statement
# ============================================================================
s3 = slides[2]
replace_text(get_textbox(s3, "TextBox 5"),
    "PROJECT TITLE: Employee Earnings Calculator (ML-Powered Salary Intelligence)")
replace_text(get_textbox(s3, "TextBox 6"),
    "Salary estimation in corporate environments is largely manual, subjective, and inconsistent. "
    "HR teams rely on experience-based guesswork rather than data-driven models, leading to "
    "compensation inequity, hiring inefficiency, and talent attrition.")
set_tf(get_textbox(s3, "TextBox 8"), [
    "o  No Data-Driven Benchmark: Companies lack objective salary benchmarks calibrated to industry, role, experience, and region.",
    "o  Black-Box Decisions: Compensation decisions are non-transparent, causing employee dissatisfaction and trust deficit.",
    "o  Manual & Error-Prone: Spreadsheet-based estimation is slow, inconsistent across departments, and not reproducible.",
    "o  Lack of Explainability: Even when ML models exist, organizations cannot explain why a particular salary was predicted.",
    "o  No Scenario Analysis: HR cannot simulate 'what-if' scenarios to understand how education or relocation affects pay.",
    "o  Scalability: Manual estimation fails at scale — evaluating hundreds of candidates simultaneously is infeasible.",
])

# ============================================================================
# Slide 4 – Objectives
# ============================================================================
s4 = slides[3]
set_tf(get_textbox(s4, "TextBox 5"), [
    "o  Accurate Prediction: Build an XGBoost / Gradient Boosting ML regression pipeline predicting annual salary with >93% R2 accuracy.",
    "o  Explainability (XAI): Implement SHAP-style feature attribution showing each factor's monetary contribution to the predicted salary.",
    "o  Interactive UI: Design a real-time web predictor with sliders and dropdowns with instant updates (<15 ms latency) using Glassmorphism UI.",
    "o  Batch Processing: Enable bulk CSV upload to predict salaries for hundreds of employees simultaneously with one-click CSV export.",
    "o  Career Optimizer: Provide AI-generated sensitivity recommendations helping employees understand how to boost earning potential.",
    "o  Multi-Currency Support: Display predicted salaries in USD ($), EUR (E), GBP (P), and INR (Rs) with live currency conversion.",
])

# ============================================================================
# Slide 5 – Technologies Used
# ============================================================================
s5 = slides[4]
set_tf(get_textbox(s5, "TextBox 5"), [
    "o  Machine Learning - Python 3.12, XGBoost, Scikit-Learn, Pandas, NumPy, Joblib for model training, serialization, and inference.",
    "o  Frontend - HTML5, Custom Glassmorphism CSS3 (backdrop-filter, neon glows), JavaScript ES6+ for all UI interactions and ML inference.",
    "o  Visualization - Chart.js for experience growth trajectory, industry benchmark radar, and feature attribution waterfall charts.",
    "o  Data Pipeline - MinMaxScaler (numerical features), LabelEncoder (categorical: education, occupation, region), Joblib persistence.",
    "o  Deployment - GitHub Pages (static hosting), Git for version control, Python HTTP Server for local development.",
])

# ============================================================================
# Slide 6 – Methodology
# ============================================================================
s6 = slides[5]
set_tf(get_textbox(s6, "TextBox 5"), [
    "o  Data Ingestion & Cleaning: 10,000+ synthetic employee profiles generated from UCI Adult Income benchmarks and BLS wage statistics.",
    "o  Feature Engineering: 9 input features - Age, Experience, Education, Occupation, Hours/Week, Work Mode, Company Size, Region, Gender.",
    "o  Preprocessing Pipeline: LabelEncoder for categoricals; MinMaxScaler for numericals; stratified 80/20 train-test split.",
    "o  Model Training: XGBoost Regressor (300 estimators, depth 6, lr=0.05); evaluated via R2 Score and RMSE on held-out test set.",
    "o  Salary Inference: Client-side JavaScript mirror of the trained model enabling real-time predictions with <15 ms latency.",
    "o  SHAP Attribution: Waterfall chart breaking down each feature's monetary contribution (positive/negative) relative to $24,000 baseline.",
    "o  Model Serialization: Trained model, scaler, and encoders exported as .joblib artifacts for reproducible inference.",
])

# ============================================================================
# Slide 7 – Results (UI Screenshot captions)
# ============================================================================
s7 = slides[6]
replace_text(get_textbox(s7, "TextBox 5"), "Salary Predictor - Interactive Dashboard")
replace_text(get_textbox(s7, "TextBox 7"), "Feature Attribution - SHAP Waterfall Chart")
replace_text(get_textbox(s7, "TextBox 9"), "Analytics - Experience Growth & Radar Chart")

# ============================================================================
# Slide 8 – Results contd. (captions)
# ============================================================================
s8 = slides[7]
replace_text(get_textbox(s8, "TextBox 5"), "Batch Predict - Bulk CSV Upload & Export")
replace_text(get_textbox(s8, "TextBox 7"), "ML Architecture - XGBoost Pipeline View")
replace_text(get_textbox(s8, "TextBox 9"), "Career Optimizer - What-If Scenario Engine")

# ============================================================================
# Slide 9 – Certificate  (image slide – leave unchanged)
# ============================================================================

# ============================================================================
# Slide 10 – Challenges & Learning Outcomes
# ============================================================================
s10 = slides[9]
set_tf(get_textbox(s10, "TextBox 6"), [
    "o  Non-Linear Salary Decay: Real-world experience gains plateau after ~15 years - modelled using logarithmic scaling to avoid overfitting.",
    "o  Async Rendering Sync: JavaScript chart initialization ran before ML prediction completed; refactored execution order with fallback HTML values.",
    "o  Windows Encoding: Python's cp1252 terminal encoding caused UnicodeEncodeError for emoji print statements; resolved with UTF-8 stream override.",
    "o  Client-Side ML Mirror: Replicating the trained XGBoost logic in pure JavaScript required careful parameter tuning to match server predictions.",
    "o  CSV Batch Performance: Parsing large CSV files in-browser caused UI freezes; resolved using chunked processing with setTimeout yielding.",
])
set_tf(get_textbox(s10, "TextBox 9"), [
    "o  ML Regression Mastery: Hands-on XGBoost, Scikit-Learn pipelines, hyperparameter tuning, and Joblib model serialization.",
    "o  Explainable AI (XAI): Implemented SHAP-style feature attribution making ML predictions interpretable for non-technical HR users.",
    "o  Full-Stack Web Dev: Built a production-quality static web app using HTML5, Glassmorphism CSS3, JavaScript ES6+, and Chart.js.",
    "o  Data Engineering: Cleaned, encoded, and scaled real-world benchmark salary data for a supervised ML regression pipeline.",
    "o  GitHub Workflow: Practiced version control, branch management, and cloud static hosting on GitHub Pages for public deployment.",
])

# ============================================================================
# Slide 11 – Future Enhancements
# ============================================================================
s11 = slides[10]
set_tf(get_textbox(s11, "TextBox 5"), [
    "o  Real Dataset Integration: Replace synthetic data with anonymized Glassdoor/LinkedIn salary datasets for higher real-world accuracy.",
    "o  REST API Backend: Deploy the XGBoost model as a FastAPI microservice enabling integration with enterprise HRMS platforms.",
    "o  True SHAP Library: Replace client-side SHAP approximation with the official shap Python library for exact Shapley values.",
    "o  Resume Parser: Add NLP-based resume parsing (spaCy / HuggingFace) to auto-populate predictor fields from uploaded PDF resumes.",
    "o  Mobile App: Develop a React Native mobile application with offline ML inference for salary benchmarking on-the-go.",
    "o  Company Dashboard: Build an HR analytics dashboard to compare internal salaries against predicted market benchmarks.",
    "o  LLM Career Coach: Integrate Gemini / GPT to generate personalized career growth roadmaps based on predicted salary gaps.",
])

# ============================================================================
# Slide 12 – Thank You
# ============================================================================
s12 = slides[11]
replace_text(get_textbox(s12, "TextBox 4"), "Employee Earnings Calculator (ML Salary Intelligence)")
replace_text(get_textbox(s12, "TextBox 5"),
    "Sanjana Chirutha  |  STU6817962b86d921746376235  |  AI & Machine Learning")
replace_text(get_textbox(s12, "TextBox 6"),
    "Edunet Foundation  o  AICTE  o  IBM SkillsBuild  o  6 Weeks Virtual Internship")

# ============================================================================
# Save
# ============================================================================
prs.save(OUTPUT)
print(f"[SUCCESS] Saved: {OUTPUT}")
