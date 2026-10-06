"""
Script to generate the Official AICTE & Edunet Foundation Internship Project Report PDF
Matching the exact visual design, typography, colors, and layout of AICTE B2 PD Certificate-2436.pdf
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

def draw_certificate_decorations(canvas_obj, doc):
    canvas_obj.saveState()
    
    # Bottom blue bar
    canvas_obj.setFillColor(colors.HexColor('#1d4ed8'))
    canvas_obj.rect(0, 0, doc.pagesize[0], 0.35 * inch, fill=True, stroke=False)
    
    # Left vertical decorative margin pattern
    canvas_obj.setFillColor(colors.HexColor('#e0f2fe'))
    canvas_obj.rect(0, 0, 0.6 * inch, doc.pagesize[1], fill=True, stroke=False)
    
    canvas_obj.setStrokeColor(colors.HexColor('#bae6fd'))
    canvas_obj.setLineWidth(1)
    canvas_obj.line(0.6 * inch, 0, 0.6 * inch, doc.pagesize[1])
    
    # Outer border line
    canvas_obj.setStrokeColor(colors.HexColor('#0284c7'))
    canvas_obj.setLineWidth(1.5)
    canvas_obj.rect(0.7 * inch, 0.5 * inch, doc.pagesize[0] - 1.4 * inch, doc.pagesize[1] - 1.0 * inch)
    
    canvas_obj.restoreState()

def build_pdf():
    pdf_filename = "Sanjana_Chirutha_AICTE_Edunet_Internship_Report.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=0.9 * inch,
        rightMargin=0.9 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch
    )

    styles = getSampleStyleSheet()
    
    # Custom Typography Styles matching reference certificate
    style_org_header = ParagraphStyle(
        'OrgHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1e3a8a')
    )
    
    style_sub_org = ParagraphStyle(
        'SubOrg',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#475569')
    )
    
    style_cert_title = ParagraphStyle(
        'CertTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=15
    )

    style_name_box = ParagraphStyle(
        'NameBox',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#0f172a')
    )

    style_body_lead = ParagraphStyle(
        'BodyLead',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=17,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#334155'),
        spaceAfter=15
    )

    style_section_heading = ParagraphStyle(
        'SecHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=12,
        spaceAfter=8
    )

    style_normal_text = ParagraphStyle(
        'NormText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=15,
        textColor=colors.HexColor('#334155')
    )

    story = []

    # --- Header Banner (Edunet & AICTE Logos representation) ---
    header_data = [
        [
            Paragraph("<b>edunet</b><font color='#991b1b'> foundation</font>", style_org_header),
            Paragraph("In collaboration with<br/><b>AICTE</b><br/><font size=8 color='#64748b'>All India Council for Technical Education</font>", ParagraphStyle('RHead', parent=style_sub_org, alignment=TA_RIGHT))
        ]
    ]
    t_head = Table(header_data, colWidths=[3.2 * inch, 3.2 * inch])
    t_head.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10)
    ]))
    story.append(t_head)
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cbd5e1'), spaceBefore=5, spaceAfter=20))

    # --- Page 1: Certificate of Completion & Project Cover Sheet ---
    story.append(Paragraph("Certificate of Completion", style_cert_title))
    story.append(Spacer(1, 10))
    story.append(Paragraph("This is to certify that", ParagraphStyle('CertSub', parent=styles['Normal'], fontName='Helvetica', fontSize=12, alignment=TA_CENTER, textColor=colors.HexColor('#64748b'), spaceAfter=15)))

    # Student Name Box
    name_table_data = [[Paragraph("Sanjana Chirutha", style_name_box)]]
    t_name = Table(name_table_data, colWidths=[6.4 * inch])
    t_name.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f1f5f9')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('ALIGN', (0,0), (-1,-1), 'CENTER')
    ]))
    story.append(t_name)
    story.append(Spacer(1, 20))

    # Certificate Statement Text (Matching exact certificate text)
    cert_text = """
    Completed 6 weeks Internship on <b>Artificial Intelligence & Machine Learning</b> in collaboration with <b>All India Council for Technical Education (AICTE)</b>, implemented by <b>Edunet Foundation</b>, from <b>18/06/2025 to 30/07/2025</b>.
    <br/><br/>
    Your participation and engagement during the training is greatly appreciated.
    <br/><br/>
    <i>Students' STU ID:</i> <b>STU6817962b86d921746376235</b>
    """
    story.append(Paragraph(cert_text, style_body_lead))
    story.append(Spacer(1, 25))

    # Signatory Table
    sig_data = [
        [
            Paragraph("<b>Nagesh Singh</b><br/>Chairman<br/>Edunet Foundation", ParagraphStyle('SigLeft', parent=styles['Normal'], fontName='Helvetica', fontSize=10, leading=14, textColor=colors.HexColor('#334155'))),
            Paragraph("In collaboration with<br/><b>IBM SkillsBuild</b>", ParagraphStyle('SigRight', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, leading=15, alignment=TA_RIGHT, textColor=colors.HexColor('#1e3a8a')))
        ]
    ]
    t_sig = Table(sig_data, colWidths=[3.2 * inch, 3.2 * inch])
    t_sig.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM')
    ]))
    story.append(t_sig)
    
    # Page Break for Report Content
    story.append(PageBreak())

    # --- Page 2: Internship Project Report Content ---
    story.append(Paragraph("INTERNSHIP CAPSTONE PROJECT REPORT", style_cert_title))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1e3a8a'), spaceBefore=5, spaceAfter=15))

    story.append(Paragraph("1. Project Title & Executive Summary", style_section_heading))
    p_summary = """
    <b>Project Title:</b> Employee Earnings Calculator (ML-Powered Salary Intelligence System)<br/>
    <b>Student Name:</b> Sanjana Chirutha (STU ID: STU6817962b86d921746376235)<br/>
    <b>Domain:</b> Artificial Intelligence & Machine Learning (6 Weeks Virtual Internship)<br/>
    <b>Organizations:</b> Edunet Foundation | AICTE | IBM SkillsBuild<br/><br/>
    <b>Executive Overview:</b> During the 6-week virtual internship from 18/06/2025 to 30/07/2025, an end-to-end Machine Learning web application was developed to predict employee compensation based on multi-variate professional attributes (Age, Work Experience, Education Level, Occupation, Hours Worked per Week, Work Arrangement, Company Scale, and Regional Market Tier).
    """
    story.append(Paragraph(p_summary, style_normal_text))
    story.append(Spacer(1, 15))

    story.append(Paragraph("2. Machine Learning Architecture & Performance Results", style_section_heading))
    
    metrics_table_data = [
        [Paragraph("<b>Metric</b>", style_normal_text), Paragraph("<b>Value / Result</b>", style_normal_text), Paragraph("<b>Description</b>", style_normal_text)],
        [Paragraph("Algorithm Selected", style_normal_text), Paragraph("Gradient Boosting / XGBoost", style_normal_text), Paragraph("Ensemble tree regression optimizer", style_normal_text)],
        [Paragraph("Model Accuracy (R² Score)", style_normal_text), Paragraph("<b>98.00%</b> (0.9800)", style_normal_text), Paragraph("High variance coverage across roles", style_normal_text)],
        [Paragraph("Root Mean Squared Error", style_normal_text), Paragraph("<b>$3,523.39</b>", style_normal_text), Paragraph("Low prediction error margin", style_normal_text)],
        [Paragraph("Inference Latency", style_normal_text), Paragraph("<b>&lt; 15 ms</b>", style_normal_text), Paragraph("Real-time client-side prediction", style_normal_text)],
        [Paragraph("Training Dataset Size", style_normal_text), Paragraph("10,000 Profiles", style_normal_text), Paragraph("Cleaned UCI & BLS benchmarks", style_normal_text)]
    ]
    t_metrics = Table(metrics_table_data, colWidths=[2.1 * inch, 2.0 * inch, 2.3 * inch])
    t_metrics.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6)
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 15))

    story.append(Paragraph("3. Candidate Test Case Prediction Results", style_section_heading))
    
    test_results_text = """
    <b>Candidate Profile Inputs Tested:</b><br/>
    • Age: 20 yrs old | Work Experience: 2 yrs<br/>
    • Education Level: Master's Degree<br/>
    • Occupation: Data Science & AI / ML<br/>
    • Work Arrangement & Company Scale: Fully Remote | Mid-Size (50-500 staff)<br/>
    • Regional Benchmark: Global Remote Benchmark (INR ₹)<br/><br/>
    <b>Live System Output Generated:</b><br/>
    • <b>Estimated Annual Earnings:</b> <font color='#16a34a'><b>₹ 13,555,746 / yr</b></font><br/>
    • <b>Monthly Take-Home:</b> ₹ 1,129,646 / month<br/>
    • <b>Hourly Rate Equivalent:</b> ₹ 6,517.19 / hr<br/>
    • <b>95% Confidence Interval:</b> ₹ 11,115,712 – ₹ 16,538,010<br/>
    • <b>Earning Percentile Rank:</b> Top 21% Tier<br/>
    • <b>Primary Key Impact Driver:</b> Occupation Level (+₹ 4,008,000)
    """
    story.append(Paragraph(test_results_text, style_normal_text))
    story.append(Spacer(1, 15))

    story.append(Paragraph("4. Key Learnings & Internship Outcomes", style_section_heading))
    outcomes_text = """
    • Mastered end-to-end Machine Learning model training, evaluation, and Joblib serialization.<br/>
    • Applied practical AI/ML concepts from <b>IBM SkillsBuild</b> and <b>Edunet Foundation</b> coursework.<br/>
    • Built custom responsive Glassmorphic user interface using HTML5, CSS3, and JavaScript.<br/>
    • Deployed live web application and GitHub repository: <b>github.com/Sanjana-Chirutha/Employee-Earnings-Calculator</b>
    """
    story.append(Paragraph(outcomes_text, style_normal_text))

    doc.build(story, onFirstPage=draw_certificate_decorations, onLaterPages=draw_certificate_decorations)
    print(f"[SUCCESS] PDF report generated successfully at {pdf_filename}")

if __name__ == '__main__':
    build_pdf()
