import sys
import subprocess

# Ensure fpdf2 is installed
try:
    from fpdf import FPDF
except ImportError:
    print("Installing fpdf2...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
    from fpdf import FPDF

class ProjectReportPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'SR NEXT - Instagram Influencer Analytics Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('helvetica', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', 1)
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('helvetica', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()

def generate_report(output_path):
    pdf = ProjectReportPDF()
    pdf.add_page()

    pdf.chapter_title('1. Executive Summary')
    body1 = (
        "This report outlines the methodology, tools, and outcomes for the SR NEXT Self-Guided "
        "Internship project. The project focuses on collecting, analyzing, and ranking Instagram "
        "influencers with under 1 million followers using Python, AI, and Machine Learning techniques. "
        "The project was successfully divided into two core phases: Data Collection and AI-Powered Analysis."
    )
    pdf.chapter_body(body1)

    pdf.chapter_title('2. Phase 1: Data Collection & Preprocessing')
    body2 = (
        "Objective: To collect public Instagram data and prepare a structured influencer dataset.\n\n"
        "Data Sources & Criteria:\n"
        "- Target: Instagram influencers and pages with less than 1M followers.\n"
        "- Dataset Size: Targeted over 2000 profiles.\n"
        "- Key Data Points: Name, Username, Followers, Engagement Rate, Contact Details, Posting Frequency, "
        "Hashtags, and Niche/Category.\n\n"
        "Tools Used:\n"
        "- Python, Pandas, BeautifulSoup, Selenium, and standard requests modules.\n\n"
        "Process:\n"
        "Data was collected by identifying target profiles based on follower count. Publicly available "
        "metrics were scraped, cleaned, and compiled into a structured CSV format. Missing values and "
        "outliers were handled during the preprocessing step to ensure data quality and integrity for the next phase."
    )
    pdf.chapter_body(body2)

    pdf.chapter_title('3. Phase 2: AI-Powered Analysis & Ranking')
    body3 = (
        "Objective: To analyze collected data, apply AI/ML techniques, and rank influencers.\n\n"
        "Feature Engineering:\n"
        "- Calculated Engagement Rate ((Likes + Comments) / Followers) based on the latest posts.\n"
        "- Extracted hashtag diversity and categorized content types.\n\n"
        "AI/ML Models:\n"
        "- NLP: Utilized for caption analysis and content classification to categorize influencers (e.g., Tech, Fashion).\n"
        "- Predictive Modeling: Employed basic ML models using scikit-learn to predict engagement trends and the "
        "likelihood of automation tool adoption based on posting patterns.\n\n"
        "Ranking Methodology:\n"
        "- A composite score was created by weighting engagement metrics and AI predictions. This score was "
        "used to generate the final ranked influencer list.\n\n"
        "Visualization:\n"
        "- A Streamlit/Plotly dashboard was developed to provide interactive visual insights into the top "
        "influencers per category."
    )
    pdf.chapter_body(body3)

    pdf.chapter_title('4. Conclusion & Deliverables')
    body4 = (
        "The project successfully demonstrated an end-to-end data pipeline: from web scraping and data cleaning to "
        "machine learning and data visualization. The final deliverables include:\n"
        "1. A comprehensive, cleaned dataset (CSV) of influencers.\n"
        "2. A final ranked list of influencers based on AI predictions.\n"
        "3. An interactive visualization dashboard."
    )
    pdf.chapter_body(body4)

    pdf.chapter_title('5. Appendix: Dashboard Outputs')
    pdf.chapter_body("Below are the screenshots of the generated dashboard and model outputs.")
    
    import os
    img_dir = r"D:\SR Next-Intern\sr-next-influencer\Output-Screenshot"
    if os.path.exists(img_dir):
        for img_name in sorted(os.listdir(img_dir)):
            if img_name.endswith(".png"):
                pdf.add_page()
                pdf.image(os.path.join(img_dir, img_name), w=190)

    pdf.output(output_path)
    print(f"PDF successfully generated at: {output_path}")

if __name__ == "__main__":
    generate_report("d:/SR Next-Intern/sr-next-influencer/Project_Report.pdf")
