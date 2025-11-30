# sample_data_generator.py
from fpdf import FPDF
import os

def make_sample_pdfs(out_dir="sample_data"):
    os.makedirs(out_dir, exist_ok=True)
    files = {
        "company_policy.pdf": """Company Policy

1. Leave Policy: Employees are entitled to 12 casual leaves per year. Sick leave is 8 days.
2. Attendance: Employees must log in by 10:00 AM.
3. Benefits: Health insurance covers immediate family.

This is sample policy text for demo.""",
        "product_manual.pdf": """Product Manual

The X200 device supports WiFi and Bluetooth.
Installation steps:
1) Unbox
2) Plug in
3) Press power.
Warranty: 1 year from purchase date.
""",
        "onboarding_guide.pdf": """Onboarding Guide

Welcome! New hires should complete:
- HR forms
- System access request
- Manager introduction meeting

Orientation schedule:
Day 1: HR
Day 2: Team
Day 3: Systems.
"""
    }
    for filename, content in files.items():
        pdf_path = os.path.join(out_dir, filename)
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        for line in content.split("\n"):
            pdf.multi_cell(0, 10, line)
        pdf.output(pdf_path)
    print(f"Sample PDFs created in: {out_dir}")

if __name__ == "__main__":
    make_sample_pdfs()
