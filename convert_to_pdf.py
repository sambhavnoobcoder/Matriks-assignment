#!/usr/bin/env python3
"""
Convert markdown analysis to PDF
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
import re

def parse_markdown(markdown_text):
    """Parse markdown text and convert to a list of reportlab elements"""
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading1_style = styles['Heading1']
    heading2_style = styles['Heading2']
    heading3_style = ParagraphStyle(
        'Heading3',
        parent=styles['Heading2'],
        fontSize=14,
        spaceAfter=6
    )
    normal_style = styles['Normal']
    
    # Split text into lines
    lines = markdown_text.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Title (# Heading)
        if line.startswith('# '):
            elements.append(Paragraph(line[2:], title_style))
            elements.append(Spacer(1, 12))
        
        # Heading 1 (## Heading)
        elif line.startswith('## '):
            elements.append(Paragraph(line[3:], heading1_style))
            elements.append(Spacer(1, 10))
        
        # Heading 2 (### Heading)
        elif line.startswith('### '):
            elements.append(Paragraph(line[4:], heading2_style))
            elements.append(Spacer(1, 8))
        
        # Heading 3 (#### Heading)
        elif line.startswith('#### '):
            elements.append(Paragraph(line[5:], heading3_style))
            elements.append(Spacer(1, 6))
        
        # List item
        elif line.startswith('- '):
            # Convert bullet points
            bullet_text = '• ' + line[2:]
            # Bold text between ** **
            bullet_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', bullet_text)
            elements.append(Paragraph(bullet_text, normal_style))
            elements.append(Spacer(1, 4))
        
        # Numbered list
        elif re.match(r'^\d+\. ', line):
            num_text = line
            # Bold text between ** **
            num_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', num_text)
            elements.append(Paragraph(num_text, normal_style))
            elements.append(Spacer(1, 4))
        
        # Regular paragraph
        elif line and not line.startswith('```'):
            # Bold text between ** **
            paragraph_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            elements.append(Paragraph(paragraph_text, normal_style))
            elements.append(Spacer(1, 6))
        
        i += 1
    
    return elements

def convert_to_pdf(markdown_file, pdf_file):
    """Convert markdown file to PDF"""
    # Read markdown file
    with open(markdown_file, 'r') as f:
        markdown_text = f.read()
    
    # Create PDF document
    doc = SimpleDocTemplate(pdf_file, pagesize=letter, 
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    
    # Parse markdown and get elements
    elements = parse_markdown(markdown_text)
    
    # Build PDF
    doc.build(elements)

if __name__ == "__main__":
    convert_to_pdf('analysis_summary.md', 'Matiks_Data_Analysis_Summary.pdf')
    print("PDF created successfully: Matiks_Data_Analysis_Summary.pdf") 