"""
Export test cases as PDF documentation
"""
from pathlib import Path
from typing import Dict
from datetime import datetime
import re

try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_LEFT, TA_CENTER
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Preformatted
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class PDFExporter:
    """
    Exports test cases as formatted PDF documentation
    """

    def __init__(self):
        """Initialize the PDF exporter"""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab is required for PDF export. Install with: pip install reportlab")

    def export(
        self,
        tests: str,
        output_path: str,
        metadata: Dict = None
    ) -> str:
        """
        Export tests to PDF format

        Args:
            tests: Test code string
            output_path: Path to save the PDF file
            metadata: Additional metadata

        Returns:
            Path to the exported file
        """
        # Ensure .pdf extension
        output_path = Path(output_path)
        if output_path.suffix != '.pdf':
            output_path = output_path.with_suffix('.pdf')

        # Create PDF document
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )

        # Build content
        story = []
        styles = getSampleStyleSheet()

        # Add custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=30,
            alignment=TA_CENTER
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12
        )

        # Title
        story.append(Paragraph("Generated Test Suite", title_style))
        story.append(Spacer(1, 12))

        # Metadata table
        if metadata:
            story.append(Paragraph("Test Suite Information", heading_style))

            metadata_data = []
            if 'user_story' in metadata:
                metadata_data.append(['User Story:', metadata['user_story']])
            if 'timestamp' in metadata:
                metadata_data.append(['Generated:', metadata['timestamp']])
            if 'coverage' in metadata:
                metadata_data.append(['Coverage:', f"{metadata['coverage']:.1%}"])
            if 'iterations' in metadata:
                metadata_data.append(['Iterations:', str(metadata['iterations'])])
            if 'test_count' in metadata:
                metadata_data.append(['Total Tests:', str(metadata['test_count'])])

            metadata_table = Table(metadata_data, colWidths=[2*inch, 4.5*inch])
            metadata_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#ecf0f1')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))

            story.append(metadata_table)
            story.append(Spacer(1, 20))

        # Parse and add test cases
        story.append(Paragraph("Test Cases", heading_style))
        story.append(Spacer(1, 12))

        test_cases = self._parse_tests(tests)

        for idx, test_case in enumerate(test_cases, 1):
            # Test case header
            test_header = f"Test Case {idx}: {test_case['name']}"
            story.append(Paragraph(test_header, styles['Heading3']))

            if test_case['description']:
                story.append(Paragraph(f"<i>{test_case['description']}</i>", styles['Normal']))

            story.append(Spacer(1, 8))

            # Test code (formatted)
            code_text = test_case['code']
            code_para = Preformatted(
                code_text,
                ParagraphStyle(
                    'Code',
                    parent=styles['Code'],
                    fontSize=8,
                    leading=10,
                    leftIndent=10,
                    textColor=colors.HexColor('#2c3e50')
                )
            )
            story.append(code_para)
            story.append(Spacer(1, 20))

        # Build PDF
        doc.build(story)

        return str(output_path)

    def _parse_tests(self, tests: str) -> list:
        """
        Parse test code into individual test cases

        Args:
            tests: Test code string

        Returns:
            List of test case dictionaries
        """
        test_cases = []

        # Split into individual test functions
        test_pattern = r'def (test_\w+)\([^)]*\):(.*?)(?=\ndef |\Z)'
        matches = re.findall(test_pattern, tests, re.DOTALL)

        for test_name, test_body in matches:
            # Extract docstring
            docstring_match = re.search(r'"""(.*?)"""', test_body, re.DOTALL)
            description = docstring_match.group(1).strip() if docstring_match else ""

            test_cases.append({
                'name': test_name,
                'description': description,
                'code': f'def {test_name}():{test_body}'
            })

        return test_cases
