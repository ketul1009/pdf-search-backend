# synthetic_data.py
def get_parsed_pdf_data():
    """Simulates data extracted from a PDF parser."""
    return {
        "pdf_id": "annual_report_2024.pdf",
        "content": [
            {
                "type": "paragraph",
                "data": "In 2024, our company achieved a record-breaking revenue of $50 million, a 25% increase from the previous year. This growth was primarily driven by the launch of our new product, 'InnovateX'.",
                "page": 2,
            },
            {
                "type": "paragraph",
                "data": "Our operational expenses were kept under control, totaling $20 million. The net profit margin stood at an impressive 30%, showcasing strong financial health and operational efficiency.",
                "page": 3,
            },
            {
                "type": "table",
                # In a real scenario, this would be structured data. We'll use a summary.
                "data": "Table on page 4 shows regional sales breakdown. North America contributed $25M, Europe $15M, and Asia $10M in revenue.",
                "page": 4,
            },
             {
                "type": "image",
                # In a real scenario, this would be an image file. We'll use a generated caption.
                "data": "A pie chart on page 5 illustrates market share, with our company holding 40%, Competitor A at 30%, and others at 30%.",
                "page": 5,
            }
        ]
    }