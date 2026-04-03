"""Convert midterm_report.md to PDF using fpdf2."""
import re
from fpdf import FPDF


class ReportPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def clean_md(text):
    """Remove markdown formatting and non-latin1 chars."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = text.replace('\\', '')
    # Unicode -> ASCII
    replacements = {
        '\u2014': '--', '\u2013': '-', '\u2018': "'", '\u2019': "'",
        '\u201c': '"', '\u201d': '"', '\u2026': '...', '\u2022': '-',
        '\u2192': '->', '\u2248': '~', '\u2265': '>=', '\u2264': '<=',
        '\u2713': '[ok]', '\u2717': '[x]',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = text.encode('latin-1', errors='replace').decode('latin-1')
    return text.strip()


def render_table(pdf, table_rows):
    """Render a markdown table with auto-sized columns and multi-line cells."""
    data_rows = []
    for row in table_rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        if all(re.match(r'^[-:]+$', c) for c in cells):
            continue
        data_rows.append([clean_md(c) for c in cells])

    if not data_rows:
        return

    num_cols = len(data_rows[0])
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    font_size = 8
    cell_padding = 2
    line_h = 4.5

    # Pad all rows to same number of columns
    for row in data_rows:
        while len(row) < num_cols:
            row.append("")

    # Calculate column widths based on content
    pdf.set_font("Helvetica", "", font_size)
    max_widths = [0] * num_cols
    for row in data_rows:
        for j, cell in enumerate(row):
            w = pdf.get_string_width(cell) + cell_padding * 2
            max_widths[j] = max(max_widths[j], w)

    # Scale to fit page width
    total = sum(max_widths)
    if total > page_w:
        col_widths = [w / total * page_w for w in max_widths]
    else:
        col_widths = max_widths

    # Ensure minimum column width
    min_w = 18
    for j in range(num_cols):
        if col_widths[j] < min_w:
            col_widths[j] = min_w

    # Re-scale after minimum enforcement
    total = sum(col_widths)
    if total > page_w:
        col_widths = [w / total * page_w for w in col_widths]

    def calc_row_height(row, widths):
        """Calculate row height needed for wrapped text."""
        max_lines = 1
        for j, cell in enumerate(row):
            usable = widths[j] - cell_padding * 2
            if usable < 5:
                usable = 5
            text_w = pdf.get_string_width(cell)
            lines = max(1, int(text_w / usable) + 1)
            max_lines = max(max_lines, lines)
        return max_lines * line_h + cell_padding

    for i, row in enumerate(data_rows):
        is_header = (i == 0)
        if is_header:
            pdf.set_font("Helvetica", "B", font_size)
            pdf.set_fill_color(235, 235, 235)
        else:
            pdf.set_font("Helvetica", "", font_size)
            if i % 2 == 0:
                pdf.set_fill_color(248, 248, 248)

        fill = is_header or (i % 2 == 0)
        row_h = calc_row_height(row, col_widths)

        # Page break check
        if pdf.get_y() + row_h > pdf.h - pdf.b_margin:
            pdf.add_page()

        y_start = pdf.get_y()
        x_offset = pdf.l_margin

        # Draw cell backgrounds and borders first
        for j in range(num_cols):
            pdf.set_xy(x_offset + sum(col_widths[:j]), y_start)
            pdf.cell(col_widths[j], row_h, "", border=1, fill=fill)

        # Then draw text with wrapping
        for j, cell in enumerate(row):
            x = x_offset + sum(col_widths[:j]) + cell_padding
            pdf.set_xy(x, y_start + cell_padding / 2)
            usable = col_widths[j] - cell_padding * 2
            pdf.multi_cell(usable, line_h, cell)

        pdf.set_xy(pdf.l_margin, y_start + row_h)

    pdf.ln(4)


def parse_md_to_pdf(md_path, pdf_path):
    with open(md_path, "r") as f:
        lines = f.readlines()

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    in_table = False
    table_rows = []

    for line in lines:
        stripped = line.rstrip("\n")

        # Table handling
        if stripped.startswith("|"):
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(stripped)
            continue
        elif in_table:
            render_table(pdf, table_rows)
            in_table = False
            table_rows = []

        # Always reset x to left margin
        pdf.set_x(pdf.l_margin)

        # Horizontal rule
        if stripped.strip() == "---":
            pdf.ln(2)
            y = pdf.get_y()
            pdf.set_draw_color(200, 200, 200)
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(4)
            continue

        # H1
        if stripped.startswith("# "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 9, clean_md(stripped[2:]))
            pdf.ln(2)
            continue

        # H2
        if stripped.startswith("## "):
            pdf.ln(6)
            pdf.set_font("Helvetica", "B", 14)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(0, 8, clean_md(stripped[3:]))
            y = pdf.get_y()
            pdf.set_draw_color(200, 200, 200)
            pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
            pdf.ln(3)
            continue

        # H3
        if stripped.startswith("### "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 7, clean_md(stripped[4:]))
            pdf.ln(1)
            continue

        # Empty line
        if stripped.strip() == "":
            pdf.ln(3)
            continue

        # Numbered list
        num_match = re.match(r'^(\d+)\.\s+(.+)', stripped)
        if num_match:
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            text = f"{num_match.group(1)}. {clean_md(num_match.group(2))}"
            pdf.multi_cell(0, 6, "  " + text)
            pdf.ln(1)
            continue

        # Bullet list
        if stripped.startswith("- "):
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            text = clean_md(stripped[2:])
            pdf.multi_cell(0, 6, "  - " + text)
            pdf.ln(1)
            continue

        # Regular paragraph
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(60, 60, 60)
        text = clean_md(stripped)
        if text:
            pdf.multi_cell(0, 6, text)

    # Flush remaining table
    if in_table:
        render_table(pdf, table_rows)

    pdf.output(pdf_path)
    print(f"Done: {pdf_path}")


if __name__ == "__main__":
    parse_md_to_pdf("midterm_report.md", "midterm_report.pdf")
