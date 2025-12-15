import pandas as pd
from collections.abc import Iterable

def dataframe_to_markdown_table(df: pd.DataFrame, output_file: str):
    """
    Convert a DataFrame into a markdown file with nested table structure.

    Rules:
    - Scalar columns (numbers/strings) -> main table columns
    - Iterable columns (lists/arrays) -> nested table
    - Dict columns -> nested table with key/value pairs
    - Column order is respected exactly as in the DataFrame
    """

    # Header
    header = """<table align="center; adding: 0; line-height: 1.1">\n<tr>\n"""
    for col in df.columns:
        header += f'  <th style="font-size: 1.2em;">{col}</th>\n'
    header += "</tr>\n"

    # Rows
    rows = []
    for _, row in df.iterrows():
        row_md = " <tr>\n"

        for col in df.columns:
            val = row[col]

            # Dict -> nested table
            if isinstance(val, dict):
                row_md += '  <td>\n'
                row_md += '   <table style="font-family: monospace; border-collapse: collapse; adding: 0; line-height: 1.1">\n'
                for k, v in val.items():
                    if isinstance(v, Iterable) and not isinstance(v, str):
                        formatted = "[" + " ".join(f"{x:.3f}" for x in v) + "]"
                    else:
                        formatted = f"{v:.3f}" if isinstance(v, float) else str(v)
                    row_md += f"    <tr><td>{k}</td><td>{formatted}</td></tr>\n"
                row_md += "   </table>\n"
                row_md += "  </td>\n"

            # Iterable (list/array) -> nested table
            elif isinstance(val, Iterable) and not isinstance(val, str):
                row_md += '  <td>\n'
                row_md += '   <table style="font-family: monospace; border-collapse: collapse; adding: 0; line-height: 1.1">\n'
                formatted = "[" + " ".join(f"{x:.3f}" for x in val) + "]"
                row_md += f"    <tr><td>{col}</td><td>{formatted}</td></tr>\n"
                row_md += "   </table>\n"
                row_md += "  </td>\n"

            # Scalar
            else:
                if isinstance(val, float):
                    row_md += f'  <td align="center" style="font-size: 1.2em;">{val:.3f}</td>\n'
                else:
                    row_md += f'  <td align="center" style="font-size: 1.2em;">{val}</td>\n'

        row_md += " </tr>\n"
        rows.append(row_md)

    footer = "</table>\n"

    markdown_content = header + "".join(rows) + footer
    with open(output_file, "w") as f:
        f.write(markdown_content)
    print(f"Markdown table saved to {output_file}")
