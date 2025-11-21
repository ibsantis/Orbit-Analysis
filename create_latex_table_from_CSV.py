import pandas as pd

# Load CSV file
csv_file = "medians_and_errors.csv"  # Replace with your file path
data = pd.read_csv(csv_file)

# Ensure the data has the correct structure
if data.shape[1] != 31:  # 1 name column + 10 main columns + 20 error columns
    raise ValueError("CSV file must have 31 columns: 1 name column + 10 main columns + 20 error columns.")

# Generate LaTeX rows, skipping the header row
latex_rows = []
for _, row in data.iterrows():
    galaxy_name = row[0]  # First column is the galaxy name
    formatted_row = [galaxy_name]  # Start the row with the galaxy name
    for i in range(1, 31, 3):  # Start after the name column and group by 3 columns
        main = row[i]
        error_lower = row[i + 1]
        error_upper = row[i + 2]
        formatted_value = f"${main}^{{{error_upper}}}_{{{error_lower}}}$"
        formatted_row.append(formatted_value)
    latex_rows.append(" & ".join(formatted_row) + " \\\\")

# Generate LaTeX table header using column names
header_row = " & ".join(data.columns[:1].tolist() + [f"Column {i}" for i in range(1, 11)]) + r" \\"

# Generate LaTeX table
latex_table = r"""
\begin{table}[ht]
\centering
\begin{tabular}{|l|""" + "c|" * 10 + r"""}
\hline
""" + header_row + r"""
\hline
""" + "\n".join(latex_rows) + r"""
\hline
\end{tabular}
\caption{Your caption here}
\label{tab:your_label}
\end{table}
"""

# Save to a .tex file
with open("table.tex", "w") as file:
    file.write(latex_table)

print("LaTeX table saved to table.tex")
