"""
-------------------------------------------------------------------------------
Script: Kruskal-Wallis with Dunn Post-hoc Analysis across Multiple Excel Sheets
-------------------------------------------------------------------------------
Author: Rafael Luiz Martins Monteiro

Description:
-------------
This script automates the process of running Kruskal-Wallis non-parametric tests
followed by Dunn's post-hoc pairwise comparisons (with Bonferroni correction)
across multiple sheets of an Excel file.

Pipeline Steps:
---------------
1. Iterate through all sheets of the input Excel file.
2. For each sheet:
    a. Test for normality (Shapiro-Wilk) per group.
    b. Test for homogeneity of variances (Levene's Test).
    c. Perform Kruskal-Wallis test for global differences.
    d. Perform Dunn's post-hoc pairwise comparisons with Bonferroni correction.
3. Save the statistical results into a structured Excel file, with a separate
   sheet for normality, homogeneity, Kruskal-Wallis, and post-hoc results for
   each original sheet.

Statistical Tests Applied:
--------------------------
- Shapiro-Wilk Test (Normality per group)
- Levene's Test (Homogeneity of variances)
- Kruskal-Wallis Test (Non-parametric global comparison)
- Dunn's Test with Bonferroni Correction (Post-hoc pairwise comparison)

-------------------------------------------------------------------------------
"""

import pandas as pd
from scipy import stats
import numpy as np
import itertools
from pathlib import Path


# -------------------------------
# Dunn's Test with Bonferroni Correction
# -------------------------------
def dunn_test(df, group_col, val_col):
    """
    Perform Dunn's post-hoc pairwise comparisons with Bonferroni correction.
    """
    unique_groups = df[group_col].unique()
    comparisons = list(itertools.combinations(unique_groups, 2))
    p_values = []

    # Perform Mann-Whitney U test for all pairwise group combinations
    for group1, group2 in comparisons:
        data1 = df[df[group_col] == group1][val_col]
        data2 = df[df[group_col] == group2][val_col]
        stat, p = stats.mannwhitneyu(data1, data2, alternative="two-sided")
        p_values.append(p)

    # Apply Bonferroni correction
    p_adjusted = np.array(p_values) * len(comparisons)
    p_adjusted = np.clip(p_adjusted, 0, 1)

    # Prepare result DataFrame
    result_df = pd.DataFrame(comparisons, columns=["Group1", "Group2"])
    result_df["p-value"] = p_values
    result_df["p-adjusted"] = p_adjusted

    return result_df


# -------------------------------
# Run Kruskal-Wallis with Normality, Homogeneity, and Post-hoc Dunn Tests
# -------------------------------
def run_kruskal_with_tests(df):
    """
    Perform Shapiro-Wilk, Levene, Kruskal-Wallis, and Dunn's post-hoc tests on the input DataFrame.
    """
    results = {}

    # Normality Test (Shapiro-Wilk) per group/column
    normality_results = {}
    for col in df.columns:
        stat, p = stats.shapiro(df[col])
        normality_results[col] = {"statistic": stat, "p_value": p}
    results["normality"] = normality_results

    # Homogeneity of variances (Levene's Test)
    stat, p = stats.levene(*[df[col] for col in df.columns])
    results["homogeneity"] = {"statistic": stat, "p_value": p}

    # Kruskal-Wallis Global Test
    kruskal_stat, kruskal_p = stats.kruskal(*[df[col] for col in df.columns])
    results["kruskal"] = {"H-statistic": kruskal_stat, "p_value": kruskal_p}

    # Post-hoc Dunn Test (Data needs to be melted into long format)
    melted_df = df.melt(var_name="group", value_name="value")
    posthoc_df = dunn_test(melted_df, "group", "value")
    results["posthoc"] = posthoc_df

    return results


# -------------------------------
# Main Execution
# -------------------------------
if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    file_path = project_root / "results_CV" / "sg_resultados_3.xlsx"

    # Load all sheets
    xls = pd.ExcelFile(file_path)
    all_sheets_results = {}

    # Iterate over all sheets to apply the statistical tests
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name).dropna()
        all_sheets_results[sheet_name] = run_kruskal_with_tests(df)

    # -------------------------------
    # Save Statistical Results to Excel
    # -------------------------------

    # Define output path for the statistical summary Excel file
    output_path = file_path.with_stem(file_path.stem + "_kruskal_stats")

    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, results in all_sheets_results.items():
            # Save Normality Test Results
            normality_df = pd.DataFrame(results["normality"]).transpose()
            normality_df.to_excel(writer, sheet_name=f"{sheet_name}_normality")

            # Save Homogeneity Test Result
            homogeneity_df = pd.DataFrame([results["homogeneity"]])
            homogeneity_df.to_excel(
                writer, sheet_name=f"{sheet_name}_homogeneity", index=False
            )

            # Save Kruskal-Wallis Test Result
            kruskal_df = pd.DataFrame([results["kruskal"]])
            kruskal_df.to_excel(writer, sheet_name=f"{sheet_name}_kruskal", index=False)

            # Save Post-hoc Dunn Test Results
            posthoc_df = results["posthoc"]
            posthoc_df.to_excel(writer, sheet_name=f"{sheet_name}_posthoc", index=False)

    print(f"[OK] Kruskal-Wallis and Dunn's post-hoc results saved to {output_path}")
