
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "shopping_trends_updated.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# 2. FUNCTION TO SAVE CHARTS
# --------------------------------------------------

def save_chart(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()
    print(f"Chart saved: {filename}")


# --------------------------------------------------
# 3. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 55)
    print("CODEALPHA DATA VISUALIZATION PROJECT")
    print("=" * 55)

    # Check dataset
    if not DATA_FILE.exists():
        print("\nERROR: Dataset not found!")
        print("Expected file:", DATA_FILE)
        print(
            "Place shopping_trends_updated.csv "
            "inside the data folder."
        )
        return

    # Load dataset
    df = pd.read_csv(DATA_FILE)

    if df.empty:
        print("ERROR: The dataset is empty.")
        return

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    print("\nDataset loaded successfully!")
    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))

    print("\nDataset columns:")
    print(df.columns.tolist())

    # Remove duplicate rows
    original_count = len(df)
    df = df.drop_duplicates().copy()
    duplicates_removed = original_count - len(df)

    # Remove extra spaces from text values
    for column in df.select_dtypes(include=["object"]).columns:
        df[column] = df[column].astype("string").str.strip()

    print("\nDuplicate rows removed:", duplicates_removed)

    # Identify columns without case sensitivity
    column_map = {
        column.lower(): column
        for column in df.columns
    }

    age_col = column_map.get("age")
    gender_col = column_map.get("gender")
    category_col = column_map.get("category")
    season_col = column_map.get("season")
    location_col = column_map.get("location")

    amount_col = next(
        (
            column for column in df.columns
            if "purchase amount" in column.lower()
        ),
        None
    )

    rating_col = next(
        (
            column for column in df.columns
            if "review rating" in column.lower()
        ),
        None
    )

    payment_col = next(
        (
            column for column in df.columns
            if "payment method" in column.lower()
        ),
        None
    )

    # Convert numeric columns safely
    if age_col:
        df[age_col] = pd.to_numeric(
            df[age_col], errors="coerce"
        )

    if amount_col:
        df[amount_col] = pd.to_numeric(
            df[amount_col].astype(str).str.replace(
                r"[$,]", "", regex=True
            ),
            errors="coerce"
        )

    if rating_col:
        df[rating_col] = pd.to_numeric(
            df[rating_col], errors="coerce"
        )

    # Save cleaned dataset
    df.to_csv(
        OUTPUT_DIR / "cleaned_shopping_trends.csv",
        index=False
    )

    # --------------------------------------------------
    # 4. CUSTOMER AGE DISTRIBUTION
    # --------------------------------------------------

    if age_col and df[age_col].notna().any():

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=age_col,
            bins=15,
            kde=True
        )

        plt.title("Customer Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Number of Customers")

        save_chart("age_distribution.png")

    # --------------------------------------------------
    # 5. PURCHASE AMOUNT DISTRIBUTION
    # --------------------------------------------------

    if amount_col and df[amount_col].notna().any():

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=amount_col,
            bins=15,
            kde=True
        )

        plt.title("Purchase Amount Distribution")
        plt.xlabel("Purchase Amount")
        plt.ylabel("Frequency")

        save_chart("purchase_amount_distribution.png")

        # Purchase amount box plot
        plt.figure(figsize=(8, 4))

        sns.boxplot(data=df, x=amount_col)

        plt.title("Purchase Amount Box Plot")
        plt.xlabel("Purchase Amount")

        save_chart("purchase_amount_boxplot.png")

    # --------------------------------------------------
    # 6. CATEGORY-WISE SPENDING
    # --------------------------------------------------

    if category_col and amount_col:

        category_data = df[
            [category_col, amount_col]
        ].dropna()

        if not category_data.empty:

            category_summary = (
                category_data.groupby(category_col)[amount_col]
                .sum()
                .sort_values(ascending=False)
            )

            plt.figure(figsize=(9, 5))

            sns.barplot(
                x=category_summary.index,
                y=category_summary.values
            )

            plt.title("Total Purchase Amount by Category")
            plt.xlabel("Category")
            plt.ylabel("Total Purchase Amount")
            plt.xticks(rotation=25, ha="right")

            save_chart("category_spending.png")

            category_summary.to_csv(
                OUTPUT_DIR / "category_spending_summary.csv",
                header=["Total Purchase Amount"]
            )

    # --------------------------------------------------
    # 7. GENDER DISTRIBUTION
    # --------------------------------------------------

    if gender_col:

        gender_counts = df[gender_col].value_counts()

        if not gender_counts.empty:

            plt.figure(figsize=(7, 5))

            sns.barplot(
                x=gender_counts.index,
                y=gender_counts.values
            )

            plt.title("Customer Distribution by Gender")
            plt.xlabel("Gender")
            plt.ylabel("Number of Customers")

            save_chart("gender_distribution.png")

    # --------------------------------------------------
    # 8. PAYMENT METHOD ANALYSIS
    # --------------------------------------------------

    if payment_col:

        payment_counts = df[payment_col].value_counts()

        if not payment_counts.empty:

            plt.figure(figsize=(9, 5))

            sns.barplot(
                x=payment_counts.index,
                y=payment_counts.values
            )

            plt.title("Most Used Payment Methods")
            plt.xlabel("Payment Method")
            plt.ylabel("Number of Purchases")
            plt.xticks(rotation=25, ha="right")

            save_chart("payment_methods.png")

    # --------------------------------------------------
    # 9. SEASONAL PURCHASE ANALYSIS
    # --------------------------------------------------

    if season_col:

        season_counts = df[season_col].value_counts()

        if not season_counts.empty:

            plt.figure(figsize=(8, 5))

            sns.barplot(
                x=season_counts.index,
                y=season_counts.values
            )

            plt.title("Purchases by Season")
            plt.xlabel("Season")
            plt.ylabel("Number of Purchases")

            save_chart("seasonal_purchases.png")

    # --------------------------------------------------
    # 10. CUSTOMER REVIEW RATINGS
    # --------------------------------------------------

    if rating_col and df[rating_col].notna().any():

        plt.figure(figsize=(8, 5))

        sns.histplot(
            data=df,
            x=rating_col,
            bins=10
        )

        plt.title("Customer Review Ratings")
        plt.xlabel("Review Rating")
        plt.ylabel("Frequency")

        save_chart("review_ratings.png")

    # --------------------------------------------------
    # 11. TOP 10 LOCATIONS
    # --------------------------------------------------

    if location_col:

        top_locations = df[location_col].value_counts().head(10)

        if not top_locations.empty:

            plt.figure(figsize=(9, 5))

            sns.barplot(
                x=top_locations.values,
                y=top_locations.index
            )

            plt.title("Top Locations by Number of Purchases")
            plt.xlabel("Number of Purchases")
            plt.ylabel("Location")

            save_chart("top_locations.png")

    # --------------------------------------------------
    # 12. CORRELATION HEATMAP
    # --------------------------------------------------

    numeric_data = df.select_dtypes(include="number")

    if numeric_data.shape[1] >= 2:

        correlation = numeric_data.corr()

        plt.figure(figsize=(10, 7))

        sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title("Correlation Heatmap")

        save_chart("correlation_heatmap.png")

    # --------------------------------------------------
    # 13. GENERATE SUMMARY REPORT
    # --------------------------------------------------

    report = [
        "CODEALPHA DATA VISUALIZATION REPORT",
        "=" * 45,
        f"Total rows after cleaning: {len(df)}",
        f"Total columns: {len(df.columns)}",
        f"Duplicate rows removed: {duplicates_removed}",
        "",
        "Missing Values by Column:",
        df.isna().sum().to_string(),
    ]

    if amount_col and df[amount_col].notna().any():

        report.extend([
            "",
            "Purchase Amount Statistics:",
            df[amount_col].describe().round(2).to_string()
        ])

    if category_col and amount_col:

        category_totals = (
            df.groupby(category_col)[amount_col]
            .sum()
            .sort_values(ascending=False)
        )

        report.extend([
            "",
            "Total Purchase Amount by Category:",
            category_totals.round(2).to_string()
        ])

    report_file = OUTPUT_DIR / "visualization_report.txt"

    report_file.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 14. COMPLETION MESSAGE
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("=" * 55)
    print("Cleaned dataset and charts saved in:")
    print(OUTPUT_DIR)
    print("Summary report: visualization_report.txt")


# Run the project
if __name__ == "__main__":
    main()
