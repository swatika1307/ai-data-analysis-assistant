import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Ask Your Data - AI Data Analyst", layout="wide")

# Title
st.title("📊 Ask Questions About Your Data")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

def generate_quick_insights(df):
    insights = []

    # Missing values insight
    missing = df.isnull().sum()
    if missing.sum() > 0:
        top_missing_col = missing.idxmax()
        top_missing_val = missing.max()
        if top_missing_val > 0:
            insights.append(f"'{top_missing_col}' has the highest missing values ({top_missing_val}).")
    else:
        insights.append("The dataset has no missing values.")

    # Better numeric column selection
    numeric_cols = [col for col in df.select_dtypes(include="number").columns if "id" not in col.lower()]
    preferred_avg_keywords = ["sales", "price", "amount", "revenue", "profit", "quantity", "rating", "score", "salary"]

    important_numeric = None
    for keyword in preferred_avg_keywords:
        for col in numeric_cols:
            if keyword in col.lower():
                important_numeric = col
                break
        if important_numeric:
            break

    if not important_numeric and numeric_cols:
        important_numeric = numeric_cols[0]

    if important_numeric:
        insights.append(
            f"The average of '{important_numeric}' is {df[important_numeric].mean():.2f}."
        )

    # Better categorical insight
    categorical_cols = [col for col in df.select_dtypes(include=["object", "category"]).columns if "id" not in col.lower()]
    useful_cats = [col for col in categorical_cols if 2 <= df[col].nunique() <= 15]

    if useful_cats:
        preferred_cat_keywords = ["category", "type", "region", "department", "segment", "size", "genre"]
        cat_col = None

        for keyword in preferred_cat_keywords:
            for col in useful_cats:
                if keyword in col.lower():
                    cat_col = col
                    break
            if cat_col:
                break

        if not cat_col:
            cat_col = useful_cats[0]

        top_cat = df[cat_col].value_counts().idxmax()
        top_cat_count = df[cat_col].value_counts().max()
        insights.append(
            f"The most frequent value in '{cat_col}' is '{top_cat}' ({top_cat_count} records)."
        )

    # Strongest correlation insight
    numeric_df = df.select_dtypes(include="number")
    numeric_df = numeric_df[[col for col in numeric_df.columns if "id" not in col.lower()]]

    if numeric_df.shape[1] >= 2:
        corr_matrix = numeric_df.corr().abs()

        strongest_pair = None
        strongest_value = 0

        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_value = corr_matrix.iloc[i, j]
                if pd.notna(corr_value) and corr_value > strongest_value:
                    strongest_value = corr_value
                    strongest_pair = (cols[i], cols[j])

        if strongest_pair and strongest_value > 0:
            insights.append(
                f"The strongest correlation is between '{strongest_pair[0]}' and '{strongest_pair[1]}' ({strongest_value:.2f})."
            )

    return insights

def generate_advanced_insights(df):
    insights = []

    numeric_cols = [
        col for col in df.select_dtypes(include="number").columns
        if "id" not in col.lower()
    ]

    categorical_cols = [
        col for col in df.select_dtypes(include=["object", "category"]).columns
        if "id" not in col.lower()
    ]

    # Missing value analysis
    missing_percent = (df.isnull().sum() / len(df)) * 100

    for col, percent in missing_percent.items():
        if percent > 30:
            insights.append(
                f"'{col}' has a high missing value rate ({percent:.1f}%), which may affect analysis quality."
            )

    # Numeric variance analysis
    for col in numeric_cols:
        if df[col].std() > df[col].mean():
            insights.append(
                f"'{col}' shows high variability, indicating inconsistent patterns or possible outliers."
            )

    # Strong correlation analysis
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]

                if abs(corr_value) > 0.7:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]

                    insights.append(
                        f"'{col1}' and '{col2}' are strongly correlated ({corr_value:.2f}), suggesting a possible relationship."
                    )

    # Dominant category detection
    for col in categorical_cols:
        if df[col].nunique() <= 15:
            top_ratio = df[col].value_counts(normalize=True).max()

            if top_ratio > 0.6:
                top_value = df[col].value_counts().idxmax()

                insights.append(
                    f"'{top_value}' dominates the '{col}' column, representing {(top_ratio * 100):.1f}% of records."
                )

    # Outlier hint
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        outliers = df[
            (df[col] < q1 - 1.5 * iqr) |
            (df[col] > q3 + 1.5 * iqr)
        ]

        if len(outliers) > 0:
            insights.append(
                f"'{col}' may contain outliers ({len(outliers)} unusual values detected)."
            )

    if not insights:
        insights.append("No major anomalies or unusual patterns were detected.")

    return insights

def calculate_dataset_health_score(df):
    score = 100
    issues = []

    total_cells = df.shape[0] * df.shape[1]

    # Missing values penalty
    missing_values = df.isnull().sum().sum()
    missing_percent = (missing_values / total_cells) * 100

    if missing_percent > 0:
        penalty = min(30, missing_percent * 0.5)
        score -= penalty
        issues.append(
            f"Missing values affect {missing_percent:.1f}% of the dataset."
        )

    # Duplicate rows penalty
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        duplicate_percent = (duplicate_count / len(df)) * 100
        penalty = min(20, duplicate_percent * 0.5)

        score -= penalty

        issues.append(
            f"{duplicate_count} duplicate rows detected."
        )

    # High cardinality categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)

        if unique_ratio > 0.9:
            score -= 2

            issues.append(
                f"'{col}' contains many unique values, which may increase data complexity."
            )

    # Too many missing values in a single column
    for col in df.columns:
        col_missing = (df[col].isnull().sum() / len(df)) * 100

        if col_missing > 50:
            score -= 5

            issues.append(
                f"'{col}' has more than 50% missing values."
            )

    # Prevent negative score
    score = max(0, round(score))

    return score, issues

def format_result_table(df_result):
    """
    Clean result tables for display.
    """
    formatted_df = df_result.copy()

    for col in formatted_df.select_dtypes(include=["float", "int"]).columns:
        formatted_df[col] = formatted_df[col].round(2)

    return formatted_df

def answer_data_question(question, df):
    """
    Simple rule-based AI analyst for dataset questions.
    Returns either:
    - a string for normal answers
    - a DataFrame for grouped/table results
    """
    question = question.lower().strip()

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    all_columns = df.columns.tolist()

    def matches_column(col_name, question_text):
        col_clean = col_name.lower().replace("_", " ")
        return col_name.lower() in question_text or col_clean in question_text

    # Basic dataset info
    if "rows" in question or "how many records" in question or "how many entries" in question:
        return f"The dataset has {df.shape[0]} rows."

    if "columns" in question or "how many columns" in question:
        return f"The dataset has {df.shape[1]} columns."

    if "column names" in question or "columns names" in question:
        return "Columns are: " + ", ".join(df.columns)

    if "missing" in question or "null" in question:
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Column", "Missing Values"]
        return missing

    if "data types" in question or "datatype" in question or "dtypes" in question:
        dtypes_df = df.dtypes.astype(str).reset_index()
        dtypes_df.columns = ["Column", "Data Type"]
        return dtypes_df

    if "summary" in question or "statistics" in question or "describe" in question:
        summary_df = df.describe(include="all").fillna("").reset_index()
        return summary_df

    # Group by + sum
    if ("total" in question or "sum" in question) and "by" in question:
        value_col = None
        group_col = None

        for col in numeric_columns:
            if matches_column(col, question):
                value_col = col
                break

        for col in all_columns:
            if matches_column(col, question) and col != value_col:
                group_col = col
                break

        if value_col and group_col:
            result = (
                df.groupby(group_col)[value_col]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )
            result.columns = [group_col, f"Total {value_col}"]
            return result.round(2)

    # Group by + average
    if ("average" in question or "mean" in question) and "by" in question:
        value_col = None
        group_col = None

        for col in numeric_columns:
            if matches_column(col, question):
                value_col = col
                break

        for col in all_columns:
            if matches_column(col, question) and col != value_col:
                group_col = col
                break

        if value_col and group_col:
            result = (
                df.groupby(group_col)[value_col]
                .mean()
                .sort_values(ascending=False)
                .reset_index()
            )
            result.columns = [group_col, f"Average {value_col}"]
            return result.round(2)

    # Group by + count
    if ("count" in question or "how many" in question) and "by" in question:
        group_col = None

        for col in all_columns:
            if matches_column(col, question):
                group_col = col
                break

        if group_col:
            result = df[group_col].value_counts().reset_index()
            result.columns = [group_col, "Count"]
            return result

    # Mean / average
    if "average" in question or "mean" in question:
        for col in numeric_columns:
            if matches_column(col, question):
                return f"The average of '{col}' is {df[col].mean():.2f}."
        return "Please mention a numeric column name to calculate the average."

    # Maximum
    if "maximum" in question or "max" in question or "highest" in question:
        for col in numeric_columns:
            if matches_column(col, question):
                return f"The maximum value in '{col}' is {df[col].max():.2f}."
        return "Please mention a numeric column name to find the maximum value."

    # Minimum
    if "minimum" in question or "min" in question or "lowest" in question:
        for col in numeric_columns:
            if matches_column(col, question):
                return f"The minimum value in '{col}' is {df[col].min():.2f}."
        return "Please mention a numeric column name to find the minimum value."

    # Sum / total
    if "sum" in question or "total" in question:
        for col in numeric_columns:
            if matches_column(col, question):
                return f"The total of '{col}' is {df[col].sum():.2f}."
        return "Please mention a numeric column name to calculate the total."

    # Unique values
    if "unique" in question:
        for col in all_columns:
            if matches_column(col, question):
                return f"Column '{col}' has {df[col].nunique()} unique values."
        return "Please mention a column name to check unique values."

    # Top values
    if "top" in question or "most common" in question or "frequent" in question:
        for col in all_columns:
            if matches_column(col, question):
                top_values = df[col].value_counts().head(5).reset_index()
                top_values.columns = [col, "Count"]
                return top_values
        return "Please mention a column name to check top values."

    return (
        "Sorry, I could not understand that question yet.\n\n"
        "Try questions like:\n"
        "- How many rows are there?\n"
        "- What are the column names?\n"
        "- How many missing values are there?\n"
        "- What is the average of sales?\n"
        "- What is the maximum of profit?\n"
        "- What is the total sales by region?\n"
        "- What is the average salary by department?\n"
        "- Count by category?"
    )
    
if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")
    # -----------------------------
    # Dataset Health Score
    # -----------------------------
    score, issues = calculate_dataset_health_score(df)

    st.subheader("🩺 Dataset Health Score")

    # Health status label
    if score >= 85:
        health_status = "Excellent"
        health_color = "🟢"

    elif score >= 70:
        health_status = "Good"
        health_color = "🟡"

    else:
        health_status = "Poor"
        health_color = "🔴"

    # Score card
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Health Score", f"{score}/100")

    with col2:
        st.metric("Dataset Status", f"{health_color} {health_status}")

    # Dataset issues
    if issues:
        st.write("### Detected Issues")

        for issue in issues[:5]:
            st.warning(issue)

    else:
        st.success("No major dataset quality issues detected.")
        
    # Dataset overview
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    # Preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), width="stretch")

    # Column names
    st.subheader("Column Names")
    st.write(", ".join(df.columns))

    # Data types
    st.subheader("Data Types")
    dtypes_df = df.dtypes.astype(str).reset_index()
    dtypes_df.columns = ["Column", "Data Type"]
    st.dataframe(dtypes_df, width="stretch")

    # Missing values
    st.subheader("Missing Values")
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ["Column", "Missing Values"]
    st.dataframe(missing_df, width="stretch")

    # Descriptive statistics
    st.subheader("Descriptive Statistics")
    stats_df = df.describe(include="all").fillna("")
    stats_df = stats_df.round(1)
    st.dataframe(stats_df.astype(str), width="stretch")

    # Data Visualization
    st.subheader("Data Visualization")

    all_numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    all_categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

    # Exclude ID columns
    numeric_columns = [col for col in all_numeric_columns if "id" not in col.lower()]
    categorical_columns = [col for col in all_categorical_columns if "id" not in col.lower()]

    # Fallback
    if not numeric_columns:
        numeric_columns = all_numeric_columns
    if not categorical_columns:
        categorical_columns = all_categorical_columns

    # Keep only useful categorical columns for bar chart
    bar_chart_columns = [col for col in categorical_columns if df[col].nunique() <= 10]

    # Fallback if none match
    if not bar_chart_columns:
        bar_chart_columns = categorical_columns

    col1, col2 = st.columns(2)

    with col1:
        chart_type = st.selectbox("Select chart type", ["Histogram", "Bar Chart"])

    with col2:
        if chart_type == "Histogram":
            selected_column = st.selectbox("Select a numeric column", numeric_columns) if numeric_columns else None
        else:
            selected_column = st.selectbox("Select a categorical column", bar_chart_columns) if bar_chart_columns else None

    left_space, center_chart, right_space = st.columns([1, 2, 1])

    with center_chart:
        if chart_type == "Histogram":
            if selected_column:
                fig, ax = plt.subplots(figsize=(5, 3))
                ax.hist(df[selected_column].dropna(), bins=10)
                ax.set_title(f"Histogram of {selected_column}", fontsize=12)
                ax.set_xlabel(selected_column, fontsize=10)
                ax.set_ylabel("Frequency", fontsize=10)
                st.pyplot(fig, width="content")
            else:
                st.warning("No numeric columns available for histogram.")

        elif chart_type == "Bar Chart":
            if selected_column:
                value_counts = df[selected_column].value_counts().head(10)

                fig, ax = plt.subplots(figsize=(5, 3))
                ax.bar(value_counts.index.astype(str), value_counts.values)
                ax.set_title(f"Bar Chart of {selected_column}", fontsize=12)
                ax.set_xlabel(selected_column, fontsize=10)
                ax.set_ylabel("Count", fontsize=10)
                plt.xticks(rotation=45, fontsize=9, ha="right")
                plt.tight_layout()
                st.pyplot(fig, width="content")
            else:
                st.warning("No suitable categorical columns available for bar chart.")

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr()
        fig, ax = plt.subplots(figsize=(4, 2.8))
        cax = ax.matshow(corr)
        fig.colorbar(cax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="left", fontsize=9)
        ax.set_yticklabels(corr.columns, fontsize=9)
        ax.set_title("Correlation Heatmap", fontsize=12, pad=12)
        plt.tight_layout()
        st.pyplot(fig, width="content")
    else:
        st.warning("Not enough numeric columns for correlation heatmap.")

    # -----------------------------
    # AI Data Analyst
    # -----------------------------
    st.subheader("🤖 Ask Questions About Your Data")

    if "question_text" not in st.session_state:
        st.session_state.question_text = ""

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # Remove ID-like columns
    filtered_numeric_cols = [col for col in numeric_cols if "id" not in col.lower()]
    filtered_categorical_cols = [col for col in categorical_cols if "id" not in col.lower()]

    # Preferred numeric keywords
    preferred_numeric_keywords = [
        "sales", "price", "amount", "revenue", "profit",
        "quantity", "score", "rating", "salary", "age", "year"
    ]

    # Preferred categorical keywords
    preferred_group_keywords = [
        "category", "type", "region", "department", "segment",
        "city", "country", "state", "gender", "size", "name", "title", "genre"
    ]

    sample_numeric = None
    sample_group = None

    # Pick smart numeric column
    for keyword in preferred_numeric_keywords:
        for col in filtered_numeric_cols:
            if keyword in col.lower():
                sample_numeric = col
                break
        if sample_numeric:
            break

    # Fallback numeric column
    if not sample_numeric and filtered_numeric_cols:
        sample_numeric = filtered_numeric_cols[0]
    elif not sample_numeric and numeric_cols:
        sample_numeric = numeric_cols[0]

    # Pick smart group column
    for keyword in preferred_group_keywords:
        for col in filtered_categorical_cols:
            if keyword in col.lower():
                sample_group = col
                break
        if sample_group:
            break

    # Fallback group column
    if not sample_group and filtered_categorical_cols:
        sample_group = filtered_categorical_cols[0]
    elif not sample_group and categorical_cols:
        sample_group = categorical_cols[0]

    # Better column for average questions
    average_numeric = None
    average_keywords = ["rating", "score", "price", "amount", "salary", "age", "year"]

    for keyword in average_keywords:
        for col in filtered_numeric_cols:
            if keyword in col.lower():
                average_numeric = col
                break
        if average_numeric:
            break

    if not average_numeric:
        average_numeric = sample_numeric

    # Better column for total/grouped sum questions
    total_numeric = None
    total_keywords = ["sales", "revenue", "amount", "profit", "quantity", "total", "price"]
    excluded_total_columns = ["age", "year", "rating", "score"]

    for keyword in total_keywords:
        for col in filtered_numeric_cols:
            # Skip unsuitable total columns
            if col.lower() in excluded_total_columns:
                continue
            if keyword in col.lower():
                total_numeric = col
                break
        if total_numeric:
            break

    if not total_numeric:
        for col in filtered_numeric_cols:
            if col.lower() not in excluded_total_columns:
                total_numeric = col
                break

    # Smarter grouping column
    group_candidates = []
    for col in filtered_categorical_cols:
        unique_count = df[col].nunique()
        if 2 <= unique_count <= 15:
            group_candidates.append(col)

    better_group = None
    for keyword in preferred_group_keywords:
        for col in group_candidates:
            if keyword in col.lower():
                better_group = col
                break
        if better_group:
            break

    if not better_group and group_candidates:
        better_group = group_candidates[0]

    if not better_group:
        better_group = sample_group

    st.write("Try a sample question:")

    col1, col2 = st.columns(2)

    with col1:
        if average_numeric and st.button(f"Average of {average_numeric}"):
            st.session_state.question_text = f"What is the average of {average_numeric}?"

        if total_numeric and better_group and st.button(f"Total {total_numeric} by {better_group}"):
            st.session_state.question_text = f"What is the total {total_numeric} by {better_group}?"

    with col2:
        if better_group and st.button(f"Count by {better_group}"):
            st.session_state.question_text = f"Count by {better_group}"

        if st.button("Missing values"):
            st.session_state.question_text = "How many missing values are there?"

    user_question = st.text_input(
        "Ask a question about your dataset:",
        value=st.session_state.question_text,
        placeholder="Example: What is the average of sales?"
    )

    st.session_state.question_text = user_question

    if st.button("Get Answer"):
        if user_question.strip():
            with st.spinner("Analyzing your data..."):
                answer = answer_data_question(user_question, df)

            st.markdown("## Answer")

            if isinstance(answer, pd.DataFrame):
                formatted_answer = format_result_table(answer)

                st.dataframe(formatted_answer, width="stretch", hide_index=True)

                csv_data = formatted_answer.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="Download Answer as CSV",
                    data=csv_data,
                    file_name="answer_table.csv",
                    mime="text/csv"
                )

                # Auto chart only for grouped answers with 2 columns
                if formatted_answer.shape[1] == 2:
                    x_col = formatted_answer.columns[0]
                    y_col = formatted_answer.columns[1]

                    if pd.api.types.is_numeric_dtype(formatted_answer[y_col]):
                        # Small insight line
                        top_label = formatted_answer.iloc[0][x_col]
                        top_value = formatted_answer.iloc[0][y_col]
                        st.info(f"Top {x_col}: {top_label} ({top_value})")

                        # Show chart only if category count is reasonable
                        chart_data = formatted_answer.copy()

                        # If too many rows, use only top 10 for chart
                        if len(chart_data) > 10:
                            chart_data = chart_data.head(10)
                            st.caption("Showing top 10 categories in chart for better readability.")

                        left_space, center_chart, right_space = st.columns([1, 2, 1])

                        with center_chart:
                            fig, ax = plt.subplots(figsize=(5, 3))
                            ax.bar(chart_data[x_col].astype(str), chart_data[y_col])

                            ax.set_title(f"{y_col} by {x_col}", fontsize=11, pad=10)
                            ax.set_xlabel(x_col, fontsize=10)
                            ax.set_ylabel(y_col, fontsize=10)
                            plt.xticks(rotation=45, ha="right", fontsize=9)
                            plt.yticks(fontsize=9)
                            plt.tight_layout()

                            st.pyplot(fig, width="content")
            else:
                st.success(answer)
        else:
            st.warning("Please enter a question first.")

    #st.subheader("📌 Insights Center")

if "latest_insights" not in st.session_state:
    st.session_state.latest_insights = []

if "advanced_insights" not in st.session_state:
    st.session_state.advanced_insights = []

if "full_report" not in st.session_state:
    st.session_state.full_report = ""

col1, col2, col3 = st.columns(3)

with col1:
    quick_button = st.button("⚡ Generate Quick Insights")

with col2:
    advanced_button = st.button("🧠 Generate Advanced Insights")

with col3:
    report_button = st.button("📄 Generate Full Report")

if quick_button:
    insights = generate_quick_insights(df)
    st.session_state.latest_insights = insights

if advanced_button:
    advanced = generate_advanced_insights(df)
    st.session_state.advanced_insights = advanced

#st.markdown("---")

if st.session_state.latest_insights:
    st.subheader("⚡ Quick Insights")
    for i, insight in enumerate(st.session_state.latest_insights, start=1):
        st.info(f"{i}. {insight}")

    insights_text = "Quick Insights Report\n\n"
    for i, insight in enumerate(st.session_state.latest_insights, start=1):
        insights_text += f"{i}. {insight}\n"

    st.download_button(
        label="Download Insights Report",
        data=insights_text,
        file_name="quick_insights_report.txt",
        mime="text/plain"
    )

#st.markdown("---")

if st.session_state.advanced_insights:
    st.subheader("🧠 Advanced AI-Style Insights")
    for i, insight in enumerate(st.session_state.advanced_insights, start=1):
        st.warning(f"{i}. {insight}")

    advanced_report = "Advanced AI-Style Insights Report\n\n"

    for i, insight in enumerate(st.session_state.advanced_insights, start=1):
        advanced_report += f"{i}. {insight}\n"

    st.download_button(
        label="Download Advanced Insights Report",
        data=advanced_report,
        file_name="advanced_insights_report.txt",
        mime="text/plain"
    )

# -----------------------------
# Full Analysis Report
# -----------------------------
#st.markdown("---")
#st.subheader("📄 Full Analysis Report")

if report_button:
    st.subheader("📄 Full Analysis Report")
    report = ""

    # Dataset overview
    report += "DATASET OVERVIEW\n"
    report += "-" * 40 + "\n"
    report += f"Rows: {df.shape[0]}\n"
    report += f"Columns: {df.shape[1]}\n\n"

    # Health score
    report += "DATASET HEALTH SCORE\n"
    report += "-" * 40 + "\n"
    report += f"Health Score: {score}/100\n"
    report += f"Status: {health_status}\n\n"

    if issues:
        report += "Detected Issues:\n"
        for issue in issues:
            report += f"- {issue}\n"

    report += "\n"

    # Quick insights
    report += "QUICK INSIGHTS\n"
    report += "-" * 40 + "\n"

    quick_insights = generate_quick_insights(df)

    for insight in quick_insights:
        report += f"- {insight}\n"

    report += "\n"

    # Advanced insights
    report += "ADVANCED INSIGHTS\n"
    report += "-" * 40 + "\n"

    advanced_insights = generate_advanced_insights(df)

    for insight in advanced_insights:
        report += f"- {insight}\n"

    report += "\n"

    # Numeric summary
    report += "NUMERIC SUMMARY\n"
    report += "-" * 40 + "\n"

    numeric_summary = df.describe().round(2).to_string()

    report += numeric_summary

    st.session_state.full_report = report

    if st.session_state.full_report:

        st.text_area(
            "Generated Report",
            st.session_state.full_report,
            height=400
        )

        st.download_button(
            label="Download Full Report",
            data=st.session_state.full_report,
            file_name="full_analysis_report.txt",
            mime="text/plain"
        )
