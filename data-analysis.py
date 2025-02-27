import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(page_title="Advanced Data Analysis App", layout="wide")

# Custom CSS to improve UI
st.markdown("""
<style>
    .main {
        padding: 1rem 2rem;
    }
    h1, h2, h3 {
        color: #1E88E5;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .plot-container {
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        padding: 10px;
        border-radius: 5px;
        background-color: white;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("📊 Comprehensive Data Analysis Tool")
st.markdown("Upload your Excel file and explore your data through various analyses and visualizations")

# File uploader
uploaded_file = st.file_uploader("Choose an Excel file (.xlsx)", type="xlsx")


# Main function
def main():
    if uploaded_file is not None:
        # Read the data
        try:
            df = pd.read_excel(uploaded_file)
            st.success("File successfully uploaded!")

            # Display tabs for different analyses
            tabs = st.tabs(["📋 Data Overview", "📊 Descriptive Statistics", "📈 Visualizations",
                            "🔥 Correlation Analysis", "📉 Advanced Analysis"])

            with tabs[0]:
                data_overview(df)

            with tabs[1]:
                descriptive_statistics(df)

            with tabs[2]:
                visualizations(df)

            with tabs[3]:
                correlation_analysis(df)

            with tabs[4]:
                advanced_analysis(df)

        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Please upload an Excel file to begin analysis.")

        # Sample data option
        if st.button("Use Sample Data"):
            # Generate sample data
            np.random.seed(42)
            sample_data = pd.DataFrame({
                'Age': np.random.normal(35, 10, 100).astype(int),
                'Income': np.random.normal(50000, 15000, 100),
                'Experience': np.random.normal(8, 5, 100),
                'Satisfaction': np.random.randint(1, 6, 100),
                'Department': np.random.choice(['Sales', 'Marketing', 'HR', 'IT', 'Finance'], 100),
                'Performance': np.random.normal(7, 2, 100)
            })

            # Save to a BytesIO object
            output = BytesIO()
            sample_data.to_excel(output, index=False)
            output.seek(0)

            # Use the sample data
            df = sample_data
            st.success("Using sample data!")

            # Display tabs for different analyses
            tabs = st.tabs(["📋 Data Overview", "📊 Descriptive Statistics", "📈 Visualizations",
                            "🔥 Correlation Analysis", "📉 Advanced Analysis"])

            with tabs[0]:
                data_overview(df)

            with tabs[1]:
                descriptive_statistics(df)

            with tabs[2]:
                visualizations(df)

            with tabs[3]:
                correlation_analysis(df)

            with tabs[4]:
                advanced_analysis(df)


def data_overview(df):
    st.header("Data Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Number of Rows:** {df.shape[0]}")
        st.write(f"**Number of Columns:** {df.shape[1]}")

    with col2:
        st.write(f"**Missing Values:** {df.isna().sum().sum()}")
        st.write(f"**Duplicated Rows:** {df.duplicated().sum()}")

    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10))

    # Column information
    st.subheader("Column Information")

    col_info = pd.DataFrame({
        'Column Name': df.columns,
        'Data Type': df.dtypes.values,
        'Non-Null Count': df.count().values,
        'Missing Values': df.isna().sum().values,
        'Unique Values': [df[col].nunique() for col in df.columns]
    })
    st.dataframe(col_info)

    # Data cleaning options
    st.subheader("Data Cleaning Options")

    if st.checkbox("Remove Duplicates"):
        df = df.drop_duplicates()
        st.write(f"Removed duplicates. New shape: {df.shape}")

    handle_missing = st.radio("Handle Missing Values",
                              ["No action", "Drop rows with missing values", "Fill numeric with mean",
                               "Fill categorical with mode"])

    if handle_missing == "Drop rows with missing values":
        df = df.dropna()
        st.write(f"Dropped rows with missing values. New shape: {df.shape}")
    elif handle_missing == "Fill numeric with mean":
        for col in df.select_dtypes(include=np.number).columns:
            df[col].fillna(df[col].mean(), inplace=True)
        st.write("Filled numeric missing values with mean.")
    elif handle_missing == "Fill categorical with mode":
        for col in df.select_dtypes(exclude=np.number).columns:
            if not df[col].empty:
                df[col].fillna(df[col].mode()[0], inplace=True)
        st.write("Filled categorical missing values with mode.")


def descriptive_statistics(df):
    st.header("Descriptive Statistics")

    # Select columns for analysis
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    if numeric_cols:
        st.subheader("Numeric Variables")

        # Multiselect for columns
        selected_numeric = st.multiselect("Select numeric columns for analysis",
                                          numeric_cols, default=numeric_cols[:min(5, len(numeric_cols))])

        if selected_numeric:
            # Display descriptive stats for selected columns
            stats_df = df[selected_numeric].describe().T
            stats_df['median'] = df[selected_numeric].median()
            stats_df['skew'] = df[selected_numeric].skew()
            stats_df['kurtosis'] = df[selected_numeric].kurtosis()
            stats_df['missing'] = df[selected_numeric].isna().sum()
            stats_df['missing_percent'] = (df[selected_numeric].isna().sum() / len(df)) * 100

            st.dataframe(stats_df)

            # Option to download statistics
            stats_csv = stats_df.to_csv(index=True)
            st.download_button(
                label="Download Statistics as CSV",
                data=stats_csv,
                file_name='descriptive_statistics.csv',
                mime='text/csv',
            )

    if categorical_cols:
        st.subheader("Categorical Variables")

        # Multiselect for categorical columns
        selected_categorical = st.multiselect("Select categorical columns for analysis",
                                              categorical_cols,
                                              default=categorical_cols[:min(5, len(categorical_cols))])

        if selected_categorical:
            for col in selected_categorical:
                st.write(f"**{col}**")

                # Get value counts and percentages
                value_counts = df[col].value_counts()
                value_percent = df[col].value_counts(normalize=True) * 100

                # Combine into a DataFrame
                cat_stats = pd.DataFrame({
                    'Count': value_counts,
                    'Percentage (%)': value_percent
                })

                st.dataframe(cat_stats)

                # Display simple bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                value_counts.plot(kind='bar', ax=ax)
                ax.set_title(f"Frequency Distribution - {col}")
                ax.set_ylabel("Count")
                ax.set_xlabel(col)
                plt.tight_layout()
                st.pyplot(fig)


def visualizations(df):
    st.header("Data Visualizations")

    # Get column types
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Visualization type selection
    viz_type = st.selectbox("Select Visualization Type",
                            ["Histogram", "Box Plot", "Scatter Plot", "Bar Chart",
                             "Line Plot", "Pie Chart", "Violin Plot", "Pair Plot"])

    if viz_type == "Histogram":
        st.subheader("Histogram")
        col = st.selectbox("Select a numeric column", numeric_cols)

        bins = st.slider("Number of bins", min_value=5, max_value=100, value=30)

        fig = px.histogram(df, x=col, nbins=bins, marginal="box",
                           title=f"Histogram of {col}")
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Box Plot":
        st.subheader("Box Plot")
        y_col = st.selectbox("Select a numeric column for Y-axis", numeric_cols)

        x_col = st.selectbox("Select a column for X-axis (Optional)",
                             ["None"] + categorical_cols)

        if x_col == "None":
            fig = px.box(df, y=y_col, title=f"Box Plot of {y_col}")
        else:
            fig = px.box(df, x=x_col, y=y_col,
                         title=f"Box Plot of {y_col} grouped by {x_col}")

        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Scatter Plot":
        st.subheader("Scatter Plot")

        col1, col2 = st.columns(2)

        with col1:
            x_col = st.selectbox("Select X-axis", numeric_cols)

        with col2:
            y_col = st.selectbox("Select Y-axis",
                                 [col for col in numeric_cols if col != x_col])

        color_col = st.selectbox("Color by (Optional)", ["None"] + df.columns.tolist())

        if color_col == "None":
            fig = px.scatter(df, x=x_col, y=y_col,
                             title=f"Scatter Plot: {y_col} vs {x_col}")
        else:
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                             title=f"Scatter Plot: {y_col} vs {x_col}, colored by {color_col}")

        # Add trendline if both columns are numeric
        if st.checkbox("Add Trend Line"):
            fig.update_traces(marker=dict(size=8))
            fig = px.scatter(df, x=x_col, y=y_col,
                             trendline="ols",
                             title=f"Scatter Plot with Trend Line: {y_col} vs {x_col}")

        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Bar Chart":
        st.subheader("Bar Chart")

        if not categorical_cols:
            st.warning("No categorical columns available for bar chart.")
            return

        x_col = st.selectbox("Select category for X-axis", categorical_cols)

        agg_option = st.radio("Select aggregation method",
                              ["Count", "Sum", "Mean", "Median"])

        y_col = None
        if agg_option != "Count":
            y_col = st.selectbox("Select numeric column for Y-axis", numeric_cols)

        # Create bar chart
        if agg_option == "Count":
            fig = px.bar(df[x_col].value_counts().reset_index(),
                         x='index', y=x_col,
                         labels={'index': x_col, x_col: 'Count'},
                         title=f"Count of {x_col}")
        elif agg_option == "Sum":
            agg_df = df.groupby(x_col)[y_col].sum().reset_index()
            fig = px.bar(agg_df, x=x_col, y=y_col,
                         title=f"Sum of {y_col} by {x_col}")
        elif agg_option == "Mean":
            agg_df = df.groupby(x_col)[y_col].mean().reset_index()
            fig = px.bar(agg_df, x=x_col, y=y_col,
                         title=f"Mean of {y_col} by {x_col}")
        else:  # Median
            agg_df = df.groupby(x_col)[y_col].median().reset_index()
            fig = px.bar(agg_df, x=x_col, y=y_col,
                         title=f"Median of {y_col} by {x_col}")

        fig.update_layout(xaxis_title=x_col)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Line Plot":
        st.subheader("Line Plot")

        if not numeric_cols:
            st.warning("No numeric columns available for line plot.")
            return

        # Check if datetime columns exist
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            # Try to convert to datetime
            elif df[col].dtypes == 'object':
                try:
                    pd.to_datetime(df[col])
                    datetime_cols.append(col)
                except:
                    pass

        if not datetime_cols:
            st.warning("No datetime columns detected. Creating a simple line plot.")

            x_col = st.selectbox("Select X-axis", numeric_cols)
            y_col = st.selectbox("Select Y-axis",
                                 [col for col in numeric_cols if col != x_col])

            # Sort by x-axis value for better line plot
            sorted_df = df.sort_values(by=x_col)

            fig = px.line(sorted_df, x=x_col, y=y_col,
                          title=f"Line Plot: {y_col} vs {x_col}")
        else:
            x_col = st.selectbox("Select datetime column for X-axis", datetime_cols)
            y_col = st.selectbox("Select numeric column for Y-axis", numeric_cols)

            # Ensure datetime format
            if not pd.api.types.is_datetime64_any_dtype(df[x_col]):
                df[x_col] = pd.to_datetime(df[x_col])

            # Group by time period if needed
            time_group = st.radio("Group by time period", ["None", "Day", "Week", "Month", "Year"])

            if time_group == "None":
                # Sort by date
                sorted_df = df.sort_values(by=x_col)
                fig = px.line(sorted_df, x=x_col, y=y_col,
                              title=f"Line Plot: {y_col} over time")
            else:
                # Group by selected time period
                if time_group == "Day":
                    grouped_df = df.groupby(df[x_col].dt.date)[y_col].mean().reset_index()
                elif time_group == "Week":
                    grouped_df = df.groupby(df[x_col].dt.isocalendar().week)[y_col].mean().reset_index()
                elif time_group == "Month":
                    grouped_df = df.groupby(df[x_col].dt.to_period('M').astype(str))[y_col].mean().reset_index()
                else:  # Year
                    grouped_df = df.groupby(df[x_col].dt.year)[y_col].mean().reset_index()

                fig = px.line(grouped_df, x=x_col, y=y_col,
                              title=f"Line Plot: Average {y_col} by {time_group}")

        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Pie Chart":
        st.subheader("Pie Chart")

        if not categorical_cols:
            st.warning("No categorical columns available for pie chart.")
            return

        cat_col = st.selectbox("Select category column", categorical_cols)

        agg_option = st.radio("Select aggregation method for pie chart",
                              ["Count", "Sum"])

        if agg_option == "Count":
            fig = px.pie(df, names=cat_col,
                         title=f"Distribution of {cat_col}")
        else:  # Sum
            value_col = st.selectbox("Select numeric column to sum", numeric_cols)
            fig = px.pie(df, names=cat_col, values=value_col,
                         title=f"Sum of {value_col} by {cat_col}")

        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Violin Plot":
        st.subheader("Violin Plot")

        if not numeric_cols:
            st.warning("No numeric columns available for violin plot.")
            return

        y_col = st.selectbox("Select numeric column (Y-axis)", numeric_cols)

        x_col = None
        if categorical_cols:
            x_col = st.selectbox("Select category for X-axis (Optional)",
                                 ["None"] + categorical_cols)

        if x_col == "None" or not x_col:
            fig = px.violin(df, y=y_col, box=True, points="all",
                            title=f"Violin Plot of {y_col}")
        else:
            fig = px.violin(df, x=x_col, y=y_col, box=True, points="all",
                            title=f"Violin Plot of {y_col} grouped by {x_col}")

        st.plotly_chart(fig, use_container_width=True)

    elif viz_type == "Pair Plot":
        st.subheader("Pair Plot")

        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for pair plot.")
            return

        # Select columns for pair plot (limit to prevent overload)
        selected_cols = st.multiselect("Select columns (max 5 recommended)",
                                       numeric_cols,
                                       default=numeric_cols[:min(4, len(numeric_cols))])

        if len(selected_cols) < 2:
            st.warning("Please select at least 2 columns.")
            return

        if len(selected_cols) > 5:
            st.warning("Too many columns may slow down performance. Consider reducing selection.")

        color_col = None
        if categorical_cols:
            color_col = st.selectbox("Color by (Optional)",
                                     ["None"] + categorical_cols)
            if color_col == "None":
                color_col = None

        # Create pair plot with Plotly
        fig = px.scatter_matrix(df,
                                dimensions=selected_cols,
                                color=color_col,
                                title="Pair Plot / Scatter Matrix")

        # Update layout for better readability
        fig.update_traces(diagonal_visible=False)
        fig.update_layout(height=800)

        st.plotly_chart(fig, use_container_width=True)


def correlation_analysis(df):
    st.header("Correlation Analysis")

    # Get numeric columns for correlation
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) < 2:
        st.warning("Need at least 2 numeric columns for correlation analysis.")
        return

    # Select columns for analysis
    selected_cols = st.multiselect("Select columns for correlation analysis",
                                   numeric_cols,
                                   default=numeric_cols[:min(6, len(numeric_cols))])

    if len(selected_cols) < 2:
        st.warning("Please select at least 2 columns.")
        return

    # Choose correlation method
    corr_method = st.radio("Select correlation method",
                           ["Pearson", "Spearman", "Kendall"])

    # Calculate correlation
    corr_matrix = df[selected_cols].corr(method=corr_method.lower())

    # Display correlation table
    st.subheader("Correlation Matrix")
    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1))

    # Heatmap
    st.subheader("Correlation Heatmap")

    fig = plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    heatmap = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                          fmt='.2f', linewidths=0.5, vmin=-1, vmax=1)
    plt.title(f'{corr_method} Correlation Heatmap')
    plt.tight_layout()
    st.pyplot(fig)

    # Pairwise correlation analysis
    st.subheader("Pairwise Correlation")

    col1, col2 = st.columns(2)

    with col1:
        x_col = st.selectbox("Select first variable", selected_cols)

    with col2:
        y_col = st.selectbox("Select second variable",
                             [col for col in selected_cols if col != x_col])

    # Scatter plot with regression line
    fig = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                     title=f"Correlation between {x_col} and {y_col}")

    # Calculate and display correlation coefficient
    corr_value = df[[x_col, y_col]].corr(method=corr_method.lower()).iloc[0, 1]
    fig.add_annotation(x=0.5, y=0.95,
                       text=f"{corr_method} correlation: {corr_value:.4f}",
                       showarrow=False,
                       font=dict(size=14),
                       xref="paper", yref="paper")

    st.plotly_chart(fig, use_container_width=True)


def advanced_analysis(df):
    st.header("Advanced Analysis")

    analysis_type = st.selectbox("Select Analysis Type",
                                 ["Principal Component Analysis (PCA)",
                                  "Distribution Analysis",
                                  "Outlier Detection",
                                  "Time Series Decomposition"])

    if analysis_type == "Principal Component Analysis (PCA)":
        pca_analysis(df)

    elif analysis_type == "Distribution Analysis":
        distribution_analysis(df)

    elif analysis_type == "Outlier Detection":
        outlier_analysis(df)

    elif analysis_type == "Time Series Decomposition":
        time_series_analysis(df)


def pca_analysis(df):
    st.subheader("Principal Component Analysis (PCA)")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.shape[1] < 2:
        st.warning("PCA requires at least 2 numeric columns.")
        return

    # Select columns for PCA
    selected_cols = st.multiselect("Select columns for PCA",
                                   numeric_df.columns.tolist(),
                                   default=numeric_df.columns.tolist()[:min(5, len(numeric_df.columns))])

    if len(selected_cols) < 2:
        st.warning("Please select at least 2 columns.")
        return

    # Get data for PCA
    X = numeric_df[selected_cols].copy()

    # Handle missing values
    if X.isna().any().any():
        st.warning("Missing values detected. They will be filled with column means.")
        X = X.fillna(X.mean())

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Number of components
    n_components = st.slider("Number of components", min_value=2,
                             max_value=min(len(selected_cols), 10), value=2)

    # Perform PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_ * 100

    # Display explained variance
    st.write("**Explained Variance by Component**")

    # Create DataFrame for explained variance
    exp_var_df = pd.DataFrame({
        'Component': [f'PC{i + 1}' for i in range(n_components)],
        'Explained Variance (%)': explained_variance,
        'Cumulative Variance (%)': np.cumsum(explained_variance)
    })

    st.dataframe(exp_var_df)

    # Plot explained variance
    fig = px.bar(exp_var_df, x='Component', y='Explained Variance (%)',
                 title="Explained Variance by Principal Component")

    fig.add_trace(go.Scatter(x=exp_var_df['Component'],
                             y=exp_var_df['Cumulative Variance (%)'],
                             mode='lines+markers', name='Cumulative Variance',
                             line=dict(color='red', width=2)))

    st.plotly_chart(fig, use_container_width=True)

    # Display PCA components
    st.write("**PCA Components Loading**")

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i + 1}' for i in range(n_components)],
        index=selected_cols
    )

    st.dataframe(loadings)

    # Plot loadings
    fig = px.imshow(loadings,
                    labels=dict(x="Principal Components", y="Features"),
                    x=[f'PC{i + 1}' for i in range(n_components)],
                    y=selected_cols,
                    color_continuous_scale='RdBu_r',
                    title="PCA Component Loadings")

    st.plotly_chart(fig, use_container_width=True)

    # Plot PCA scatter plot for first two components
    if n_components >= 2:
        st.subheader("PCA Scatter Plot (First Two Components)")

        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(
            X_pca[:, :2],
            columns=['PC1', 'PC2']
        )

        # Add color column if available
        color_col = None
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        if categorical_cols:
            color_col = st.selectbox("Color points by (optional)",
                                     ["None"] + categorical_cols)

            if color_col != "None":
                pca_df[color_col] = df[color_col].values

        # Create scatter plot
        if color_col and color_col != "None":
            fig = px.scatter(pca_df, x='PC1', y='PC2', color=color_col,
                             title=f"PCA Scatter Plot (PC1 vs PC2, colored by {color_col})")
        else:
            fig = px.scatter(pca_df, x='PC1', y='PC2',
                             title="PCA Scatter Plot (PC1 vs PC2)")

        # Add axis labels with explained variance
        fig.update_xaxes(title=f"PC1 ({explained_variance[0]:.2f}%)")
        fig.update_yaxes(title=f"PC2 ({explained_variance[1]:.2f}%)")

        # Add loading vectors
        if st.checkbox("Show feature loadings on scatter plot"):
            loadings_x = loadings['PC1'].values
            loadings_y = loadings['PC2'].values

            for i, feature in enumerate(selected_cols):
                fig.add_shape(
                    type='line',
                    x0=0, y0=0,
                    x1=loadings_x[i] * 3,  # Scaling for visibility
                    y1=loadings_y[i] * 3,  # Scaling for visibility
                    line=dict(color='red', width=1, dash='dot')
                )

                fig.add_annotation(
                    x=loadings_x[i] * 3.5,  # Position text at end of arrow
                    y=loadings_y[i] * 3.5,
                    text=feature,
                    showarrow=False,
                    font=dict(color='red')
                )

        st.plotly_chart(fig, use_container_width=True)


def distribution_analysis(df):
    st.subheader("Distribution Analysis")

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available for distribution analysis.")
        return

    # Select column for analysis
    col = st.selectbox("Select column for distribution analysis", numeric_cols)

    # Remove NaN values
    data = df[col].dropna()

    col1, col2 = st.columns(2)

    with col1:
        # Descriptive statistics
        st.write("**Descriptive Statistics**")
        stats = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
            'Value': [
                data.mean(),
                data.median(),
                data.std(),
                data.min(),
                data.max(),
                data.skew(),
                data.kurtosis()
            ]
        })
        st.dataframe(stats)

    with col2:
        # Calculate quartiles and IQR
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        st.write("**Quantiles**")
        quantiles = pd.DataFrame({
            'Quantile': ['0% (Min)', '25%', '50% (Median)', '75%', '100% (Max)', 'IQR'],
            'Value': [
                data.min(),
                q1,
                data.median(),
                q3,
                data.max(),
                iqr
            ]
        })
        st.dataframe(quantiles)

    # Distribution plots
    st.write("**Distribution Visualization**")

    plot_type = st.radio("Select plot type",
                         ["Histogram with KDE", "Box Plot", "Violin Plot", "QQ Plot"])

    if plot_type == "Histogram with KDE":
        bin_count = st.slider("Number of bins", min_value=5, max_value=100, value=30)

        fig = plt.figure(figsize=(10, 6))
        sns.histplot(data, kde=True, bins=bin_count)
        plt.title(f"Histogram with KDE for {col}")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

    elif plot_type == "Box Plot":
        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(x=data)
        plt.title(f"Box Plot for {col}")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

    elif plot_type == "Violin Plot":
        fig = plt.figure(figsize=(10, 6))
        sns.violinplot(x=data)
        plt.title(f"Violin Plot for {col}")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

    elif plot_type == "QQ Plot":
        from scipy import stats as scipy_stats

        fig = plt.figure(figsize=(10, 6))

        # Create Q-Q plot
        scipy_stats.probplot(data, dist="norm", plot=plt)
        plt.title(f"Q-Q Plot for {col}")
        plt.grid(True, alpha=0.3)

        st.pyplot(fig)

    # Additional distribution characteristics
    st.write("**Test for Normality**")

    from scipy import stats as scipy_stats

    # Perform Shapiro-Wilk test for normality
    if len(data) <= 5000:  # Shapiro-Wilk works best for smaller samples
        stat, p_value = scipy_stats.shapiro(data.sample(min(len(data), 5000)))
        test_name = "Shapiro-Wilk"
    else:
        # For larger datasets, use K-S test
        stat, p_value = scipy_stats.kstest(data, 'norm')
        test_name = "Kolmogorov-Smirnov"

    # Display test results
    normality_result = pd.DataFrame({
        'Test': [test_name],
        'Statistic': [stat],
        'p-value': [p_value],
        'Interpretation': [
            "Data likely follows a normal distribution" if p_value > 0.05 else
            "Data likely does not follow a normal distribution"
        ]
    })

    st.dataframe(normality_result)

    if p_value <= 0.05:
        st.info("Since the p-value is ≤ 0.05, we reject the null hypothesis that the data is normally distributed.")
    else:
        st.info(
            "Since the p-value is > 0.05, we cannot reject the null hypothesis that the data is normally distributed.")


def outlier_analysis(df):
    st.subheader("Outlier Detection")

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available for outlier detection.")
        return

    # Select column for analysis
    col = st.selectbox("Select column for outlier detection", numeric_cols)

    # Remove NaN values
    data = df[col].dropna()

    # Select outlier detection method
    method = st.radio("Select outlier detection method",
                      ["Z-Score", "IQR (Interquartile Range)", "Both"])

    # Z-Score Analysis
    if method in ["Z-Score", "Both"]:
        st.write("**Z-Score Method**")

        z_threshold = st.slider("Z-Score threshold",
                                min_value=1.0, max_value=5.0, value=3.0, step=0.1,
                                key="z_score_slider")

        # Calculate Z-scores
        z_scores = (data - data.mean()) / data.std()
        outliers_z = data[abs(z_scores) > z_threshold]

        if not outliers_z.empty:
            st.write(f"Found {len(outliers_z)} outliers using Z-Score method (threshold: ±{z_threshold})")

            # Show outliers table
            outlier_df = pd.DataFrame({
                'Index': outliers_z.index,
                'Value': outliers_z.values,
                'Z-Score': z_scores[outliers_z.index].values
            })

            st.dataframe(outlier_df)

            # Visualize outliers
            fig = px.scatter(x=data.index, y=data,
                             title=f"Z-Score Outliers for {col}")

            # Add outliers as different markers
            fig.add_scatter(x=outliers_z.index, y=outliers_z,
                            mode='markers', marker=dict(color='red', size=10),
                            name='Outliers')

            # Add threshold lines
            mean_val = data.mean()
            std_val = data.std()

            fig.add_hline(y=mean_val + z_threshold * std_val, line_dash="dash",
                          line_color="red", annotation_text=f"+{z_threshold} SD")
            fig.add_hline(y=mean_val - z_threshold * std_val, line_dash="dash",
                          line_color="red", annotation_text=f"-{z_threshold} SD")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No outliers found using Z-Score method with threshold ±{z_threshold}")

    # IQR Analysis
    if method in ["IQR (Interquartile Range)", "Both"]:
        st.write("**IQR Method**")

        iqr_multiplier = st.slider("IQR multiplier",
                                   min_value=1.0, max_value=3.0, value=1.5, step=0.1,
                                   key="iqr_slider")

        # Calculate IQR bounds
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        outliers_iqr = data[(data < lower_bound) | (data > upper_bound)]

        if not outliers_iqr.empty:
            st.write(f"Found {len(outliers_iqr)} outliers using IQR method (multiplier: {iqr_multiplier})")

            # Show outliers table
            outlier_df = pd.DataFrame({
                'Index': outliers_iqr.index,
                'Value': outliers_iqr.values,
                'Type': ['Below lower bound' if val < lower_bound else 'Above upper bound'
                         for val in outliers_iqr.values]
            })

            st.dataframe(outlier_df)

            # Visualize outliers
            fig = px.box(data, title=f"Box Plot with Outliers for {col}")

            # Add scatter points for outliers
            fig.add_scatter(x=[0] * len(outliers_iqr), y=outliers_iqr,
                            mode='markers', marker=dict(color='red', size=8),
                            name='Outliers')

            st.plotly_chart(fig, use_container_width=True)

            # Show outlier distribution
            fig = px.scatter(x=data.index, y=data,
                             title=f"IQR Outliers for {col}")

            # Add outliers as different markers
            fig.add_scatter(x=outliers_iqr.index, y=outliers_iqr,
                            mode='markers', marker=dict(color='red', size=10),
                            name='Outliers')

            # Add threshold lines
            fig.add_hline(y=upper_bound, line_dash="dash",
                          line_color="red", annotation_text=f"Upper bound ({upper_bound:.2f})")
            fig.add_hline(y=lower_bound, line_dash="dash",
                          line_color="red", annotation_text=f"Lower bound ({lower_bound:.2f})")

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No outliers found using IQR method with multiplier {iqr_multiplier}")

    # Provide options for handling outliers
    if method == "Both" and (not outliers_z.empty or not outliers_iqr.empty):
        st.subheader("Outlier Comparison")

        if not outliers_z.empty and not outliers_iqr.empty:
            # Find common outliers
            common_outliers = set(outliers_z.index).intersection(set(outliers_iqr.index))

            st.write(f"**Common outliers detected by both methods:** {len(common_outliers)}")

            # Show Venn diagram
            from matplotlib_venn import venn2

            fig, ax = plt.subplots(figsize=(8, 6))
            venn = venn2([set(outliers_z.index), set(outliers_iqr.index)],
                         set_labels=('Z-Score', 'IQR'))
            plt.title("Outlier Detection Comparison")
            st.pyplot(fig)

    # Add options for handling outliers
    if st.checkbox("Show outlier handling options"):
        st.write("**Outlier Handling Options**")

        handling_method = st.radio(
            "Select method to handle outliers",
            ["Remove outliers", "Cap outliers", "Replace with mean/median", "No action"]
        )

        # Function to get the combined outliers from both methods
        def get_combined_outliers():
            if method == "Z-Score":
                return outliers_z.index
            elif method == "IQR (Interquartile Range)":
                return outliers_iqr.index
            else:  # Both
                return list(set(outliers_z.index) | set(outliers_iqr.index))

        if handling_method != "No action":
            outlier_indices = get_combined_outliers()

            if len(outlier_indices) == 0:
                st.info("No outliers to handle.")
            else:
                # Create a copy of the dataframe for demonstration
                modified_df = df.copy()

                if handling_method == "Remove outliers":
                    modified_df = modified_df.drop(outlier_indices)
                    action_text = "removed"

                elif handling_method == "Cap outliers":
                    if method in ["IQR (Interquartile Range)", "Both"]:
                        # Cap using IQR bounds
                        modified_df.loc[modified_df[modified_df[col] > upper_bound].index, col] = upper_bound
                        modified_df.loc[modified_df[modified_df[col] < lower_bound].index, col] = lower_bound
                    else:
                        # Cap using Z-score
                        mean_val = data.mean()
                        std_val = data.std()
                        modified_df.loc[modified_df[modified_df[
                                                        col] > mean_val + z_threshold * std_val].index, col] = mean_val + z_threshold * std_val
                        modified_df.loc[modified_df[modified_df[
                                                        col] < mean_val - z_threshold * std_val].index, col] = mean_val - z_threshold * std_val

                    action_text = "capped"

                elif handling_method == "Replace with mean/median":
                    replacement = st.radio("Replace with:", ["Mean", "Median"])

                    if replacement == "Mean":
                        replacement_value = data.mean()
                    else:  # Median
                        replacement_value = data.median()

                    modified_df.loc[outlier_indices, col] = replacement_value
                    action_text = f"replaced with {replacement.lower()}"

                st.write(f"**Result after outliers {action_text}:**")

                # Show before and after histograms
                col1, col2 = st.columns(2)

                with col1:
                    st.write("Before:")
                    fig = px.histogram(df[col].dropna(), nbins=30,
                                       title=f"Original Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.write("After:")
                    fig = px.histogram(modified_df[col].dropna(), nbins=30,
                                       title=f"Modified Distribution of {col}")
                    st.plotly_chart(fig, use_container_width=True)

                # Show summary statistics
                st.write("**Summary Statistics Comparison:**")

                summary_comparison = pd.DataFrame({
                    'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Skewness'],
                    'Before': [
                        df[col].dropna().count(),
                        df[col].dropna().mean(),
                        df[col].dropna().median(),
                        df[col].dropna().std(),
                        df[col].dropna().min(),
                        df[col].dropna().max(),
                        df[col].dropna().skew()
                    ],
                    'After': [
                        modified_df[col].dropna().count(),
                        modified_df[col].dropna().mean(),
                        modified_df[col].dropna().median(),
                        modified_df[col].dropna().std(),
                        modified_df[col].dropna().min(),
                        modified_df[col].dropna().max(),
                        modified_df[col].dropna().skew()
                    ]
                })

                st.dataframe(summary_comparison)


def time_series_analysis(df):
    st.subheader("Time Series Analysis")

    # Check if there are datetime columns
    datetime_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
        # Try to convert to datetime
        elif df[col].dtypes == 'object':
            try:
                pd.to_datetime(df[col])
                datetime_cols.append(col)
            except:
                pass

    if not datetime_cols:
        st.warning("No datetime columns detected. Time series analysis requires a datetime column.")
        return

    # Select datetime column
    date_col = st.selectbox("Select date/time column", datetime_cols)

    # Make sure it's datetime format
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except:
        st.error(f"Failed to convert {date_col} to datetime format.")
        return

    # Select value column
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        st.warning("No numeric columns available for time series analysis.")
        return

    value_col = st.selectbox("Select value column to analyze", numeric_cols)

    # Time resampling
    freq_options = {
        "Original (no resampling)": None,
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Quarterly": "Q",
        "Yearly": "Y"
    }

    resample_freq = st.selectbox("Select time aggregation", list(freq_options.keys()))

    # Aggregation method
    agg_method = st.selectbox("Select aggregation method",
                              ["Mean", "Sum", "Min", "Max", "Median", "Count"])

    # Prepare time series data
    try:
        # Sort by date
        ts_df = df[[date_col, value_col]].sort_values(by=date_col)

        # Set date as index
        ts_df = ts_df.set_index(date_col)

        # Resample if selected
        if freq_options[resample_freq] is not None:
            if agg_method == "Mean":
                ts_df = ts_df.resample(freq_options[resample_freq]).mean()
            elif agg_method == "Sum":
                ts_df = ts_df.resample(freq_options[resample_freq]).sum()
            elif agg_method == "Min":
                ts_df = ts_df.resample(freq_options[resample_freq]).min()
            elif agg_method == "Max":
                ts_df = ts_df.resample(freq_options[resample_freq]).max()
            elif agg_method == "Median":
                ts_df = ts_df.resample(freq_options[resample_freq]).median()
            else:  # Count
                ts_df = ts_df.resample(freq_options[resample_freq]).count()

        # Reset index for plotting
        ts_df = ts_df.reset_index()

        # Basic time series plot
        st.write("**Time Series Plot**")

        # Interactive plot with Plotly
        fig = px.line(ts_df, x=date_col, y=value_col,
                      title=f"Time Series: {value_col} over time")

        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Time series decomposition
        if st.checkbox("Show time series decomposition"):
            from statsmodels.tsa.seasonal import seasonal_decompose

            # Set date as index again for decomposition
            ts_data = ts_df.set_index(date_col)[value_col]

            if len(ts_data) < 4:
                st.warning("Not enough data points for decomposition. Need at least 4 points.")
            else:
                # Determine appropriate period for decomposition
                if freq_options[resample_freq] is None:
                    # If using original data, try to infer frequency
                    period_options = {
                        "Auto-detect": None,
                        "Daily (7)": 7,
                        "Weekly (52)": 52,
                        "Monthly (12)": 12,
                        "Quarterly (4)": 4
                    }
                else:
                    # For resampled data
                    if freq_options[resample_freq] == "D":
                        period_options = {
                            "Weekly (7)": 7,
                            "Bi-weekly (14)": 14,
                            "Monthly (30)": 30,
                            "Custom": "custom"
                        }
                    elif freq_options[resample_freq] == "W":
                        period_options = {
                            "Monthly (4)": 4,
                            "Quarterly (13)": 13,
                            "Yearly (52)": 52,
                            "Custom": "custom"
                        }
                    elif freq_options[resample_freq] == "M":
                        period_options = {
                            "Quarterly (3)": 3,
                            "Yearly (12)": 12,
                            "Custom": "custom"
                        }
                    elif freq_options[resample_freq] in ["Q", "Y"]:
                        period_options = {
                            "Yearly (4)": 4,
                            "Custom": "custom"
                        }
                    else:
                        period_options = {
                            "Auto-detect": None,
                            "Custom": "custom"
                        }

                period_selection = st.selectbox("Select seasonality period", list(period_options.keys()))

                if period_selection == "Custom":
                    period = st.number_input("Enter custom period", min_value=2, max_value=len(ts_data) // 2, value=12)
                else:
                    period = period_options[period_selection]

                    # Auto-detect if selected
                    if period is None:
                        # Simple heuristic for period detection
                        if len(ts_data) >= 730:  # At least 2 years of daily data
                            period = 365  # Annual cycle for daily data
                        elif len(ts_data) >= 60:  # At least 2 years of monthly data
                            period = 12  # Annual cycle for monthly data
                        elif len(ts_data) >= 16:  # At least 2 years of quarterly data
                            period = 4  # Annual cycle for quarterly data
                        elif len(ts_data) >= 14:  # At least 2 weeks of daily data
                            period = 7  # Weekly cycle
                        else:
                            period = 4  # Default fallback

                # Make sure period is not too large compared to data length
                period = min(period, len(ts_data) // 2)

                try:
                    # Perform decomposition
                    decomposition = seasonal_decompose(ts_data, model='additive', period=period)

                    # Create DataFrames for the components
                    trend = decomposition.trend
                    seasonal = decomposition.seasonal
                    residual = decomposition.resid

                    # Plot components
                    st.write(f"**Time Series Decomposition (Period: {period})**")

                    fig = plt.figure(figsize=(12, 10))
                    plt.subplot(411)
                    plt.plot(ts_data, label='Original')
                    plt.legend(loc='upper left')
                    plt.title('Time Series Decomposition')

                    plt.subplot(412)
                    plt.plot(trend, label='Trend')
                    plt.legend(loc='upper left')

                    plt.subplot(413)
                    plt.plot(seasonal, label='Seasonality')
                    plt.legend(loc='upper left')

                    plt.subplot(414)
                    plt.plot(residual, label='Residuals')
                    plt.legend(loc='upper left')

                    plt.tight_layout()
                    st.pyplot(fig)

                    # Interactive decomposition with Plotly
                    decomp_df = pd.DataFrame({
                        'Date': ts_data.index,
                        'Original': ts_data.values,
                        'Trend': trend.values,
                        'Seasonal': seasonal.values,
                        'Residual': residual.values
                    }).dropna()

                    # Plot original and trend
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=decomp_df['Date'], y=decomp_df['Original'],
                                             mode='lines', name='Original'))
                    fig.add_trace(go.Scatter(x=decomp_df['Date'], y=decomp_df['Trend'],
                                             mode='lines', name='Trend', line=dict(width=3)))

                    fig.update_layout(title='Original Time Series and Trend',
                                      xaxis_title='Date',
                                      yaxis_title='Value')

                    st.plotly_chart(fig, use_container_width=True)

                    # Plot seasonality
                    fig = px.line(decomp_df, x='Date', y='Seasonal',
                                  title='Seasonal Component')
                    st.plotly_chart(fig, use_container_width=True)

                    # Plot residuals
                    fig = px.scatter(decomp_df, x='Date', y='Residual',
                                     title='Residual Component')

                    # Add zero line
                    fig.add_hline(y=0, line_dash="dash", line_color="red")

                    st.plotly_chart(fig, use_container_width=True)

                    # Seasonality patterns (only for appropriate frequencies)
                    if period >= 4 and period <= 52:
                        # Group by period component to show seasonality pattern
                        seasonal_pattern = seasonal.groupby(seasonal.index.month if period == 12 else
                                                            seasonal.index.dayofweek if period == 7 else
                                                            seasonal.index % period).mean()

                        fig = px.line(x=seasonal_pattern.index, y=seasonal_pattern.values,
                                      title='Seasonal Pattern',
                                      labels={'x': 'Period Component', 'y': 'Average Effect'})

                        st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Decomposition error: {e}")
                    st.info("Try adjusting the period or using a different aggregation level.")

        # Show autocorrelation/partial autocorrelation
        if st.checkbox("Show autocorrelation analysis"):
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

            # Set date as index again
            ts_data = ts_df.set_index(date_col)[value_col]

            # Remove missing values
            ts_data = ts_data.dropna()

            if len(ts_data) < 3:
                st.warning("Not enough data points for autocorrelation analysis.")
            else:
                lags = st.slider("Number of lags", min_value=5, max_value=min(50, len(ts_data) - 1), value=20)

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Autocorrelation Function (ACF)**")
                    fig = plt.figure(figsize=(10, 6))
                    plot_acf(ts_data, lags=lags, alpha=0.05, ax=plt.gca())
                    plt.title("Autocorrelation Function")
                    st.pyplot(fig)

                with col2:
                    st.write("**Partial Autocorrelation Function (PACF)**")
                    fig = plt.figure(figsize=(10, 6))
                    plot_pacf(ts_data, lags=lags, alpha=0.05, ax=plt.gca())
                    plt.title("Partial Autocorrelation Function")
                    st.pyplot(fig)

                st.info("**Interpretation:** Significant spikes in ACF indicate seasonality patterns. "
                        "PACF helps identify the order of an autoregressive model. "
                        "Spikes crossing the blue lines are statistically significant.")

    except Exception as e:
        st.error(f"Error in time series analysis: {e}")


# Run the app
if __name__ == "__main__":
    main()