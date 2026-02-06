import tempfile
import csv
import re 
import streamlit as st
import pandas as pd
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckdb import DuckDbTools
from agno.tools.pandas import PandasTools

# Helper: sanitize table names for DuckDB (based on file name)
def sanitize_table_name(filename: str, index: int) -> str:
    name = filename.rsplit(".", 1)[0]  # drop extension
    name = name.lower()
    # replace non-alphanumeric with underscore
    name = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
    # ensure it’s not empty and avoid collisions using index
    if not name:
        name = f"table_{index}"
    else:
        name = f"{name}_{index}"
    return name

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep as is if conversion fails
                    pass
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # Save the DataFrame to the temporary CSV file with quotes around string fields
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Streamlit app
st.title("📊 Data Analyst Agent (Multi-Table + SQL + Charts)")

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    openai_key = st.text_input("Enter your OpenAI API key:", type="password")
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("API key saved!")
    else:
        st.warning("Please enter your OpenAI API key to proceed.")

# File upload widget – now supports multiple files
uploaded_files = st.file_uploader(
    "Upload one or more CSV or Excel files",
    type=["csv", "xlsx"],
    accept_multiple_files=True,
)

# Ensure we have a place to store last generated SQL (optional, for future use)
if "generated_code" not in st.session_state:
    st.session_state.generated_code = None

tables_info = {}  # table_name -> df

if uploaded_files and "openai_key" in st.session_state:
    st.markdown("### 📂 Uploaded Datasets")
    
    # Initialize DuckDbTools once
    duckdb_tools = DuckDbTools()

    # Process each uploaded file into its own DuckDB table
    for idx, uploaded_file in enumerate(uploaded_files):
        temp_path, columns, df = preprocess_and_save(uploaded_file)
        
        if temp_path and columns and df is not None:
            table_name = sanitize_table_name(uploaded_file.name, idx)
            
            st.write(f"#### Table: `{table_name}` (from `{uploaded_file.name}`)")
            st.write("Preview:")
            st.dataframe(df)
            st.write("Columns:", list(df.columns))
            
            # Load into DuckDB under its own table name
            duckdb_tools.load_local_csv_to_table(
                path=temp_path,
                table=table_name,
            )
            
            tables_info[table_name] = df
        else:
            st.warning(f"Skipping file `{uploaded_file.name}` due to preprocessing error.")

    if tables_info:
        # Initialize the Agent with DuckDB and Pandas tools
        # Updated system message to ask model to always show the SQL used
        data_analyst_agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key=st.session_state.openai_key),
            tools=[duckdb_tools, PandasTools()],
            system_message=(
                "You are an expert data analyst. "
                "You have access to one or more DuckDB tables (e.g., 'uploaded_data', table names based on filenames). "
                "Use these tables to answer user queries. Generate SQL queries using DuckDB tools to solve the user's query. "
                "Always include the final SQL query (or queries) that you used inside a ```sql ... ``` fenced code block "
                "before your natural language explanation. Provide clear and concise answers with the results."
            ),
            markdown=True,
        )

        st.markdown("---")
        st.subheader("❓ Ask Questions About Your Data")

        # Main query input widget
        user_query = st.text_area(
            "Ask a query about the data (you can refer to tables by name):",
            placeholder="Example: Which customer made the highest total purchase across all tables?"
        )
        
        st.info("💡 Tip: You can mention table names explicitly, e.g., `FROM sales_0` or `FROM customers_1`.")

        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('Processing your query with the agent...'):
                        # Get the response from the agent
                        response = data_analyst_agent.run(user_query)

                        # Extract the content from the response object
                        if hasattr(response, 'content'):
                            response_content = response.content
                        else:
                            response_content = str(response)

                    # Display the response in Streamlit
                    st.markdown("### 🧠 Agent Response")
                    st.markdown(response_content)

                    # Try to extract any ```sql ... ``` block from the response for transparency
                    sql_match = re.search(r"```sql(.*?)```", response_content, re.DOTALL | re.IGNORECASE)
                    if sql_match:
                        sql_code = sql_match.group(1).strip()
                        st.session_state.generated_code = sql_code  # store for potential future usage

                        st.markdown("### 📜 SQL Used by the Agent")
                        st.code(sql_code, language="sql")
                    else:
                        st.info("ℹ No SQL block detected in the response. Make sure your query requires data access.")

                except Exception as e:
                    st.error(f"Error generating response from the agent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")

        # ---- Quick visualization section ----
        st.markdown("---")
        st.subheader("📊 Quick Visualization")

        if tables_info:
            table_names = list(tables_info.keys())
            selected_table = st.selectbox(
                "Select a table to visualize:",
                options=table_names,
            )

            df_vis = tables_info[selected_table]
            all_cols = list(df_vis.columns)
            numeric_cols = list(df_vis.select_dtypes(include="number").columns)

            if not all_cols:
                st.warning("No columns available in this table to visualize.")
            else:
                chart_type = st.selectbox(
                    "Chart type:",
                    options=["Bar", "Line", "Area"],
                    index=0,
                )

                x_col = st.selectbox(
                    "X-axis column:",
                    options=all_cols,
                )

                if numeric_cols:
                    y_col = st.selectbox(
                        "Y-axis column (numeric):",
                        options=numeric_cols,
                    )
                else:
                    y_col = None
                    st.warning("No numeric columns found for Y-axis. Aggregations may not be available.")

                if st.button("Generate Chart"):
                    try:
                        if y_col is None:
                            st.error("Please select a table with at least one numeric column for Y-axis.")
                        else:
                            # Simple aggregation: sum by x_col
                            chart_df = df_vis.groupby(x_col)[y_col].sum().reset_index()
                            chart_df = chart_df.set_index(x_col)

                            if chart_type == "Bar":
                                st.bar_chart(chart_df[y_col])
                            elif chart_type == "Line":
                                st.line_chart(chart_df[y_col])
                            elif chart_type == "Area":
                                st.area_chart(chart_df[y_col])

                    except Exception as e:
                        st.error(f"Error generating chart: {e}")
        else:
            st.info("Upload at least one valid dataset to enable visualization.")

elif uploaded_files and "openai_key" not in st.session_state:
    st.warning("Please enter and save your OpenAI API key in the sidebar first.")
