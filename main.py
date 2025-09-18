import os
import time
import logging
import psycopg2
import pandas as pd
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not set in .env")
    st.stop()

client = OpenAI(api_key=api_key)

# Database connection info from .env
db_config = {
    "host": os.environ.get("DB_HOST"),
    "port": os.environ.get("DB_PORT"),
    "database": os.environ.get("DB_NAME"),
    "user": os.environ.get("DB_USER"),
    "password": os.environ.get("DB_PASSWORD")
}

# Connect to PostgreSQL
try:
    conn = psycopg2.connect(**db_config)
    logging.info("Connected to PostgreSQL database successfully.")
except Exception as e:
    st.error(f"Error connecting to database: {e}")
    st.stop()

# Streamlit UI
st.title("FEC Contributions Q&A")
user_input = st.text_input("Enter your question:")

def call_model(prompt, max_retries=3):
    """Call OpenAI GPT to generate SQL or answer"""
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model="gpt-5-mini",
                input=[{"role": "user", "content": prompt}]
            )
            return resp.output_text if hasattr(resp, "output_text") else resp['output'][0]['content'][0]['text']
        except Exception as e:
            wait = 2 ** attempt
            logging.warning(f"Model call failed (attempt {attempt+1}): {e}; retrying in {wait}s")
            time.sleep(wait)
    return None

def execute_sql(sql):
    """Execute SQL query on the database and return as DataFrame"""
    try:
        df = pd.read_sql_query(sql, conn)
        return df
    except Exception as e:
        logging.error(f"SQL execution error: {e}")
        return pd.DataFrame()

if st.button("Ask"):
    if not user_input.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating SQL..."):
            # Step 1: Generate SQL from question
            sql_prompt = f"""
            You are an AI helper specialized in U.S. Federal Election Commission (FEC) contribution data.
            Translate the following natural language question into a **safe SQL query**:
            Question: "{user_input}"
            Return only the SQL query.
            """
            sql_query = call_model(sql_prompt)

        if not sql_query:
            st.error("Error generating SQL.")
        else:
            st.info("Executing SQL query...")
            df_result = execute_sql(sql_query)

            if df_result.empty:
                result_prompt = f"""
                User asked: "{user_input}".
                The database returned no results.
                Politely explain no matching records were found.
                """
            else:
                result_prompt = f"""
                User asked: "{user_input}".
                SQL results:
                {df_result.to_string(index=False)}
                Provide a concise, professional summary of these results.
                """

            with st.spinner("Generating answer..."):
                answer = call_model(result_prompt)
                st.subheader("Answer:")
                st.write(answer)
