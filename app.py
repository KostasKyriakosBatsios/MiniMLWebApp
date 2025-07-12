import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Mini ML Web App",
    layout="wide",
)

st.title("🔧 Mini ML Web App")
st.sidebar.title("⚙️ Options")
st.sidebar.write("Choose or upload a CSV file and observe the data.")
st.markdown("Choose or Upload a '.csv' file to begin.")

st.sidebar.subheader("📁 Load dataset from 'data' folder.")
data_folder = "data"
csv_files = [f for f in os.listdir(data_folder) if f.endswith(".csv")]

selected_file = st.sidebar.selectbox("Select a CSV file from the 'data' folder.",["--"] +  csv_files)

st.sidebar.subheader("📤 Or upload a CSV file of your choice.")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

df = None

# --- Chosen file
if selected_file != "--":
    df = pd.read_csv(os.path.join(data_folder, selected_file))
    st.success(f"✅ File {selected_file} loaded successfully from 'data' folder!")

# --- Uploaded file
elif uploaded_file is not None:
    filename = uploaded_file.name
    save_path = os.path.join(data_folder, filename)

    if os.path.exists(save_path):
        st.warning(f"⚠️ File {filename} already exists in 'data' folder. Please choose a different name.")
    else:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success("✅ File uploaded successfully!")
    df = pd.read_csv(uploaded_file)

# --- Either chosen or uploaded file
if df is not None:
    # --- Preview of data
    st.write("👁️ Data Preview:")
    st.dataframe(df.head(10))

    # --- Data info
    st.subheader("📐 Data Info")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Shape (rows, columns)**")
        st.write(df.shape)
    
    with col2:
        st.markdown("**Data type of each column**")
        st.write(df.dtypes)

    with col3:
        st.markdown("**Missing values per column**")
        st.write(df.isnull().sum())