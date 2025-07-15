import io
import joblib
import os
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

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
filename = None
preprocessed_folder = "preprocessed"

# --- Chosen file
if selected_file != "--":
    filename = selected_file
    save_path = os.path.join(data_folder, filename)
    try:
        df = pd.read_csv(save_path)
    except Exception as e:
        st.error(f"❌ Error reading file {filename}: {e}")

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

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Error reading uploaded file: {e}")

# --- Either chosen or uploaded file
if df is not None:
    st.subheader(f"📋 Dataset: `{filename}`")
    st.write("👁️ Data Preview:")
    df_preview = df.head(10).copy()
    df_preview.index.name = "Row #"
    df_preview.index += 1
    st.dataframe(df_preview)


    with st.expander("🔹 Dataset Info (df.info())"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    st.subheader("🔹 Descriptive statistics")
    st.write(df.describe())

    st.subheader("🔹 Class labels frequency")
    st.write(df['class'].value_counts())

    st.subheader("🔹 Missing values check")
    st.write(df.isnull().sum())

    le = LabelEncoder()
    class_column = st.selectbox("Select target column (class):", df.columns)

    df['class_encoded'] = le.fit_transform(df[class_column])
    st.subheader("🔹 Class convertion to numeric")
    st.write(df[[class_column, 'class_encoded']].drop_duplicates())

    # --- Save preprocessed data and LabelEncoder
    st.subheader("💾 Save Preprocessed Data")
    if st.button("📥 Save for Training"):
        # Ensure preprocessed folder exists
        if not os.path.exists(preprocessed_folder):
            os.makedirs(preprocessed_folder)
    
        preprocessed_filename = f"preprocessed_{filename}"
        preprocessed_path = os.path.join(preprocessed_folder, preprocessed_filename)
        label_encoder_filename = f"label_encoder_{filename.split('.')[0]}.pkl"
        label_encoder_path = os.path.join(preprocessed_folder, label_encoder_filename)
    
        try:
            # Save preprocessed CSV
            df.to_csv(preprocessed_path, index=False)
    
            # Save the LabelEncoder object
            joblib.dump(le, label_encoder_path)
    
            st.success(f"✅ Saved preprocessed data to `{preprocessed_path}`")
            st.success(f"✅ Saved LabelEncoder to `{label_encoder_path}`")
        except Exception as e:
            st.error(f"❌ Failed to save preprocessed data or encoder: {e}")

st.header("🤖 Train ML Model")

# Check for preprocessed data
preprocessed_files = [f for f in os.listdir(preprocessed_folder) if f.startswith("preprocessed_")]
selected_preprocessed_file = st.selectbox("Select preprocessed dataset for training", ["--"] + preprocessed_files)

if selected_preprocessed_file != "--":
    try:
        preprocessed_path = os.path.join(preprocessed_folder, selected_preprocessed_file)
        df_train = pd.read_csv(preprocessed_path)

        class_column_options = [col for col in df_train.columns if col.startswith("class")]
        target_col = st.selectbox("Select target column", class_column_options)

        X = df_train.drop(columns=class_column_options)
        y = df_train[target_col]

        test_size = st.slider("Test set size (%)", min_value=10, max_value=50, value=20, step=5) / 100
        model_type = st.selectbox("Choose model", ["Logistic Regression", "Random Forest"])

        if st.button("🚀 Train Model"):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if model_type == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = RandomForestClassifier(n_estimators=100)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            st.success(f"✅ Model trained! Accuracy on test set: `{acc:.4f}`")

            # Save model
            model_filename = f"{model_type.replace(' ', '_').lower()}_{selected_preprocessed_file.split('.')[0]}.pkl"
            model_path = os.path.join("models", model_filename)
            if not os.path.exists("models"):
                os.makedirs("models")

            joblib.dump(model, model_path)
            st.info(f"💾 Model saved to `{model_path}`")

    except Exception as e:
        st.error(f"❌ Failed to train model: {e}")