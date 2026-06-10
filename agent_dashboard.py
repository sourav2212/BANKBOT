import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Agent Dashboard",
    page_icon="👤",
    layout="wide"
)

st.title("Human Agent Dashboard")
st.caption("Escalated conversations requiring human attention")

log_file = "escalation_log.csv"

if not os.path.isfile(log_file):
    st.info("No escalations yet. Flagged conversations will appear here automatically.")
    st.stop()

df = pd.read_csv(log_file)
df = df.sort_values("Timestamp", ascending=False).reset_index(drop=True)

# --- Summary metrics at the top ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Escalations", len(df))
with col2:
    st.metric("Avg Sentiment Score", f"{df['Score'].mean():.2f}")
with col3:
    most_recent = df["Timestamp"].iloc[0]
    st.metric("Most Recent", most_recent)

st.divider()

# --- Most urgent case highlighted ---
worst_idx = df["Score"].idxmin()
worst = df.iloc[worst_idx]
st.error(
    f"Most urgent message: \"{worst['Message']}\"  "
    f"(score: {worst['Score']})"
)

st.divider()

# --- Full escalation table ---
st.subheader("All Escalated Conversations")
st.dataframe(
    df,
    use_container_width=True,
    column_config={
        "Score": st.column_config.ProgressColumn(
            "Sentiment Score",
            min_value=-1,
            max_value=1,
            format="%.2f"
        )
    }
)

# --- Download button ---
csv_data = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download escalation log as CSV",
    data=csv_data,
    file_name="escalation_log.csv",
    mime="text/csv"
)