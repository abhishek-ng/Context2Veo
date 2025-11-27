import streamlit as st
import google.generativeai as genai
import re

# --------------------------------------
# Load API key from Streamlit Secrets
# --------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

MODEL = "models/gemini-2.5-pro"


# --------------------------------------
# Helper: Generate full numbered prompt sequence
# --------------------------------------
def generate_prompt_sequence(context, num_prompts):
    system_prompt = f"""
    You are a cinematic prompt generator.

    Task:
    - Expand this short context: "{context}"
    - Generate EXACTLY {num_prompts} cinematic video prompts.
    - Each prompt is a new shot of a long cinematic sequence.
    - Use highly visual descriptions, camera motion, lighting, mood.

    Format STRICTLY like:
    1. <prompt>
    2. <prompt>
    ...
    {num_prompts}. <prompt>
    """

    model = genai.GenerativeModel(MODEL)
    result = model.generate_content(system_prompt)
    return result.text


# --------------------------------------
# Helper: Split generated text into list of prompts
# --------------------------------------
def split_numbered_prompts(text):
    """
    Extract prompts that follow the pattern:
    1. something
    2. something
    3. something
    """

    pattern = r"\d+\.\s*(.+?)(?=\n\d+\.|\Z)"
    matches = re.findall(pattern, text, flags=re.S)

    # Clean whitespace
    return [m.strip() for m in matches]


# --------------------------------------
# Streamlit UI
# --------------------------------------
st.set_page_config(page_title="Prompt Sequencer", layout="wide")

st.title("🎬 Multi-Shot Video Prompt Generator")
st.write("Generate multiple cinematic prompts for Veo from a tiny idea.")

context = st.text_input(
    "Enter your short context (4–5 words)",
    placeholder="Example: futuristic desert city"
)

num_prompts = st.slider("Number of prompts", 5, 40, 12)

generate_btn = st.button("Generate Prompt Sequence 🚀")

# --------------------------------------
# Run when clicked
# --------------------------------------
if generate_btn:
    if not context.strip():
        st.error("Please enter context first.")
        st.stop()

    with st.spinner("Generating prompt sequence..."):
        full_text = generate_prompt_sequence(context, num_prompts)

    # Split into individual prompts
    prompt_list = split_numbered_prompts(full_text)

    st.success("Generated! Showing individual prompts below 👇")

    # --------------------------------------
    # UI: Show each prompt separately
    # --------------------------------------
    st.subheader("🎞️ Sequential Prompts")

    for i, p in enumerate(prompt_list, start=1):
        st.markdown(f"### 🎥 Shot {i}")
        st.code(p, language="markdown")
        st.divider()

    # Show raw sequence too (optional)
    st.subheader("🧾 Full Combined Output")
    st.code(full_text, language="markdown")
