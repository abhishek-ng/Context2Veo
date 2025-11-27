import streamlit as st
import google.generativeai as genai

# --------------------------------------
# Load API key from Streamlit Secrets
# --------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# Choose model (best for creative prompt generation)
MODEL = "models/gemini-2.5-pro"   # or "models/gemini-2.0-flash-thinking-exp"

# --------------------------------------
# Helper function: generate prompt sequence
# --------------------------------------
def generate_prompt_sequence(context, num_prompts):
    system_prompt = f"""
    You are a cinematic prompt generator.

    Task:
    - Expand the short context: "{context}"
    - Generate a sequential list of highly detailed cinematic prompts.
    - Each prompt represents the next shot of a long video scene.
    - Make them rich, consistent, realistic, and cinematic.
    - Use strong visual descriptions, camera motions, lighting, mood.

    Format strictly as:
    1. <prompt>
    2. <prompt>
    ...
    """

    model = genai.GenerativeModel(MODEL)
    result = model.generate_content(system_prompt)

    return result.text


# --------------------------------------
# Streamlit UI
# --------------------------------------
st.set_page_config(page_title="Prompt Sequencer", layout="wide")

st.title("🎬 Multi-Shot Video Prompt Generator")
st.write("Convert a small idea into a sequence of cinematic prompts for Veo.")

context = st.text_input(
    "Enter your short context (4–5 words)",
    placeholder="Example: futuristic desert city"
)

num_prompts = st.slider("Number of prompts needed", 5, 40, 12)

generate_btn = st.button("Generate Prompt Sequence 🚀")

# --------------------------------------
# Run when clicked
# --------------------------------------
if generate_btn:
    if not context.strip():
        st.error("Please enter context first.")
        st.stop()

    with st.spinner("Generating prompt sequence..."):
        sequence = generate_prompt_sequence(context, num_prompts)

    st.subheader("🎞️ Generated Prompt Sequence")
    st.code(sequence, language="markdown")

    st.success("Done! Copy your sequence to generate multi-shot Veo videos.")
