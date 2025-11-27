import streamlit as st
import google.generativeai as genai
import os

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
genai.configure(api_key = st.secrets["GEMINI_API_KEY"])

# -------------------------------------------------
# Helper: Load text templates
# -------------------------------------------------
def load_prompt(path):
    return open(path, "r", encoding="utf-8").read()

extractor_template = load_prompt("prompts/extractor.txt")
enhancer_template = load_prompt("prompts/enhancer.txt")
assembler_template = load_prompt("prompts/assembler.txt")

# Gemini Model
model = genai.GenerativeModel("models/gemini-2.5-pro")

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Context2Veo - Prompt Generator", layout="wide")

st.title("🎬 Context2Veo — Video Prompt Generator")
st.write("Convert a simple idea into a cinematic prompt ready for Veo.")

# User input
context = st.text_area(
    "Describe your idea (context)",
    placeholder="Example: A peaceful village morning with fog, birds, and a slow camera pan...",
    height=150
)

generate_btn = st.button("Generate Veo Prompt 🚀")

# -------------------------------------------------
# Logic: 3-step transformation pipeline
# -------------------------------------------------
def step_extract(context_text):
    """Step 1: Convert raw context → structured details."""
    prompt = extractor_template.replace("{{context}}", context_text)
    result = model.generate_content(prompt)
    return result.text.strip()

def step_enhance(details_text):
    """Step 2: Expand details with cinematic clarity."""
    prompt = enhancer_template.replace("{{details}}", details_text)
    result = model.generate_content(prompt)
    return result.text.strip()

def step_assemble(enhanced_text):
    """Step 3: Create final Veo-ready video prompt."""
    prompt = assembler_template.replace("{{enhanced}}", enhanced_text)
    result = model.generate_content(prompt)
    return result.text.strip()

# -------------------------------------------------
# Run Pipeline on Button Click
# -------------------------------------------------
if generate_btn:
    if not context.strip():
        st.error("Please enter some context.")
        st.stop()

    with st.spinner("Extracting details..."):
        extracted = step_extract(context)

    with st.spinner("Enhancing scene..."):
        enhanced = step_enhance(extracted)

    with st.spinner("Building final Veo prompt..."):
        final_prompt = step_assemble(enhanced)

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------
    st.subheader("🧩 Extracted Details")
    st.code(extracted, language="markdown")

    st.subheader("✨ Enhanced Version")
    st.code(enhanced, language="markdown")

    st.subheader("🎥 Final Veo Prompt")
    st.code(final_prompt, language="markdown")

    st.success("Done! Copy your final Veo prompt above.")
