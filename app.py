import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

# -------------------------------------------------
# Load environment variables
# -------------------------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ_API_KEY is missing in .env file!")
    st.stop()

client = Groq(api_key=groq_api_key)

# -------------------------------------------------
# Helper: Load text templates
# -------------------------------------------------
def load_prompt(path):
    return open(path, "r", encoding="utf-8").read()

extractor_template = load_prompt("prompts/extractor.txt")
enhancer_template = load_prompt("prompts/enhancer.txt")
assembler_template = load_prompt("prompts/assembler.txt")


# -------------------------------------------------
# Groq LLM Helper function
# -------------------------------------------------
def groq_generate(prompt: str) -> str:
    completion = client.chat.completions.create(
        model="mixtral-8x7b-32768",   # or "llama3-70b-8192"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return completion.choices[0].message["content"]


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(page_title="Context2Veo - Prompt Generator", layout="wide")

st.title("🎬 Context2Veo — Video Prompt Generator (Groq Edition)")
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
    prompt = extractor_template.replace("{{context}}", context_text)
    return groq_generate(prompt)

def step_enhance(details_text):
    prompt = enhancer_template.replace("{{details}}", details_text)
    return groq_generate(prompt)

def step_assemble(enhanced_text):
    prompt = assembler_template.replace("{{enhanced}}", enhanced_text)
    return groq_generate(prompt)


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
    st.code(extracted)

    st.subheader("✨ Enhanced Version")
    st.code(enhanced)

    st.subheader("🎥 Final Veo Prompt")
    st.code(final_prompt)

    st.success("Done! Copy your final Veo prompt above.")
