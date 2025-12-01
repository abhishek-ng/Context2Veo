import streamlit as st
import google.generativeai as genai
from huggingface_hub import InferenceClient
import json


# -------------------------------------------------
# Load API Keys
# -------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
HF_TOKEN = st.secrets["HF_TOKEN"]

hf_client = InferenceClient(
    "ali-vilab/text-to-video-ms-1.7b",
    token=HF_TOKEN,
)

MODEL_NAME = "models/gemini-2.5-pro"


# -------------------------------------------------
# Load Prompt Templates
# -------------------------------------------------
def load_prompt(path):
    return open(path, "r", encoding="utf-8").read()


extractor_template = load_prompt("prompts/extractor.txt")
enhancer_template = load_prompt("prompts/enhancer.txt")
assembler_template = load_prompt("prompts/assembler.txt")


# -------------------------------------------------
# Gemini Text Generation
# -------------------------------------------------
def gemini_generate(prompt: str):
    model = genai.GenerativeModel(MODEL_NAME)

    response = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json"
        }
    )

    return response.text



# -------------------------------------------------
# Processing Steps
# -------------------------------------------------
def step_extract(context_text):
    prompt = extractor_template.replace("{{context}}", context_text)
    return gemini_generate(prompt)


def step_enhance(details_text):
    prompt = enhancer_template.replace("{{details}}", details_text)
    return gemini_generate(prompt)


def step_assemble(enhanced_text):
    prompt = assembler_template.replace("{{enhanced}}", enhanced_text)
    return gemini_generate(prompt)


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(
    page_title="Context → JSON Prompt Generator",
    layout="wide"
)

st.title("Context → JSON Prompt Generator")

context = st.text_area("Enter your video idea / scene context:", height=150)
generate_btn = st.button("Generate JSON Prompt")


if generate_btn:

    if not context.strip():
        st.error("Please enter some context first.")
        st.stop()

    # -----------------------------------------
    # EXTRACTION
    # -----------------------------------------
    with st.spinner("Extracting structured details..."):
        extracted = step_extract(context)

    # -----------------------------------------
    # ENHANCEMENT
    # -----------------------------------------
    with st.spinner("Enhancing cinematic richness..."):
        enhanced = step_enhance(extracted)

    # -----------------------------------------
    # FINAL ASSEMBLER → JSON
    # -----------------------------------------
    with st.spinner("Assembling final JSON prompt..."):
        final_prompt = step_assemble(enhanced)

    # -----------------------------------------
    # Display
    # -----------------------------------------
    st.subheader("🧩 Extracted Details")
    st.code(extracted)

    st.subheader("✨ Enhanced Cinematic Details")
    st.code(enhanced)

    st.subheader("📦 Final JSON Prompt (Ready for Video Models)")
    st.code(final_prompt, language="json")
