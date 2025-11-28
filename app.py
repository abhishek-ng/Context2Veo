import streamlit as st
import google.generativeai as genai
from huggingface_hub import InferenceClient

# -------------------------------------------------
# Load API keys from Streamlit Secrets
# -------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
HF_TOKEN = st.secrets["HF_TOKEN"]

# HuggingFace client (Novita provider)
hf_client = InferenceClient(
    provider="novita",
    api_key=HF_TOKEN,
)

MODEL_NAME = "models/gemini-2.5-pro"

# -------------------------------------------------
# Helper: Load text templates
# -------------------------------------------------
def load_prompt(path):
    return open(path, "r", encoding="utf-8").read()

extractor_template = load_prompt("prompts/extractor.txt")
enhancer_template = load_prompt("prompts/enhancer.txt")
assembler_template = load_prompt("prompts/assembler.txt")


# -------------------------------------------------
# Helper to call Gemini model
# -------------------------------------------------
def gemini_generate(prompt: str) -> str:
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text


# -------------------------------------------------
# 3-STEP PIPELINE
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
st.set_page_config(page_title="Gemini → Wan Video Generator", layout="wide")

st.title("🎬 Gemini Prompt Generator + Wan 2.1 Video Creator")
st.write("Give a context → generate cinematic prompt → create video with Wan 2.1.")


context = st.text_area(
    "Describe your idea (context)",
    placeholder="Example: A peaceful village morning with fog, birds, and a slow camera pan...",
    height=150
)

generate_btn = st.button("Generate Prompt + Video 🚀")


# -------------------------------------------------
# Run Everything
# -------------------------------------------------
if generate_btn:

    if not context.strip():
        st.error("Please enter some context.")
        st.stop()

    # -------- Step 1 --------
    with st.spinner("Extracting structured details..."):
        extracted = step_extract(context)

    # -------- Step 2 --------
    with st.spinner("Enhancing cinematic richness..."):
        enhanced = step_enhance(extracted)

    # -------- Step 3 --------
    with st.spinner("Building final Veo-style prompt..."):
        final_prompt = step_assemble(enhanced)

    # Display outputs
    st.subheader("🧩 Extracted Details")
    st.code(extracted, language="markdown")

    st.subheader("✨ Enhanced Version")
    st.code(enhanced, language="markdown")

    st.subheader("🎥 Final Video Prompt (Gemini → Wan)")
    st.code(final_prompt, language="markdown")

    st.success("Prompt generation complete!")

    # -------------------------------------------------
    # Generate Video (WAN T2V)
    # -------------------------------------------------
    st.subheader("🎞️ Generating Video with Wan 2.1...")

    with st.spinner("Generating video (this takes ~20–40 seconds)..."):
        video = client.text_to_video(
                    prompt=final_prompt,
                    model="stabilityai/stable-video-diffusion-text2vid",
                )

    st.video(video_bytes)
    st.success("Video generated successfully!")
