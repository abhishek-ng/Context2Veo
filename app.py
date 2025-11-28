import streamlit as st
import google.generativeai as genai
from huggingface_hub import InferenceClient

# -------------------------------------------------
# Load API Keys from Streamlit Secrets
# -------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
HF_TOKEN = st.secrets["HF_TOKEN"]

# HuggingFace Client (NO PROVIDER)
hf_client = InferenceClient(token=HF_TOKEN)

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
# Gemini Text Generation Helper
# -------------------------------------------------
def gemini_generate(prompt: str) -> str:
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text

# -------------------------------------------------
# 3-Step Prompt Pipeline
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
st.set_page_config(page_title="Gemini → SVD Video Generator", layout="wide")

st.title("🎬 Gemini Prompt Generator + Stable Video Diffusion Creator")
st.write("Give context → generate cinematic prompt → auto-create video using Stable Video Diffusion.")

context = st.text_area(
    "Describe your idea",
    placeholder="Example: A peaceful village morning with fog, birds, and a slow camera pan...",
    height=150
)

generate_btn = st.button("Generate Prompt + Video 🚀")

# -------------------------------------------------
# Execution
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
    with st.spinner("Building final cinematic prompt..."):
        final_prompt = step_assemble(enhanced)

    # Show prompt pipeline
    st.subheader("🧩 Extracted Details")
    st.code(extracted)

    st.subheader("✨ Enhanced Version")
    st.code(enhanced)

    st.subheader("🎥 Final Video Prompt")
    st.code(final_prompt)

    st.success("Prompt generation complete!")

    # -------------------------------------------------
    # Generate Video (Stable Video Diffusion)
    # -------------------------------------------------
    st.subheader("🎞️ Generating Video with Stable Video Diffusion...")

    with st.spinner("Creating video... This may take 20–40 seconds."):

        # Text → Video
        video_bytes = hf_client.text_to_video(
            prompt=final_prompt,
            model="stabilityai/stable-video-diffusion-text2vid",
        )

    st.video(video_bytes)
    st.success("Video generated successfully!")
