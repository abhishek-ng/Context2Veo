import streamlit as st
import google.generativeai as genai
from huggingface_hub import InferenceClient

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
    response = model.generate_content(prompt)
    return response.text


# -------------------------------------------------
# Steps
# -------------------------------------------------
def step_extract(context_text):
    return gemini_generate(extractor_template.replace("{{context}}", context_text))

def step_enhance(details_text):
    return gemini_generate(enhancer_template.replace("{{details}}", details_text))

def step_assemble(enhanced_text):
    return gemini_generate(assembler_template.replace("{{enhanced}}", enhanced_text))


# -------------------------------------------------
# Streamlit
# -------------------------------------------------
st.set_page_config(page_title="Gemini → HunyuanVideo", layout="wide")

st.title("🎬 Gemini Prompt Generator + Hunyuan Video Creator")

context = st.text_area("Enter your idea", height=150)
generate_btn = st.button("Generate Prompt + Video 🚀")

if generate_btn:

    if not context.strip():
        st.error("Enter context first.")
        st.stop()

    with st.spinner("Extracting..."):
        extracted = step_extract(context)

    with st.spinner("Enhancing..."):
        enhanced = step_enhance(extracted)

    with st.spinner("Finalizing prompt..."):
        final_prompt = step_assemble(enhanced)

    st.subheader("🧩 Extracted")
    st.code(extracted)

    st.subheader("✨ Enhanced")
    st.code(enhanced)

    st.subheader("🎥 Final Prompt")
    st.code(final_prompt)

    st.subheader("🎞 Generating Video (Hunyuan)...")

    with st.spinner("Rendering video..."):
        video_bytes = hf_client.text_to_video(final_prompt)

    st.video(video_bytes)
    st.success("Video created successfully!")
