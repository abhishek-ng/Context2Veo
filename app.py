import streamlit as st
import google.generativeai as genai
import json

# -------------------------------------------------
# Load API Keys
# -------------------------------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
HF_TOKEN = st.secrets["HF_TOKEN"]

# (Prepared for future video generation – not used yet in this file)
hf_client = InferenceClient(
    "ali-vilab/text-to-video-ms-1.7b",
    token=HF_TOKEN,
)

MODEL_NAME = "models/gemini-2.5-pro"


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_prompt(path: str) -> str:
    """Load a prompt template from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def gemini_generate(prompt: str) -> str:
    """Call Gemini with JSON-only response."""
    model = genai.GenerativeModel(MODEL_NAME)

    response = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json"
        }
    )
    # Gemini returns a JSON string here
    return response.text


def pretty_json(text: str) -> str:
    """Prettify a JSON string for display; fallback to raw if parsing fails."""
    try:
        obj = json.loads(text)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return text


# -------------------------------------------------
# Load Prompt Templates
# -------------------------------------------------
# Long-form pipeline templates
story_bible_template = load_prompt("prompts/story_bible.txt")
shot_list_template = load_prompt("prompts/shot_list_generator.txt")
shot_enhancer_template = load_prompt("prompts/shot_enhancer.txt")
long_video_assembler_template = load_prompt("prompts/long_video_assembler.txt")

# (Optional: legacy single-shot templates if you still want them)
# extractor_template = load_prompt("prompts/extractor.txt")
# enhancer_template = load_prompt("prompts/enhancer.txt")
# assembler_template = load_prompt("prompts/assembler.txt")


# -------------------------------------------------
# Processing Steps — Long-form Pipeline
# -------------------------------------------------
def step_story_bible(context_text: str) -> str:
    """
    Phase 1: Generate GLOBAL STORY BIBLE from user context.
    Returns: JSON string.
    """
    prompt = story_bible_template.replace("{{context}}", context_text)
    return gemini_generate(prompt)


def step_shot_list(story_bible_json: str) -> str:
    """
    Phase 2: Generate SHOT LIST (sequence of 24 shots).
    Returns: JSON string.
    """
    prompt = shot_list_template.replace("{{bible}}", story_bible_json)
    return gemini_generate(prompt)


def step_expand_shots(shot_list_json: str, story_bible_json: str) -> str:
    """
    Phase 3: Expand each shot into a full cinematic prompt.
    Returns: JSON string (array of shots).
    """
    prompt = (
        shot_enhancer_template
        .replace("{{bible}}", story_bible_json)
        .replace("{{shots}}", shot_list_json)
    )
    return gemini_generate(prompt)


def step_long_assemble(expanded_shots_json: str) -> str:
    """
    Phase 4: Assemble expanded shots into MASTER long-video JSON.
    Returns: JSON string.
    """
    prompt = long_video_assembler_template.replace("{{expanded}}", expanded_shots_json)
    return gemini_generate(prompt)


# -------------------------------------------------
# (Optional) Placeholder for future video generation
# -------------------------------------------------
# def generate_video_clip_from_prompt(prompt: str):
#     """
#     Example placeholder for calling the HF text-to-video model.
#     You can adapt this to the exact API signature you prefer.
#     """
#     result = hf_client.post(json={"inputs": prompt})
#     # handle result (e.g., save as file) here
#     return result


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(
    page_title="Long-form Video Prompt Generator",
    layout="wide"
)

st.title("🎬 Context → 2-Min Multi-Shot Video Prompt Generator")

st.markdown(
    """
Turn a single **story context** into a full **2-minute cinematic shot sequence**.

Pipeline:
1. Build a **Story Bible** (characters, style, rules)  
2. Generate a sequential **Shot List** (24 shots)  
3. Expand each shot into a **cinematic video prompt**  
4. Assemble everything into a **Master JSON** ready for text-to-video models  
"""
)

context = st.text_area(
    "Enter your video idea / scene context:",
    height=160,
    placeholder="Example: In his cluttered study, Isaac Newton contemplates the falling apple..."
)

col1, col2 = st.columns([1, 2])

with col1:
    generate_btn = st.button("Generate Long-Form JSON")

with col2:
    st.info("Each shot is ~8 seconds. 24 shots ≈ 2 minutes.")


if generate_btn:
    if not context.strip():
        st.error("Please enter some context first.")
        st.stop()

    # -----------------------------------------
    # PHASE 1 — STORY BIBLE
    # -----------------------------------------
    with st.spinner("📘 Building global story bible..."):
        story_bible_json = step_story_bible(context)
        story_bible_pretty = pretty_json(story_bible_json)

    # -----------------------------------------
    # PHASE 2 — SHOT LIST
    # -----------------------------------------
    with st.spinner("🎞️ Generating sequential shot list..."):
        shot_list_json = step_shot_list(story_bible_json)
        shot_list_pretty = pretty_json(shot_list_json)

    # -----------------------------------------
    # PHASE 3 — CINEMATIC EXPANSION
    # -----------------------------------------
    with st.spinner("🎬 Expanding each shot into cinematic prompts..."):
        expanded_shots_json = step_expand_shots(shot_list_json, story_bible_json)
        expanded_shots_pretty = pretty_json(expanded_shots_json)

    # -----------------------------------------
    # PHASE 4 — MASTER LONG-VIDEO JSON
    # -----------------------------------------
    with st.spinner("📦 Assembling final 2-minute video JSON..."):
        master_json = step_long_assemble(expanded_shots_json)
        master_json_pretty = pretty_json(master_json)

    # -----------------------------------------
    # Display Results
    # -----------------------------------------
    st.markdown("---")

    st.subheader("📘 Story Bible (Global Consistency)")
    st.code(story_bible_pretty, language="json")

    st.subheader("🎞️ Shot List (Storyboard of 24 Shots)")
    st.code(shot_list_pretty, language="json")

    st.subheader("🎬 Expanded Cinematic Shots (Per-Shot Prompts)")
    st.code(expanded_shots_pretty, language="json")

    st.subheader("📦 Final Master JSON (Ready for Text-to-Video Models)")
    st.code(master_json_pretty, language="json")

    st.success("Done! You can now use each per-shot prompt with your text-to-video model and stitch the clips together.")
