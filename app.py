import streamlit as st
import google.generativeai as genai
import json
from pathlib import Path

# -------------------------------------------------
# Config
# -------------------------------------------------
MODEL_NAME = "models/gemini-2.5-pro"

# Expect: st.secrets["GEMINI_API_KEY"] set in Streamlit
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def load_prompt(relative_path: str) -> str:
    """
    Load a prompt template from the prompts/ directory
    using a path relative to the project root.
    """
    base_dir = Path(__file__).resolve().parent
    full_path = base_dir / relative_path
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()


def gemini_generate(prompt: str, json_mode: bool = False) -> str:
    """
    Call Gemini with the given prompt.
    If json_mode=True, request JSON mime type.
    """
    model = genai.GenerativeModel(MODEL_NAME)

    if json_mode:
        resp = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"},
        )
    else:
        resp = model.generate_content(prompt)

    return resp.text


def pretty_json(text: str) -> str:
    """
    Try to pretty-print JSON; if it fails, return original text.
    """
    try:
        obj = json.loads(text)
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return text


# -------------------------------------------------
# Load prompt templates
# -------------------------------------------------
story_bible_template = load_prompt("prompts/story_bible.txt")
extract_visual_style_template = load_prompt("prompts/extract_visual_style.txt")
extract_shot_features_template = load_prompt("prompts/extract_shot_features.txt")
merge_global_template = load_prompt("prompts/merge_global.txt")


# -------------------------------------------------
# Pipeline steps
# -------------------------------------------------
def step_storyboard(context_text: str) -> str:
    """
    Step 1: From raw context → 20-scene storyboard (plain text).
    Uses story_bible.txt (which we’re using as storyboard writer).
    """
    prompt = story_bible_template.replace("{{context}}", context_text)
    return gemini_generate(prompt, json_mode=False)


def step_extract_visual_style(storyboard_text: str) -> str:
    """
    Step 2: From storyboard → global visual style + characters (JSON).
    Uses extract_visual_style.txt.
    """
    prompt = extract_visual_style_template.replace("{{storyboard}}", storyboard_text)
    return gemini_generate(prompt, json_mode=True)


def step_extract_shot_features(storyboard_text: str) -> str:
    """
    Step 3: From storyboard → per-scene features (plain structured text).
    Uses extract_shot_features.txt.
    """
    prompt = extract_shot_features_template.replace("{{storyboard}}", storyboard_text)
    return gemini_generate(prompt, json_mode=False)


def step_merge_global(global_json_text: str, scene_features_text: str) -> str:
    """
    Step 4: Merge global visual style + per-scene features
    into a final JSON with 20 scenes, each including:
      - action, subject, behavior, camera_motion, motion_dynamics,
        mood_emotion, sound_sensory, additional_notes
      - cinematography_style, lighting_style, color_ambience, characters

    Uses merge_global.txt.
    """
    prompt = (
        merge_global_template
        .replace("{{global}}", global_json_text)
        .replace("{{scenes}}", scene_features_text)
    )
    return gemini_generate(prompt, json_mode=True)


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.set_page_config(
    page_title="Multi-Shot Cinematic Prompt Builder",
    layout="wide",
)

st.title("🎬 Context → 20-Scene Cinematic JSON Prompt")

st.markdown(
    """
This app turns a single story **context** into a structured **20-scene cinematic plan**:

1. **Storyboard** (20 scenes, 2–3 lines each)  
2. **Global visual style & character appearance**  
3. **Per-scene features** (action, behavior, mood, etc.)  
4. **Merged JSON** with all features for every scene  
"""
)

context = st.text_area(
    "Enter your story context:",
    height=160,
    placeholder="Example: A young boy practices hockey alone on a frozen pond, preparing for his first big tournament...",
)

generate_btn = st.button("Generate 20-Scene Cinematic JSON")


if generate_btn:
    if not context.strip():
        st.error("Please enter some context first.")
        st.stop()

    # -----------------------------------------
    # Step 1 — Storyboard
    # -----------------------------------------
    with st.spinner("✏️ Generating 20-scene storyboard..."):
        storyboard = step_storyboard(context)

    # -----------------------------------------
    # Step 2 — Global visual style
    # -----------------------------------------
    with st.spinner("🎨 Extracting global visual style & characters..."):
        global_style_json = step_extract_visual_style(storyboard)
        global_style_pretty = pretty_json(global_style_json)

    # -----------------------------------------
    # Step 3 — Per-scene features
    # -----------------------------------------
    with st.spinner("🧩 Extracting per-scene features..."):
        scene_features_text = step_extract_shot_features(storyboard)

    # -----------------------------------------
    # Step 4 — Merge into final per-scene JSON
    # -----------------------------------------
    with st.spinner("📦 Merging global style into all 20 scenes..."):
        merged_json = step_merge_global(global_style_json, scene_features_text)
        merged_pretty = pretty_json(merged_json)

    # -----------------------------------------
    # Display sections
    # -----------------------------------------
    st.markdown("---")

    st.subheader("📝 Storyboard (20 Scenes)")
    st.text(storyboard)

    st.subheader("🎬 Final 20-Scene Cinematic JSON (with global consistency)")
    st.code(merged_pretty, language="json")

    st.success("Done! Each scene now contains both local actions and global cinematic style.")
