# 🎬 Multi-Shot Cinematic JSON Generator  
### Turn a single story context into a fully structured 20-scene cinematic JSON pipeline

This project transforms raw story context into a **20-scene cinematic structure**, complete with:

- **Storyboard (20 scenes, 2–3 lines each)**  
- **Global visual style + character appearance**  
- **Per-scene action + behavior + mood + motion features**  
- **Merged JSON with global + local cinematic elements**  

This pipeline is designed for **long-form AI video generation**, where consistency across multiple 5–8 second clips is essential.

---

# 🚀 Features

## ✔ 1. Storyboard Generator  
Creates a clean 20-scene storyboard using `story_bible.txt`.  
Each scene is:

- 2–3 lines  
- simple visual storytelling  
- no cinematography terms  
- sequential narrative flow  

---

## ✔ 2. Global Visual Style Extractor  
Using `extract_visual_style.txt`, the system infers:

- Cinematography style  
- Lighting style  
- Color ambience / grading  
- Full character appearance  
- Global mood & tone  

All extracted from the storyboard.

---

## ✔ 3. Per-Scene Feature Extractor  
`extract_shot_features.txt` extracts per-scene:

- Action  
- Subject  
- Behavior  
- Camera motion  
- Motion dynamics  
- Mood & emotion  
- Sound / sensory cues  
- Additional notes  

---

## ✔ 4. Scene Merger → Final Cinematic JSON  
`merge_global.txt` merges:

- Global visual style  
- Character appearance  
- Per-scene features  

Into a **single 20-scene cinematic JSON**, perfect for text-to-video models.

---

# 📂 Project Structure

.
├── prompts/
│ ├── story_bible.txt
│ ├── extract_visual_style.txt
│ ├── extract_shot_features.txt
│ ├── merge_global.txt
│
├── app.py
├── requirements.txt
├── LICENSE
├── .gitignore
└── README.md
