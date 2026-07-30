# ♻️ Sort the Trash! — Kids' Educational Game

An interactive Streamlit game that teaches kids to sort trash correctly,
powered by the model we trained (EfficientNetB0) on 6 classes:
biological, cardboard, glass, metal, paper, plastic.

## About the Project

**Sort the Trash!** is an interactive educational game designed to teach children how to sort waste correctly in a fun and engaging way. The game uses a deep learning model based on **EfficientNetB0**, trained to recognize six categories of waste:

- 🌱 Biological
- 📦 Cardboard
- 🍾 Glass
- 🥫 Metal
- 📄 Paper
- 🧴 Plastic

Players are shown a real image of a waste item and must choose the correct recycling bin. The AI model predicts the waste category, and the player earns points for correct answers while learning proper recycling habits. The game includes a scoring system, streak bonuses, sound effects, and colorful visual feedback to make learning enjoyable for children.

## How to run

1. Put these files together in one folder:
   ```
   recycle_game/
   ├── app.py
   ├── model_builder.py
   ├── requirements.txt
   ├── garbage_classifier_effnetb0.weights.h5   <-- from the training notebook
   ├── class_names.json                          <-- from the training notebook
   ├── samples/
   │   ├── biological/   (a few .jpg/.png images)
   │   ├── cardboard/
   │   ├── glass/
   │   ├── metal/
   │   ├── paper/
   │   └── plastic/
   └── assets/
       └── sounds/
           ├── correct.wav   <-- short "ding"/cheer sound
           └── wrong.wav     <-- short "buzz"/oops sound
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the game:
   ```
   streamlit run app.py
   ```

## How to play

- A real trash photo is shown to the child.
- The child clicks the bin they think is correct (♻️ Organic / Cardboard / Glass / Metal / Paper / Plastic).
- The model actually classifies the image. If the child's choice matches the model:
  - ✅ +10 points, a confetti celebration, and a +5 bonus every 3 correct answers in a row.
- If it doesn't match:
  - ❌ -5 points, with an encouraging message showing the correct classification.
- Score and streak are always visible at the top, with a "Start over" button in the sidebar.

---

## 👥 Team

- **Zeyad Abdalkader**  
  LinkedIn: https://www.linkedin.com/in/zeyad-abdalkader/

- **Kareem Okeil**  
  LinkedIn: https://www.linkedin.com/in/kareem-okeil/

- **Ali Elzihdany**  
  LinkedIn: https://www.linkedin.com/in/ali-elzihdany

- **Mazen Samir**  
  LinkedIn: https://www.linkedin.com/in/mazen--samir

---

## 🎥 Demo Video

📹 **Gameplay Demo:** [Watch the demo](assets/demo.mp4)