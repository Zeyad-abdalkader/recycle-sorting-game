# ♻️ Sort the Trash! — Kids' Educational Game

An interactive Streamlit game that teaches kids to sort trash correctly,
powered by the model we trained (EfficientNetB0) on 6 classes:
biological, cardboard, glass, metal, paper, plastic.

## How to run

1. Put these files together in one folder:
   ```
   recycle_game/
   ├── app.py
   ├── model_builder.py
   ├── requirements.txt
   ├── garbage_classifier_effnetb0.weights.h5   <-- from the training notebook
   ├── class_names.json                          <-- from the training notebook
   └── samples/
       ├── biological/   (a few .jpg/.png images)
       ├── cardboard/
       ├── glass/
       ├── metal/
       ├── paper/
       └── plastic/
   ```

2. If you have `test_df` from the training notebook, you can generate the
   `samples/` folder from it with this snippet:

   In the training notebook, save weights-only instead of the full model
   (this avoids Keras/TensorFlow version mismatches between machines):
   ```python
   model.save_weights("garbage_classifier_effnetb0.weights.h5")
   ```

   ```python
   import shutil, os

   SAMPLES_OUT = "samples"
   N_PER_CLASS = 15  # images per class to use in the game

   for c in class_names:
       out_dir = os.path.join(SAMPLES_OUT, c)
       os.makedirs(out_dir, exist_ok=True)
       class_imgs = test_df[test_df['label'] == c]['filepath'].sample(
           min(N_PER_CLASS, (test_df['label'] == c).sum()), random_state=42
       )
       for p in class_imgs:
           shutil.copy(p, out_dir)

   print("Game sample images ready ✅")
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the game:
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

## Optional customization

- Change `CORRECT_POINTS` / `WRONG_POINTS` / `STREAK_BONUS` near the top of `app.py` to tune scoring.
- Change `BIN_META` to adjust colors, icons, or bin names.
- To add sound on correct/wrong answers, drop in `.mp3` files and play them with `st.audio` or JS `Audio()`.
