"""
Kids' educational game: sort trash into the correct recycling bin
=====================================================================
The child sees a real photo (glass, paper, plastic, etc.) and clicks on
the bin they think is correct. The trained model (EfficientNetB0)
actually classifies the image, and if the child's choice matches the
model's prediction they earn points and a reward; if not, they get a
friendly nudge to try again.
 
Run:
    streamlit run app.py

Requirements (place next to app.py):
    - garbage_classifier_effnetb0.keras   (the trained model)
    - class_names.json                     (["biological","cardboard","glass","metal","paper","plastic"])
    - samples/<class_name>/*.jpg           (folder of real images grouped by class, used in the game)
"""

import base64
import json
import random
import time
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from tensorflow import keras

from model_builder import build_model

# ----------------------------------------------------------------------
# General settings
# ----------------------------------------------------------------------
APP_DIR = Path(__file__).parent
# Weights-only file (avoids Keras version mismatch issues across machines)
MODEL_PATH = APP_DIR / "garbage_classifier_effnetb0.weights.h5"
CLASS_NAMES_PATH = APP_DIR / "class_names.json"
SAMPLES_DIR = APP_DIR / "samples"
IMG_SIZE = 224

CORRECT_POINTS = 10
WRONG_POINTS = -5
STREAK_BONUS = 5  # bonus points every 3 correct answers in a row

# Sound effects (place .wav files here)
SOUND_DIR = Path("assets/sounds")
CORRECT_SOUND_FILE = "correct.wav"
WRONG_SOUND_FILE = "wrong.wav"

# Reaction video shown after every answer (right or wrong)
REACTION_VIDEO_URL = "https://youtu.be/XFbLZyEq_mg?si=eiu0JhEEZWONHjt2"

# Metadata for each bin: display name, icon, and accent color
BIN_META = {
    "biological": {"name": "Organic",   "emoji": "🍂", "color": "#6d4c41"},
    "cardboard":  {"name": "Cardboard", "emoji": "📦", "color": "#c17a3f"},
    "glass":      {"name": "Glass",     "emoji": "🍾", "color": "#2e7d32"},
    "metal":      {"name": "Metal",     "emoji": "🥫", "color": "#78909c"},
    "paper":      {"name": "Paper",     "emoji": "📄", "color": "#1e88e5"},
    "plastic":    {"name": "Plastic",   "emoji": "🧴", "color": "#fbc02d"},
}

st.set_page_config(page_title="Sort the Trash!", page_icon="♻️", layout="wide")


# ----------------------------------------------------------------------
# Load the model and sample images (cached for performance)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model_and_classes():
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = build_model(num_classes=len(class_names), img_size=IMG_SIZE)
    model.load_weights(MODEL_PATH)
    return model, class_names


@st.cache_data
def load_sample_pool(_class_names):
    """Returns dict: class_name -> list of image paths"""
    pool = {}
    for c in _class_names:
        class_dir = SAMPLES_DIR / c
        if class_dir.exists():
            imgs = [
                str(p) for p in class_dir.iterdir()
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")
            ]
            pool[c] = imgs
    return pool


def predict_image(model, class_names, img_path):
    img = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)  # model was trained without manual rescaling
    probs = model.predict(arr, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], float(probs[pred_idx]), probs


# ----------------------------------------------------------------------
# CSS styling (colorful bins, bounce animation, image card)
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Baloo 2', sans-serif;
    }

    .score-banner {
        background: linear-gradient(135deg, #43a047, #1e88e5);
        border-radius: 20px;
        padding: 14px 24px;
        color: white;
        text-align: center;
        font-size: 26px;
        font-weight: 700;
        box-shadow: 0 6px 16px rgba(0,0,0,0.15);
        margin-bottom: 18px;
    }

    .trash-card {
        background: white;
        border-radius: 24px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        border: 4px dashed #90a4ae;
        animation: floaty 2.5s ease-in-out infinite;
    }

    @keyframes floaty {
        0%   { transform: translateY(0px); }
        50%  { transform: translateY(-8px); }
        100% { transform: translateY(0px); }
    }

    .bin-slot {
        position: relative;
        text-align: center;
        padding-top: 16px;
    }

    .bin-lid {
        width: 76%;
        height: 16px;
        margin: 0 auto -6px auto;
        border-radius: 8px;
        position: relative;
        z-index: 3;
        filter: brightness(0.72);
        transform-origin: 15% center;
        box-shadow: 0 3px 0 rgba(0,0,0,0.25);
        transition: transform 0.2s ease;
    }
    .bin-lid::after {
        content: "";
        position: absolute;
        left: 50%;
        top: 50%;
        width: 26%;
        height: 55%;
        background: rgba(0,0,0,0.4);
        border-radius: 50%;
        transform: translate(-50%, -50%);
    }

    .bin-slot.opening .bin-lid {
        animation: binLidOpen 0.5s ease forwards;
    }
    .bin-slot.opening .bin-button button {
        animation: binBounce 0.55s ease;
    }

    @keyframes binLidOpen {
        0%   { transform: rotate(0deg) translateY(0); }
        50%  { transform: rotate(-35deg) translateY(-8px); }
        100% { transform: rotate(-22deg) translateY(-6px); }
    }
    @keyframes binBounce {
        0%, 100% { transform: translateY(0) scale(1); }
        30%      { transform: translateY(-5px) scale(1.04); }
        60%      { transform: translateY(3px) scale(0.97); }
    }

    .bin-fall-icon {
        position: absolute;
        left: 50%;
        top: -50px;
        font-size: 34px;
        z-index: 4;
        animation: trashDropIn 0.75s ease forwards;
        animation-delay: 0.15s;
        opacity: 0;
    }
    @keyframes trashDropIn {
        0%   { top: -50px; opacity: 1; transform: translateX(-50%) scale(1) rotate(0deg); }
        65%  { top: 28px;  opacity: 1; transform: translateX(-50%) scale(0.7) rotate(20deg); }
        100% { top: 42px;  opacity: 0; transform: translateX(-50%) scale(0.35) rotate(35deg); }
    }

    .bin-button button {
        width: 100%;
        height: 110px;
        border: none;
        font-size: 18px;
        font-weight: 700;
        color: white;
        clip-path: polygon(10% 0%, 90% 0%, 100% 100%, 0% 100%);
        border-radius: 4px 4px 16px 16px;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 6px 0 rgba(0,0,0,0.2);
    }
    .bin-button button:hover:not(:disabled) {
        transform: translateY(-4px) scale(1.03);
        box-shadow: 0 10px 0 rgba(0,0,0,0.2);
    }
    .bin-button button:active:not(:disabled) {
        transform: translateY(2px) scale(0.98);
        box-shadow: 0 2px 0 rgba(0,0,0,0.2);
    }
    .bin-button button:disabled {
        opacity: 0.55;
    }

    .feedback-correct {
        background: #e8f5e9;
        border: 3px solid #43a047;
        color: #2e7d32;
        border-radius: 16px;
        padding: 14px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        animation: pop 0.4s ease;
    }
    .feedback-wrong {
        background: #ffebee;
        border: 3px solid #e53935;
        color: #c62828;
        border-radius: 16px;
        padding: 14px;
        text-align: center;
        font-size: 22px;
        font-weight: 700;
        animation: shake 0.4s ease;
    }

    @keyframes pop {
        0% { transform: scale(0.7); opacity: 0; }
        100% { transform: scale(1); opacity: 1; }
    }
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-8px); }
        75% { transform: translateX(8px); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_sound_b64(filename: str) -> str | None:
    """Reads a .wav file from assets/sounds and returns it as base64, or
    None if the file isn't there yet (so the app doesn't crash if sounds
    haven't been added)."""
    path = SOUND_DIR / filename
    if not path.exists():
        return None
    data = path.read_bytes()
    return base64.b64encode(data).decode()


def play_sound(filename: str):
    """Autoplays a short sound effect. This runs right after a button
    click (Next / drop), which counts as user interaction, so browsers
    allow the audio to play with sound."""
    if not st.session_state.get("sound_on", True):
        return
    b64 = load_sound_b64(filename)
    if b64 is None:
        return  # sound file not found, skip silently
    st.markdown(
        f'<audio autoplay="true" style="display:none">'
        f'<source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )


def confetti_burst():
    """Fires a JS confetti celebration when the child answers correctly"""
    components.html(
        """
        <canvas id="confetti-canvas" style="position:fixed;top:0;left:0;width:100%;height:100%;
        pointer-events:none;z-index:9999;"></canvas>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/canvas-confetti/1.9.2/confetti.browser.min.js"></script>
        <script>
            const canvas = document.getElementById('confetti-canvas');
            const myConfetti = confetti.create(canvas, { resize: true, useWorker: true });
            myConfetti({
                particleCount: 140,
                spread: 90,
                origin: { y: 0.6 }
            });
        </script>
        """,
        height=0,
        width=0,
    )


# ----------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------
def init_state(class_names, pool):
    if "score" not in st.session_state:
        st.session_state.score = 0
        st.session_state.streak = 0
        st.session_state.round_num = 1
        st.session_state.feedback = None  # ("correct"/"wrong", message)
        st.session_state.used_images = set()
        st.session_state.current_image = None
        st.session_state.current_true_label = None
        st.session_state.last_clicked_bin = None
        pick_new_image(class_names, pool)


def pick_new_image(class_names, pool):
    available_classes = [c for c in class_names if pool.get(c)]
    if not available_classes:
        st.session_state.current_image = None
        return

    true_label = random.choice(available_classes)
    candidates = [p for p in pool[true_label] if p not in st.session_state.used_images]
    if not candidates:
        # reset the used pool once the sample set is exhausted
        st.session_state.used_images = set()
        candidates = pool[true_label]

    img_path = random.choice(candidates)
    st.session_state.used_images.add(img_path)
    st.session_state.current_image = img_path
    st.session_state.current_true_label = true_label
    st.session_state.feedback = None
    st.session_state.last_clicked_bin = None


def handle_bin_click(chosen_bin, model, class_names):
    img_path = st.session_state.current_image
    pred_label, confidence, _ = predict_image(model, class_names, img_path)

    st.session_state.last_clicked_bin = chosen_bin
    is_correct = (chosen_bin == pred_label)

    if is_correct:
        st.session_state.streak += 1
        bonus = STREAK_BONUS if st.session_state.streak % 3 == 0 else 0
        st.session_state.score += CORRECT_POINTS + bonus
        msg = (
            f"Correct! 🎉 The model agrees, it's **{BIN_META[pred_label]['name']}** "
            f"({confidence*100:.0f}% sure) +{CORRECT_POINTS + bonus} points"
        )
        st.session_state.feedback = ("correct", msg)
    else:
        st.session_state.streak = 0
        st.session_state.score = max(0, st.session_state.score + WRONG_POINTS)
        msg = (
            f"Nice try! The model thinks it's actually **{BIN_META[pred_label]['name']}** "
            f"{BIN_META[pred_label]['emoji']}, not {BIN_META[chosen_bin]['name']}. "
            f"Try again! {WRONG_POINTS} points"
        )
        st.session_state.feedback = ("wrong", msg)

    st.session_state.round_num += 1


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------
def main():
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        st.error(
            "Model files not found. Make sure `garbage_classifier_effnetb0.weights.h5` "
            "and `class_names.json` are placed next to app.py."
        )
        return

    model, class_names = load_model_and_classes()
    pool = load_sample_pool(class_names)

    if not any(pool.values()):
        st.error(
            "No sample images found. Add a `samples/<class_name>/` folder with images "
            "next to app.py, e.g. samples/glass/bottle1.jpg"
        )
        return

    init_state(class_names, pool)

    tab_play, tab_upload = st.tabs(["🎮 Play the Game", "📸 Try Your Own Photo!"])

    with tab_play:
        render_game_tab(model, class_names, pool)

    with tab_upload:
        render_upload_tab(model, class_names)

    # -------------------- Sidebar: stats and restart --------------------
    with st.sidebar:
        st.header("📊 Your Stats")
        st.metric("Score", st.session_state.score)
        st.metric("Current streak", st.session_state.streak)
        st.metric("Rounds played", st.session_state.round_num - 1)
        st.divider()
        st.session_state.sound_on = st.toggle(
            "🔊 Sound effects", value=st.session_state.get("sound_on", True)
        )
        st.divider()
        if st.button("🔄 Start over", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def render_game_tab(model, class_names, pool):
    # -------------------- Header / score banner --------------------
    st.markdown(
        f"""
        <div class="score-banner">
            ♻️ Sort the Trash! &nbsp;|&nbsp; Score: {st.session_state.score} 🌟
            &nbsp;|&nbsp; Streak: {st.session_state.streak} 🔥
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1.4], gap="large")

    # -------------------- Image card --------------------
    with col_left:
        if st.session_state.current_image:
            st.markdown('<div class="trash-card">', unsafe_allow_html=True)
            st.image(st.session_state.current_image, use_container_width=True)
            st.markdown(
                "<p style='font-size:20px; font-weight:700; color:#546e7a;'>"
                "🤔 What kind of trash is this? Drop it in the right bin!</p>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No images available right now.")

        # -------------------- Feedback message --------------------
        if st.session_state.feedback:
            kind, msg = st.session_state.feedback
            css_class = "feedback-correct" if kind == "correct" else "feedback-wrong"
            st.markdown(f'<div class="{css_class}">{msg}</div>', unsafe_allow_html=True)
            if kind == "correct":
                confetti_burst()
                play_sound(CORRECT_SOUND_FILE)
            else:
                play_sound(WRONG_SOUND_FILE)

            with st.expander("🎬 Watch this!", expanded=True):
                st.video(REACTION_VIDEO_URL)

            if st.button("➡️ Next image", use_container_width=True):
                pick_new_image(class_names, pool)
                st.rerun()

    # -------------------- The six bins --------------------
    with col_right:
        st.markdown(
            "<p style='font-size:20px; font-weight:700; color:#37474f;'>Pick a bin:</p>",
            unsafe_allow_html=True,
        )
        bin_cols = st.columns(3)
        for i, c in enumerate(class_names):
            meta = BIN_META.get(c, {"name": c, "emoji": "🗑️", "color": "#607d8b"})
            is_opening = (
                st.session_state.feedback is not None
                and st.session_state.last_clicked_bin == c
            )
            with bin_cols[i % 3]:
                slot_class = "bin-slot opening" if is_opening else "bin-slot"
                st.markdown(f'<div class="{slot_class}">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="bin-lid" style="background:{meta["color"]};"></div>',
                    unsafe_allow_html=True,
                )
                if is_opening:
                    st.markdown(
                        f'<div class="bin-fall-icon">{meta["emoji"]}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"""
                    <style>
                    div[data-testid="stButton"] > button#bin_{c} {{
                        background-color: {meta['color']};
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="bin-button">', unsafe_allow_html=True)
                clicked = st.button(
                    f"{meta['emoji']}\n{meta['name']}",
                    key=f"bin_{c}",
                    use_container_width=True,
                    disabled=st.session_state.feedback is not None,
                )
                st.markdown("</div>", unsafe_allow_html=True)  # close bin-button
                st.markdown("</div>", unsafe_allow_html=True)  # close bin-slot
                if clicked:
                    handle_bin_click(c, model, class_names)
                    st.rerun()


def render_upload_tab(model, class_names):
    st.markdown(
        """
        <div class="score-banner" style="background: linear-gradient(135deg, #8e24aa, #ec407a);">
            📸 Upload a photo and let the robot guess what it is!
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a photo of some trash (jpg, jpeg, or png)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info("👆 Upload a picture to see what the robot thinks it is!")
        return

    col_img, col_result = st.columns([1, 1.2], gap="large")

    with col_img:
        st.markdown('<div class="trash-card">', unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        with st.spinner("🤖 The robot is thinking..."):
            img = Image.open(uploaded_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.expand_dims(np.array(img).astype("float32"), axis=0)
            probs = model.predict(arr, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        pred_label = class_names[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        meta = BIN_META.get(pred_label, {"name": pred_label, "emoji": "🗑️", "color": "#607d8b"})

        st.markdown(
            f"""
            <div class="feedback-correct" style="font-size:26px;">
                {meta['emoji']} It's <b>{meta['name']}</b>!<br>
                <span style="font-size:16px;">The robot is {confidence:.0f}% sure</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if confidence >= 80:
            confetti_burst()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**How sure is the robot about each type?**")
        for i, c in enumerate(class_names):
            c_meta = BIN_META.get(c, {"name": c, "emoji": "🗑️"})
            st.progress(
                float(probs[i]),
                text=f"{c_meta['emoji']} {c_meta['name']} — {probs[i]*100:.0f}%",
            )


if __name__ == "__main__":
    main()
