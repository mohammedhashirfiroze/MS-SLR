"""
MS-SLR: Motion-Signature Sign Language Recognition
Real-Time Edge AI for Accessible Communication

Author: Mohammed Hashir Firoze
Institution: University of Wollongong in Dubai
Project: Bachelors's Capstone Project - Sign Language Recognition System
Date: February 2026
GitHub: https://github.com/mohammedhashirfiroze/MS-SLR

Description:
    Real-time sign language recognition system achieving 89.5% accuracy with 
    sub-100ms latency on commodity hardware. Implements motion-signature temporal 
    feature extraction and hierarchical stabilization for production-grade 
    performance without GPU requirements.
    
    This system addresses the critical communication gap affecting 466 million 
    people worldwide with hearing loss by providing privacy-first, real-time 
    sign language translation suitable for healthcare, education, and public services.

Technical Innovation:
    - Motion-signature features: Captures velocity & variance over 10-frame windows
    - Temporal stabilization: Confidence-weighted voting reduces frame-level noise
    - Privacy-first architecture: No video storage, only skeletal landmarks
    - Edge deployment: CPU-only inference (<450MB RAM, <100ms latency)

Performance:
    - Top-1 Accuracy: 89.47% (Test set)
    - Top-5 Accuracy: 98.44%
    - Cross-Validation: 72.33% ± 0.42%
    - Real-world User Study: 72% accuracy with non-expert signers
    - Vocabulary: 150 curated ASL signs

Usage:
    python main.py
    
    Controls:
        A - Toggle auto mode (hands-free operation)
        SPACE - Manually add current prediction
        G - Process with AI grammar correction
        C - Clear sentence buffer
        Q - Quit application
"""

import cv2
import mediapipe as mp
import pickle
import numpy as np
import pyttsx3
import threading
from google import genai
from collections import deque  # Faster append/pop than list for rolling windows
import time
from queue import Queue
import json

# ============================================================================
# Configuration Loading
# ============================================================================
# External config allows tuning without code changes - important for deployment

try:
    with open('config.json', 'r') as f:
        config = json.load(f)
    GEMINI_API_KEY = config['gemini_api_key']
    CONFIDENCE_THRESHOLD = config['confidence_threshold']
    PREDICTION_SMOOTHING_WINDOW = config['smoothing_window']
    STABILITY_FRAMES = config['stability_frames']
    camera_indices_to_try = config['camera_indices']  # Try multiple indices since cameras vary by system
    USE_MOTION_SIGNATURE = config.get('use_motion_signature', True)
    MOTION_WINDOW_SIZE = config.get('motion_window_size', 10)  # 10 frames = 333ms at 30 FPS
    TWO_HAND_MODE = config.get('two_hand_mode', True)
    MODEL_PATH = config.get('model_path', "model_mslr_150.p")
    SENTENCE_BUFFER_SIZE = config.get('sentence_buffer_size', 20)
except FileNotFoundError:
    # Fallback to defaults if config missing - ensures system still runs
    print("⚠ config.json not found, using defaults")
    GEMINI_API_KEY = ""
    CONFIDENCE_THRESHOLD = 0.32  # Set to 32% after testing - balances precision/recall
    PREDICTION_SMOOTHING_WINDOW = 18  # Found 18 frames optimal for stability
    STABILITY_FRAMES = 20  # Requires 667ms of consistency before committing
    camera_indices_to_try = [1, 0, 2]
    USE_MOTION_SIGNATURE = True
    MOTION_WINDOW_SIZE = 10
    TWO_HAND_MODE = True
    MODEL_PATH = "model_mslr_150.p"
    SENTENCE_BUFFER_SIZE = 20

# UI color scheme - BGR format for OpenCV
COLOR_PRIMARY = (255, 140, 0)  # Orange for primary actions
COLOR_SUCCESS = (0, 255, 120)  # Green for high confidence predictions
COLOR_WARNING = (0, 255, 255)  # Yellow for medium confidence
COLOR_DANGER = (0, 80, 255)    # Red for low confidence
COLOR_TEXT = (255, 255, 255)   # White for general text

print("="*70)
print("MS-SLR: MOTION-SIGNATURE SIGN LANGUAGE RECOGNITION")
print("="*70)

# ============================================================================
# Model & System Initialization
# ============================================================================

print("\n[1/4] Loading trained model...", end=" ", flush=True)
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)  # ExtraTreesClassifier with 250 trees
    print(f"✓ Loaded ({len(model.classes_)} signs)")
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    print("Make sure model file exists. Run train_model_150.py first.")
    exit(1)

print("[2/4] Initializing hand tracking...", end=" ", flush=True)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Track both hands - critical for two-hand signs
    min_detection_confidence=0.7,  # Higher threshold reduces false positives
    min_tracking_confidence=0.7    # Maintains smooth tracking once detected
)
print("✓")

print("[3/4] Initializing text-to-speech...", end=" ", flush=True)
voice_engine = pyttsx3.init()
voice_engine.setProperty('rate', 160)  # 160 WPM for natural-sounding speech
print("✓")

print("[4/4] Connecting to AI grammar service...", end=" ", flush=True)
if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
        print("✓ Connected")
    except Exception as e:
        gemini_client = None
        print(f"⚠ Skipped ({e})")
else:
    gemini_client = None
    print("⚠ No API key (grammar disabled)")

# Camera initialization with fallback to multiple indices
print("\nSearching for camera...", end=" ", flush=True)
cap = None
for idx in camera_indices_to_try:
    cap = cv2.VideoCapture(idx)
    time.sleep(0.5)  # Brief delay for camera to initialize
    if cap.isOpened():
        ret, _ = cap.read()
        if ret:
            print(f"✓ Found at index {idx}")
            break
        cap.release()

if not cap or not cap.isOpened():
    print("\n✗ ERROR: No camera detected!")
    print("Check that:")
    print("  - Camera is connected")
    print("  - No other application is using it")
    print("  - Camera permissions are granted")
    exit(1)

print("\n" + "="*70)
print("SYSTEM READY")
print("="*70)
print("\nKEYBOARD CONTROLS:")
print("  A     = Toggle AUTO MODE (hands-free sign detection)")
print("  SPACE = Manually add current prediction")
print("  G     = Process sentence with AI grammar correction")
print("  C     = Clear sentence buffer")
print("  Z     = Undo last word")
print("  M/H   = Show quick help")
print("  Q     = Quit")
print("\nTIP: Auto mode is great for demos, manual mode for precise control")
print("="*70 + "\n")

# ============================================================================
# Global State Variables
# ============================================================================
# Using deques for O(1) append/popleft operations vs O(n) for lists

prediction_buffer = deque(maxlen=PREDICTION_SMOOTHING_WINDOW)  # Rolling window for temporal averaging
confidence_buffer = deque(maxlen=PREDICTION_SMOOTHING_WINDOW)
sentence_words = []  # Accumulated predictions forming sentence
last_prediction = None  # Track prediction consistency across frames
stable_frames = 0  # Counter for consecutive identical predictions
is_speaking = False  # Flag to prevent overlapping TTS
is_processing_grammar = False  # Flag to prevent concurrent grammar requests
motion_buffer = deque(maxlen=MOTION_WINDOW_SIZE)  # 10-frame window for motion signature
last_auto_word = None  # Debounce mechanism for auto mode
last_auto_time = 0.0
status_message = ""  # On-screen status display
status_message_time = 0.0
auto_mode = False  # Auto-detection vs manual capture
last_sentence_word_count = 0  # Track changes for auto-grammar trigger
last_auto_grammar_time = 0.0

# Thread-safe queues for async operations
voice_queue = Queue()
grammar_queue = Queue()

# ============================================================================
# Async Worker Threads
# ============================================================================
# Separate threads prevent blocking main loop during I/O operations

def voice_worker():
    """
    Background thread for TTS - prevents blocking during speech synthesis.
    Using thread instead of multiprocessing since pyttsx3 has initialization issues with fork.
    """
    global is_speaking
    while True:
        text = voice_queue.get()
        if text == "STOP":  # Shutdown signal
            break
        try:
            is_speaking = True
            voice_engine.say(text)
            voice_engine.runAndWait()  # Blocks this thread, not main loop
            is_speaking = False
        except Exception as e:
            print(f"TTS error: {e}")
            is_speaking = False

voice_thread = threading.Thread(target=voice_worker, daemon=True)
voice_thread.start()

def speak(text):
    """Queue text for speech synthesis without blocking."""
    voice_queue.put(text)

def grammar_worker():
    """
    Background thread for AI grammar correction - API calls can take 1-3 seconds.
    Daemon thread ensures clean shutdown if main thread exits.
    """
    global is_processing_grammar
    while True:
        words = grammar_queue.get()
        if words == "STOP":
            break
        try:
            is_processing_grammar = True
            raw_text = " ".join(words)
            
            # Prompt engineering: explicit instruction for gloss-to-English conversion
            prompt = f"""Convert these sign language glosses into a natural English sentence:

{raw_text}

Natural sentence:"""
            
            response = gemini_client.models.generate_content(
                model='gemini-2.0-flash-exp',  # Using Flash for low latency
                contents=prompt
            )
            
            # Strip quotes that LLM sometimes adds
            corrected = response.text.strip().strip('"\'')
            speak(corrected)
            is_processing_grammar = False
        except Exception as e:
            print(f"Grammar API error: {e}")
            is_processing_grammar = False
            # Fallback: speak raw glosses if AI fails
            speak(" ".join(words))

grammar_thread = threading.Thread(target=grammar_worker, daemon=True)
grammar_thread.start()

def process_grammar(words):
    """
    Queue sentence for grammar correction.
    Returns True if queued successfully, False if no grammar service available.
    """
    if gemini_client and words:
        grammar_queue.put(words.copy())  # Copy to avoid race condition
        return True
    elif words:
        # No AI available - just speak raw glosses
        speak(" ".join(words))
        return True
    return False

def show_status(message, duration=2.0):
    """
    Display temporary status message on screen.
    Using timestamp-based expiry instead of frame counter for consistency.
    """
    global status_message, status_message_time
    status_message = message
    status_message_time = time.time() + duration

# ============================================================================
# Feature Extraction Functions
# ============================================================================

def extract_two_hand_features(results):
    """
    Extract and order hand landmarks for consistency.
    
    Two-hand signs require consistent left/right ordering across frames.
    Strategy: Sort by wrist X-coordinate (left hand has smaller X).
    This normalization is critical for model generalization.
    
    Returns:
        126-element list: [hand_a (63 coords), hand_b (63 coords)]
        If only one hand detected, fills other with zeros.
    """
    hand_a = [0.0] * 63  # 21 landmarks × 3 coords (x,y,z)
    hand_b = [0.0] * 63
    hands_data = []

    # Extract raw coordinates from MediaPipe results
    for hand_landmarks in results.multi_hand_landmarks:
        coords = []
        for landmark in hand_landmarks.landmark:
            coords.extend([landmark.x, landmark.y, landmark.z])
        hands_data.append(coords)

    if len(hands_data) == 1:
        # Single hand - place in hand_a slot
        hand_a = hands_data[0]
    elif len(hands_data) >= 2:
        # Two hands detected - sort by wrist X position for consistency
        wrist_x = [hand.landmark[0].x for hand in results.multi_hand_landmarks[:2]]
        first_idx = 0 if wrist_x[0] <= wrist_x[1] else 1  # Left hand first
        second_idx = 1 - first_idx
        hand_a = hands_data[first_idx]
        hand_b = hands_data[second_idx]

    return hand_a + hand_b

def normalize_frame_features(features):
    """
    Wrist-relative normalization for person-independent recognition.
    
    Problem: Different people have different hand sizes and signing positions.
    Solution: Translate to wrist origin and scale to unit sphere.
    
    This is the key transformation enabling >70% cross-signer generalization.
    Without this, model would overfit to training signers' hand sizes.
    """
    features = np.array(features, dtype=np.float32)
    # Split into individual hands (63 features each)
    hands = [features[:63], features[63:]] if features.shape[0] > 63 else [features]

    normalized_hands = []
    for hand in hands:
        # Skip if hand not detected (all zeros)
        if np.allclose(hand, 0.0):
            normalized_hands.append(np.zeros_like(hand))
            continue
        
        points = hand.reshape(21, 3)  # Convert to 21 (x,y,z) points
        wrist = points[0]  # Wrist is landmark 0
        
        # Translate to wrist origin
        centered = points - wrist
        
        # Scale to unit sphere (max distance = 1.0)
        scale = np.linalg.norm(centered, axis=1).max()
        if scale <= 1e-6:  # Prevent division by zero
            scale = 1.0
        normalized = centered / scale
        
        normalized_hands.append(normalized.reshape(-1))

    return np.concatenate(normalized_hands) if len(normalized_hands) > 1 else normalized_hands[0]

def compute_motion_signature(window):
    """
    Motion-signature temporal feature extraction - PRIMARY INNOVATION.
    
    Traditional approaches: Analyze single frame hand pose.
    Our approach: Statistical descriptors over 10-frame window (333ms).
    
    Extracted features:
        - Mean (μ): Average spatial configuration
        - Velocity (v): Motion dynamics (first derivative)
        - Variance (σ²): Movement consistency
    
    Result: 378-dimensional vector (126 spatial × 3 temporal) that captures
    both WHAT the hand shape is and HOW it moves.
    
    Example: "HELP" and "SORRY" have similar hand shapes but different motions.
    Static features alone achieve 78% accuracy. Motion features: 89.5%.
    """
    window = np.array(window, dtype=np.float32)
    
    # Spatial feature: Average position over window
    mean = window.mean(axis=0)
    
    # Temporal feature: Velocity (frame-to-frame differences)
    deltas = np.diff(window, axis=0)  # Compute differences between consecutive frames
    velocity = deltas.mean(axis=0)
    
    # Consistency feature: Movement variability
    variance = window.var(axis=0)
    
    # Concatenate into single feature vector
    return np.concatenate([mean, velocity, variance])

def get_smoothed_prediction(current_pred, current_conf):
    """
    Temporal smoothing via majority voting over prediction buffer.
    
    Problem: Per-frame predictions jitter between similar signs.
    Solution: Track last N predictions and select most common.
    
    Using simple majority vote rather than weighted average since
    empirical testing showed it's more robust to sudden motion changes.
    """
    prediction_buffer.append(current_pred)
    confidence_buffer.append(current_conf)
    
    # Count occurrences of each prediction
    pred_counts = {}
    for pred in prediction_buffer:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    
    # Select most frequent prediction
    most_common = max(pred_counts, key=pred_counts.get)
    avg_confidence = np.mean(confidence_buffer)
    
    return most_common, avg_confidence

def draw_overlay_text(frame, text, pos, size=0.6, color=(255, 255, 255), thickness=2, bg=True):
    """
    Draw text with semi-transparent background for readability.
    
    Simple but critical for UX - ensures text readable regardless of background.
    Using addWeighted for smooth transparency instead of alpha channel.
    """
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)
    
    if bg:
        # Create overlay copy and blend
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (pos[0] - 5, pos[1] - text_height - 5),
                     (pos[0] + text_width + 5, pos[1] + baseline + 5),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

# ============================================================================
# Main Processing Loop
# ============================================================================

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera read failed - exiting")
        break
    
    frame = cv2.flip(frame, 1)  # Mirror for natural interaction
    height, width = frame.shape[:2]
    
    # MediaPipe requires RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    
    # Initialize prediction variables
    current_prediction = None
    current_confidence = 0.0
    top_5_predictions = []
    top_5_confidences = []
    
    # ========================================================================
    # Hand Detection & Classification
    # ========================================================================
    
    if results.multi_hand_landmarks:
        # Render hand landmarks on frame
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Extract features based on mode
        if TWO_HAND_MODE:
            features = extract_two_hand_features(results)
        else:
            # Single-hand mode (fallback for one-handed signs)
            features = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                features.extend([landmark.x, landmark.y, landmark.z])

        # Apply normalization
        features = normalize_frame_features(features)

        # Generate motion signature if enabled
        if USE_MOTION_SIGNATURE:
            motion_buffer.append(features)
            if len(motion_buffer) < MOTION_WINDOW_SIZE:
                # Need full window before first prediction
                frame_count += 1
                continue
            
            # Compute temporal features over window
            motion_features = compute_motion_signature(motion_buffer)
            features_array = motion_features.reshape(1, -1)
        else:
            # Static features only (baseline comparison)
            features_array = np.array(features).reshape(1, -1)

        # Model inference
        prediction_proba = model.predict_proba(features_array)[0]
        
        # Extract top-5 for display
        top_5_indices = np.argsort(prediction_proba)[-5:][::-1]
        top_5_predictions = model.classes_[top_5_indices].tolist()
        top_5_confidences = prediction_proba[top_5_indices].tolist()
        
        current_prediction = top_5_predictions[0]
        current_confidence = top_5_confidences[0]
        
        # Apply temporal smoothing
        current_prediction, current_confidence = get_smoothed_prediction(
            current_prediction, current_confidence
        )
        
        # Track stability (consecutive identical predictions)
        if current_prediction == last_prediction:
            stable_frames += 1
        else:
            stable_frames = 0
        
        last_prediction = current_prediction

        # ====================================================================
        # Auto-Add Logic (for auto mode)
        # ====================================================================
        # Only commit prediction if:
        #   1. Confidence exceeds threshold (32% empirically optimal)
        #   2. Stable for required frames (20 frames = 667ms)
        #   3. Different from last word OR sufficient time elapsed (debounce)
        
        if auto_mode and (current_prediction and 
            current_confidence > CONFIDENCE_THRESHOLD and 
            stable_frames >= STABILITY_FRAMES):
            
            now = time.monotonic()  # Monotonic clock for reliable timing
            
            # Debounce: 1.5s minimum between identical words
            if current_prediction != last_auto_word or (now - last_auto_time) > 1.5:
                sentence_words.append(current_prediction)
                
                # Limit buffer to prevent unbounded growth
                if len(sentence_words) > SENTENCE_BUFFER_SIZE:
                    sentence_words = sentence_words[-SENTENCE_BUFFER_SIZE:]
                
                last_auto_word = current_prediction
                last_auto_time = now
                show_status(f"Auto-added: {current_prediction}", 1.0)
    
    # ========================================================================
    # Auto Grammar Processing
    # ========================================================================
    # Trigger conditions:
    #   - At least 3 words (minimum for meaningful sentence)
    #   - 5 seconds elapsed OR 5+ words accumulated
    #   - Sentence changed since last processing
    
    if auto_mode and sentence_words and not is_processing_grammar:
        current_time = time.time()
        words_changed = len(sentence_words) != last_sentence_word_count
        
        should_auto_process = (
            len(sentence_words) >= 3 and
            (current_time - last_auto_grammar_time > 5.0 or len(sentence_words) >= 5) and
            words_changed
        )
        
        if should_auto_process:
            if process_grammar(sentence_words):
                last_auto_grammar_time = current_time
                last_sentence_word_count = len(sentence_words)
                show_status("Auto-processing with AI...", 2.0)
    
    # ========================================================================
    # UI Rendering
    # ========================================================================
    # Minimal overlay design to not obstruct video feed
    
    # Top-left: Mode indicator
    if auto_mode:
        draw_overlay_text(frame, "AUTO MODE ON", (10, 30), 0.6, COLOR_SUCCESS, 2, bg=True)
        y_start = 60
    else:
        y_start = 30
    
    # Top-left: Current prediction with confidence-based coloring
    if current_prediction:
        # Color coding: Green >50%, Yellow 30-50%, Red <30%
        conf_color = COLOR_SUCCESS if current_confidence > 0.5 else \
                     (COLOR_WARNING if current_confidence > 0.3 else COLOR_DANGER)
        pred_text = f"{current_prediction} ({current_confidence*100:.0f}%)"
        draw_overlay_text(frame, pred_text, (10, y_start), 0.8, conf_color, 2, bg=True)
    
    # Top-right: Top-5 predictions (helpful for debugging and user confidence)
    y_offset = 30
    for i, (pred, conf) in enumerate(zip(top_5_predictions[:5], top_5_confidences[:5])):
        conf_color = COLOR_SUCCESS if conf > 0.5 else \
                     (COLOR_WARNING if conf > 0.3 else COLOR_DANGER)
        text = f"#{i+1} {pred} {conf*100:.0f}%"
        draw_overlay_text(frame, text, (width - 250, y_offset), 0.5, conf_color, 1, bg=True)
        y_offset += 28
    
    # Bottom-left: Current sentence (show last 5 words to prevent overflow)
    sentence_text = " ".join(sentence_words[-5:]) if sentence_words else "[Empty - Start signing]"
    draw_overlay_text(frame, f"Sentence: {sentence_text}", (10, height - 70), 0.6, COLOR_TEXT, 2, bg=True)
    
    # Bottom-left-2: Temporary status messages
    if status_message and time.time() < status_message_time:
        draw_overlay_text(frame, status_message, (10, height - 40), 0.6, COLOR_PRIMARY, 2, bg=True)
    
    # Bottom-center: Grammar processing indicator
    if is_processing_grammar:
        draw_overlay_text(frame, "AI Processing...", (width//2 - 80, height - 40), 0.7, COLOR_WARNING, 2, bg=True)
    
    # Bottom-right: Stability meter (helps user understand when to hold still)
    status = f"Stable: {stable_frames}/{STABILITY_FRAMES}"
    draw_overlay_text(frame, status, (width - 200, height - 40), 0.5, COLOR_TEXT, 1, bg=True)
    
    # Display frame
    cv2.imshow('MS-SLR: Sign Language Recognition', frame)
    
    # ========================================================================
    # Keyboard Input Handling
    # ========================================================================
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        # Quit application
        break
    
    elif key == ord('a'):
        # Toggle auto mode
        auto_mode = not auto_mode
        if auto_mode:
            show_status("AUTO MODE ON - Signs will auto-add & process", 3.0)
            print("Switched to AUTO mode")
        else:
            show_status("AUTO MODE OFF - Use SPACE to add manually", 3.0)
            print("Switched to MANUAL mode")
    
    elif key == ord(' '):
        # Manual add current prediction
        if current_prediction:
            sentence_words.append(current_prediction)
            show_status(f"Added: {current_prediction}", 1.5)
            print(f"Manually added: {current_prediction}")
        else:
            show_status("No prediction to add", 1.5)
    
    elif key == ord('g'):
        # Trigger grammar correction
        if sentence_words:
            if process_grammar(sentence_words):
                show_status("Processing grammar with AI...", 3.0)
                print(f"Processing: {sentence_words}")
            else:
                show_status("Grammar service unavailable", 1.5)
        else:
            show_status("No sentence to process", 1.5)
    
    elif key == ord('c'):
        # Clear sentence buffer
        if sentence_words:
            sentence_words.clear()
            last_sentence_word_count = 0
            show_status("Sentence cleared", 1.5)
            print("Cleared sentence")
        else:
            show_status("Sentence already empty", 1.0)
    
    elif key == ord('z'):
        # Undo last word
        if sentence_words:
            removed = sentence_words.pop()
            show_status(f"Removed: {removed}", 1.5)
            print(f"Undid: {removed}")
        else:
            show_status("Nothing to undo", 1.0)
    
    elif key == ord('m') or key == ord('h'):
        # Show quick help
        show_status("A=Auto G=Grammar SPACE=Add C=Clear Z=Undo Q=Quit", 3.0)
    
    frame_count += 1

# ============================================================================
# Cleanup
# ============================================================================

print("\nShutting down...")
cap.release()
cv2.destroyAllWindows()

# Signal worker threads to stop
voice_queue.put("STOP")
grammar_queue.put("STOP")

# Wait briefly for threads to finish
time.sleep(0.5)

print("✓ Cleanup complete")
print("Thank you for using MS-SLR!")
