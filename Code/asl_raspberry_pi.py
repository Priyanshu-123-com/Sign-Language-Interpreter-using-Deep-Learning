#!/usr/bin/env python3
"""
ASL Alphabet Recognition System - Raspberry Pi 4 Optimized
Optimized for ARM architecture and limited resources
"""

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
import time
import logging
import json
import os
from datetime import datetime
from typing import Optional, Tuple, Dict, List

# Configure logging for Pi
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/pi/asl_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ASLRaspberryPiRecognizer:
    """ASL Recognition System optimized for Raspberry Pi 4"""
    
    def __init__(self, config_file: str = "/home/pi/asl_config.json"):
        """Initialize the ASL recognition system for Pi"""
        self.config = self._load_config(config_file)
        self._setup_mediapipe()
        self._setup_tts()
        self._setup_ui()
        
        # Recognition state
        self.current_letter = ""
        self.current_word = ""
        self.letter_frames = 0
        self.last_letter = ""
        self.recognition_history = []
        
        # Performance metrics
        self.start_time = time.time()
        self.total_recognitions = 0
        self.accuracy_score = 0.0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        logger.info("ASL Alphabet Recognition System initialized for Raspberry Pi 4")
    
    def _load_config(self, config_file: str) -> Dict:
        """Load configuration optimized for Pi"""
        default_config = {
            "camera_index": 0,
            "camera_width": 320,      # Reduced resolution for Pi
            "camera_height": 240,
            "min_confidence": 0.6,    # Lower threshold for Pi
            "min_frames": 6,          # Fewer frames needed
            "voice_enabled": True,
            "voice_rate": 120,        # Slower speech for Pi
            "ui_width": 720,          # Smaller UI for Pi
            "ui_height": 360,
            "hand_area_x": 100,
            "hand_area_y": 50,
            "hand_area_w": 200,
            "hand_area_h": 200,
            "save_history": True,
            "log_level": "INFO",
            "max_fps": 15,            # Limit FPS for Pi
            "enable_gpu": False,      # Disable GPU acceleration on Pi
            "memory_optimization": True
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {config_file}")
                return {**default_config, **config}
            except Exception as e:
                logger.warning(f"Failed to load config: {e}. Using defaults.")
        
        # Save default config
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        logger.info(f"Default configuration saved to {config_file}")
        return default_config
    
    def _setup_mediapipe(self):
        """Initialize MediaPipe with Pi optimizations"""
        try:
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=self.config["min_confidence"],
                min_tracking_confidence=0.4,  # Lower for Pi
                model_complexity=0  # Use simplest model for Pi
            )
            self.mp_drawing = mp.solutions.drawing_utils
            logger.info("MediaPipe hands detection initialized for Pi")
        except Exception as e:
            logger.error(f"Failed to initialize MediaPipe: {e}")
            raise
    
    def _setup_tts(self):
        """Initialize text-to-speech for Pi"""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.config["voice_rate"])
            self.voice_enabled = self.config["voice_enabled"]
            logger.info("Text-to-speech engine initialized for Pi")
        except Exception as e:
            logger.warning(f"TTS initialization failed: {e}")
            self.voice_enabled = False
    
    def _setup_ui(self):
        """Setup UI parameters for Pi"""
        self.ui_width = self.config["ui_width"]
        self.ui_height = self.config["ui_height"]
        self.panel_width = 300
        self.camera_width = self.config["camera_width"]
        self.camera_height = self.config["camera_height"]
        
        # Hand detection area
        self.hand_x = self.config["hand_area_x"]
        self.hand_y = self.config["hand_area_y"]
        self.hand_w = self.config["hand_area_w"]
        self.hand_h = self.config["hand_area_h"]
    
    def detect_hand_closed(self, landmarks) -> bool:
        """Check if a hand is closed (fist)"""
        thumb_up = landmarks[4].y < landmarks[3].y
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y
        
        return not (index_up or middle_up or ring_up or pinky_up)
    
    def detect_alphabet_gesture(self, landmarks) -> Optional[str]:
        """Detect ASL alphabet gestures - simplified for Pi"""
        # Get finger states
        thumb_up = landmarks[4].y < landmarks[3].y
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y
        
        # Simplified gesture detection for better Pi performance
        finger_count = sum([index_up, middle_up, ring_up, pinky_up])
        
        # A - Fist (all fingers down, thumb across palm)
        if finger_count == 0 and landmarks[4].x < landmarks[3].x:
            return "A"
        
        # B - All fingers up, thumb down
        if finger_count == 4 and not thumb_up:
            return "B"
        
        # C - Curved hand (simplified)
        if finger_count == 1 and landmarks[4].x > landmarks[3].x:
            return "C"
        
        # D - Only index finger up
        if finger_count == 1 and index_up and not thumb_up:
            return "D"
        
        # E - All fingers down, thumb up
        if finger_count == 0 and thumb_up:
            return "E"
        
        # F - Thumb and index touching, other fingers up
        if finger_count == 2 and landmarks[4].x < landmarks[3].x:
            return "F"
        
        # G - Index and thumb pointing
        if finger_count == 1 and landmarks[4].x > landmarks[3].x:
            return "G"
        
        # H - Index and middle finger up
        if finger_count == 2 and index_up and middle_up:
            return "H"
        
        # I - Only pinky up
        if finger_count == 1 and pinky_up and not thumb_up:
            return "I"
        
        # J - Same as I (simplified)
        if finger_count == 1 and pinky_up and not thumb_up:
            return "J"
        
        # K - Index and middle finger up, thumb between them
        if finger_count == 2 and index_up and middle_up and thumb_up:
            return "K"
        
        # L - Index and thumb up
        if finger_count == 1 and index_up and thumb_up:
            return "L"
        
        # M - Thumb up, others down (simplified)
        if finger_count == 0 and thumb_up:
            return "M"
        
        # N - Thumb up, others down (simplified)
        if finger_count == 0 and thumb_up:
            return "N"
        
        # O - Thumb and fingers form circle
        if finger_count == 1 and landmarks[4].x < landmarks[3].x:
            return "O"
        
        # P - Thumb and index down, others up
        if finger_count == 3 and not index_up and thumb_up:
            return "P"
        
        # Q - Thumb and pinky up
        if finger_count == 1 and pinky_up and thumb_up:
            return "Q"
        
        # R - Index and middle crossed
        if finger_count == 2 and index_up and middle_up and landmarks[8].x > landmarks[12].x:
            return "R"
        
        # S - Fist with thumb over fingers
        if finger_count == 0 and thumb_up:
            return "S"
        
        # T - Thumb up, others down (simplified)
        if finger_count == 0 and thumb_up:
            return "T"
        
        # U - Index and middle up
        if finger_count == 2 and index_up and middle_up and not thumb_up:
            return "U"
        
        # V - Index and middle up spread apart
        if (finger_count == 2 and index_up and middle_up and not thumb_up and
            abs(landmarks[8].x - landmarks[12].x) > 0.05):
            return "V"
        
        # W - Index, middle, ring up
        if finger_count == 3 and index_up and middle_up and ring_up:
            return "W"
        
        # X - Index finger bent (simplified)
        if finger_count == 0 and not thumb_up:
            return "X"
        
        # Y - Thumb and pinky up
        if finger_count == 1 and pinky_up and thumb_up:
            return "Y"
        
        # Z - Index finger up (simplified)
        if finger_count == 1 and index_up and not thumb_up:
            return "Z"
        
        return None
    
    def recognize_letter(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize letter from image - optimized for Pi"""
        # Resize image for better Pi performance
        if self.config["memory_optimization"]:
            image = cv2.resize(image, (self.camera_width, self.camera_height))
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        letter = None
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            # Check for two hands closed (space functionality)
            if len(results.multi_hand_landmarks) == 2:
                left_hand_closed = self.detect_hand_closed(results.multi_hand_landmarks[0].landmark)
                right_hand_closed = self.detect_hand_closed(results.multi_hand_landmarks[1].landmark)
                
                if left_hand_closed and right_hand_closed:
                    return "SPACE", 0.95
            
            # Process each hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks (simplified for Pi)
                if not self.config["memory_optimization"]:
                    self.mp_drawing.draw_landmarks(
                        image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Detect alphabet gesture
                letter = self.detect_alphabet_gesture(hand_landmarks.landmark)
                if letter:
                    confidence = 0.8  # Fixed confidence for Pi
                    break
        
        return letter, confidence
    
    def say_text(self, text: str):
        """Convert text to speech (Pi optimized)"""
        if not self.voice_enabled:
            return
        
        def speak():
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                logger.warning(f"TTS error: {e}")
        
        # Use daemon thread for Pi
        thread = threading.Thread(target=speak)
        thread.daemon = True
        thread.start()
    
    def calculate_fps(self):
        """Calculate and log FPS for Pi monitoring"""
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Log every 30 frames
            current_time = time.time()
            fps = 30 / (current_time - self.fps_start_time)
            logger.info(f"FPS: {fps:.1f}")
            self.fps_start_time = current_time
    
    def create_ui_panel(self, frame: np.ndarray, letter: str, confidence: float, 
                       hand_detected: bool) -> np.ndarray:
        """Create UI panel optimized for Pi"""
        # Create main display
        main_panel = np.zeros((self.ui_height, self.ui_width, 3), dtype=np.uint8)
        
        # Simple background for Pi
        main_panel[:] = [20, 20, 30]
        
        # Copy camera feed
        main_panel[:self.camera_height, :self.camera_width] = frame
        
        # Create info panel
        info_panel = np.zeros((self.ui_height, self.panel_width, 3), dtype=np.uint8)
        info_panel[:] = [25, 25, 35]
        
        # Add border
        cv2.rectangle(info_panel, (0, 0), (self.panel_width-1, self.ui_height-1), (100, 100, 150), 1)
        
        # Title (smaller for Pi)
        cv2.putText(info_panel, "ASL ALPHABET", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)
        cv2.putText(info_panel, "RECOGNITION", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)
        
        # Current letter display
        if letter:
            letter_color = (0, 255, 0) if letter != "SPACE" else (255, 165, 0)
            cv2.putText(info_panel, f"LETTER: {letter}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, letter_color, 2)
        else:
            cv2.putText(info_panel, "LETTER: --", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        
        # Confidence bar (simplified)
        conf_width = int(200 * confidence)
        cv2.rectangle(info_panel, (10, 100), (210, 115), (50, 50, 50), -1)
        if conf_width > 0:
            conf_color = (0, 255, 0) if confidence > 0.8 else (255, 255, 0) if confidence > 0.6 else (255, 0, 0)
            cv2.rectangle(info_panel, (10, 100), (10 + conf_width, 115), conf_color, -1)
        cv2.putText(info_panel, f"Conf: {confidence:.1%}", (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Word display
        cv2.putText(info_panel, "WORD:", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        word_display = self.current_word if self.current_word else "Start typing..."
        word_color = (255, 255, 255) if self.current_word else (150, 150, 150)
        cv2.putText(info_panel, word_display, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, word_color, 1)
        
        # Statistics (simplified)
        cv2.putText(info_panel, f"Count: {self.total_recognitions}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Acc: {self.accuracy_score:.1%}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions (simplified)
        cv2.putText(info_panel, "INSTRUCTIONS:", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(info_panel, "Show ASL letters", (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(info_panel, "Close hands = SPACE", (10, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Controls (simplified)
        cv2.putText(info_panel, "Q-Quit V-Voice", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(info_panel, "C-Clear S-Save", (10, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Hand detection area
        cv2.rectangle(frame, (self.hand_x, self.hand_y), 
                     (self.hand_x + self.hand_w, self.hand_y + self.hand_h), (0, 255, 0), 2)
        cv2.putText(frame, "HAND AREA", (self.hand_x, self.hand_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Hand detection status
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        status_text = "HAND" if hand_detected else "NO HAND"
        cv2.putText(frame, status_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Combine panels
        main_panel[:self.ui_height, self.camera_width:] = info_panel
        
        return main_panel
    
    def run(self):
        """Main application loop optimized for Pi"""
        logger.info("Starting ASL Alphabet Recognition System on Raspberry Pi 4")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.config["camera_index"])
        if not cap.isOpened():
            logger.error("Failed to open camera")
            return
        
        # Set camera properties for Pi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        cap.set(cv2.CAP_PROP_FPS, self.config["max_fps"])
        
        logger.info(f"Camera {self.config['camera_index']} initialized for Pi")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read from camera")
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Recognize letter
                letter, confidence = self.recognize_letter(frame)
                
                # Update recognition state
                if letter and confidence > self.config["min_confidence"]:
                    if letter == self.current_letter:
                        self.letter_frames += 1
                    else:
                        self.current_letter = letter
                        self.letter_frames = 1
                else:
                    self.letter_frames = 0
                    self.current_letter = ""
                
                # Add letter to word when recognized
                if (self.letter_frames >= self.config["min_frames"] and 
                    self.current_letter != self.last_letter):
                    
                    if self.current_letter == "SPACE":
                        self.current_word += " "
                        self.say_text("space")
                    else:
                        self.current_word += self.current_letter
                        self.say_text(self.current_letter)
                    
                    # Update statistics
                    self.total_recognitions += 1
                    self.accuracy_score = (self.accuracy_score * (self.total_recognitions - 1) + confidence) / self.total_recognitions
                    
                    # Save to history
                    self.recognition_history.append({
                        "timestamp": time.time(),
                        "letter": self.current_letter,
                        "confidence": confidence,
                        "word": self.current_word
                    })
                    
                    logger.info(f"Recognized: {self.current_letter} (confidence: {confidence:.2f})")
                    self.last_letter = self.current_letter
                
                # Check for hand detection
                hand_detected = letter is not None and confidence > 0.5
                
                # Create and show UI
                display_frame = self.create_ui_panel(frame, self.current_letter, confidence, hand_detected)
                cv2.imshow("ASL Alphabet Recognition - Pi", display_frame)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('v'):
                    self.voice_enabled = not self.voice_enabled
                    logger.info(f"Voice {'enabled' if self.voice_enabled else 'disabled'}")
                elif key == ord('c'):
                    self.current_word = ""
                    logger.info("Word cleared")
                elif key == ord('s'):
                    self.save_recognition_history()
                    logger.info("Recognition history saved")
        
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {e}")
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            self.save_recognition_history()
            logger.info("Application closed")
    
    def save_recognition_history(self):
        """Save recognition history to file"""
        if not self.config["save_history"]:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/pi/recognition_history_{timestamp}.json"
        
        history_data = {
            "session_start": self.start_time,
            "session_duration": time.time() - self.start_time,
            "total_recognitions": self.total_recognitions,
            "accuracy_score": self.accuracy_score,
            "recognitions": self.recognition_history
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(history_data, f, indent=2)
            logger.info(f"Recognition history saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

def main():
    """Main entry point for Pi"""
    try:
        recognizer = ASLRaspberryPiRecognizer()
        recognizer.run()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"Error: {e}")
        print("Make sure you have installed all required packages:")
        print("pip install opencv-python mediapipe pyttsx3 numpy")

if __name__ == "__main__":
    main()
