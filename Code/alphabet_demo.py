#!/usr/bin/env python3
"""
ASL Alphabet Recognition Demo
Focuses specifically on recognizing American Sign Language alphabet gestures
"""

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from threading import Thread
import sys
import os

class AlphabetRecognizer:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Support for two hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize text-to-speech
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
            self.voice_enabled = True
        except:
            print("⚠ Text-to-speech not available")
            self.voice_enabled = False
        
        # ASL Alphabet mapping
        self.alphabet_map = {
            'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F',
            'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L',
            'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R',
            'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X',
            'Y': 'Y', 'Z': 'Z'
        }
        
        self.last_letter = ""
        self.letter_count = 0
        self.min_letter_frames = 8  # Faster recognition for letters
        self.current_word = ""
        
    def detect_hand_closed(self, landmarks):
        """Check if a hand is closed (fist)"""
        thumb_up = landmarks[4].y < landmarks[3].y
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y
        
        # Hand is closed if no fingers are up
        return not (index_up or middle_up or ring_up or pinky_up)
        
    def detect_alphabet_gestures(self, landmarks):
        """Detect ASL alphabet gestures using MediaPipe landmarks"""
        # Get finger states
        thumb_up = landmarks[4].y < landmarks[3].y
        index_up = landmarks[8].y < landmarks[6].y
        middle_up = landmarks[12].y < landmarks[10].y
        ring_up = landmarks[16].y < landmarks[14].y
        pinky_up = landmarks[20].y < landmarks[18].y
        
        # Get finger positions for more complex gestures
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # A - Fist (all fingers down, thumb across palm)
        if not index_up and not middle_up and not ring_up and not pinky_up:
            if landmarks[4].x < landmarks[3].x:  # Thumb across
                return "A"
        
        # B - All fingers up, thumb down
        if index_up and middle_up and ring_up and pinky_up and not thumb_up:
            return "B"
        
        # C - Curved hand (thumb and index form C shape)
        if (landmarks[4].x > landmarks[3].x and  # Thumb out
            landmarks[8].y < landmarks[6].y and  # Index up
            landmarks[12].y > landmarks[10].y and # Middle down
            landmarks[16].y > landmarks[14].y and # Ring down
            landmarks[20].y > landmarks[18].y):   # Pinky down
            return "C"
        
        # D - Only index finger up
        if index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            return "D"
        
        # E - All fingers down, thumb up
        if not index_up and not middle_up and not ring_up and not pinky_up and thumb_up:
            return "E"
        
        # F - Thumb and index touching, other fingers up
        if (landmarks[4].x < landmarks[3].x and  # Thumb position
            landmarks[8].y < landmarks[6].y and  # Index up
            landmarks[12].y < landmarks[10].y and # Middle up
            landmarks[16].y > landmarks[14].y and # Ring down
            landmarks[20].y > landmarks[18].y):   # Pinky down
            return "F"
        
        # G - Index and thumb pointing (like gun)
        if (landmarks[4].x > landmarks[3].x and  # Thumb out
            landmarks[8].y < landmarks[6].y and  # Index up
            landmarks[12].y > landmarks[10].y and # Middle down
            landmarks[16].y > landmarks[14].y and # Ring down
            landmarks[20].y > landmarks[18].y):   # Pinky down
            return "G"
        
        # H - Index and middle finger up, others down
        if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
            return "H"
        
        # I - Only pinky up
        if not index_up and not middle_up and not ring_up and pinky_up and not thumb_up:
            return "I"
        
        # J - I gesture with movement (simplified as I)
        if not index_up and not middle_up and not ring_up and pinky_up and not thumb_up:
            return "J"
        
        # K - Index and middle finger up, thumb between them
        if (index_up and middle_up and not ring_up and not pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "K"
        
        # L - Index and thumb up, others down
        if (index_up and not middle_up and not ring_up and not pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "L"
        
        # M - Thumb between ring and pinky, others down
        if (not index_up and not middle_up and not ring_up and not pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "M"
        
        # N - Thumb between middle and ring, others down
        if (not index_up and not middle_up and not ring_up and not pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "N"
        
        # O - Thumb and fingers form circle
        if (landmarks[4].x < landmarks[3].x and  # Thumb position
            landmarks[8].y < landmarks[6].y and  # Index up
            landmarks[12].y > landmarks[10].y and # Middle down
            landmarks[16].y > landmarks[14].y and # Ring down
            landmarks[20].y > landmarks[18].y):   # Pinky down
            return "O"
        
        # P - Thumb and index down, others up
        if not index_up and middle_up and ring_up and pinky_up and not thumb_up:
            return "P"
        
        # Q - Thumb and pinky up, others down
        if (not index_up and not middle_up and not ring_up and pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "Q"
        
        # R - Index and middle crossed, others down
        if (index_up and middle_up and not ring_up and not pinky_up and not thumb_up and
            landmarks[8].x > landmarks[12].x):  # Index over middle
            return "R"
        
        # S - Fist with thumb over fingers
        if not index_up and not middle_up and not ring_up and not pinky_up and thumb_up:
            return "S"
        
        # T - Thumb between index and middle, others down
        if (not index_up and not middle_up and not ring_up and not pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "T"
        
        # U - Index and middle up, others down
        if index_up and middle_up and not ring_up and not pinky_up and not thumb_up:
            return "U"
        
        # V - Index and middle up spread apart, others down
        if (index_up and middle_up and not ring_up and not pinky_up and not thumb_up and
            abs(landmarks[8].x - landmarks[12].x) > 0.05):  # Fingers spread
            return "V"
        
        # W - Index, middle, ring up, others down
        if index_up and middle_up and ring_up and not pinky_up and not thumb_up:
            return "W"
        
        # X - Index finger bent, others down
        if (not index_up and not middle_up and not ring_up and not pinky_up and not thumb_up and
            landmarks[8].y > landmarks[6].y and landmarks[8].y < landmarks[5].y):  # Index bent
            return "X"
        
        # Y - Thumb and pinky up, others down
        if (not index_up and not middle_up and not ring_up and pinky_up and 
            landmarks[4].y < landmarks[3].y):  # Thumb up
            return "Y"
        
        # Z - Index finger draws Z shape (simplified as index up)
        if index_up and not middle_up and not ring_up and not pinky_up and not thumb_up:
            return "Z"
        
        return None
    
    def say_text(self, text):
        """Convert text to speech"""
        if not self.voice_enabled:
            return
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except:
            pass  # Ignore TTS errors
    
    def recognize_letter(self, image):
        """Recognize letter from image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        letter = None
        confidence = 0
        
        if results.multi_hand_landmarks:
            # Check for two hands closed (space functionality)
            if len(results.multi_hand_landmarks) == 2:
                left_hand_closed = self.detect_hand_closed(results.multi_hand_landmarks[0].landmark)
                right_hand_closed = self.detect_hand_closed(results.multi_hand_landmarks[1].landmark)
                
                if left_hand_closed and right_hand_closed:
                    return "SPACE", 0.95
            
            # Process each hand
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Detect alphabet gesture
                letter = self.detect_alphabet_gestures(hand_landmarks.landmark)
                if letter:
                    confidence = 0.9
                    
        return letter, confidence
    
    def run_demo(self):
        """Run the alphabet recognition demo"""
        print("=" * 60)
        print("ASL Alphabet Recognition Demo")
        print("=" * 60)
        print("Supported letters: A-Z")
        print("Make clear ASL alphabet gestures")
        print("Close both hands for SPACE")
        print("\nControls:")
        print("- Press 'q' to quit")
        print("- Press 'v' to toggle voice")
        print("- Press 'c' to clear current word")
        print("- Press 'h' to show this help")
        print("=" * 60)
        
        # Try different camera indices
        cap = None
        for camera_index in [0, 1, 2]:
            print(f"Trying camera {camera_index}...")
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"✓ Using camera {camera_index}")
                    break
                else:
                    cap.release()
            else:
                cap.release()
        
        if cap is None or not cap.isOpened():
            print("✗ No camera found!")
            return
            
        voice_enabled = True
        current_letter = ""
        letter_frames = 0
        show_help = False
        
        print("\nStarting alphabet recognition...")
        print("Make sure your hand is visible in the camera!")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            frame_height, frame_width = frame.shape[:2]
            
            # Recognize letter
            letter, confidence = self.recognize_letter(frame)
            
            # Update letter recognition
            if letter is not None and confidence > 0.7:
                if letter == current_letter:
                    letter_frames += 1
                else:
                    current_letter = letter
                    letter_frames = 1
            else:
                letter_frames = 0
                current_letter = ""
            
            # Add letter to word when recognized
            if letter_frames >= self.min_letter_frames and current_letter != self.last_letter:
                if current_letter == "SPACE":
                    print(f"Recognized: SPACE (confidence: {confidence:.2f})")
                    if voice_enabled:
                        Thread(target=self.say_text, args=("space",)).start()
                    self.current_word += " "
                else:
                    print(f"Recognized: {current_letter} (confidence: {confidence:.2f})")
                    if voice_enabled:
                        Thread(target=self.say_text, args=(current_letter,)).start()
                    self.current_word += current_letter
                
                self.last_letter = current_letter
            
            # Enhanced UI Design
            display_height = frame_height
            display_width = frame_width
            panel_width = 400
            
            # Create main panel
            main_panel = np.zeros((display_height, display_width + panel_width, 3), dtype=np.uint8)
            
            # Add gradient background
            for i in range(display_height):
                intensity = int(20 + (i / display_height) * 30)
                main_panel[i, :] = [intensity, intensity, intensity]
            
            # Copy camera feed
            main_panel[:display_height, :display_width] = frame
            
            # Create info panel
            info_panel = np.zeros((display_height, panel_width, 3), dtype=np.uint8)
            info_panel[:] = [25, 25, 35]  # Dark blue background
            
            # Add border to info panel
            cv2.rectangle(info_panel, (0, 0), (panel_width-1, display_height-1), (100, 100, 150), 2)
            
            # Title
            cv2.putText(info_panel, "ASL ALPHABET", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 150), 2)
            cv2.putText(info_panel, "RECOGNITION", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 150), 2)
            
            # Current letter display
            if current_letter:
                letter_color = (0, 255, 0) if current_letter != "SPACE" else (255, 165, 0)
                cv2.putText(info_panel, f"LETTER: {current_letter}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, letter_color, 2)
            else:
                cv2.putText(info_panel, "LETTER: --", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
            
            # Confidence bar
            conf_width = int(300 * confidence)
            cv2.rectangle(info_panel, (20, 140), (320, 160), (50, 50, 50), -1)
            if conf_width > 0:
                conf_color = (0, 255, 0) if confidence > 0.8 else (255, 255, 0) if confidence > 0.6 else (255, 0, 0)
                cv2.rectangle(info_panel, (20, 140), (20 + conf_width, 160), conf_color, -1)
            cv2.putText(info_panel, f"Confidence: {confidence:.1%}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Word display
            cv2.putText(info_panel, "WORD:", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            word_display = self.current_word if self.current_word else "Start typing..."
            word_color = (255, 255, 255) if self.current_word else (150, 150, 150)
            cv2.putText(info_panel, word_display, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, word_color, 1)
            
            # Instructions
            cv2.putText(info_panel, "INSTRUCTIONS:", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(info_panel, "• Show ASL letters clearly", (20, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_panel, "• Close both hands for SPACE", (20, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(info_panel, "• Keep hand in center", (20, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Controls
            cv2.putText(info_panel, "CONTROLS:", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(info_panel, "Q - Quit  |  V - Voice", (20, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(info_panel, "C - Clear  |  H - Help", (20, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Voice status indicator
            voice_color = (0, 255, 0) if voice_enabled else (255, 0, 0)
            cv2.circle(info_panel, (350, 430), 8, voice_color, -1)
            cv2.putText(info_panel, "VOICE", (320, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, voice_color, 1)
            
            # Help text overlay
            if show_help:
                help_panel = np.zeros((200, 300, 3), dtype=np.uint8)
                help_panel[:] = [0, 0, 0]
                cv2.rectangle(help_panel, (0, 0), (299, 199), (100, 100, 100), 2)
                cv2.putText(help_panel, "HELP", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                help_text = [
                    "q - Quit",
                    "v - Toggle voice", 
                    "c - Clear word",
                    "h - Toggle help"
                ]
                for i, text in enumerate(help_text):
                    cv2.putText(help_panel, text, (10, 60 + i*25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                # Overlay help panel
                info_panel[50:250, 50:350] = help_panel
            
            # Combine panels
            main_panel[:display_height, display_width:] = info_panel
            
            cv2.imshow("ASL Alphabet Recognition", main_panel)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('v'):
                voice_enabled = not voice_enabled
                print(f"Voice {'enabled' if voice_enabled else 'disabled'}")
            elif key == ord('c'):
                self.current_word = ""
                print("Word cleared")
            elif key == ord('h'):
                show_help = not show_help
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nFinal word: {self.current_word}")
        print("Demo ended!")

def main():
    try:
        recognizer = AlphabetRecognizer()
        recognizer.run_demo()
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed all required packages:")
        print("pip install opencv-python mediapipe pyttsx3 numpy")

if __name__ == "__main__":
    main()
