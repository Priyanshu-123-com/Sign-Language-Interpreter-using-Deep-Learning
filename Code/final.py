import cv2, pickle
import numpy as np
import tensorflow as tf
import os
import sqlite3, pyttsx3
from threading import Thread
import mediapipe as mp

# Initialize text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Try to load the CNN model, if not available use MediaPipe fallback
model = None
use_mediapipe = False

try:
    from keras.models import load_model
    model = load_model('cnn_model_keras2.h5')
    print("✓ CNN model loaded successfully")
except Exception as e:
    print(f"⚠ CNN model not found: {e}")
    print("⚠ Falling back to MediaPipe-based recognition")
    use_mediapipe = True
    
    # Initialize MediaPipe hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,  # Support for two hands
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

def get_hand_hist():
	try:
		with open("hist", "rb") as f:
			hist = pickle.load(f)
		return hist
	except FileNotFoundError:
		print("⚠ Hand histogram file not found. Using default skin detection.")
		return None

def get_image_size():
	try:
		img = cv2.imread('gestures/0/100.jpg', 0)
		return img.shape
	except:
		# Default image size if gestures folder doesn't exist
		return (50, 50)

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	try:
		conn = sqlite3.connect("gesture_db.db")
		cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
		cursor = conn.execute(cmd)
		for row in cursor:
			return row[0]
	except:
		# Fallback gesture names if database is not available
		gesture_names = {
			0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
			6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "A", 11: "B",
			12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
			19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P",
			26: "Q", 27: "R", 28: "S", 29: "T", 30: "U", 31: "V", 32: "W",
			33: "X", 34: "Y", 35: "Z"
		}
		return gesture_names.get(pred_class, f"Gesture_{pred_class}")

# MediaPipe-based gesture recognition functions
def count_fingers_mediapipe(landmarks):
	"""Count extended fingers using MediaPipe landmarks"""
	tips = [4, 8, 12, 16, 20]  # Finger tip landmarks
	fingers = []
	
	# Thumb (special case)
	if landmarks[tips[0]].x > landmarks[tips[0] - 1].x:
		fingers.append(1)
	else:
		fingers.append(0)
		
	# Other fingers
	for i in range(1, 5):
		if landmarks[tips[i]].y < landmarks[tips[i] - 2].y:
			fingers.append(1)
		else:
			fingers.append(0)
			
	return sum(fingers)

def detect_special_gestures_mediapipe(landmarks):
	"""Detect special gestures using MediaPipe landmarks"""
	# Thumbs up detection
	if (landmarks[4].y < landmarks[3].y and  # Thumb up
		landmarks[8].y > landmarks[6].y and  # Index down
		landmarks[12].y > landmarks[10].y and # Middle down
		landmarks[16].y > landmarks[14].y and # Ring down
		landmarks[20].y > landmarks[18].y):   # Pinky down
		return "Thumbs Up"
		
	# Thumbs down detection
	if (landmarks[4].y > landmarks[3].y and  # Thumb down
		landmarks[8].y > landmarks[6].y and  # Index down
		landmarks[12].y > landmarks[10].y and # Middle down
		landmarks[16].y > landmarks[14].y and # Ring down
		landmarks[20].y > landmarks[18].y):   # Pinky down
		return "Thumbs Down"
		
	# Peace sign (V sign)
	if (landmarks[8].y < landmarks[6].y and  # Index up
		landmarks[12].y < landmarks[10].y and # Middle up
		landmarks[16].y > landmarks[14].y and # Ring down
		landmarks[20].y > landmarks[18].y and # Pinky down
		landmarks[4].y > landmarks[3].y):     # Thumb down
		return "Peace"
		
	# OK sign (thumb and index form circle)
	if (landmarks[4].x < landmarks[3].x and  # Thumb position
		landmarks[8].y < landmarks[6].y and  # Index up
		landmarks[12].y > landmarks[10].y and # Middle down
		landmarks[16].y > landmarks[14].y and # Ring down
		landmarks[20].y > landmarks[18].y):   # Pinky down
		return "OK"
		
	return None

def detect_hand_closed(landmarks):
	"""Check if a hand is closed (fist)"""
	thumb_up = landmarks[4].y < landmarks[3].y
	index_up = landmarks[8].y < landmarks[6].y
	middle_up = landmarks[12].y < landmarks[10].y
	ring_up = landmarks[16].y < landmarks[14].y
	pinky_up = landmarks[20].y < landmarks[18].y
	
	# Hand is closed if no fingers are up
	return not (index_up or middle_up or ring_up or pinky_up)

def detect_alphabet_gestures_mediapipe(landmarks):
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
	
	# M - Thumb between ring and pinky, others down (simplified as fist with thumb)
	if (not index_up and not middle_up and not ring_up and not pinky_up and 
		landmarks[4].y < landmarks[3].y):  # Thumb up
		return "M"
	
	# N - Thumb between middle and ring, others down (simplified as fist with thumb)
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
	
	# T - Thumb between index and middle, others down (simplified as fist with thumb)
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

def recognize_gesture_mediapipe(image, alphabet_mode=False):
	"""Recognize gesture using MediaPipe"""
	if not use_mediapipe:
		return None, 0.0
		
	rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	results = hands.process(rgb_image)
	
	if results.multi_hand_landmarks:
		# Check for two hands closed (space functionality)
		if len(results.multi_hand_landmarks) == 2:
			left_hand_closed = detect_hand_closed(results.multi_hand_landmarks[0].landmark)
			right_hand_closed = detect_hand_closed(results.multi_hand_landmarks[1].landmark)
			
			if left_hand_closed and right_hand_closed:
				return "SPACE", 0.95
		
		# Process each hand
		for hand_landmarks in results.multi_hand_landmarks:
			# Draw hand landmarks
			mp_drawing.draw_landmarks(
				image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
			
			# Check for alphabet gestures first if in alphabet mode
			if alphabet_mode:
				alphabet_gesture = detect_alphabet_gestures_mediapipe(hand_landmarks.landmark)
				if alphabet_gesture:
					return alphabet_gesture, 0.9
			
			# Count fingers
			finger_count = count_fingers_mediapipe(hand_landmarks.landmark)
			
			# Check for special gestures
			special_gesture = detect_special_gestures_mediapipe(hand_landmarks.landmark)
			
			if special_gesture:
				return special_gesture, 0.9
			elif 0 <= finger_count <= 5:
				return str(finger_count), 0.8
				
	return None, 0.0

def get_pred_from_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	text = ""
	
	# Try CNN model first if available
	if model is not None:
		if w1 > h1:
			save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
		elif h1 > w1:
			save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
		pred_probab, pred_class = keras_predict(model, save_img)
		if pred_probab*100 > 70:
			text = get_pred_text_from_db(pred_class)
	
	return text

def get_pred_from_image(image, alphabet_mode=False):
	"""Get prediction from image using available methods"""
	text = ""
	confidence = 0.0
	
	# Try MediaPipe first if available
	if use_mediapipe:
		gesture, conf = recognize_gesture_mediapipe(image, alphabet_mode)
		if gesture and conf > 0.7:
			text = gesture
			confidence = conf
	
	# If MediaPipe didn't work or CNN is preferred, try CNN
	if not text and model is not None:
		# Use the existing contour-based approach
		img, contours, thresh = get_img_contour_thresh(image)
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				text = get_pred_from_contour(contour, thresh)
				confidence = 0.8 if text else 0.0
	
	return text, confidence

def get_operator(pred_text):
	try:
		pred_text = int(pred_text)
	except:
		return ""
	operator = ""
	if pred_text == 1:
		operator = "+"
	elif pred_text == 2:
		operator = "-"
	elif pred_text == 3:
		operator = "*"
	elif pred_text == 4:
		operator = "/"
	elif pred_text == 5:
		operator = "%"
	elif pred_text == 6:
		operator = "**"
	elif pred_text == 7:
		operator = ">>"
	elif pred_text == 8:
		operator = "<<"
	elif pred_text == 9:
		operator = "&"
	elif pred_text == 0:
		operator = "|"
	return operator

hist = get_hand_hist()
x, y, w, h = 300, 100, 300, 300
is_voice_on = True

def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	if hist is not None:
		# Use histogram-based detection
		dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
		cv2.filter2D(dst,-1,disc,dst)
		blur = cv2.GaussianBlur(dst, (11,11), 0)
		blur = cv2.medianBlur(blur, 15)
		thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	else:
		# Use color-based skin detection as fallback
		lower_skin = np.array([0, 20, 70], dtype=np.uint8)
		upper_skin = np.array([20, 255, 255], dtype=np.uint8)
		mask = cv2.inRange(imgHSV, lower_skin, upper_skin)
		
		# Morphological operations to clean up the mask
		kernel = np.ones((5, 5), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		thresh = mask
	
	thresh = cv2.merge((thresh,thresh,thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	thresh = thresh[y:y+h, x:x+w]
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	return img, contours, thresh

def say_text(text):
	if not is_voice_on:
		return
	while engine._inLoop:
		pass
	engine.say(text)
	engine.runAndWait()

def calculator_mode(cam):
	global is_voice_on
	flag = {"first": False, "operator": False, "second": False, "clear": False}
	count_same_frames = 0
	first, operator, second = "", "", ""
	pred_text = ""
	calc_text = ""
	info = "Enter first number"
	Thread(target=say_text, args=(info,)).start()
	count_clear_frames = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_pred_text = pred_text
		
		# Use the new prediction method
		pred_text, confidence = get_pred_from_image(img)
		
		if old_pred_text == pred_text:
			count_same_frames += 1
		else:
			count_same_frames = 0

		if pred_text == "C":
			if count_same_frames > 5:
				count_same_frames = 0
				first, second, operator, pred_text, calc_text = '', '', '', '', ''
				flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
				info = "Enter first number"
				Thread(target=say_text, args=(info,)).start()

		elif pred_text == "Best of Luck " and count_same_frames > 15:
			count_same_frames = 0
			if flag['clear']:
				first, second, operator, pred_text, calc_text = '', '', '', '', ''
				flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
				info = "Enter first number"
				Thread(target=say_text, args=(info,)).start()
			elif second != '':
				flag['second'] = True
				info = "Clear screen"
				#Thread(target=say_text, args=(info,)).start()
				second = ''
				flag['clear'] = True
				try:
					calc_text += "= "+str(eval(calc_text))
				except:
					calc_text = "Invalid operation"
				if is_voice_on:
					speech = calc_text
					speech = speech.replace('-', ' minus ')
					speech = speech.replace('/', ' divided by ')
					speech = speech.replace('**', ' raised to the power ')
					speech = speech.replace('*', ' multiplied by ')
					speech = speech.replace('%', ' mod ')
					speech = speech.replace('>>', ' bitwise right shift ')
					speech = speech.replace('<<', ' bitwise leftt shift ')
					speech = speech.replace('&', ' bitwise and ')
					speech = speech.replace('|', ' bitwise or ')
					Thread(target=say_text, args=(speech,)).start()
			elif first != '':
				flag['first'] = True
				info = "Enter operator"
				Thread(target=say_text, args=(info,)).start()
				first = ''

		elif pred_text != "Best of Luck " and pred_text.isnumeric():
			if flag['first'] == False:
				if count_same_frames > 15:
					count_same_frames = 0
					Thread(target=say_text, args=(pred_text,)).start()
					first += pred_text
					calc_text += pred_text
			elif flag['operator'] == False:
				operator = get_operator(pred_text)
				if count_same_frames > 15:
					count_same_frames = 0
					flag['operator'] = True
					calc_text += operator
					info = "Enter second number"
					Thread(target=say_text, args=(info,)).start()
					operator = ''
			elif flag['second'] == False:
				if count_same_frames > 15:
					Thread(target=say_text, args=(pred_text,)).start()
					second += pred_text
					calc_text += pred_text
					count_same_frames = 0	

		if count_clear_frames == 30:
			first, second, operator, pred_text, calc_text = '', '', '', '', ''
			flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
			info = "Enter first number"
			Thread(target=say_text, args=(info,)).start()
			count_clear_frames = 0

		# Enhanced UI for Calculator Mode
		display_height = 480
		display_width = 640
		panel_width = 400
		
		# Create main panel
		main_panel = np.zeros((display_height, display_width + panel_width, 3), dtype=np.uint8)
		
		# Add gradient background
		for i in range(display_height):
			intensity = int(20 + (i / display_height) * 30)
			main_panel[i, :] = [intensity, intensity, intensity]
		
		# Copy camera feed
		main_panel[:display_height, :display_width] = img
		
		# Create info panel
		info_panel = np.zeros((display_height, panel_width, 3), dtype=np.uint8)
		info_panel[:] = [25, 25, 35]  # Dark blue background
		
		# Add border to info panel
		cv2.rectangle(info_panel, (0, 0), (panel_width-1, display_height-1), (100, 100, 150), 2)
		
		# Title
		cv2.putText(info_panel, "CALCULATOR", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 150), 2)
		cv2.putText(info_panel, "MODE", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 150), 2)
		
		# Current input display
		if pred_text:
			cv2.putText(info_panel, f"INPUT: {pred_text}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
		else:
			cv2.putText(info_panel, "INPUT: --", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
		
		# Operator display
		if operator:
			cv2.putText(info_panel, f"OP: {operator}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
		else:
			cv2.putText(info_panel, "OP: --", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
		
		# Calculation display
		cv2.putText(info_panel, "CALCULATION:", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
		calc_display = calc_text if calc_text else "0"
		cv2.putText(info_panel, calc_display, (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
		
		# Status
		cv2.putText(info_panel, f"STATUS: {info}", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
		
		# Instructions
		cv2.putText(info_panel, "INSTRUCTIONS:", (20, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
		cv2.putText(info_panel, "• Show numbers 0-9", (20, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		cv2.putText(info_panel, "• Use operators +,-,*,/", (20, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		cv2.putText(info_panel, "• Close both hands = result", (20, 385), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
		
		# Controls
		cv2.putText(info_panel, "CONTROLS:", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
		cv2.putText(info_panel, "Q - Quit  |  T - Text Mode", (20, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		cv2.putText(info_panel, "A - Alphabet  |  V - Voice", (20, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		
		# Voice status indicator
		voice_color = (0, 255, 0) if is_voice_on else (255, 0, 0)
		cv2.circle(info_panel, (350, 430), 8, voice_color, -1)
		cv2.putText(info_panel, "VOICE", (320, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, voice_color, 1)
		
		# Hand detection area indicator
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
		cv2.putText(img, "HAND AREA", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		
		# Add hand detection status
		hand_detected = len(contours) > 0 and max([cv2.contourArea(c) for c in contours]) > 1000 if contours else False
		status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
		status_text = "HAND DETECTED" if hand_detected else "NO HAND"
		cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
		
		# Combine panels
		main_panel[:display_height, display_width:] = info_panel
		
		# Show the enhanced display
		cv2.imshow("Calculator Mode", main_panel)
		keypress = cv2.waitKey(1)
		if keypress == ord('q') or keypress == ord('t') or keypress == ord('a'):
			break
		if keypress == ord('v') and is_voice_on:
			is_voice_on = False
		elif keypress == ord('v') and not is_voice_on:
			is_voice_on = True

	if keypress == ord('t'):
		return 1
	elif keypress == ord('a'):
		return 3
	else:
		return 0

def text_mode(cam):
	global is_voice_on
	text = ""
	word = ""
	count_same_frame = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_text = text
		
		# Use the new prediction method
		text, confidence = get_pred_from_image(img)
		
		if old_text == text:
			count_same_frame += 1
		else:
			count_same_frame = 0

		if count_same_frame > 20:
			if len(text) == 1:
				Thread(target=say_text, args=(text, )).start()
			word = word + text
			if word.startswith('I/Me '):
				word = word.replace('I/Me ', 'I ')
			elif word.endswith('I/Me '):
				word = word.replace('I/Me ', 'me ')
			count_same_frame = 0

		# Check if hand is not visible (for clearing word)
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) < 1000:
				if word != '':
					Thread(target=say_text, args=(word, )).start()
				text = ""
				word = ""
		else:
			if word != '':
				Thread(target=say_text, args=(word, )).start()
			text = ""
			word = ""
		# Enhanced UI for Text Mode
		display_height = 480
		display_width = 640
		panel_width = 400
		
		# Create main panel
		main_panel = np.zeros((display_height, display_width + panel_width, 3), dtype=np.uint8)
		
		# Add gradient background
		for i in range(display_height):
			intensity = int(20 + (i / display_height) * 30)
			main_panel[i, :] = [intensity, intensity, intensity]
		
		# Copy camera feed
		main_panel[:display_height, :display_width] = img
		
		# Create info panel
		info_panel = np.zeros((display_height, panel_width, 3), dtype=np.uint8)
		info_panel[:] = [25, 25, 35]  # Dark blue background
		
		# Add border to info panel
		cv2.rectangle(info_panel, (0, 0), (panel_width-1, display_height-1), (100, 100, 150), 2)
		
		# Title
		cv2.putText(info_panel, "GESTURE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 150), 2)
		cv2.putText(info_panel, "RECOGNITION", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 150), 2)
		
		# Current gesture display
		if text:
			cv2.putText(info_panel, f"GESTURE: {text}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
		else:
			cv2.putText(info_panel, "GESTURE: --", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
		
		# Word display
		cv2.putText(info_panel, "TEXT:", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
		word_display = word if word else "Start gesturing..."
		word_color = (255, 255, 255) if word else (150, 150, 150)
		cv2.putText(info_panel, word_display, (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, word_color, 1)
		
		# Instructions
		cv2.putText(info_panel, "INSTRUCTIONS:", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
		cv2.putText(info_panel, "• Show clear gestures", (20, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		cv2.putText(info_panel, "• Keep hand in green box", (20, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		cv2.putText(info_panel, "• Hold gesture steady", (20, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		
		# Controls
		cv2.putText(info_panel, "CONTROLS:", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
		cv2.putText(info_panel, "Q - Quit  |  C - Calculator", (20, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		cv2.putText(info_panel, "A - Alphabet  |  V - Voice", (20, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		
		# Voice status indicator
		voice_color = (0, 255, 0) if is_voice_on else (255, 0, 0)
		cv2.circle(info_panel, (350, 430), 8, voice_color, -1)
		cv2.putText(info_panel, "VOICE", (320, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, voice_color, 1)
		
		# Hand detection area indicator
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
		cv2.putText(img, "HAND AREA", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		
		# Add hand detection status
		hand_detected = len(contours) > 0 and max([cv2.contourArea(c) for c in contours]) > 1000 if contours else False
		status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
		status_text = "HAND DETECTED" if hand_detected else "NO HAND"
		cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
		
		# Combine panels
		main_panel[:display_height, display_width:] = info_panel
		
		# Show the enhanced display
		cv2.imshow("Gesture Recognition", main_panel)
		keypress = cv2.waitKey(1)
		if keypress == ord('q') or keypress == ord('c') or keypress == ord('a'):
			break
		if keypress == ord('v') and is_voice_on:
			is_voice_on = False
		elif keypress == ord('v') and not is_voice_on:
			is_voice_on = True

	if keypress == ord('c'):
		return 2
	elif keypress == ord('a'):
		return 3
	else:
		return 0

def alphabet_mode(cam):
	"""Alphabet recognition mode using ASL gestures"""
	global is_voice_on
	text = ""
	word = ""
	count_same_frame = 0
	alphabet_mode = True
	
	print("Alphabet Recognition Mode")
	print("Supported letters: A-Z")
	print("Press 'q' to quit, 'v' to toggle voice, 'c' to switch to text mode")
	
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_text = text
		
		# Use alphabet mode for prediction
		text, confidence = get_pred_from_image(img, alphabet_mode=True)
		
		if old_text == text:
			count_same_frame += 1
		else:
			count_same_frame = 0

		if count_same_frame > 15:  # Faster recognition for letters
			if text == "SPACE":
				Thread(target=say_text, args=("space", )).start()
				word = word + " "
				count_same_frame = 0
			elif len(text) == 1 and text.isalpha():
				Thread(target=say_text, args=(text, )).start()
				word = word + text
				count_same_frame = 0

		# Check if hand is not visible (for clearing word)
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) < 1000:
				if word != '':
					Thread(target=say_text, args=(word, )).start()
				text = ""
				word = ""
		else:
			if word != '':
				Thread(target=say_text, args=(word, )).start()
			text = ""
			word = ""
			
		# Enhanced UI Design
		# Create main display area
		display_height = 480
		display_width = 640
		panel_width = 400
		
		# Create main panel
		main_panel = np.zeros((display_height, display_width + panel_width, 3), dtype=np.uint8)
		
		# Add gradient background
		for i in range(display_height):
			intensity = int(20 + (i / display_height) * 30)
			main_panel[i, :] = [intensity, intensity, intensity]
		
		# Copy camera feed
		main_panel[:display_height, :display_width] = img
		
		# Create info panel
		info_panel = np.zeros((display_height, panel_width, 3), dtype=np.uint8)
		info_panel[:] = [25, 25, 35]  # Dark blue background
		
		# Add border to info panel
		cv2.rectangle(info_panel, (0, 0), (panel_width-1, display_height-1), (100, 100, 150), 2)
		
		# Title
		cv2.putText(info_panel, "ASL ALPHABET", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 150), 2)
		cv2.putText(info_panel, "RECOGNITION", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 150), 2)
		
		# Current letter display
		if text:
			letter_color = (0, 255, 0) if text != "SPACE" else (255, 165, 0)
			cv2.putText(info_panel, f"LETTER: {text}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, letter_color, 2)
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
		word_display = word if word else "Start typing..."
		word_color = (255, 255, 255) if word else (150, 150, 150)
		cv2.putText(info_panel, word_display, (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, word_color, 1)
		
		# Instructions
		cv2.putText(info_panel, "INSTRUCTIONS:", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
		cv2.putText(info_panel, "• Show ASL letters clearly", (20, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		cv2.putText(info_panel, "• Close both hands for SPACE", (20, 345), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
		cv2.putText(info_panel, "• Keep hand in green box", (20, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		
		# Controls
		cv2.putText(info_panel, "CONTROLS:", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
		cv2.putText(info_panel, "Q - Quit  |  C - Text Mode", (20, 425), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		cv2.putText(info_panel, "V - Toggle Voice", (20, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
		
		# Voice status indicator
		voice_color = (0, 255, 0) if is_voice_on else (255, 0, 0)
		cv2.circle(info_panel, (350, 430), 8, voice_color, -1)
		cv2.putText(info_panel, "VOICE", (320, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, voice_color, 1)
		
		# Hand detection area indicator
		cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
		cv2.putText(img, "HAND AREA", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		
		# Add hand detection status
		hand_detected = len(contours) > 0 and max([cv2.contourArea(c) for c in contours]) > 1000 if contours else False
		status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
		status_text = "HAND DETECTED" if hand_detected else "NO HAND"
		cv2.putText(img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
		
		# Combine panels
		main_panel[:display_height, display_width:] = info_panel
		
		# Show the enhanced display
		cv2.imshow("ASL Alphabet Recognition", main_panel)
		
		keypress = cv2.waitKey(1)
		if keypress == ord('q') or keypress == ord('c'):
			break
		if keypress == ord('v') and is_voice_on:
			is_voice_on = False
		elif keypress == ord('v') and not is_voice_on:
			is_voice_on = True

	if keypress == ord('c'):
		return 1  # Switch to text mode
	else:
		return 0  # Quit

def recognize():
	cam = cv2.VideoCapture(1)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	
	print("=" * 60)
	print("Sign Language Recognition System")
	print("=" * 60)
	print("Available modes:")
	print("1. Text Mode (t) - General gesture recognition")
	print("2. Calculator Mode (c) - Mathematical operations")
	print("3. Alphabet Mode (a) - ASL alphabet recognition")
	print("Press 'q' to quit")
	print("=" * 60)
	
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(cam)
		elif keypress == 2:
			keypress = calculator_mode(cam)
		elif keypress == 3:
			keypress = alphabet_mode(cam)
		else:
			break

# Initialize the model if available
if model is not None:
	try:
		keras_predict(model, np.zeros((50, 50), dtype = np.uint8))
	except:
		pass

recognize()
