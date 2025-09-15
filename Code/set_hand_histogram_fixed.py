import cv2
import numpy as np
import pickle

def build_squares(img):
    x, y, w, h = 420, 140, 10, 10
    d = 10
    imgCrop = None
    crop = None
    
    for i in range(10):
        for j in range(5):
            if imgCrop is None:
                imgCrop = img[y:y+h, x:x+w]
            else:
                imgCrop = np.hstack((imgCrop, img[y:y+h, x:x+w]))
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 1)
            x += w + d
        if crop is None:
            crop = imgCrop
        else:
            crop = np.vstack((crop, imgCrop)) 
        imgCrop = None
        x = 420
        y += h + d
    return crop

def get_hand_hist():
    print("Setting up hand histogram...")
    print("Instructions:")
    print("1. Place your hand in the green squares area")
    print("2. Press 'c' to capture hand color")
    print("3. Press 's' to save and exit")
    print("4. Press 'q' to quit without saving")
    
    # Try different camera indices
    cam = None
    for camera_index in [0, 1, 2]:
        print(f"Trying camera {camera_index}...")
        cam = cv2.VideoCapture(camera_index)
        if cam.isOpened():
            ret, frame = cam.read()
            if ret:
                print(f"✓ Camera {camera_index} is working!")
                break
            else:
                cam.release()
        else:
            cam.release()
    
    if cam is None or not cam.isOpened():
        print("✗ No working camera found!")
        return
    
    x, y, w, h = 300, 100, 300, 300
    flagPressedC, flagPressedS = False, False
    imgCrop = None
    hist = None
    
    print("\nCamera window should open now. If not, check your display settings.")
    
    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to read from camera")
            break
            
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (640, 480))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Draw hand region rectangle
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, "Hand Region", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        keypress = cv2.waitKey(1) & 0xFF
        
        if keypress == ord('c'):		
            if imgCrop is not None:
                hsvCrop = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2HSV)
                flagPressedC = True
                hist = cv2.calcHist([hsvCrop], [0, 1], None, [180, 256], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                print("✓ Hand color captured! Press 's' to save or 'c' to recapture.")
            else:
                print("✗ No hand region captured yet. Make sure your hand is in the green area.")
                
        elif keypress == ord('s'):
            if hist is not None:
                flagPressedS = True	
                break
            else:
                print("✗ No histogram to save. Press 'c' first to capture hand color.")
                
        elif keypress == ord('q'):
            print("Exiting without saving...")
            break
            
        if flagPressedC and hist is not None:	
            dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
            dst1 = dst.copy()
            disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
            cv2.filter2D(dst,-1,disc,dst)
            blur = cv2.GaussianBlur(dst, (11,11), 0)
            blur = cv2.medianBlur(blur, 15)
            ret,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            thresh = cv2.merge((thresh,thresh,thresh))
            cv2.imshow("Hand Detection Preview", thresh)
        
        if not flagPressedS:
            imgCrop = build_squares(img)
        
        # Add instructions on the image
        cv2.putText(img, "Press 'c' to capture hand color", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "Press 's' to save and exit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(img, "Press 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Set hand histogram", img)
    
    cam.release()
    cv2.destroyAllWindows()
    
    if flagPressedS and hist is not None:
        with open("hist", "wb") as f:
            pickle.dump(hist, f)
        print("✓ Hand histogram saved successfully!")
    else:
        print("✗ Hand histogram not saved.")

if __name__ == "__main__":
    get_hand_hist()
