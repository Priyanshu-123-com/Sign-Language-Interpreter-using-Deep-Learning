import cv2
import numpy as np
import pickle, os, sqlite3, random

image_x, image_y = 50, 50

def get_hand_hist():
    with open("hist", "rb") as f:
        hist = pickle.load(f)
    return hist

def init_create_folder_database():
    # create the folder and database if not exist
    if not os.path.exists("gestures"):
        os.mkdir("gestures")
        print("✓ Created 'gestures' folder")
    
    if not os.path.exists("gesture_db.db"):
        conn = sqlite3.connect("gesture_db.db")
        create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
        conn.execute(create_table_cmd)
        conn.commit()
        print("✓ Created gesture database")
    else:
        print("✓ Gesture database already exists")

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

def store_in_db(g_id, g_name):
    conn = sqlite3.connect("gesture_db.db")
    cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, \'%s\')" % (g_id, g_name)
    try:
        conn.execute(cmd)
        print(f"✓ Added gesture {g_id}: {g_name} to database")
    except sqlite3.IntegrityError:
        choice = input("g_id already exists. Want to change the record? (y/n): ")
        if choice.lower() == 'y':
            cmd = "UPDATE gesture SET g_name = \'%s\' WHERE g_id = %s" % (g_name, g_id)
            conn.execute(cmd)
            print(f"✓ Updated gesture {g_id}: {g_name}")
        else:
            print("Doing nothing...")
            return
    conn.commit()
    conn.close()
    
def store_images(g_id):
    total_pics = 1200
    hist = get_hand_hist()
    
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
    create_folder("gestures/"+str(g_id))
    pic_no = 0
    flag_start_capturing = False
    frames = 0
    
    print(f"\nCapturing gesture {g_id}...")
    print("Instructions:")
    print("1. Place your hand in the green rectangle")
    print("2. Press 'c' to start/stop capturing")
    print("3. Make sure your hand is clearly visible")
    print("4. The system will capture 1200 images automatically")
    print("5. Press 'q' to quit early")
    
    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to read from camera")
            break
            
        img = cv2.flip(img, 1)
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        cv2.filter2D(dst,-1,disc,dst)
        blur = cv2.GaussianBlur(dst, (11,11), 0)
        blur = cv2.medianBlur(blur, 15)
        thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        thresh = cv2.merge((thresh,thresh,thresh))
        thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        thresh = thresh[y:y+h, x:x+w]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

        if len(contours) > 0:
            contour = max(contours, key = cv2.contourArea)
            if cv2.contourArea(contour) > 10000 and frames > 50 and flag_start_capturing:
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                pic_no += 1
                save_img = thresh[y1:y1+h1, x1:x1+w1]
                if w1 > h1:
                    save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
                elif h1 > w1:
                    save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
                save_img = cv2.resize(save_img, (image_x, image_y))
                rand = random.randint(0, 10)
                if rand % 2 == 0:
                    save_img = cv2.flip(save_img, 1)
                cv2.imwrite("gestures/"+str(g_id)+"/"+str(pic_no)+".jpg", save_img)
                print(f"Captured image {pic_no}/{total_pics}", end='\r')

        # Draw hand region rectangle
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        
        # Add status text
        if flag_start_capturing:
            cv2.putText(img, "CAPTURING...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0))
        else:
            cv2.putText(img, "Press 'c' to start capturing", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255))
        
        cv2.putText(img, f"Images: {pic_no}/{total_pics}", (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
        cv2.putText(img, "Press 'c' to start/stop, 'q' to quit", (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255))
        
        cv2.imshow("Capturing gesture", img)
        cv2.imshow("Hand Detection", thresh)
        
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('c'):
            if flag_start_capturing == False:
                flag_start_capturing = True
                print("\n✓ Started capturing...")
            else:
                flag_start_capturing = False
                frames = 0
                print(f"\n✓ Stopped capturing. Captured {pic_no} images.")
        elif keypress == ord('q'):
            print(f"\n✓ Quit early. Captured {pic_no} images.")
            break
            
        if flag_start_capturing == True:
            frames += 1
        if pic_no == total_pics:
            print(f"\n✓ Completed! Captured {total_pics} images.")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print(f"✓ Gesture {g_id} capture completed!")

def main():
    print("=" * 60)
    print("Sign Language Gesture Capture Tool")
    print("=" * 60)
    
    # Initialize folders and database
    init_create_folder_database()
    
    print("\nThis tool will help you capture training data for your sign language recognition system.")
    print("You'll need to capture multiple gestures (letters, numbers, etc.)")
    
    while True:
        print("\n" + "-" * 40)
        g_id = input("Enter gesture ID (number, e.g., 0, 1, 2...): ")
        if not g_id.isdigit():
            print("Please enter a valid number!")
            continue
            
        g_name = input("Enter gesture name/text (e.g., 'A', 'Hello', '1'): ")
        if not g_name.strip():
            print("Please enter a valid gesture name!")
            continue
            
        store_in_db(g_id, g_name)
        store_images(g_id)
        
        continue_capture = input("\nDo you want to capture another gesture? (y/n): ")
        if continue_capture.lower() != 'y':
            break
    
    print("\n✓ Gesture capture session completed!")
    print("Next steps:")
    print("1. Run 'python load_images.py' to prepare training data")
    print("2. Run 'python cnn_model_train.py' to train the model")
    print("3. Run 'python final.py' to test recognition")

if __name__ == "__main__":
    main()
