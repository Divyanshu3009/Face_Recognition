import cv2

def generate_dataset():
    face_detect = cv2.CascadeClassifier("C:/Users/divya/OneDrive/Desktop/LEAP/Face Recognition/haarcascade_frontalface_default.xml")
    
    def face_cropped(img):
        clour_change = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detect.detectMultiScale(clour_change,1.3,3)
         
        if faces is ():
            return None
        for(x,y,w,h) in faces:
            cropped_face = img[y: y+h, x:x+w]
            return cropped_face
        
    cap = cv2.VideoCapture(0)
    user_id = 3
    img_id = 0
    
    while True:
        ret,frame = cap.read()
   
        if face_cropped(frame) is not None:
            img_id+= 1
            photo = cv2.resize(face_cropped(frame),(250,250))
            face = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
            file_path = "C:/Users/divya/OneDrive/Desktop/LEAP/Face Recognition/Data/" +str(user_id)+ "." +str(img_id)+ ".jpg"
            cv2.imwrite(file_path,face)
            cv2.putText(face,str(img_id),(50,50),cv2.FONT_HERSHEY_PLAIN,1, (0,255,0),2)
            
            cv2.imshow("Cropped Face", face)
            if cv2.waitKey(1)==13 or int(img_id)==200:
                break
    cap.release()
    cv2.destroyAllWindows() 
    print ("Collection of sample is compleated. . . .")

generate_dataset()
