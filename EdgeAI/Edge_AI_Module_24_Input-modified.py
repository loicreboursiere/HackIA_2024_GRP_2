#!/usr/bin/env python
# coding: utf-8

from tkinter import *
import torch
from torchvision import transforms
import face_recognition
import face_recognition
import cv2
import time
import numpy as np
import os
from PIL import ImageTk, Image 
from ultralytics import YOLO
from torch2trt import TRTModule
import collections
import telegram
import asyncio


LOADED = False
BASE_PATH = os.path.abspath(os.path.dirname(__file__))

# Classification Models
Base_Classifier = 'models/firenet.pt'                            # Initial Model for Classification
Edge_Classifier = 'compressed_models/firenet.pth'                # Compressed Model for Classification

# Detection models 
Base_Detector = 'models/best (5).pt'  #Le Top 'models/best (5).pt'                              # Initial Model for Detection                                  
Edge_Detector = 'compressed_models/best.engine'                  # Compressed Model for Detection





TOKEN = "**********************************************"
chat_id = '**********'



bot = telegram.Bot(token=TOKEN)


async def send_message(text, chat_id):
    async with bot:
        await bot.send_message(text=text, chat_id=chat_id)


async def send_document(document, chat_id):
    async with bot:
        await bot.send_document(document=document, chat_id=chat_id)


async def send_photo(photo, chat_id):
    async with bot:
        await bot.send_photo(photo=photo, chat_id=chat_id)


async def send_video(video, chat_id):
    async with bot:
        await bot.send_video(video=video, chat_id=chat_id)


async def authentification():
    """
    Fonction qui permet de s'identifier lors du lancement de la Jetson
    """

    # Load a sample picture and learn how to recognize it.
    sidi_image = face_recognition.load_image_file("faces/sidi.jpg")
    sidi_face_encoding = face_recognition.face_encodings(sidi_image)[0]
    rabie_image = face_recognition.load_image_file("faces/rabie.png")
    rabie_face_encoding = face_recognition.face_encodings(rabie_image)[0]
    francesco_image = face_recognition.load_image_file("faces/francesco.png")
    francesco_face_encoding = face_recognition.face_encodings(francesco_image)[0]
    berengere_image = face_recognition.load_image_file("faces/berengere.png")
    berengere_face_encoding = face_recognition.face_encodings(berengere_image)[0]
    loic_image = face_recognition.load_image_file("faces/rabie.png")
    loic_face_encoding = face_recognition.face_encodings(loic_image)[0]
 
    # Create arrays of known face encodings and their names
    known_face_encodings = [
        sidi_face_encoding,
        rabie_face_encoding,
        francesco_face_encoding,
        berengere_face_encoding,
        loic_face_encoding,
    ]
    known_face_names = [
		"Sidi",
		"Rabie",
		"Francesco",
		"Berengere",
		"Loïc",
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    authorized = False
    video_capture = cv2.VideoCapture(0)
    SEUIL = 50
    lst_cpt = []
    cmp=0
    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        cmp+=1
        # Only process every other frame of video to save
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            code = cv2.COLOR_BGR2RGB
            rgb_small_frame = cv2.cvtColor(rgb_small_frame, code)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding
                )
                name = "Unknown"
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding
                )
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    if name in known_face_names:
                        lst_cpt.append(name)
                    if len(lst_cpt) == SEUIL:
                        if len(set(lst_cpt)) == 1:
                            authorized = True
                            video_capture.release()
                            cv2.destroyAllWindows()
                            return authorized
                        else:
                            lst_cpt = []
                    face_names.append(name)
        process_this_frame = not process_this_frame
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1
            )

        # Display the resulting image
        cv2.imshow("Video", frame)
        if cmp>200:
        	cv2.imwrite("tempo.jpg", frame) 
        	await send_message(text="Attention ! Tentative d'accès à votre machine par un inconnu.", chat_id=chat_id)
        	await send_photo(photo=open('./tempo.jpg', 'rb'), chat_id=chat_id)
        	
        	return False
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return authorized



def fire_detection(classifier_path, detector_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # fire model
    fire_model = torch.load(classifier_path)                          # This load allows just to know the model type 
    # if quantized model: load with TRTModule
    if isinstance(fire_model, collections.OrderedDict):
    	fire_model = TRTModule()
    	fire_model.load_state_dict(torch.load(classifier_path))
    # yolo model 
    yolo_model = YOLO(detector_path, verbose=False)
    #yolo_model=torch.hub.load('ultralytics/yolov5','custom',path='/home/ilia/HackIA24_Input/models/yolov8n.pt')

    
    
    fire_model.train(False)
    fire_model.to(device)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]
    )
    classes = ["Start fire", "No fire", "Fire" ]
    average_fps = []
    videos_to_test = ["fire1.mp4"]
    for video_to_test in videos_to_test:
        cam_port = os.path.join(BASE_PATH, "videos_tests/" + video_to_test)
        cam = cv2.VideoCapture(cam_port)
        #
        prev_frame_time = 0
        new_frame_time = 0
        #
        while cam.isOpened():
            success, img = cam.read()
            if not success:
                break
            data = transform(img).to(device)
            data.unsqueeze_(0)
            with torch.inference_mode():
                classe = fire_model(data).cpu().detach().numpy()
            
            # si classe = fire ou smoke alors run le modèle de détection YOLO pour détecter emplacement du feu
            if np.argmax(classe) == 0 or np.argmax(classe) == 2:
                with torch.no_grad():
                    results = yolo_model.predict(img, imgsz=224)
                boxes = results[0].boxes.xyxy.tolist()
                classes_yolo = results[0].boxes.cls.tolist()
                names = results[0].names
                confidences = results[0].boxes.conf.tolist()

                # Iterate through the results
                for box, cls, conf in zip(boxes, classes_yolo, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    confidence = conf
                    detected_class = cls
                    name = names[int(cls)]
                    # Draw the bounding box and label
                    cv2.rectangle(
                        img, (x1, y1), (x2, y2), (255, 128, 0), 2
                    )
                    cv2.putText(
                        img,
                        f"{name} {confidence:.2f}",
                        (x1+2, y2-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 128, 0),
                        2,
                    )

            cv2.putText(
                img,
                classes[np.argmax(classe)],
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                thickness=3,
            )
            
            # calculate FPS and display it
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            average_fps.append(fps)
            fps = str(fps)
            cv2.putText(
                img,
                "FPS: " + fps,
                (500, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Fire Detection", cv2.resize(img, (640, 480)))
            
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cam.release()
        print(f'Average FPS = {sum(average_fps) / len(average_fps):.2f} FPS') 
    cv2.destroyAllWindows()
    del fire_model

def t(nom):
    	cap=cv2.VideoCapture('/home/ilia/HackIA24_Input/XAI/'+nom)
    	if not cap.isOpened():
    		print("Erreur: Impossible d'ouvrir la vidéo.")
	    	exit()
    	while cap.isOpened():
    		ret, frame = cap.read()  # ret: indicateur de succès, frame: l'image de la vidéo

    		if not ret:
    			print("Fin de la vidéo ou erreur de lecture.")
    			break

	    # Afficher la frame
    		cv2.imshow('Frame', frame)

	    # Appuyer sur 'q' pour quitter
    		if cv2.waitKey(25) & 0xFF == ord('q'):
    			break
    	cap.release()
    	cv2.destroyAllWindows()
    	return
	
def XAI():
    	if True:
		    xai_window = Tk()
		    xai_window.resizable(width=False, height=False)
		    xai_window.geometry("400x300")
		    xai_window.title("XAI Options")

		    main_frame = Frame(xai_window, relief=RIDGE, borderwidth=2)
		    main_frame.config(background="lightblue")
		    main_frame.pack(fill=BOTH, expand=1)

		    label_msg = Label(
			main_frame, text=("Choose an option:"), bg="lightblue", font=("Helvetica 20 bold"), height=2
		    )
		    label_msg.pack(side=TOP, pady=20)

		    but_start_fire = Button(
			main_frame,
			padx=5,
			pady=5,
			width=20,
			height=2,
			bg="white",
			fg="black",
			relief=RAISED,
			command=lambda: t("start_fire.avi"),
			text="Start Fire",
			font=("helvetica 16 bold"),
		    )
		    but_start_fire.pack(side=TOP, pady=5)

		    but_no_fire = Button(
			main_frame,
			padx=5,
			pady=5,
			width=20,
			height=2,
			bg="white",
			fg="black",
			relief=RAISED,
			command=lambda: t("no_fire.avi"),
			text="No Fire",
			font=("helvetica 16 bold"),
		    )
		    but_no_fire.pack(side=TOP, pady=5)

		    but_fire = Button(
			main_frame,
			padx=5,
			pady=5,
			width=20,
			height=2,
			bg="white",
			fg="black",
			relief=RAISED,
			command=lambda: t("Fire.avi"),
			text="Fire",
			font=("helvetica 16 bold"),
		    )
		    but_fire.pack(side=TOP, pady=5)

		    xai_window.mainloop()
    	cap=cv2.VideoCapture('/media/ilia/FATTY/HackIA24_Input/XAI/Fire_analysis.mp4')
    	if not cap.isOpened():
    		print("Erreur: Impossible d'ouvrir la vidéo.")
	    	exit()
    	while cap.isOpened():
    		ret, frame = cap.read()  # ret: indicateur de succès, frame: l'image de la vidéo

    		if not ret:
    			print("Fin de la vidéo ou erreur de lecture.")
    			break

	    # Afficher la frame
    		cv2.imshow('Frame', frame)

	    # Appuyer sur 'q' pour quitter
    		if cv2.waitKey(25) & 0xFF == ord('q'):
    			break
    	cap.release()
    	cv2.destroyAllWindows()
    	return 



def main():
    # Etape 1: authentification pour accéder à l'interface graphique

    authorized = authentification()
    #authorized=True
    if not authorized:
        print("No authorization")
        from sys import exit
        exit(0)
    

    # Etape 2: interface graphique
    root = Tk()
    root.resizable(width=False, height=False)
    root.geometry("1000x650")
    root.title("Edge AI System")
    
    # Main frame
    main_frame = Frame(root, relief=RIDGE, borderwidth=2)
    main_frame.config(background="green2")
    main_frame.pack(fill=BOTH, expand=1)
    #
    # Welcome message for user
    label_msg = Label(
        main_frame, text=("Welcome!"), bg="green2", font=("Helvetica 24 bold"), height=2
    )
    label_msg.pack(side=TOP)
    label_msg2 = Label(
        main_frame,
        text=("Hello, you are well authorized, congrats !"),
        bg="green2",
        font=("Helvetica 22 bold"),
    )
    label_msg2.pack(side=TOP)
    
    # add logos to the interface
    logo1 = Image.open("logos/all_logos.png")
    logo1 = logo1.resize((990, 90), Image.LANCZOS) 
    logo1 = ImageTk.PhotoImage(logo1)
    logo_label1 = Label(main_frame, image=logo1)
    logo_label1.image = logo1
    logo_label1.pack(side=BOTTOM)

    
    # Ajout texte
    label_msg3 = Label(
        main_frame,
        text=("Initial version"),
        bg="green2",
        fg="black",
        font=("Helvetica 20 bold"),
    )
    label_msg3.place(x=220, y=200)
    label_msg4 = Label(
        main_frame,
        text=("Compressed version"),
        bg="green2",
        fg="black",
        font=("Helvetica 20 bold"),
    )
    label_msg4.place(x=580, y=200)
    # Menu
    but1 = Button(
        main_frame,
        padx=5,
        pady=5,
        width=25,
        height=2,
        bg="white",
        fg="black",
        relief=RAISED,
        command=lambda: fire_detection(Base_Classifier, Base_Detector),
        text="Fire detection",
        font=("helvetica 16 bold"),
    )

    but2 = Button(
        main_frame,
        padx=5,
        pady=5,
        # bd=5,
        height=2,
        width=25,
        bg="white",
        fg="black",
        relief=RAISED,
        command=XAI,
        text="XAI",
        font=("helvetica 16 bold"),
    )

    but3 = Button(
        main_frame,
        padx=5,
        pady=5,
        height=2,
        # bd=5,
        width=25,
        bg="white",
        fg="black",
        relief=RAISED,
        command=lambda: fire_detection(Edge_Classifier, Edge_Detector),
        text="Fire detection",
        font=("helvetica 16 bold"),
    )

    but4 = Button(
        main_frame,
        padx=5,
        pady=5,
        height=2,
        # bd=5,
        width=25,
        bg="white",
        fg="black",
        relief=RAISED,
        command=XAI,
        text="XAI",
        font=("helvetica 16 bold"),
    )
    but1.place(x=150, y=280)
    but2.place(x=150, y=380)
    but3.place(x=550, y=280)
    but4.place(x=550, y=380)

    but5 = Button(
        main_frame,
        padx=5,
        pady=5,
        # bd=5,
        height=2,
        width=14,
        bg="white",
        fg="black",
        relief=RAISED,
        command=root.destroy,
        text="Exit",
        font=("helvetica 15 bold"),
    )
    but5.place(x=800, y=480)

    root.mainloop()

async def fct():
	#authorized=True
	authorized =await authentification()
	#print(authorized)
	if not authorized:
	    # Unauthorized access window
	    root = Tk()
	    root.resizable(width=False, height=False)
	    root.geometry("700x700")
	    root.title("Unauthorized Access")

	    # Main frame
	    main_frame = Frame(root, relief=RIDGE, borderwidth=2)
	    main_frame.config(background="red")
	    main_frame.pack(fill=BOTH, expand=1)

	    # Warning message for user
	    label_msg = Label(
		main_frame,
		text=("Warning!"),
		bg="red",
		fg="white",
		font=("Helvetica 24 bold"),
		height=2,
	    )
	    label_msg.pack(side=TOP, pady=20)
	    label_msg2 = Label(
		main_frame,
		text=("You are not authorized to access this machine."),
		bg="red",
		fg="white",
		font=("Helvetica 18 bold"),
	    )
	    label_msg2.pack(side=TOP)
	    label_msg3 = Label(
		main_frame,
		text=("A capture of your face has been sent to the administrator."),
		bg="red",
		fg="white",
		font=("Helvetica 16 bold"),
	    )
	    label_msg3.pack(side=TOP, pady=20)

	    # Exit button
	    but_exit = Button(
		main_frame,
		padx=5,
		pady=5,
		height=2,
		width=14,
		bg="white",
		fg="black",
		relief=RAISED,
		command=root.destroy,
		text="Exit",
		font=("helvetica 15 bold"),
	    )
	    but_exit.pack(side=BOTTOM, pady=20)

	    root.mainloop()
			
		
		
		#print("No authorization")
		#from sys import exit
		#exit(0)
	else:
		main()
		
asyncio.run(fct())
