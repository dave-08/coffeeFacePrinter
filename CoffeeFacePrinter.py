import cv2
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk
import os
import asyncio
import websockets
import base64
from PIL import Image
import json
import numpy as np


class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Capture")

        self.file_path = " "
        self.running = True

        # Create a label to display the webcam feed
        self.video_label = Label(root)
        self.video_label.pack()

        # Add buttons for taking a picture and exiting
        self.capture_button = Button(root, text="Capture & Print", command=self.start_capture_image)
        self.capture_button.pack(pady=10)

        self.exit_button = Button(root, text="Exit", command=self.exit_app)
        self.exit_button.pack(pady=10)

        self.entryPanel = Label(root,bg="grey",height=5,width=90)
        self.entryPanel.pack(pady=10)
    

        

        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)  # 0 for the default webcam

        self.update_frame()

    def update_frame(self):
        """Update the video frame in the GUI."""
        if not self.cap.isOpened():
            return  # Exit if the webcam is not open
        
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to an image compatible with Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the video label with the new frame
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        # Call this method again after a short delay
        if self.running:
            self.root.after(10, self.update_frame)

    def start_capture_image(self):
        """Start the asynchronous capture and WebSocket handling."""
        asyncio.create_task(self.capture_image())

    async def capture_image(self):
        """Capture the current frame, save it, and send it via WebSocket."""
        ret, frame = self.cap.read()
        if ret:
            self.file_path = os.getcwd() + '\camImg.jpg'
            self.print_path = os.getcwd() + "\printImg.png"
            if self.file_path:
                cv2.imwrite(self.file_path, frame)
                print(f"Image saved to {self.file_path}")
                self.capture_button.config(state=tk.DISABLED)
                self.process_image(self.file_path,self.print_path)
            
            ip = "192.168.1.113"  # IP address of the coffee maker
            port = "8888"
            await self.websocket_client(ip, port)

    async def websocket_client(self, ip, port):
        uri = f"ws://{ip}:{port}"
        try:
            async with websockets.connect(uri) as websocket:
                print(f"Connected to {uri}")

                # Check if the machine is idle
                comm_msg = {
                    "code": 1,
                    "tag": 1
                }
                await websocket.send(json.dumps(comm_msg))
                self.writeMsg(f"Message sent: {json.dumps(comm_msg)}")
                print(f"Message sent: {json.dumps(comm_msg)}")

                response = await websocket.recv()
                print(f"Response: {response}")
                self.writeMsg(response)

                if 'machine is idle' in response.lower():
                    self.writeMsg("Machine is idle.")
                    print("Machine is idle.")
                    
                    image_data = self.image_to_print(self.print_path)
                    size = 75
                    imagedata = {
                        "code": 2,
                        "tag": 1,
                        "data": {
                            "size": size,
                            "img": image_data
                        }
                    }
                    await websocket.send(json.dumps(imagedata))

                    response_prev = None
                    while True:
                        loop_msg = {
                            "code": 2,
                            "tag": 1
                        }
                        await websocket.send(json.dumps(loop_msg))
                        response = await websocket.recv()
                        self.writeMsg(response)

                        if response != response_prev:
                            print(response)
                            self.writeMsg(response)
                        response_prev = response

                        if "printing succeeded" in response.lower():
                            print("Printing Succeeded")
                            self.writeMsg("Printing Succeeded")
                            break
                        elif "printing failed" in response.lower():
                            print("Printing Failed")
                            self.writeMsg("Printing Failed")
                            break

                else:
                    print("Machine is not idle.")
                    self.writeMsg("Machine is not idle.")
        except Exception as e:
            print(f"Connection error: {e}")
            self.writeMsg(f"Connection error: {e}")
        self.capture_button.config(state=tk.NORMAL)

    def image_to_print(self, image_path):
        """Convert the image to Base64 format."""
        try:
            with Image.open(image_path) as img:
                breakpoint
                # Save to buffer in JPEG format
                from io import BytesIO
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_data = buffered.getvalue()

                # Encode to Base64
                base64_string = base64.b64encode(img_data).decode("utf-8")
                return f"data:image/jpeg;base64,{base64_string}"
            
        except Exception as e:
            print(f"Error: {e}")
            return None

    def detect_face(self,image):
        """Detect face and return the bounding box."""
        # Load the pre-trained Haar Cascade Classifier for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None  # No face detected
        
        # Get the first detected face (assuming there's only one face)
        (x, y, w, h) = faces[0]
        
        return (x, y, w, h)

    def crop_and_resize(self,image, face_bbox, output_size=(1800, 1800)):
    
        if face_bbox is None:
            return None
        
        x, y, w, h = face_bbox
        
        # Calculate the center of the face
        face_center = (x + w//2, y + h//2)
        
        marginScale = 2.5
        # Define the cropping area (center the face)
        crop_margin = int((max(w, h)* marginScale)) # Use the larger dimension for cropping
        crop_x1 = max(face_center[0] - crop_margin // 2, 0)
        crop_y1 = max(face_center[1] - crop_margin // 2, 0)
        crop_x2 = min(face_center[0] + crop_margin // 2, image.shape[1])
        crop_y2 = min(face_center[1] + crop_margin // 2, image.shape[0])
        
        cropped_image = image[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # Resize the cropped image to 800x800
        resized_image = cv2.resize(cropped_image, output_size)
        
        return resized_image

    def apply_circle_mask(self,image):
    
        # Create a black mask of the same size as the image
        mask = np.zeros_like(image)

        # Create a circular mask
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        radius = min(width, height) // 2

        # Create a mask for the circle
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)

        # Add an alpha channel
        image_with_alpha = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        image_with_alpha[:, :, 3] = mask  # Set the alpha channel to the circular mask

        return image_with_alpha

    def process_image(self,image_path, output_path):
        """Load an image, detect the face, crop and resize it, and apply a circular mask."""
        # Read the image
        image = cv2.imread(image_path)
        
        # Detect the face
        face_bbox = self.detect_face(image)
        
        # Crop and resize the image
        cropped_resized_image = self.crop_and_resize(image, face_bbox)
        
        if cropped_resized_image is None:
            print("No face detected.")
            return
        
        # Apply the circular mask
        circular_image = self.apply_circle_mask(cropped_resized_image)
        
        cv2.imwrite(output_path, circular_image)

    def writeMsg (self,message):
        msgonPanel = "Note :" + message
        self.entryPanel.config(text=msgonPanel)

    def exit_app(self):
        """Release the webcam and close the application."""
        self.running = False  # Stop the update loop
        self.cap.release()
        self.root.destroy()


# Run the app with asyncio event loop
if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)

    # Run the Tkinter mainloop alongside asyncio
    async def run_tk():
        while True:
            root.update()
            await asyncio.sleep(0.01)
    asyncio.run(run_tk())