import cv2
import tkinter as tk
from tkinter import Toplevel, Button, Label, Entry
from PIL import Image, ImageTk
import os
import asyncio
import websockets
import base64
from PIL import Image
import json
import numpy as np
import logging

# Initialize logging
logging.basicConfig(
    filename="webcam_app.log",  # Log file name
    level=logging.INFO,  # Log level (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format="%(asctime)s [%(levelname)s] %(message)s",  # Log format
    datefmt="%Y-%m-%d %H:%M:%S"  # Date format
)

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam Capture")
        self.file_path = ""
        self.running = True

        # Default settings
        self.ip_address = "192.168.1.113"
        self.port = "8888"
        self.cup_diameter = "75"

        # Handle the close button (X)
        self.root.protocol("WM_DELETE_WINDOW", self.exit_app)

        # Create a label to display the webcam feed
        self.video_label = Label(root)
        self.video_label.pack()

        # Add settings button
        self.settings_button = Button(root, text="⚙️ Settings", command=self.open_settings_window)
        self.settings_button.pack(pady=10)

        # Add buttons for capturing and exiting
        self.capture_button = Button(root, text="Capture & Print", command=self.start_capture_image)
        self.capture_button.pack(pady=10)

        self.exit_button = Button(root, text="Exit", command=self.exit_app)
        self.exit_button.pack(pady=10)

        self.entryPanel = Label(root, bg="grey", height=5, width=90)
        self.entryPanel.pack(pady=10)

        # Initialize the webcam
        self.cap = cv2.VideoCapture(0)  # 0 for the default webcam

        self.update_frame()

    def update_frame(self):
        """Update the video frame in the GUI."""
        if not self.cap.isOpened():
            return

        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        if self.running:
            self.root.after(10, self.update_frame)

    def open_settings_window(self):
        """Open the settings window."""
        settings_window = Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("300x200")

        # Labels and Entry fields for settings
        Label(settings_window, text="IP Address:").grid(row=0, column=0, pady=5, padx=5, sticky="w")
        ip_entry = Entry(settings_window, width=20)
        ip_entry.insert(0, self.ip_address)
        ip_entry.grid(row=0, column=1, pady=5, padx=5)

        Label(settings_window, text="Port Number:").grid(row=1, column=0, pady=5, padx=5, sticky="w")
        port_entry = Entry(settings_window, width=20)
        port_entry.insert(0, self.port)
        port_entry.grid(row=1, column=1, pady=5, padx=5)

        Label(settings_window, text="Cup Diameter (mm):").grid(row=2, column=0, pady=5, padx=5, sticky="w")
        cup_dia_entry = Entry(settings_window, width=20)
        cup_dia_entry.insert(0, self.cup_diameter)
        cup_dia_entry.grid(row=2, column=1, pady=5, padx=5)

        # Save button
        save_button = Button(settings_window, text="Save", command=lambda: self.save_settings(
        ip_entry.get(), port_entry.get(), cup_dia_entry.get(), settings_window))
        save_button.grid(row=3, column=1, pady=10)

    def save_settings(self, ip, port, cup_dia, settings_window):
        """Save the settings entered by the user."""
        self.ip_address = ip
        self.port = port
        self.cup_diameter = cup_dia
        logging.info(f"Settings updated: IP={self.ip_address}, Port={self.port}, Cup Diameter={self.cup_diameter}")
        self.writeMsg(f"Settings updated: IP={self.ip_address}, Port={self.port}, Cup Diameter={self.cup_diameter}")
        settings_window.destroy()

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
                logging.info(f"Image saved to {self.file_path}")
                self.capture_button.config(state=tk.DISABLED)
                self.process_image(self.file_path,self.print_path)
            
            ip = self.ip_address  # IP address of the coffee maker
            port = self.port
            await self.websocket_client(ip, port)

    async def websocket_client(self, ip, port):
        uri = f"ws://{ip}:{port}"
        try:
            async with websockets.connect(uri) as websocket:
                logging.info(f"Connected to {uri}")

                # Check if the machine is idle
                comm_msg = {
                    "code": 1,
                    "tag": 1
                }
                await websocket.send(json.dumps(comm_msg))
                logging.info(f"Message sent: {json.dumps(comm_msg)}")

                response = await websocket.recv()
                logging.info(f"Response: {response}")
                self.writeMsg(response)

                if 'machine is idle' in response.lower():
                    logging.info("Machine is idle.")
                    
                    image_data = self.image_to_print(self.print_path)
                    size = int(self.cup_diameter)
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
                            logging.info(response)
                            self.writeMsg(response)
                        response_prev = response

                        if "printing succeeded" in response.lower():
                            logging.info("Printing Succeeded")
                            self.writeMsg("Printing Succeeded")
                            break
                        elif "printing failed" in response.lower():
                            logging.info("Printing Failed")
                            self.writeMsg("Printing Failed")
                            break

                else:
                    logging.info("Machine is not idle.")
                    self.writeMsg("Machine is not idle.")
        except Exception as e:
            logging.info(f"Connection error: {e}")
            self.writeMsg(f"Connection error: {e}")
            

        self.capture_button.config(state=tk.NORMAL)

    def image_to_print(self, image_path):
        """Convert the image to Base64 format."""
        try:
            with Image.open(image_path) as img:
                # Save to buffer in JPEG format
                from io import BytesIO
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_data = buffered.getvalue()

                # Encode to Base64
                base64_string = base64.b64encode(img_data).decode("utf-8")
                return f"data:image/jpeg;base64,{base64_string}"
        except Exception as e:
            logging.info(f"Error: {e}")
            return None

    def detect_face(self,image):
        """Detect face and return the bounding box."""
        # Load the pre-trained Haar Cascade Classifier for face detection
        current_directory = os.getcwd()  # Get the current working directory
        cascade_path = os.path.join(current_directory,'_internal', 'haarcascade_frontalface_default.xml')

        #dist\CoffeeFacePrinter\_internal
        logging.info("Pre-Trained Face detection file "+cascade_path)
        
        face_cascade = cv2.CascadeClassifier(cascade_path)

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None  # No face detected
        
        # Get the first detected face (assuming there's only one face)
        (x, y, w, h) = faces[0]
        
        return (x, y, w, h)

    def crop_and_resize(self,image, face_bbox, output_size=(800, 800)):
    
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
    
        """Apply a circular mask with a transparent background to the given image."""
        # Convert image to RGBA (adds an alpha channel for transparency)
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)

        # Create a blank mask with the same dimensions as the image (including alpha channel)
        mask = np.zeros_like(image_rgba)

        # Create a circular mask
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        radius = min(width, height) // 2

        # Draw a white-filled circle on the alpha channel of the mask
        cv2.circle(mask, center, radius, (255, 255, 255, 255), -1)

        # Combine the image with the mask to retain only the circular area
        circular_image = cv2.bitwise_and(image_rgba, mask)

        # Set the alpha channel of the non-circular area to 0 (fully transparent)
        circular_image[mask[..., 3] == 0] = [0, 0, 0, 0]

        return circular_image

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
        if self.cap.isOpened():
            self.cap.release()  # Release the webcam
        self.root.destroy()  # Close the application window


# Run the app with asyncio event loop
if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)

    async def tk_async_mainloop():
        while app.running:
            try:
                root.update()
                await asyncio.sleep(0.01)  # Allow other tasks to run
            except tk.TclError as e:
                if "application has been destroyed" in str(e):
                    break  # Exit when the app is closed

    loop = asyncio.get_event_loop()
    loop.run_until_complete(tk_async_mainloop())