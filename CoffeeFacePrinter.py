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
            self.file_path = os.getcwd() + "/printImg.jpg"
            if self.file_path:
                cv2.imwrite(self.file_path, frame)
                print(f"Image saved to {self.file_path}")
                self.capture_button.config(state=tk.DISABLED)

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
                print(f"Message sent: {json.dumps(comm_msg)}")

                response = await websocket.recv()
                print(f"Response: {response}")
                self.writeMsg(response)

                if 'machine is idle' in response.lower():
                    print("Machine is idle.")
                    image_data = self.image_to_print(self.file_path)
                    size = 70
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
        except Exception as e:
            print(f"Connection error: {e}")
        self.capture_button.config(state=tk.NORMAL)

    def image_to_print(self, image_path):
        """Convert the image to Base64 format."""
        try:
            with Image.open(image_path) as img:
                # Resize the image
                img = img.resize((800, 800), Image.Resampling.LANCZOS)

                # Convert to grayscale
                img = img.convert("L")

                # Save to buffer in JPEG format
                from io import BytesIO
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_data = buffered.getvalue()

                # Encode to Base64
                base64_string = base64.b64encode(img_data).decode("utf-8")
                return f"data:image/jpeg;base64,{base64_string}"
        except Exception as e:
            print(f"Error: {e}")
            return None

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
