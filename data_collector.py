
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

import os
import time
import numpy as np
from PIL import Image, ImageTk, ImageGrab
import tkinter as tk

from screeninfo import get_monitors

SAVE_PATH = os.getenv("SAVE_PATH", default="collected_data/")

def capture_screenshot():
    return ImageGrab.grab()

class DataCollector:
    def __init__(self, master):
        self.master = master
        master.title("Data Collection for Icon Locator")

        # Detect monitors and set primary and secondary
        monitors = get_monitors()
        self.primary_monitor = monitors[0]
        self.secondary_monitor = monitors[1] if len(monitors) > 1 else monitors[0]

        # Set the GUI to open in the center of the secondary monitor
        master.geometry(f"+{self.secondary_monitor.x}+{self.secondary_monitor.y}")

        # Capture a screenshot of the primary monitor
        left, top, right, bottom = self.primary_monitor.x, self.primary_monitor.y, self.primary_monitor.width, self.primary_monitor.height
        self.screenshot = ImageGrab.grab(bbox=(left, top, right, bottom))

        # Display a button to initiate overlaying the screenshot on the primary monitor
        self.overlay_button = tk.Button(master, text="Overlay Screenshot", command=self.overlay_screenshot)
        self.overlay_button.pack()

    def overlay_screenshot(self):
        # Open a full screen, borderless window on the primary monitor to display the screenshot
        overlay_window = tk.Toplevel(self.master)
        overlay_window.attributes('-fullscreen', True)
        overlay_window.attributes('-topmost', True)
        overlay_window.overrideredirect(1)  # Make the window borderless

        self.screenshot_image = ImageTk.PhotoImage(self.screenshot)
        label = tk.Label(overlay_window, image=self.screenshot_image)
        label.pack()

        # Bind a click event to capture data
        label.bind("<Button-1>", self.capture_data)

    def capture_data(self, event):
        # Capture pixel data around the clicked location
        region = (event.x - 32, event.y - 32, event.x + 32, event.y + 32)
        icon_region = self.screenshot.crop(region)

        # Save the captured region (this can be further modified to save to a database)
        icon_region.save(os.path.join(SAVE_PATH, f"icon_{time.time()}.png"))

if __name__ == "__main__":
    root = tk.Tk()
    collector = DataCollector(root)
    root.mainloop()
