import json
import random
import os
import sys
import time
import tkinter as tk
from tkinter import Tk, Label, Button, filedialog, Toplevel
from PIL import Image, ImageTk, ImageGrab
from CNN import IconLocatorCNN
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from screeninfo import get_monitors
from datetime import datetime
import pyautogui
from stable_baselines3 import PPO
from tkinter import simpledialog
from stable_baselines3.common.callbacks import BaseCallback

def load_screenshots(directory_path):
    screenshot_paths = [os.path.join(directory_path, fname) for fname in os.listdir(directory_path) if fname.endswith('.jpg')]
    screenshots = [Image.open(path) for path in screenshot_paths]
    return screenshots



class ReinforcementTrainer:
    def __init__(self, model, state_dim, action_dim):
        self.model = model
        self.agent = PPO("MlpPolicy", self.model, verbose=1)
    
    def train(self, similarity_scores, rewards):
        # Train the agent using similarity scores and rewards
        self.agent.learn(total_timesteps=len(similarity_scores), callback=CustomCallback(similarity_scores, rewards))

class CustomCallback(BaseCallback):
    def __init__(self, similarity_scores, rewards):
        self.similarity_scores = similarity_scores
        self.rewards = rewards
        self.step = 0
    
    def _on_step(self):
        self.model.update(similarity_score=self.similarity_scores[self.step], reward=self.rewards[self.step])
        self.step += 1
        return True  # Continue training

class IconLocator:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define the device
        self.model = IconLocatorCNN().to(self.device)  # Move the model to the device
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
    
    def locate(self, screenshot):
        screenshot_tensor = self.transform(screenshot).unsqueeze(0).to(self.device)
        predicted_coordinates = self.model(screenshot_tensor).squeeze(0).cpu().detach().numpy()
        return predicted_coordinates

    class IconLocatorTrainer:
        def __init__(self, model, state_dim, action_dim):
            self.model = model
            self.agent = PPO("MlpPolicy", self.model, verbose=1)
        
        def train(self, similarity_scores, rewards):
            # Train the agent using similarity scores and rewards
            self.agent.learn(total_timesteps=len(similarity_scores), callback=CustomCallback(similarity_scores, rewards))
       


class win_ui_int:



    def __init__(self, master):
        self.master = master
        master.title("WinUI-Int - Icon Selector")
        self.continuous_locating_enabled = False  # Add this variable and initialize it to False

        # Detect monitors and set primary and secondary
        monitors = get_monitors()
        self.primary_monitor = monitors[0]
        self.secondary_monitor = monitors[1] if len(monitors) > 1 else monitors[0]
       


        # Set GUI position to secondary monitor if available
        if len(monitors) > 1:
            # Center the GUI on the secondary monitor
            x_center = self.secondary_monitor.x + (self.secondary_monitor.width // 2) - 150
            y_center = self.secondary_monitor.y + (self.secondary_monitor.height // 2) - 200
            master.geometry(f"+{x_center}+{y_center}")


        
        # Capture screenshot of the primary monitor and save it
        #primary_monitor = self.monitors[0]
        #self.screenshot = ImageGrab.grab(bbox=(primary_monitor.x, primary_monitor.y, primary_monitor.width, primary_monitor.height))
        #self.screenshot_path = f"E:\\training_image\\Screenshot {datetime.now().strftime('%Y-%m-%d %H%M%S')}.jpg"
        #self.screenshot.save(self.screenshot_path)
         
        # Create a window to display the screenshot (no image yet, just setting up the window)
        #self.screenshot_window = Toplevel(self.master)
        #self.screenshot_window.overrideredirect(1)
        #self.screenshot_window.geometry(f"{primary_monitor.width}x{primary_monitor.height}+{primary_monitor.x}+{primary_monitor.y}")
        #self.screenshot_window.bind("<Button-1>", self.select_icon) # Bind the select_icon method to capture click events
        
        self.screenshot_window_button = Button(self.master, text="Create Screenshot Window", command=self.create_screenshot_window)
        self.screenshot_window_button.pack(pady=10)

        
        # Placeholder label for GUI
        self.coordinates_label = Label(master, text="Selected Coordinates: None")
        self.coordinates_label.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

        # Initialize the neural network model, loss criterion, and optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Define the device
        self.model = IconLocatorCNN().to(self.device)  # Move the model to the device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training exercise context
        self.training_context = "Find the Windows Start Menu"
        self.context_label = Label(master, text=f"Training Exercise: {self.training_context}")
        self.context_label.pack()

        # Button to capture and show screenshot on primary display
        self.capture_button = Button(master, text="Capture Screenshot", command=self.show_screenshot)
        self.capture_button.pack()
        
        # Feedback Label
        self.feedback_label = Label(master, text="")
        self.feedback_label.pack(pady=10)
        
        # Save Model Button
        self.save_button = Button(master, text="Save Model", command=self.save_model)
        self.save_button.pack(pady=10)

        # Load Model Button
        self.load_button = Button(master, text="Load Model", command=self.load_model)
        self.load_button.pack(pady=10)

        # Start Prediction Button
        self.predict_button = Button(master, text="Start Prediction", command=self.start_prediction)
        self.predict_button.pack(pady=10)

        # Exit Button
        self.exit_button = Button(master, text="Exit", command=master.quit)
        self.exit_button.pack(pady=10)

        # Flag for the first click
        self.first_click = True
        self.target_icon_position = None
        
        # Training button
        self.train_button = Button(master, text="Start Training", command=self.start_training)
        self.train_button.pack(pady=10)
        
        # Training with feedback button
        self.locate_with_feedback_button = Button(master, text="Locate Start Menu (Feedback)", command=self.locate_start_menu_with_feedback)
        self.locate_with_feedback_button.pack(pady=10)

        # Locate Start Menu Button
        self.locate_button = Button(master, text="Locate Start Menu", command=lambda: self.locate_start_menu(get_monitors()))
        self.locate_button.pack(pady=10)
        
        #Visualize the model data
        self.visualize_activations_button = Button(master, text="Visualize Activations", command=self.visualize_current_activations)
        self.visualize_activations_button.pack(pady=10)
        
        

        # Create a frame to display the analysis buffer
        self.analysis_frame = tk.Frame(master, width=200, height=200, bg="white")
        self.analysis_frame.pack()

        # Button to capture and label data
        self.capture_label_data_button = Button(master, text="Capture and Label Data", command=self.capture_and_label_data)
        self.capture_label_data_button.pack()
        
        

    def update_analysis_frame(self, analysis_buffer):
        # Convert analysis buffer to an image and display it in the frame
        analysis_image = Image.fromarray(analysis_buffer)
        analysis_image = ImageTk.PhotoImage(analysis_image)
        analysis_label = Label(self.analysis_frame, image=analysis_image)
        analysis_label.image = analysis_image  # Keep a reference to prevent garbage collection
        analysis_label.pack()

    def capture_and_label_data(self):
        # Capture the analysis buffer and ask the user for a label
        analysis_buffer = capture_analysis_buffer()
        label = ask_for_label_from_user()

        # Save the labeled data
        self.save_labeled_data(label, analysis_buffer)

        

    def capture_analysis_buffer(self):
        cursor_x, cursor_y = pyautogui.position()
        buffer_size = 200  # Size of the analysis buffer

        # Capture the surrounding pixels around the cursor
        left = max(cursor_x - buffer_size // 2, 0)
        right = min(cursor_x + buffer_size // 2, primary_monitor.width)
        top = max(cursor_y - buffer_size // 2, 0)
        bottom = min(cursor_y + buffer_size // 2, primary_monitor.height)

        # Capture the pixels using pyautogui
        screenshot = pyautogui.screenshot(region=(left, top, right - left, bottom - top))

        # Convert the screenshot to an analysis buffer
        analysis_buffer = np.array(screenshot)

        return analysis_buffer
    
    def ask_for_label_from_user(self):
        # Display a simple input dialog to get the label from the user
        label = simpledialog.askstring("Label", "Enter a label for the captured data:")
        return label
    
    def start_prediction(self):
        pass  # Placeholder for now

    def locate_start_menu(self, monitors):
        # 1. Capture the current screenshot of the primary monitor
        primary_monitor = monitors[0]
        screenshot = ImageGrab.grab(bbox=(primary_monitor.x, primary_monitor.y, primary_monitor.width, primary_monitor.height))
    
        # 2. Process the screenshot
        input_tensor = self.preprocess_image(screenshot).to(self.device)
    
        # 3. Predict the location using the trained model
        with torch.no_grad():
            predicted_offsets = self.model(input_tensor).cpu().numpy().squeeze()
    
        # Calculate the predicted icon position based on the center of the primary monitor
        center_x = primary_monitor.width // 2
        center_y = primary_monitor.height // 2
        predicted_x = center_x + predicted_offsets[0]
        predicted_y = center_y + predicted_offsets[1]

        # Ensure the predicted position is within the confines of the primary monitor
        predicted_x = max(min(predicted_x, primary_monitor.width), 0)
        predicted_y = max(min(predicted_y, primary_monitor.height), 0)

        # 4. Snap the mouse to a random location on the primary monitor
        random_x = random.randint(0, primary_monitor.width)
        random_y = random.randint(0, primary_monitor.height)
        pyautogui.moveTo(random_x, random_y, duration=0.5)

        # Pause for a moment to simulate a "thinking" delay
        time.sleep(0.5)

        # 5. Then move to the predicted location of the start menu, slowly
        pyautogui.moveTo(predicted_x, predicted_y, duration=5)
        print("Locating start menu...")

    def create_screenshot_window(self):
        primary_monitor = self.monitors[0]
        self.screenshot_window = Toplevel(self.master)
        self.screenshot_window.overrideredirect(1)
        self.screenshot_window.geometry(f"{primary_monitor.width}x{primary_monitor.height}+{primary_monitor.x}+{primary_monitor.y}")
        self.screenshot_window.bind("<Button-1>", self.select_icon)

        
    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.Resize((136, 136)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor_image = preprocess(image).unsqueeze(0)
        return tensor_image

    def save_model(self):
        torch.save(self.model.state_dict(), "model.pth")
        with open("model_context.json", 'w') as f:
            json.dump({"context": self.training_context}, f)

    def load_model(self):
        self.model.load_state_dict(torch.load("model.pth"))
        with open("model_context.json", 'r') as f:
            data = json.load(f)
            self.training_context = data["context"]
            self.context_label['text'] = f"Training Exercise: {self.training_context}"

    def start_prediction(self):
        # Logic to use the trained model to make predictions
        pass

    def show_screenshot(self):
        # Ensure the screenshot_window exists before attempting to show the screenshot
        if not hasattr(self, 'screenshot_window'):
            self.create_screenshot_window()

        # Capture screenshot of the primary monitor and save it
        primary_monitor = self.monitors[0]
        self.screenshot = ImageGrab.grab(bbox=(primary_monitor.x, primary_monitor.y, primary_monitor.width, primary_monitor.height))
        self.screenshot_path = f"E:\\training_image\\Screenshot {datetime.now().strftime('%Y-%m-%d %H%M%S')}.jpg"
        self.screenshot.save(self.screenshot_path)
        self.screenshot_label_image = ImageTk.PhotoImage(self.screenshot)
        label = Label(self.screenshot_window, image=self.screenshot_label_image)
        label.pack()
        
        # Show the screenshot window
        self.screenshot_window.lift()
        self.master.wait_window(self.screenshot_window) 
        

        # Button to capture and label data
        self.capture_label_button = Button(master, text="Capture & Label Data", command=self.capture_labeled_data)
        self.capture_label_button.pack(pady=10)
        # List to store labeled training data
        self.labeled_training_data = []

        # ... (rest of the initialization code)

    def select_icon(self, event):
        self.coordinates_label['text'] = f"Selected Coordinates: {event.x}, {event.y}"
        if self.first_click:
            self.first_click = False
            self.target_icon_position = (event.x, event.y)
            self.feedback_label.config(text="Click on the icon to train the model")
        else:
            self.screenshot_window.destroy()
            self.feedback_label.config(text="Ready to train")


    def save_labeled_data(self, label, data):
        # Save the labeled data and label to the list
        self.labeled_training_data.append((label, data))


    def start_training(self):
        screenshot_directory = os.path.dirname(self.screenshot_path)  # Extract directory from the full path
        trainer = Trainer(self.screenshot, self.target_icon_position, self.model, self.criterion, self.optimizer, self.monitors, self.feedback_label, self.device, screenshot_directory, self.labeled_training_data)
        trainer.start_training_loop()
        def locate_start_menu(self):
            # 1. Capture the current screenshot
            primary_monitor = self.monitors[0]
            screenshot = ImageGrab.grab(bbox=(primary_monitor.x, primary_monitor.y, primary_monitor.width, primary_monitor.height))
        
            # 2. Process the screenshot
            input_tensor = self.preprocess_image(screenshot).to(self.device)
        
            # 3. Predict the location using the trained model
            with torch.no_grad():
                predicted_offsets = self.model(input_tensor).cpu().numpy().squeeze()
        
            # Calculate the predicted icon position
            predicted_x = primary_monitor.width // 2 + predicted_offsets[0]
            predicted_y = primary_monitor.height // 2 + predicted_offsets[1]

            # 4. Move the mouse cursor to the predicted location
            pyautogui.moveTo(predicted_x, predicted_y)
            print("Locating start menu...")
           
            
    def visualize_current_activations(self):
        # 1. Take a screenshot of the current screen
        primary_monitor = self.monitors[0]
        screenshot = ImageGrab.grab(bbox=(primary_monitor.x, primary_monitor.y, primary_monitor.width, primary_monitor.height))
    
        # 2. Preprocess the screenshot
        screenshot_tensor = self.preprocess_image(screenshot).to(self.device)
    
        # 3. Pass the screenshot through the model to get the activations
        outputs, activations = self.model(screenshot_tensor, return_intermediate=True)
    
        # 4. Visualize the activations
        self.model.visualize_activations(activations)  # This is the change made.
        
    def locate_start_menu_with_feedback(self):
        while True:
            # 1. Move the mouse to the predicted location of the Start menu
            self.locate_start_menu()

            # 2. Capture a screenshot
            primary_monitor = self.monitors[0]
            screenshot = ImageGrab.grab(bbox=(primary_monitor.x, primary_monitor.y, primary_monitor.width, primary_monitor.height))

            # 3. Check if the Start menu image is present
            is_present = is_start_menu_present(screenshot)

            # 4. Assign cookies (rewards) based on success
            if is_present:
                # Give a positive reward
                print("Found Start menu! + cookie")
            else:
                # Give a negative reward
                print("Start menu not found. - cookie")

            # Add some delay before the next attempt (if not in continuous locating mode)
        if not self.continuous_locating_enabled:
            time.sleep(2)  # Adjust as needed
            
    def toggle_continuous_locating(self):
        self.continuous_locating_enabled = not self.continuous_locating_enabled
        if self.continuous_locating_enabled:
            self.locate_continuous_button.config(text="Continuous Locate: ON")
            self.start_continuous_locating()  # Start the continuous locating loop
        else:
            self.locate_continuous_button.config(text="Continuous Locate: OFF")
    
        self.locate_continuous_button = Button(master, text="Continuous Locate: OFF", command=self.toggle_continuous_locating)
        self.locate_continuous_button.pack(pady=10)
        
    def start_continuous_locating(self):
        while self.continuous_locating_enabled:
            self.locate_start_menu_with_feedback()
            time.sleep(2)  # Add a delay between attempts
       

    def locate_start_menu_with_feedback(self):
        # 1. Capture the current screenshot
        primary_monitor = self.monitors[0]
        screenshot = ImageGrab.grab(bbox=(primary_monitor.x, primary_monitor.y, primary_monitor.width, primary_monitor.height))
        
        # 2. Process the screenshot
        input_tensor = self.preprocess_image(screenshot).to(self.device)
        
        # 3. Predict the location using the trained model
        with torch.no_grad():
            predicted_offsets = self.model(input_tensor).cpu().numpy().squeeze()
        
        # Calculate the predicted icon position
        predicted_x = primary_monitor.width // 2 + predicted_offsets[0]
        predicted_y = primary_monitor.height // 2 + predicted_offsets[1]

        # 4. Move the mouse cursor to the predicted location
        pyautogui.moveTo(predicted_x, predicted_y)

        # 5. Capture surrounding pixels and display the sample window
        cursor_position = (predicted_x, predicted_y)
        captured_data = self.capture_surrounding_pixels(screenshot, cursor_position)
        self.display_sample_window(captured_data)



    def compare_images(image1, image2):
        # Calculate structural similarity index
        diff = ImageChops.difference(image1, image2)
        mse = np.mean(np.array(diff) ** 2)
        return 1 / mse  # Higher value indicates higher similarity

    def open_sample_window(self, captured_region, similarity_score):
        # Create a new window (Tkinter Toplevel) to display sample data
        sample_window = Toplevel(self.master)

        # Display captured region and Start menu image in the sample window
        captured_photo = ImageTk.PhotoImage(captured_region)
        start_menu_photo = ImageTk.PhotoImage(self.start_menu_image)
        Label(sample_window, image=captured_photo).pack()
        Label(sample_window, image=start_menu_photo).pack()

        # Buttons for user interaction
        success_button = Button(sample_window, text="Success", command=lambda: self.user_feedback(sample_window, "success", similarity_score))
        failure_button = Button(sample_window, text="Failure", command=lambda: self.user_feedback(sample_window, "failure", similarity_score))
        success_button.pack()
        failure_button.pack()

    def user_feedback(self, sample_window, feedback, similarity_score):
        # Close the sample window
        sample_window.destroy()

        # Collect feedback and captured data for training
        if feedback == "success":
            reward = 1.0
        elif feedback == "failure":
            reward = -1.0
        else:
            reward = 0.0
        
        # Use similarity score and reward as inputs for model training
        self.train_model_with_feedback(similarity_score, reward)

        # Collect feedback and captured data for training
        # Update your training data collection here

    def get_model_prediction(self, captured_region):
        # Preprocess captured region and get model prediction
        captured_tensor = self.preprocess_image(captured_region).to(self.device)
        with torch.no_grad():
            predicted_offset = self.model(captured_tensor).cpu().numpy().squeeze()
        return compare_images(captured_tensor, self.start_menu_image)

    def train_model_with_feedback(self, similarity_score, reward):
        self.similarity_scores.append(similarity_score)
        self.rewards.append(reward)
    
    def start_reinforcement_training(self):
        reinforcement_trainer = ReinforcementTrainer(self.model, state_dim, action_dim)
        reinforcement_trainer.train(self.similarity_scores, self.rewards)

    def capture_surrounding_pixels(self, screenshot, cursor_position, window_size=30):
        cursor_x, cursor_y = cursor_position
        left = max(cursor_x - window_size, 0)
        right = min(cursor_x + window_size, screenshot.width)
        top = max(cursor_y - window_size, 0)
        bottom = min(cursor_y + window_size, screenshot.height)
        
        captured_data = screenshot.crop((left, top, right, bottom))
        return captured_data


    def display_sample_window(self, captured_data):
        sample_window = Toplevel(self.master)
        sample_window.title("Sample Window")

        # Display the captured data
        captured_data_image = ImageTk.PhotoImage(captured_data)
        label = Label(sample_window, image=captured_data_image)
        label.pack()

        # Create buttons for user feedback
        success_button = Button(sample_window, text="Success", command=lambda: self.user_feedback(sample_window, "success", similarity_score))
        failure_button = Button(sample_window, text="Failure", command=lambda: self.user_feedback(sample_window, "failure", similarity_score))
        neutral_button = Button(sample_window, text="Neutral", command=lambda: self.user_feedback(sample_window, "neutral", similarity_score))
        
        success_button.pack(side="left", padx=10)
        failure_button.pack(side="left", padx=10)
        neutral_button.pack(side="left", padx=10)

        # Calculate similarity score using your chosen similarity metric
        similarity_score = calculate_similarity(captured_data, target_start_menu_image)  # Replace with your similarity calculation function

    def capture_labeled_data(self):
        # Capture the current screenshot
        primary_monitor = self.monitors[0]
        screenshot = ImageGrab.grab(bbox=(primary_monitor.x, primary_monitor.y, primary_monitor.width, primary_monitor.height))

        # Get the current cursor position
        cursor_x, cursor_y = pyautogui.position()

        # Capture surrounding pixels
        captured_data = self.capture_surrounding_pixels(screenshot, (cursor_x, cursor_y))

        # Display the sample window for labeling
        sample_window = Toplevel(self.master)
        sample_window.title("Label Sample Window")

        # Display the captured data
        captured_data_image = ImageTk.PhotoImage(captured_data)
        label = Label(sample_window, image=captured_data_image)
        label.pack()

        # Entry field for labeling
        label_entry = Entry(sample_window)
        label_entry.pack()

        # Button to save the labeled data
        save_button = Button(sample_window, text="Save", command=lambda: self.save_labeled_data(label_entry.get(), captured_data))
        save_button.pack()

    def save_labeled_data(self, label, data):
        # Save the labeled data and label to a file or database
        # You can implement your own saving mechanism here
        print("Label:", label)
        print("Data:", data)


class Trainer:
    def __init__(self, screenshot, icon_coordinates, model, criterion, optimizer, monitors, feedback_label, device, screenshot_directory, labeled_training_data):
        self.screenshot = screenshot
        self.icon_coordinates = icon_coordinates
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.monitors = monitors
        self.feedback_label = feedback_label
        self.device = device  # Set the device attribute here
        self.screenshot_directory = screenshot_directory

    def get_training_data(self):
        # Collect training data from screenshots, cursor positions, and labels
        training_data = []

        
        # Add labeled training data to the training dataset
        for label, data in self.labeled_training_data:
            training_data.append((data, self.icon_coordinates, label))

        return training_data

    def preprocess_image(self, image):
        preprocess = transforms.Compose([
            transforms.Resize((136, 136)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor_image = preprocess(image).unsqueeze(0)
        return tensor_image


    def train_model(self, batch_screenshots, batch_cursor_positions, batch_actual_offsets):
        # Process each screenshot and stack them to form a batch tensor
        input_tensors = [self.preprocess_image(screenshot) for screenshot in batch_screenshots]
        input_tensor = torch.stack(input_tensors).to(self.device)
        
        # Prepare labeled training data
        labeled_data = [(data, self.icon_coordinates, label) for label, data in self.labeled_training_data]

        # Combine regular training data and labeled training data
        combined_data = training_data + labeled_data
        for data, cursor_position, _ in combined_data:
            # Process each training data point and perform training
            input_tensors = [self.preprocess_image(data) for _ in range(batch_size)]
            input_tensor = torch.stack(input_tensors).to(self.device)
            


        # Calculate target offsets for each data point in the batch and stack them
        target_offsets = [[actual_offsets[0] - cursor_position[0], actual_offsets[1] - cursor_position[1]] 
                          for cursor_position, actual_offsets in zip(batch_cursor_positions, batch_actual_offsets)]
        target_tensor = torch.tensor(target_offsets).float().to(self.device)
    
        # Forward pass
        predicted_offsets = self.model(input_tensor)
    
        # Compute loss
        loss = self.criterion(predicted_offsets, target_tensor)
    
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
        self.feedback_label.config(text=f"Loss: {loss.item():.4f}")




    def start_training_loop(self):
        num_epochs = 100
        batch_size = 32
        screenshot_directory = "E:\\training_image"
        all_screenshots = load_screenshots(self.screenshot_directory)

        for epoch in range(num_epochs):
            # Reshuffle the screenshots at the start of each epoch
            random.shuffle(all_screenshots)
        
            for _ in range(len(all_screenshots) // batch_size):  # Number of batches
                # Randomly select screenshots for the current batch
                batch_screenshots = random.sample(all_screenshots, batch_size)

                # Generate random cursor positions for the current batch
                batch_cursor_positions = [(random.randint(0, self.monitors[0].width), 
                                           random.randint(0, self.monitors[0].height)) for _ in range(batch_size)]

                # The target remains the same for all images in the batch
                batch_actual_offsets = [self.icon_coordinates for _ in range(batch_size)]

                # Train the model on the current batch
                self.train_model(batch_screenshots, batch_cursor_positions, batch_actual_offsets)

            print(f"Training Epoch: {epoch + 1}/{num_epochs}")

        print("Training Complete!")


root = Tk()
app = win_ui_int(root)
root.mainloop()
