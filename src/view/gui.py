"""
ModelMaster - View Component

This module implements the GUI for the ModelMaster application using customtkinter.
It handles user interactions and displays information using the controller to manage the logic.
"""

import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import messagebox
import customtkinter as ctk
import os
from src.controller.model_controller import ModelController


class ModelMasterGUI:
    """Main GUI class for the ModelMaster application."""
    
    def __init__(self):
        """Initialize the GUI application."""
        # Configure app theme and appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.app = ctk.CTk()
        self.app.geometry("400x500")
        self.app.resizable(False, False)
        self.app.title("ModelMaster")
        
        # Initialize controller
        self.controller = ModelController()
        
        # Show main screen
        self.show_main_screen()
    
    def run(self):
        """Run the application main loop."""
        self.app.mainloop()
    
    #------------------------------------------{Common UI Components}------------------------------------------#

    def create_frame(self):
        """Creates and places a centered frame for controls."""
        frame = ctk.CTkFrame(master=self.app, width=300, height=400, corner_radius=10)
        frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    def create_label(self, name, position, size=20):
        """Creates a text label at the specified position.
        
        Args:
            name: Label text content
            position: [x, y] relative position
            size: Font size in pixels
        """
        label = ctk.CTkLabel(master=self.app, text=name, text_color="#FFFFFF", 
                              font=("Arial", size), width=200, height=25)
        label.place(relx=position[0], rely=position[1], anchor=tk.CENTER)

    def create_button(self, name, function, position, width=200):
        """Creates a button with the specified callback at the given position.
        
        Args:
            name: Button text
            function: Callback function
            position: [x, y] relative position
            width: Button width in pixels
        """
        btn = ctk.CTkButton(master=self.app, text=name, width=width, command=function)
        btn.place(relx=position[0], rely=position[1], anchor=tk.CENTER) 

    def create_back_button(self, function):
        """Creates a standard back button with the provided callback."""
        btn = ctk.CTkButton(master=self.app, text="Back", width=80, command=function)
        btn.place(relx=0.25, rely=0.85, anchor=tk.CENTER)

    def create_dropdown_menu(self, values, callback, pos, default_text="Select an option"): 
        """Creates a dropdown menu with the given options.
        
        Args:
            values: List of option values
            callback: Function to call when option selected
            pos: [x, y] relative position
            default_text: Initial placeholder text
        """
        combobox = ctk.CTkOptionMenu(master=self.app, values=values, width=200, command=callback) 
        combobox.place(relx=pos[0], rely=pos[1], anchor=tk.CENTER) 
        combobox.set(default_text)
    
    #------------------------------------------{Utility Functions}------------------------------------------#

    def is_valid_split_value(self, input):
        """Validates that a test-train split value is between 0 and 1."""
        try:
            return 0 < float(input) < 1
        except ValueError:
            return False

    def clear_screen(self):
        """Removes all widgets from the current screen."""
        for widget in self.app.winfo_children():
            widget.destroy()

    def load_dataset(self):
        """Opens file dialog to load a CSV dataset."""
        csv_path = askopenfilename(filetypes=[("CSV files", "*.csv")])
        if csv_path:  # Check if a file was selected
            result = self.controller.load_dataset(csv_path)
            if result["success"]:
                messagebox.showinfo("Success", result["message"])
            else:
                messagebox.showerror("Error", result["message"])

    def show_app_info(self):
        """Displays information about the application."""
        app_info = ("Model Master - Advanced Learning Laboratory for Intelligent Networks\n\n"
                   "This application is a comprehensive tool for machine learning model development and evaluation, "
                   "featuring multiple classification algorithms including SVM, KNN, Decision Trees, and Logistic Regression.\n\n"
                   "Key Features:\n"
                   "• Intuitive dataset loading and preprocessing\n"
                   "• Customizable model parameters and configurations\n"
                   "• Detailed performance metrics and visualization tools\n"
                   "• Model persistence for saving and loading trained models\n\n"
                   "Developed as part of research at [Institution] by the ALL IN team.\n"
                   "Version 1.0.0")
        messagebox.showinfo("Model Master Info", app_info)

    def save_model(self):
        """Opens file dialog to save the current model as a pickle file."""
        file_path = asksaveasfilename(filetypes=[("Pickle file", ".pkl")], defaultextension=".pkl")
        if file_path:  # Check if a path was selected
            result = self.controller.save_model(file_path)
            if result["success"]:
                messagebox.showinfo("Success", result["message"])
            else:
                messagebox.showerror("Error", result["message"])
     
    #------------------------------------------{Results Display}---------------------------------------------#

    def show_model_results(self, algorithm):
        """Displays the evaluation metrics of the tested model.
        
        Args:
            algorithm: String identifier of the algorithm
        """
        metrics = self.controller.get_metrics()
        if not metrics or metrics["accuracy"] is None:
            messagebox.showerror("Test Error", "Please test the model first!")
            return
        
        self.clear_screen()
        self.create_frame()
        self.create_label("Results", [.5, .18]) 
        self.create_label("Accuracy", [.5, .27]) 
        self.create_label(f"{metrics['accuracy']:.4f}", [.5, .32]) 
        self.create_label("Precision", [.5, .42]) 
        self.create_label(f"{metrics['precision']:.4f}", [.5, .47]) 
        self.create_label("Recall", [.5, .57]) 
        self.create_label(f"{metrics['recall']:.4f}", [.5, .62]) 
        self.create_label("F1 Score", [.5, .72])
        self.create_label(f"{metrics['f1_score']:.4f}", [.5, .77])  
        self.create_back_button(lambda: self.show_train_test_screen(algorithm))

        def display_confusion_matrix():
            """Displays the confusion matrix visualization."""
            self.controller.display_confusion_matrix()
              
        def return_to_home():
            """Resets the app state and returns to the main screen."""
            self.controller = ModelController()  # Reset controller
            self.show_main_screen()
        
        self.create_button("Home", return_to_home, [.75, .85], 80)   
        self.create_button("Confusion Matrix", display_confusion_matrix, [.50, .85], 80) 
        self.create_button("Save Model", self.save_model, [.5, .95], 80)         

    #------------------------------------------{Training Screens}-------------------------------------------#

    def show_train_test_screen(self, algorithm_code):
        """Shows the train-test screen for a specific algorithm.
        
        Args:
            algorithm_code: String identifier of the selected algorithm
        """
        self.clear_screen()
        self.create_frame()
        self.create_label("Train & Test", [.5, .2])
        self.create_label("Set train-test split ratio", [.5, .3])
        
        slider = ctk.CTkSlider(self.app, from_=0, to=1, orientation="horizontal", number_of_steps=10)
        slider.place(relx=.5, rely=.4, anchor=tk.CENTER) 
        slider.set(0.2)  # Default to commonly used test size
        
        value_label = ctk.CTkLabel(self.app, text=f"Ratio: {slider.get():.2f}", font=("Arial", 10))
        value_label.place(relx=.5, rely=.45, anchor=tk.CENTER) 
        
        def update_label(val):
            """Updates the value label when slider is moved."""
            value_label.configure(text=f"Ratio: {val:.2f}")
        
        slider.configure(command=lambda val: update_label(val))
        
        def train_model():
            """Trains the selected algorithm with the current configuration."""
            split_ratio = slider.get()
            if not self.is_valid_split_value(split_ratio):
                messagebox.showerror("Test Size Error", "Please enter a value between 0 and 1!")
                return
            
            kwargs = {}
            if algorithm_code == "svm" and "kernel" in globals():
                kwargs["kernel"] = kernel
            elif algorithm_code == "knn" and "n_neighbors" in globals():
                kwargs["n_neighbors"] = n_neighbors 
            elif algorithm_code == "tree" and "max_depth" in globals():
                kwargs["max_depth"] = max_depth
            
            result = self.controller.train_model(algorithm_code, split_ratio, **kwargs)
            if result["success"]:
                messagebox.showinfo("Success", result["message"])
            else:
                messagebox.showerror("Error", result["message"])
                     
        def test_model():
            """Tests the trained model."""
            result = self.controller.test_model()
            if result["success"]:
                messagebox.showinfo("Success", result["message"])
            else:
                messagebox.showerror("Error", result["message"])
        
        self.create_button("Train", train_model, [.5, .5])
        self.create_button("Test", test_model, [.5, .6])
        self.create_button("Show Results", lambda: self.show_model_results(algorithm_code), [.5, .7])

        def go_back():
            """Navigates back to the appropriate configuration screen."""
            if algorithm_code == "svm":
                self.show_svm_config()  
            elif algorithm_code == "knn":
                self.show_knn_config()  
            elif algorithm_code == "logistic_reg" or algorithm_code == "linear_reg":
                self.show_algorithm_selection()   
            elif algorithm_code == "tree":
                self.show_decision_tree_config()            
        
        self.create_back_button(go_back)            

    def show_svm_config(self):
        """Shows the configuration screen for SVM algorithm."""
        self.clear_screen()
        self.create_frame()
        self.create_label("SVM Configuration", [.5, .3]) 

        def set_kernel_type(choice): 
            """Sets the selected kernel type."""
            global kernel
            kernel = choice

        def submit():
            """Validates configuration and proceeds to train-test screen."""
            if "kernel" not in globals():
                messagebox.showerror("Kernel Error", "Please select a kernel type!")
                return
            self.controller.select_algorithm("svm")
            self.show_train_test_screen("svm")

        self.create_dropdown_menu(["linear", "rbf", "poly", "sigmoid"], set_kernel_type, [.5, .4], "Select kernel") 
        self.create_button("Submit", submit, [.5, .7]) 
        self.create_back_button(self.show_algorithm_selection)

    def show_decision_tree_config(self):
        """Shows the configuration screen for Decision Tree algorithm."""
        self.clear_screen()
        self.create_frame()
        self.create_label("Decision Tree Configuration", [.5, .3]) 
        self.create_label("Enter max depth", [.5, .4]) 
     
        entry = ctk.CTkEntry(master=self.app, width=200, height=25, corner_radius=10)
        entry.place(relx=.5, rely=.5, anchor=tk.CENTER) 
        entry.insert(0, "5")  # Default common value
     
        def submit(): 
            """Validates configuration and proceeds to train-test screen."""
            input_value = entry.get()
            if input_value == "": 
                messagebox.showerror("Input Error", "Please enter a max depth value!") 
                return
            
            if not input_value.isdigit(): 
                messagebox.showerror("Input Error", "Please enter a valid number!") 
                return
                
            global max_depth
            max_depth = int(input_value)
            self.controller.select_algorithm("tree")
            self.show_train_test_screen("tree")
     
        self.create_button("Submit", submit, [.5, .7]) 
        self.create_back_button(self.show_algorithm_selection) 

    def show_knn_config(self):
        """Shows the configuration screen for KNN algorithm."""
        self.clear_screen()
        self.create_frame()
        self.create_label("KNN Configuration", [.5, .3]) 
        self.create_label("Enter number of neighbors", [.5, .4]) 
     
        entry = ctk.CTkEntry(master=self.app, width=200, height=25, corner_radius=10) 
        entry.place(relx=.5, rely=.5, anchor=tk.CENTER)
        entry.insert(0, "5")  # Default common value 
     
        def submit(): 
            """Validates configuration and proceeds to train-test screen."""
            input_value = entry.get() 
            if input_value == "": 
                messagebox.showerror("Input Error", "Please enter the number of neighbors!") 
                return
                
            if not input_value.isdigit(): 
                messagebox.showerror("Input Error", "Please enter a valid number!") 
                return

            global n_neighbors
            n_neighbors = int(input_value)
            self.controller.select_algorithm("knn")
            self.show_train_test_screen("knn")
     
        self.create_button("Submit", submit, [.5, .7]) 
        self.create_back_button(self.show_algorithm_selection) 

    def show_algorithm_selection(self):
        """Shows the algorithm selection screen."""
        self.clear_screen()
        self.create_frame()
        self.create_label("Select Your Algorithm", [.5, .2])
        self.create_button("Decision Tree", self.show_decision_tree_config, [.5, .3])
        
        def show_linear_reg():
            self.controller.select_algorithm("linear_reg")
            self.show_train_test_screen("linear_reg")
            
        def show_logistic_reg():
            self.controller.select_algorithm("logistic_reg")
            self.show_train_test_screen("logistic_reg")
            
        self.create_button("Linear Regression", show_linear_reg, [.5, .4])
        self.create_button("Logistic Regression", show_logistic_reg, [.5, .5])
        self.create_button("KNN", self.show_knn_config, [.5, .6])
        self.create_button("SVM", self.show_svm_config, [.5, .7])
        self.create_back_button(self.show_dataset_screen)
        
    def show_dataset_screen(self):
        """Shows the dataset loading and feature selection screen."""
        self.clear_screen()
        self.create_frame()
        self.create_label("Load Your Dataset", [.5, .37])
        self.create_label("Enter number of features", [.5, .45], 15)
        self.create_button("Load Dataset", self.load_dataset, [0.25, 0.75], width=80)
        
        entry = ctk.CTkEntry(master=self.app, width=200, height=25, corner_radius=10)
        entry.place(relx=.5, rely=.5, anchor=tk.CENTER)

        def submit(): 
            """Validates the dataset and feature count before proceeding."""
            n_features = entry.get()
            
            if not hasattr(self.controller.current_model, 'dataset') or self.controller.current_model.dataset is None: 
                messagebox.showerror("Dataset Error", "Please load your dataset first!") 
                return
                
            if n_features == "": 
                messagebox.showerror("Input Error", "Please enter the number of features!") 
                return
        
            if not n_features.isdigit(): 
                messagebox.showerror("Input Error", "Please enter a valid number!") 
                return
                
            # Check if number of features is valid based on dataset
            max_features = self.controller.current_model.dataset.shape[1] - 1  # All columns except target
            if int(n_features) > max_features:
                messagebox.showerror("Input Error", 
                                    f"Number of features cannot exceed the available features ({max_features})!")
                return
                
            self.controller.set_feature_count(int(n_features))
            self.show_algorithm_selection()
            
        self.create_button("Create Model", submit, [.75, .75], 80)
        self.create_button("Load Model", self.show_model_loader, [.75, .85], 80)
        self.create_back_button(self.show_main_screen)
     
    def show_model_loader(self):
        """Shows the screen for loading a previously saved model."""
        self.clear_screen()
        self.create_frame()
        self.create_label("Model Information", [.5, .18]) 

        def load_model_file():
            """Loads a model file and displays its information."""
            model_path = askopenfilename(filetypes=[("Pickle file", ".pkl")])
            if not model_path:  # Check if a file was selected
                return
             
            result = self.controller.load_model(model_path)
            if result["success"]:
                model_info = result["model_info"]
                
                # Display model details
                self.create_label("Algorithm Type", [.5, .27])
                algorithm_label = ctk.CTkLabel(self.app, 
                                             text=f"< {model_info['class_name']} >", 
                                             font=("Arial", 20))
                algorithm_label.place(relx=.5, rely=.33, anchor=tk.CENTER) 
                
                # Show specific parameters based on model type
                if 'n_neighbors' in model_info["parameters"]:
                    param_label = ctk.CTkLabel(self.app, 
                                             text=f"Number of Neighbors: {model_info['parameters']['n_neighbors']}", 
                                             font=("Arial", 20))
                    param_label.place(relx=.5, rely=.42, anchor=tk.CENTER) 
                    
                if 'max_depth' in model_info["parameters"]:
                    param_label = ctk.CTkLabel(self.app, 
                                             text=f"Max Depth: {model_info['parameters']['max_depth']}", 
                                             font=("Arial", 20))
                    param_label.place(relx=.5, rely=.42, anchor=tk.CENTER)
                    
                messagebox.showinfo("Success", result["message"])
            else:
                messagebox.showerror("Error", result["message"])

        self.create_button("Load", load_model_file, [.5, .7])
        self.create_back_button(self.show_dataset_screen)

    def show_main_screen(self):
        """Shows the main welcome screen."""
        self.clear_screen()
        self.create_frame()
        self.create_label("Welcome to Model Master", [.5, .3])
        self.create_button("Start", self.show_dataset_screen, [.5, .5])
        self.create_button("About", self.show_app_info, [0.25, 0.85], 80)
