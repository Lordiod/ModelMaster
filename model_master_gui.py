"""
ModelMaster - Machine Learning Model Explorer GUI

A graphical application for data scientists and machine learning enthusiasts to easily
load datasets, select features, train classification algorithms, and evaluate model performance.

This application provides an intuitive interface for comparing different classification models
including SVM, KNN, Decision Trees, and Regression algorithms.
"""

import tkinter 
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import messagebox
import pickle
import customtkinter 
import algorithms
import pandas as pd 
import os

# Configure app theme and appearance
customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

app = customtkinter.CTk() 
app.geometry("400x500") 
app.resizable(False, False)
app.title("ModelMaster")

#------------------------------------------{Common UI Components}------------------------------------------#

def create_frame():
    """Creates and places a centered frame for controls."""
    frame = customtkinter.CTkFrame(master=app, width=300, height=400, corner_radius=10)
    frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

def create_label(name, position, size=20):
    """Creates a text label at the specified position.
    
    Args:
        name: Label text content
        position: [x, y] relative position
        size: Font size in pixels
    """
    label = customtkinter.CTkLabel(master=app, text=name, text_color="#FFFFFF", 
                                  font=("Arial", size), width=200, height=25)
    label.place(relx=position[0], rely=position[1], anchor=tkinter.CENTER)

def create_button(name, function, position, width=200):
    """Creates a button with the specified callback at the given position.
    
    Args:
        name: Button text
        function: Callback function
        position: [x, y] relative position
        width: Button width in pixels
    """
    btn = customtkinter.CTkButton(master=app, text=name, width=width, command=function)
    btn.place(relx=position[0], rely=position[1], anchor=tkinter.CENTER) 

def create_back_button(function):
    """Creates a standard back button with the provided callback."""
    btn = customtkinter.CTkButton(master=app, text="Back", width=80, command=function)
    btn.place(relx=0.25, rely=0.85, anchor=tkinter.CENTER)

def create_dropdown_menu(values, callback, pos, default_text="Select an option"): 
    """Creates a dropdown menu with the given options.
    
    Args:
        values: List of option values
        callback: Function to call when option selected
        pos: [x, y] relative position
        default_text: Initial placeholder text
    """
    global choice 
    combobox = customtkinter.CTkOptionMenu(master=app, values=values, width=200, command=callback) 
    combobox.place(relx=pos[0], rely=pos[1], anchor=tkinter.CENTER) 
    combobox.set(default_text)  

#------------------------------------------{Utility Functions}------------------------------------------#

def is_valid_split_value(input):
    """Validates that a test-train split value is between 0 and 1."""
    try:
        return 0 < float(input) < 1
    except ValueError:
        return False

def clear_screen():
    """Removes all widgets from the current screen."""
    for widget in app.winfo_children():
        widget.destroy()

def load_dataset():
    """Opens file dialog to load a CSV dataset."""
    csv_path = askopenfilename(filetypes=[("CSV files", "*.csv")])
    if csv_path:  # Check if a file was selected
        try:
            algorithms.dataset = pd.read_csv(csv_path)
            # Display filename to user
            filename = os.path.basename(csv_path)
            messagebox.showinfo("Success", f"Dataset '{filename}' loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {str(e)}")

def show_app_info():
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

def save_model():
    """Opens file dialog to save the current model as a pickle file."""
    file_path = asksaveasfilename(filetypes=[("Pickle file", ".pkl")], defaultextension=".pkl")
    if file_path:  # Check if a path was selected
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(algorithms.model, file)
            # Display success message with filename
            filename = os.path.basename(file_path)
            messagebox.showinfo("Success", f"Model saved as '{filename}'")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
 
#------------------------------------------{Results Display}---------------------------------------------#

def show_model_results(algorithm):
    """Displays the evaluation metrics of the tested model.
    
    Args:
        algorithm: String identifier of the algorithm
    """
    if not hasattr(algorithms, 'accuracy'):
        messagebox.showerror("Test Error", "Please test the model first!")
        return
    
    clear_screen()
    create_frame()
    create_label("Results", [.5, .18]) 
    create_label("Accuracy", [.5, .27]) 
    create_label(f"{algorithms.accuracy:.4f}", [.5, .32]) 
    create_label("Precision", [.5, .42]) 
    create_label(f"{algorithms.precision:.4f}", [.5, .47]) 
    create_label("Recall", [.5, .57]) 
    create_label(f"{algorithms.recall:.4f}", [.5, .62]) 
    create_label("F1 Score", [.5, .72])
    create_label(f"{algorithms.f1_score:.4f}", [.5, .77])  
    create_back_button(lambda: show_train_test_screen(algorithm))

    def display_confusion_matrix():
        """Displays the confusion matrix visualization."""
        algorithms.display_confusion_matrix(algorithms.conf_matrix)
          
    def return_to_home():
        """Resets the app state and returns to the main screen."""
        try:
            algorithms.dataset = None
            del algorithms.model
            del algorithms.accuracy
            del algorithms.precision
            del algorithms.recall
            del algorithms.f1_score
            del algorithms.conf_matrix
        except AttributeError:
            pass  # Some attributes might not exist yet
        show_main_screen()
    
    create_button("Home", return_to_home, [.75, .85], 80)   
    create_button("Confusion Matrix", display_confusion_matrix, [.50, .85], 80) 
    create_button("Save Model", save_model, [.5, .95], 80)         

#------------------------------------------{Training Screens}-------------------------------------------#

def show_train_test_screen(algorithm_code):
    """Shows the train-test screen for a specific algorithm.
    
    Args:
        algorithm_code: String identifier of the selected algorithm
    """
    clear_screen()
    create_frame()
    create_label("Train & Test", [.5, .2])
    create_label("Set train-test split ratio", [.5, .3])
    
    slider = customtkinter.CTkSlider(app, from_=0, to=1, orientation="horizontal", number_of_steps=10)
    slider.place(relx=.5, rely=.4, anchor=tkinter.CENTER) 
    slider.set(0.0)  # Default to commonly used test size
    
    value_label = customtkinter.CTkLabel(app, text=f"Ratio: {slider.get()}", font=("Arial", 10))
    value_label.place(relx=.5, rely=.45, anchor=tkinter.CENTER) 
    
    def update_label(val):
        """Updates the value label when slider is moved."""
        value_label.configure(text=f"Ratio: {val:.2f}")
    
    slider.configure(command=lambda val: update_label(val))
    
    def train_model():
        """Trains the selected algorithm with the current configuration."""
        split_ratio = slider.get()
        if not is_valid_split_value(split_ratio):
            messagebox.showerror("Test Size Error", "Please enter a value between 0 and 1!")
            return
        
        try:
            if algorithm_code == "svm":
                algorithms.SVM(kernel, float(split_ratio))
            elif algorithm_code == "knn":
                algorithms.knn(n_neighbors, float(split_ratio)) 
            elif algorithm_code == "log_reg":
                algorithms.logistic_regression(float(split_ratio))
            elif algorithm_code == "lin_reg":
                algorithms.linear_regression(float(split_ratio))
            elif algorithm_code == "tree":
                algorithms.decision_tree(max_depth, float(split_ratio))
        except Exception as e:
            messagebox.showerror("Training Error", f"Failed to train model: {str(e)}")
                     
    create_button("Train", train_model, [.5, .5])
    create_button("Test", algorithms.tester, [.5, .6])
    create_button("Show Results", lambda: show_model_results(algorithm_code), [.5, .7])

    def go_back():
        """Navigates back to the appropriate configuration screen."""
        try:
            del algorithms.model
        except AttributeError:
            pass
            
        if algorithm_code == "svm":
            show_svm_config()  
        elif algorithm_code == "knn":
            show_knn_config()  
        elif algorithm_code == "log_reg" or algorithm_code == "lin_reg":
            show_algorithm_selection()   
        elif algorithm_code == "tree":
            show_decision_tree_config()            
    
    create_back_button(go_back)            

def show_svm_config():
    """Shows the configuration screen for SVM algorithm."""
    clear_screen()
    create_frame()
    create_label("SVM Configuration", [.5, .3]) 

    def set_kernel_type(choice): 
        """Sets the selected kernel type."""
        global kernel
        kernel = choice

    def submit():
        """Validates configuration and proceeds to train-test screen."""
        if "kernel" not in globals():
            messagebox.showerror("Kernel Error", "Please select a kernel type!")
            return
        show_train_test_screen("svm")

    create_dropdown_menu(["linear", "rbf", "poly", "sigmoid"], set_kernel_type, [.5, .4], "Select kernel") 
    create_button("Submit", submit, [.5, .7]) 
    create_back_button(show_algorithm_selection)

def show_decision_tree_config():
    """Shows the configuration screen for Decision Tree algorithm."""
    clear_screen()
    create_frame()
    create_label("Decision Tree Configuration", [.5, .3]) 
    create_label("Enter max depth", [.5, .4]) 
 
    entry = customtkinter.CTkEntry(master=app, width=200, height=25, corner_radius=10)
    entry.place(relx=.5, rely=.5, anchor=tkinter.CENTER) 
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
        show_train_test_screen("tree")
 
    create_button("Submit", submit, [.5, .7]) 
    create_back_button(show_algorithm_selection) 

def show_knn_config():
    """Shows the configuration screen for KNN algorithm."""
    clear_screen()
    create_frame()
    create_label("KNN Configuration", [.5, .3]) 
    create_label("Enter number of neighbors", [.5, .4]) 
 
    entry = customtkinter.CTkEntry(master=app, width=200, height=25, corner_radius=10) 
    entry.place(relx=.5, rely=.5, anchor=tkinter.CENTER)
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
        show_train_test_screen("knn")
 
    create_button("Submit", submit, [.5, .7]) 
    create_back_button(show_algorithm_selection) 

def show_algorithm_selection():
    """Shows the algorithm selection screen."""
    clear_screen()
    create_frame()
    create_label("Select Your Algorithm", [.5, .2])
    create_button("Decision Tree", show_decision_tree_config, [.5, .3])
    create_button("Linear Regression", lambda: show_train_test_screen('lin_reg'), [.5, .4])
    create_button("Logistic Regression", lambda: show_train_test_screen('log_reg'), [.5, .5])
    create_button("KNN", show_knn_config, [.5, .6])
    create_button("SVM", show_svm_config, [.5, .7])
    create_back_button(show_dataset_screen)
    
def show_dataset_screen():
    """Shows the dataset loading and feature selection screen."""
    clear_screen()
    create_frame()
    create_label("Load Your Dataset", [.5, .37])
    create_label("Enter number of features", [.5, .45], 15)
    create_button("Load Dataset", load_dataset, [0.25, 0.75], width=80)
    
    entry = customtkinter.CTkEntry(master=app, width=200, height=25, corner_radius=10)
    entry.place(relx=.5, rely=.5, anchor=tkinter.CENTER)

    def submit(): 
        """Validates the dataset and feature count before proceeding."""
        n_features = entry.get()
        if algorithms.dataset is None: 
            messagebox.showerror("Dataset Error", "Please load your dataset first!") 
            return
            
        if n_features == "": 
            messagebox.showerror("Input Error", "Please enter the number of features!") 
            return
    
        if not n_features.isdigit(): 
            messagebox.showerror("Input Error", "Please enter a valid number!") 
            return
            
        # Check if number of features is valid based on dataset
        max_features = algorithms.dataset.shape[1] - 1  # All columns except target
        if int(n_features) > max_features:
            messagebox.showerror("Input Error", 
                                f"Number of features cannot exceed the available features ({max_features})!")
            return
            
        algorithms.num_feature = int(n_features)
        show_algorithm_selection()
        
    create_button("Create Model", submit, [.75, .75], 80)
    create_button("Load Model", show_model_loader, [.75, .85], 80)
    create_back_button(show_main_screen)
 
def show_model_loader():
    """Shows the screen for loading a previously saved model."""
    clear_screen()
    create_frame()
    create_label("Model Information", [.5, .18]) 

    def load_model_file():
        """Loads a model file and displays its information."""
        model_path = askopenfilename(filetypes=[("Pickle file", ".pkl")])
        if not model_path:  # Check if a file was selected
            return
         
        try:   
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            algorithms.model = model
            
            # Display model details
            create_label("Algorithm Type", [.5, .27])
            algorithm_label = customtkinter.CTkLabel(app, 
                                                    text=f"< {model.__class__.__name__} >", 
                                                    font=("Arial", 20))
            algorithm_label.place(relx=.5, rely=.33, anchor=tkinter.CENTER) 
            
            # Show specific parameters based on model type
            if model.__class__.__name__ in ['KNeighborsClassifier', 'KNeighborsRegressor']:
                param_label = customtkinter.CTkLabel(app, 
                                                    text=f"Number of Neighbors: {model.n_neighbors}", 
                                                    font=("Arial", 20))
                param_label.place(relx=.5, rely=.42, anchor=tkinter.CENTER) 
                
            if model.__class__.__name__ == 'DecisionTreeClassifier':
                param_label = customtkinter.CTkLabel(app, 
                                                    text=f"Max Depth: {model.max_depth}", 
                                                    font=("Arial", 20))
                param_label.place(relx=.5, rely=.42, anchor=tkinter.CENTER)
                
            # Display load success message
            filename = os.path.basename(model_path)
            messagebox.showinfo("Success", f"Model '{filename}' loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    create_button("Load", load_model_file, [.5, .7])
    create_back_button(show_dataset_screen)

def show_main_screen():
    """Shows the main welcome screen."""
    clear_screen()
    create_frame()
    create_label("Welcome to Model Master", [.5, .3])
    create_button("Start", show_dataset_screen, [.5, .5])
    create_button("About", show_app_info, [0.25, 0.85], 80)

# Initialize the application
if __name__ == "__main__":
    show_main_screen()
    app.mainloop()