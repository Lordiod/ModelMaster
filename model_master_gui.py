import tkinter 
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import messagebox
import pickle
import customtkinter 
import algorithms
import pandas as pd 

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("green")

app = customtkinter.CTk() 
app.geometry("400x500") 
app.resizable(False, False)
app.title("ALL IN")

#------------------------------------------{Common UI Components}------------------------------------------#

def create_frame():
   frame = customtkinter.CTkFrame(master=app, width=300, height=400, corner_radius=10)
   frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

def create_label(name, position, size=20):
    label = customtkinter.CTkLabel(master=app, text=name, text_color="#FFFFFF", font=("Arial", size), width=200, height=25)
    label.place(relx=position[0], rely=position[1], anchor=tkinter.CENTER)

def create_button(name, function, position, width=200):
    btn = customtkinter.CTkButton(master=app, text=name, width=width, command=function)
    btn.place(relx=position[0], rely=position[1], anchor=tkinter.CENTER) 

def create_back_button(function):
    btn = customtkinter.CTkButton(master=app, text="Back", width=80, command=function)
    btn.place(relx=0.25, rely=0.85, anchor=tkinter.CENTER)

def create_dropdown_menu(values, callback, pos): 
    global choice 
    combobox = customtkinter.CTkOptionMenu(master=app, values=values, width=200, command=callback) 
    combobox.place(relx=pos[0], rely=pos[1], anchor=tkinter.CENTER) 
    combobox.set('choose kernel')  

#------------------------------------------{Utility Functions}------------------------------------------#

def is_valid_split_value(input):
    try:
        return 0 < float(input) < 1
    except ValueError:
        return False

def clear_screen():
    for widget in app.winfo_children():
        widget.destroy()

def load_dataset():
    csv_path = askopenfilename(filetypes=[("CSV files", "*.csv")])
    if csv_path:  # Check if a file was selected
        algorithms.dataset = pd.read_csv(csv_path)   

def show_app_info():
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
    file_path = asksaveasfilename(filetypes=[("Pickle file", ".pkl")], defaultextension=".pkl")
    if file_path:  # Check if a path was selected
        with open(file_path, 'wb') as file:
            pickle.dump(algorithms.model, file)
 
#------------------------------------------{Results Display}---------------------------------------------#

def show_model_results(algorithm):
    if not hasattr(algorithms, 'accuracy'):
        tkinter.messagebox.showerror("Test Error", "Please test the model first!")
        return
    
    clear_screen()
    create_frame()
    create_label("Results", [.5, .18]) 
    create_label("Accuracy", [.5, .27]) 
    create_label(str(algorithms.accuracy), [.5, .32]) 
    create_label("Precision", [.5, .42]) 
    create_label(str(algorithms.precision), [.5, .47]) 
    create_label("Recall", [.5, .57]) 
    create_label(algorithms.recall, [.5, .62]) 
    create_label("F1 Score", [.5, .72])
    create_label(algorithms.f1_score, [.5, .77])  
    create_back_button(lambda: show_train_test_screen(algorithm))

    def display_confusion_matrix():
        algorithms.display_confusion_matrix(algorithms.conf_matrix)
          
    def return_to_home():
        algorithms.dataset = None
        del algorithms.model
        del algorithms.accuracy
        del algorithms.precision
        del algorithms.recall
        del algorithms.f1_score
        del algorithms.conf_matrix
        show_main_screen()
    
    create_button("Home", return_to_home, [.75, .85], 80)   
    create_button("Confusion Matrix", display_confusion_matrix, [.50, .85], 80) 
    create_button("Save Model", save_model, [.5, .95], 80)         

#------------------------------------------{Training Screens}-------------------------------------------#

def show_train_test_screen(algorithm_code):
    clear_screen()
    create_frame()
    create_label("Train & Test", [.5, .2])
    create_label("Set train-test split ratio", [.5, .3])
    
    slider = customtkinter.CTkSlider(app, from_=0, to=1, orientation="horizontal", number_of_steps=10)
    slider.place(relx=.5, rely=.4, anchor=tkinter.CENTER) 
    
    value_label = customtkinter.CTkLabel(app, text="Ratio: {}".format(slider.get()), font=("Arial", 10))
    value_label.place(relx=.5, rely=.45, anchor=tkinter.CENTER) 
    
    def update_label(val):
        value_label.configure(text="Ratio: {}".format(val))
    
    slider.configure(command=lambda val: update_label(val))
    
    def train_model():
        split_ratio = slider.get()
        if not is_valid_split_value(split_ratio):
            tkinter.messagebox.showerror("Test Size Error", "Please enter a value between 0 and 1!")
            return
        
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
                     
    create_button("Train", train_model, [.5, .5])
    create_button("Test", algorithms.tester, [.5, .6])
    create_button("Show Results", lambda: show_model_results(algorithm_code), [.5, .7])

    def go_back():
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
    clear_screen()
    create_frame()
    create_label("SVM Configuration", [.5, .3]) 

    def set_kernel_type(choice): 
        global kernel
        kernel = choice

    def submit():
        if "kernel" not in globals():
            tkinter.messagebox.showerror("Kernel Error", "Please select a kernel type!")
            return
        show_train_test_screen("svm")

    create_dropdown_menu(["linear", "rbf"], set_kernel_type, [.5, .4]) 
    create_button("Submit", submit, [.5, .7]) 
    create_back_button(show_algorithm_selection)

def show_decision_tree_config():
    clear_screen()
    create_frame()
    create_label("Decision Tree Configuration", [.5, .3]) 
    create_label("Enter max depth", [.5, .4]) 
 
    entry = customtkinter.CTkEntry(master=app, width=200, height=25, corner_radius=10)
    entry.place(relx=.5, rely=.5, anchor=tkinter.CENTER) 
 
    def submit(): 
        input_value = entry.get()
        if input_value == "": 
            tkinter.messagebox.showerror("Input Error", "Please enter a max depth value!") 
            return
        
        if not input_value.isdigit(): 
            tkinter.messagebox.showerror("Input Error", "Please enter a valid number!") 
            return
            
        global max_depth
        max_depth = int(input_value)
        show_train_test_screen("tree")
 
    create_button("Submit", submit, [.5, .7]) 
    create_back_button(show_algorithm_selection) 

def show_knn_config():
    clear_screen()
    create_frame()
    create_label("KNN Configuration", [.5, .3]) 
    create_label("Enter number of neighbors", [.5, .4]) 
 
    entry = customtkinter.CTkEntry(master=app, width=200, height=25, corner_radius=10) 
    entry.place(relx=.5, rely=.5, anchor=tkinter.CENTER) 
 
    def submit(): 
        input_value = entry.get() 
        if input_value == "": 
            tkinter.messagebox.showerror("Input Error", "Please enter the number of neighbors!") 
            return
            
        if not input_value.isdigit(): 
            tkinter.messagebox.showerror("Input Error", "Please enter a valid number!") 
            return

        global n_neighbors
        n_neighbors = int(input_value)
        show_train_test_screen("knn")
 
    create_button("Submit", submit, [.5, .7]) 
    create_back_button(show_algorithm_selection) 

def show_algorithm_selection():
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
    clear_screen()
    create_frame()
    create_label("Load Your Dataset", [.5, .37])
    create_label("Enter number of features", [.5, .45], 15)
    create_button("Load Dataset", load_dataset, [0.25, 0.75], width=80)
    
    entry = customtkinter.CTkEntry(master=app, width=200, height=25, corner_radius=10)
    entry.place(relx=.5, rely=.5, anchor=tkinter.CENTER)

    def submit(): 
        n_features = entry.get()
        if algorithms.dataset is None: 
            tkinter.messagebox.showerror("Dataset Error", "Please load your dataset first!") 
            return
            
        if n_features == "": 
            tkinter.messagebox.showerror("Input Error", "Please enter the number of features!") 
            return
    
        if not n_features.isdigit(): 
            tkinter.messagebox.showerror("Input Error", "Please enter a valid number!") 
            return
            
        algorithms.num_feature = int(n_features)
        show_algorithm_selection()
        
    create_button("Create Model", submit, [.75, .75], 80)
    create_button("Load Model", show_model_loader, [.75, .85], 80)
    create_back_button(show_main_screen)
 
def show_model_loader():
    clear_screen()
    create_frame()
    create_label("Model Information", [.5, .18]) 

    def load_model_file():
        model_path = askopenfilename(filetypes=[("Pickle file", ".pkl")])
        if not model_path:  # Check if a file was selected
            return
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        algorithms.model = model
        
        create_label("Algorithm Type", [.5, .27])
        algorithm_label = customtkinter.CTkLabel(app, text="< {} >".format(model.__class__.__name__), font=("Arial", 20))
        algorithm_label.place(relx=.5, rely=.33, anchor=tkinter.CENTER) 
        
        # Show specific parameters based on model type
        if model.__class__.__name__ in ['KNeighborsClassifier', 'KNeighborsRegressor']:
            param_label = customtkinter.CTkLabel(app, text="Number of Neighbors: {}".format(model.n_neighbors), font=("Arial", 20))
            param_label.place(relx=.5, rely=.42, anchor=tkinter.CENTER) 
            
        if model.__class__.__name__ == 'DecisionTreeClassifier':
            param_label = customtkinter.CTkLabel(app, text="Max Depth: {}".format(model.max_depth), font=("Arial", 20))
            param_label.place(relx=.5, rely=.42, anchor=tkinter.CENTER)    

    create_button("Load", load_model_file, [.5, .7])
    create_back_button(show_dataset_screen)

def show_main_screen():
    clear_screen()
    create_frame()
    create_label("Welcome to Model Master", [.5, .3])
    create_button("Start", show_dataset_screen, [.5, .5])
    create_button("About", show_app_info, [0.25, 0.85], 80)

# Initialize the application
if __name__ == "__main__":
    show_main_screen()
    app.mainloop()