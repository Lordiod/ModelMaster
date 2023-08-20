import tkinter 
from tkinter.filedialog import askopenfilename , asksaveasfilename
from tkinter import messagebox
import pickle
import customtkinter 
import ELalgo
import pandas as pd 

customtkinter.set_appearance_mode("dark"); 
customtkinter.set_default_color_theme("green"); 


app = customtkinter.CTk() 
app.geometry("400x500") 
app.resizable(False,False)
app.title("ALL IN")

#------------------------------------------{Common Objects}------------------------------------------#

# Frame creator
def frame_creator():
   frame = customtkinter.CTkFrame(master=app,width=300,height=400,corner_radius=10)
   frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

# Label creator 
def label_creator(name,position,size=20):
    label = customtkinter.CTkLabel(master=app,text=name,text_color="#FFFFFF",font=("Arial",size), width=200,height=25 )
    label.place(relx=position[0],rely=position[1], anchor=tkinter.CENTER)

# Button creator
def btn_creator(name,function,position, width=200):
  btn = customtkinter.CTkButton(master=app,text=name,width=width,command=function); 
  btn.place(relx=position[0],rely=position[1],anchor=tkinter.CENTER) 

# Back button creator
def backbtn_creator(function):
  btn = customtkinter.CTkButton(master=app,text="Back",width=80, command=function); 
  btn.place(relx=0.25,rely=0.85, anchor=tkinter.CENTER)

# option menu for svm
def option_menu(values, callback, pos): 
    global choice 
    combobox = customtkinter.CTkOptionMenu(master=app,values=values,width=200,command=callback) 
    combobox.place(relx=pos[0],rely=pos[1],anchor=tkinter.CENTER) 
    combobox.set('choose kernel')  

#------------------------------------------{Test & Train exception handling}--------------------------#

def in_range(input):
    try:
        return 0< float(input) <1
    except ValueError:
        return False

#------------------------------------------{Frame killer}-----------------------------------------------#

def ELkiller():
    for w in app.winfo_children():
        w.destroy();

#------------------------------------------{Dataset loader}---------------------------------------------#

def ELloader():
    csv_path = askopenfilename(filetypes=[("CSV files", "*.csv")])
    ELalgo.dataset=pd.read_csv(csv_path)   

#------------------------------------------{info}-------------------------------------------------------#

def info():
    all_in_info= "This app is created to train and test your model using some algorithms and shows you the accuracy and the precision of the model!!"
    messagebox.showinfo("EL INFO",f"{all_in_info}")

#------------------------------------------{Model Saver}------------------------------------------------#

def saver():
    file_path = asksaveasfilename(filetypes=[("Pickle file",".pkl")],defaultextension=".pkl")
    with open(file_path, 'wb') as file:
        pickle.dump(ELalgo.model, file)
 
#------------------------------------------{Frames}-----------------------------------------------------#

def show_result(alg):
      
      if not hasattr(ELalgo, 'accuracy'):
        tkinter.messagebox.showerror("Test Error","Please test the data first !! ")
        return
      ELkiller(); 
      frame_creator(); 
      label_creator("Results", [.5,.18]) 
      label_creator("Accuracy", [.5,.27]) 
      label_creator(str(ELalgo.accuracy), [.5,.32]) 
      label_creator("Precision", [.5,.42]) 
      label_creator(str(ELalgo.precision), [.5,.47]) 
      label_creator("Recall", [.5,.57]) 
      label_creator(ELalgo.recall, [.5,.62]) 
      label_creator("F1 Score", [.5,.72])
      label_creator(ELalgo.f1_score, [.5,.77])  
      backbtn_creator(lambda:trainer(alg))

      def Conf_matrix():
          ELalgo.display_confusion_matrix(ELalgo.conf_matrix)
          
      def back_home():
          ELalgo.dataset= None
          del ELalgo.model
          del ELalgo.accuracy
          del ELalgo.precision
          del ELalgo.recall
          del ELalgo.f1_score
          del ELalgo.conf_matrix
          ELmain()
      btn_creator("Home", back_home,[.75,.85], 80)   
      btn_creator("CM",Conf_matrix,[.50,.85], 80) 
      btn_creator("Save Model",saver,[.5,.95],80)         


def trainer(alg):
    ELkiller()
    frame_creator()
    label_creator("Train & Test",[.5,.2])
    label_creator("Enter the size between 0 and 1",[.5,.3])
    slider = customtkinter.CTkSlider(app, from_=0, to=1, orientation="horizontal", number_of_steps=10)
    slider.place(relx=.5, rely=.4, anchor=tkinter.CENTER) 
    label = customtkinter.CTkLabel(app, text="Slider value: {}".format(slider.get()), font=("Arial", 10))
    label.place(relx=.5, rely=.45, anchor=tkinter.CENTER) 
    def update_label(val):
        label.configure(text="Slider value: {}".format(val))
    
    slider.configure(command=lambda val: update_label(val))
    
    def elinput():
        input = slider.get()
        if not in_range(input):
            tkinter.messagebox.showerror("Test size Error","Please enter a Value between 0 and 1!!")
            return
        
        if alg=="s":
            ELalgo.SVM(kernel,float(input))
        elif alg=="knn":
            ELalgo.knn(n_neighbors,float(input)) 
        elif alg=="lor":
            ELalgo.logistic_regression(float(input))  
        elif alg=="t":
            ELalgo.decision_tree(max_depth,float(input))   
                     
    
    btn_creator("Train", elinput, [.5, .5])
    btn_creator("Test", ELalgo.tester, [.5, .6])
    btn_creator("show results", lambda: show_result(alg), [.5, .7])

    def back_fun():
        try:
            del ELalgo.model
        except AttributeError:
            pass
        if alg=="s":
            SVM_window()  
        elif alg=="knn":
            KNN_window()  
        elif alg=="lor":
            ELalgo_selector()   
        elif alg=="t":
            decision_tree_window()           
    backbtn_creator(back_fun)            

def SVM_window():
  ELkiller(); 
  frame_creator(); 
  label_creator("SVM", [.5,.3]) 

  #function that assigns the selected choice to the global variable kernel 
  def kernelmenu_callback(choice): 
      global kernel
      kernel = choice

  def Submit():
    if "kernel" not in globals():
      tkinter.messagebox.showerror("Kernel Error","PLEASE SELECT A KERNEL !! ")
      return
    trainer("s")

  option_menu(["linear", "rbf"], kernelmenu_callback, [.5,.4]) 
  btn_creator("Submit", Submit, [.5,.7]) 
  backbtn_creator(ELalgo_selector)

def decision_tree_window():
  ELkiller(); 
  frame_creator(); 
  label_creator("Decision Tree", [.5,.3]) 
  label_creator("Enter max depth", [.5,.4]) 
 
  entry = customtkinter.CTkEntry(master=app,width=200,height=25,corner_radius=10)
  entry.place(relx=.5, rely=.5, anchor=tkinter.CENTER) 
 
  def submit(): 
    input = entry.get()
    if input == "": 
      tkinter.messagebox.showerror("Features Error ","ENTER NUMBER OF FEATURES, PLEASE!! ") 
      return
    
    if input.isdigit() == False: 
      tkinter.messagebox.showerror("Features Error ","INVALID NUMBER !! ") 
      return
    global max_depth
    max_depth=int(input)
    trainer("t")

 
  btn_creator("Submmit", submit, [.5,.7]); 

  backbtn_creator(ELalgo_selector) 

def KNN_window():
  ELkiller(); 
  frame_creator(); 
  label_creator("KNN", [.5,.3]) 
  label_creator("Enter number of neighbors", [.5,.4]) 
 
  entry = customtkinter.CTkEntry(master=app,width=200,height=25,corner_radius=10) 
  entry.place(relx=.5, rely=.5, anchor=tkinter.CENTER) 
 
  def submit(): 
    input = entry.get() 
    if input == "": 
      tkinter.messagebox.showerror("Neighbors Error ","ENTER NUMBER OF NEIGHBORS, PLEASE!! ") 
      return
    if input.isdigit() == False: 
      tkinter.messagebox.showerror("Neighbors Error ","ONLY NUMBERS!! ") 
      return

    global n_neighbors
    n_neighbors = int(input)
    trainer("knn")
 
  btn_creator("Submit", submit, [.5,.7]) 
  backbtn_creator(ELalgo_selector) 

def ELalgo_selector():
    ELkiller()
    frame_creator()
    label_creator("Select Your Algorithm",[.5,.3])
    btn_creator("Decision Tree",decision_tree_window, [.5,.4])
    btn_creator("SVM",SVM_window,[.5,.7])
    btn_creator("KNN",KNN_window,[.5,.6])
    btn_creator("Logistic Regression",lambda:trainer('lor'),[.5,.5])
    backbtn_creator(loader)
    
def loader():
    ELkiller()
    frame_creator()
    label_creator("Load Your Dataset",[.5,.37])
    label_creator("Enter number of features",[.5,.45],15)
    btn_creator("Load Dataset", ELloader,[0.25,0.75],width=80)
    entry = customtkinter.CTkEntry(master=app,width=200,height=25,corner_radius=10)
    entry.place(relx=.5, rely=.5, anchor=tkinter.CENTER)

    def submit(): 
        n_features = entry.get()
        if ELalgo.dataset is None: 
            tkinter.messagebox.showerror("Dataset Error","LOAD YOUR DATASET FIRST!!") 
            return
        if n_features == "": 
            tkinter.messagebox.showerror("Features Error ","ENTER NUMBER OF FEATURES!! ") 
            return
    
        if  n_features.isdigit() == False: 
            tkinter.messagebox.showerror("Features Error ","ONLY NUMBERS!! ") 
            return
        ELalgo.num_feature = int(n_features)
        ELalgo_selector()
    btn_creator("Create Model",submit,[.75,.75],80)
    btn_creator("Model loader",mod_loader,[.75,.85],80)
    backbtn_creator(ELmain)
 
def mod_loader():
    ELkiller()
    frame_creator()
    label_creator("Model Info", [.5,.18]) 


    def LL():
        model_path = askopenfilename(filetypes=[("Pickle file",".pkl")])
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        ELalgo.model = model
        label_creator("Algorithm Used",[.5,.27])
        label1 = customtkinter.CTkLabel(app,text="< {} >".format(model.__class__.__name__) , font=("Arial", 20))
        label1.place(relx=.5, rely=.33, anchor=tkinter.CENTER) 
        if model.__class__.__name__ == 'KNeighborsClassifier' or model.__class__.__name__ == 'KNeighborsRegressor':
            label2 = customtkinter.CTkLabel(app,text="Number of Neighbors: {} ".format(model.n_neighbors) , font=("Arial", 20))
            label2.place(relx=.5, rely=.42, anchor=tkinter.CENTER) 
        if model.__class__.__name__ == 'DecisionTreeClassifier':
            label3 = customtkinter.CTkLabel(app,text="Max Depth: {} ".format(model.max_depth) , font=("Arial", 20))
            label3.place(relx=.5, rely=.42, anchor=tkinter.CENTER)    

    btn_creator("Load",LL,[.5,.7])
    backbtn_creator(loader)





#def trained_model_resault():
#    if not hasattr(ELalgo, 'accuracy'):
#        tkinter.messagebox.showerror("Test Error","Please test the data first !! ")
#        return
#    ELkiller(); 
#    frame_creator(); 
#    label_creator("Results", [.5,.18]) 
#    label_creator("Accuracy", [.5,.27]) 
#    label_creator(str(ELalgo.accuracy), [.5,.32]) 
#    label_creator("Precision", [.5,.42]) 
#    label_creator(str(ELalgo.precision), [.5,.47]) 
#    label_creator("Recall", [.5,.57]) 
#    label_creator(ELalgo.recall, [.5,.62]) 
#    label_creator("F1 Score", [.5,.72])
#    label_creator(ELalgo.f1_score, [.5,.77]) 
#    def back_home():
#        ELalgo.dataset= None
#        del ELalgo.model
#        del ELalgo.accuracy
#        del ELalgo.precision
#        del ELalgo.recall
#        del ELalgo.f1_score
#        del ELalgo.conf_matrix
#        ELmain()
#    btn_creator("Home", back_home,[.75,.85], 80)

#def MlLoader():
#    if ELalgo.dataset is None: 
#        tkinter.messagebox.showerror("Dataset Error","LOAD YOUR DATASET FIRST!!") 
#        return   
#    ELkiller()
#    frame_creator()
#    label_creator("Trained Model Loader",[.5,.3])
#    def LL():
#        model_path = askopenfilename(filetypes=[("Pickle file",".pkl")])
#        with open(model_path, 'rb') as file:
#            model = pickle.load(file)
#        ELalgo.model = model
#        ELalgo.test_ready_model()   
#    btn_creator("Load",LL,[.5,.7])
#    btn_creator("Evaluate",trained_model_resault,[.5,.8])
    

def ELmain():
    ELkiller()
    frame_creator()
    label_creator("Welcome to el ALL IN app",[.5,.3])
    btn_creator("Start",loader,[.5,.5])
    btn_creator("EL INFO",info,[0.25,0.85],80)

ELmain()
app.mainloop()