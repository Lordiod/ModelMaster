# ALL-IN
this project helps you to choose the best classification algorithm that better suits your dataset 

first it asks you for the preprocessed dataset and it will put it in the RFE(Recursive Feature Elementaion) to rank the best features,
then it will ask you for the classification algorithm you want to evaluate,
and then if you chose "KNN" it will ask you for the number of neighbours, same thing goes to decition tree , it asks you for the max depth,
then it goes to testing window which asks you the ratio of data you want to train and test the model on.

after you train and test your model it will show you the accuracy, percision, recall and f1 score of the model , also you can check for its confusion matrix

you can save the model using pickle library and load it to check the algorithm has been used (a better model loader will be added in the future)

after using my app will help you to choose the best classification algorithm that suits your dataset.
