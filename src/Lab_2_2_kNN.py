# Laboratory practice 2.2: KNN classification
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()
import numpy as np  
import seaborn as sns


def minkowski_distance(a, b, p=2):
    """
    Compute the Minkowski distance between two arrays.

    Args:
        a (np.ndarray): First array.
        b (np.ndarray): Second array.
        p (int, optional): The degree of the Minkowski distance. Defaults to 2 (Euclidean distance).

    Returns:
        float: Minkowski distance between arrays a and b.
    """
    distance=0
    for i in range(len(a)):
        distance+=abs(a[i]-b[i])**p
    distance=distance**(1/p)
    
    return distance


# k-Nearest Neighbors Model

# - [K-Nearest Neighbours](https://scikit-learn.org/stable/modules/neighbors.html#classification)
# - [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)


class knn:
    def __init__(self):
        self.k = None
        self.p = None
        self.x_train = None
        self.y_train = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, k: int = 5, p: int = 2):
        """
        Fit the model using X as training data and y as target values.

        You should check that all the arguments shall have valid values:
            X and y have the same number of rows.
            k is a positive integer.
            p is a positive integer.

        Args:
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Target values.
            k (int, optional): Number of neighbors to use. Defaults to 5.
            p (int, optional): The degree of the Minkowski distance. Defaults to 2.
        """
      
        if X_train.shape[0]!=y_train.shape[0]:
            raise ValueError("Length of X_train and y_train must be equal.")

        elif  k<0 or p<0:
            raise ValueError("k and p must be positive integers.")
           
        else:
            self.x_train=X_train
            self.y_train=y_train
            self.k=k
            self.p=p
            

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the provided data.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class labels.
        """
      
        predicted_labels=[]
  
        for i,data_sample in enumerate(X):
            distances=[]
            neighbours_index=[]
            for i in range(len(self.x_train)):
                distance=minkowski_distance(data_sample,self.x_train[i],self.p)
                distances.append(distance)
                neighbours_index.append(i)
            distances=np.array(distances)
            neighbours_index=np.array(neighbours_index)
            indices_ordenados = np.argsort(distances)
            distances = distances[indices_ordenados]
            neighbours_index = neighbours_index[indices_ordenados]
            suma=0
            for n in neighbours_index[:self.k]:
                if self.y_train[n]:
                    suma+=1
            p=suma/self.k
            if p>1/2:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
                
            
        

        return np.array(predicted_labels)

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict the class probabilities for the provided data.

        Each class probability is the amount of each label from the k nearest neighbors
        divided by k.

        Args:
            X (np.ndarray): data samples to predict their labels.

        Returns:
            np.ndarray: Predicted class probabilities.
        """
        class_probs=[]
  
        for i,data_sample in enumerate(X):
            distances=[]
            neighbours_index=[]
            for i in range(len(self.x_train)):
                distance=minkowski_distance(data_sample,self.x_train[i],self.p)
                distances.append(distance)
                neighbours_index.append(i)
            distances=np.array(distances)
            neighbours_index=np.array(neighbours_index)
            indices_ordenados = np.argsort(distances)
            distances = distances[indices_ordenados]
            neighbours_index = neighbours_index[indices_ordenados]
            suma=0
            for n in neighbours_index[:self.k]:
                if self.y_train[n]:
                    suma+=1
            p=suma/self.k
            class_probs.append([1-p,p])

        return np.array(class_probs)

    def compute_distances(self, point: np.ndarray) -> np.ndarray:
        """Compute distance from a point to every point in the training dataset

        Args:
            point (np.ndarray): data sample.

        Returns:
            np.ndarray: distance from point to each point in the training dataset.
        """
        distances=[]
        for dataset_point in self.x_train:
            distances.append(minkowski_distance(point,dataset_point))
        return np.array(distances)


    def get_k_nearest_neighbors(self, distances: np.ndarray) -> np.ndarray:
        """Get the k nearest neighbors indices given the distances matrix from a point.

        Args:
            distances (np.ndarray): distances matrix from a point whose neighbors want to be identified.

        Returns:
            np.ndarray: row indices from the k nearest neighbors.

        Hint:
            You might want to check the np.argsort function.
        """
        return np.argsort(distances)[:self.k]

    def most_common_label(self, knn_labels: np.ndarray) -> int:
        """Obtain the most common label from the labels of the k nearest neighbors

        Args:
            knn_labels (np.ndarray): labels from the k nearest neighbors

        Returns:
            int: most common label
        """
        ones=0
        zeros=0
        for label in knn_labels:
            if label==1:
                ones+=1
            else:
                zeros+=1
        if ones>zeros:
            return 1
        else:
            return 0

    def __str__(self):
        """
        String representation of the kNN model.
        """
        return f"kNN model (k={self.k}, p={self.p})"



def plot_2Dmodel_predictions(X, y, model, grid_points_n):
    """
    Plot the classification results and predicted probabilities of a model on a 2D grid.

    This function creates two plots:
    1. A classification results plot showing True Positives, False Positives, False Negatives, and True Negatives.
    2. A predicted probabilities plot showing the probability predictions with level curves for each 0.1 increment.

    Args:
        X (np.ndarray): The input data, a 2D array of shape (n_samples, 2), where each row represents a sample and each column represents a feature.
        y (np.ndarray): The true labels, a 1D array of length n_samples.
        model (classifier): A trained classification model with 'predict' and 'predict_proba' methods. The model should be compatible with the input data 'X'.
        grid_points_n (int): The number of points in the grid along each axis. This determines the resolution of the plots.

    Returns:
        None: This function does not return any value. It displays two plots.

    Note:
        - This function assumes binary classification and that the model's 'predict_proba' method returns probabilities for the positive class in the second column.
    """
    # Map string labels to numeric
    unique_labels = np.unique(y)
    num_to_label = {i: label for i, label in enumerate(unique_labels)}

    # Predict on input data
    preds = model.predict(X)

    # Determine TP, FP, FN, TN
    tp = (y == unique_labels[1]) & (preds == unique_labels[1])
    fp = (y == unique_labels[0]) & (preds == unique_labels[1])
    fn = (y == unique_labels[1]) & (preds == unique_labels[0])
    tn = (y == unique_labels[0]) & (preds == unique_labels[0])

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Classification Results Plot
    ax[0].scatter(X[tp, 0], X[tp, 1], color="green", label=f"True {num_to_label[1]}")
    ax[0].scatter(X[fp, 0], X[fp, 1], color="red", label=f"False {num_to_label[1]}")
    ax[0].scatter(X[fn, 0], X[fn, 1], color="blue", label=f"False {num_to_label[0]}")
    ax[0].scatter(X[tn, 0], X[tn, 1], color="orange", label=f"True {num_to_label[0]}")
    ax[0].set_title("Classification Results")
    ax[0].legend()

    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_points_n),
        np.linspace(y_min, y_max, grid_points_n),
    )

    # # Predict on mesh grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    # Use Seaborn for the scatter plot
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="Set1", ax=ax[1])
    ax[1].set_title("Classes and Estimated Probability Contour Lines")

    # Plot contour lines for probabilities
    cnt = ax[1].contour(xx, yy, probs, levels=np.arange(0, 1.1, 0.1), colors="black")
    ax[1].clabel(cnt, inline=True, fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()



def evaluate_classification_metrics(y_true, y_pred, positive_label):
    """
    Calculate various evaluation metrics for a classification model.

    Args:
        y_true (array-like): True labels of the data.
        positive_label: The label considered as the positive class.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    Metrics Calculated:
        - Confusion Matrix: [TN, FP, FN, TP]
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - Precision: TP / (TP + FP)
        - Recall (Sensitivity): TP / (TP + FN)
        - Specificity: TN / (TN + FP)
        - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    """
 
    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    y_pred_mapped = np.array([1 if label == positive_label else 0 for label in y_pred])

    # Confusion Matrix
    tp=0
    fp=0
    tn=0
    fn=0
    for i in range(len(y_true_mapped)):
        if y_pred_mapped[i]:
            if  y_true_mapped[i]:
                tp+=1
            else:
                fp+=1
        if not y_pred_mapped[i]:
            if not y_true_mapped[i]:
                tn+=1
            else:
                fn+=1

    
   
    # Accuracy
    if (tp + tn + fp + fn) >0:
        accuracy=(tp + tn) / (tp + tn + fp + fn)
    else:
        accuracy=0

    # Precision
    if (tp + fp)>0:
        precision= tp / (tp + fp)
    else:
        precision=0

    # Recall (Sensitivity)
    if (tp + fn)>0:
        recall=tp / (tp + fn)
    else:
        recall=0

    # Specificity
    if (tn + fp)>0:
        specificity=tn / (tn + fp)
    else:
        specificity=0

    # F1 Score
    if (precision + recall)>0:
        f1=2 * (precision * recall) / (precision + recall)
    else:
        f1=0

    return {
        "Confusion Matrix": [tn, fp, fn, tp],
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Specificity": specificity,
        "F1 Score": f1,
    }



def plot_calibration_curve(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot a calibration curve to evaluate the accuracy of predicted probabilities.

    This function creates a plot that compares the mean predicted probabilities
    in each bin with the fraction of positives (true outcomes) in that bin.
    This helps assess how well the probabilities are calibrated.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class (positive_label).
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label that is considered the positive class.
                                    This is used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins to use for grouping predicted probabilities.
                                Defaults to 10. Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "bin_centers": Array of the center values of each bin.
            - "true_proportions": Array of the fraction of positives in each bin

    """
    # calculamos la probabilidad que habra en cada intervalo
    rangos_probs=1/n_bins


    bin_centers=[] # aqui iran los centros de los bins
    intervalos=[] # aqui guardaremos el limite superior e inferior de cada intervalo

    contador=0 # variable auxiliar
    for i in range(n_bins):
        bin_centers.append((contador+contador+rangos_probs)/2)
        intervalos.append([contador,contador+rangos_probs])
        contador+=rangos_probs


    # Creamos una lista de listas donde hay un alista por cada bin
    # en cada sub lista metermos el indice de las obervaciones de y_prob 
    # que caen dentro de su respectivo rango de probabilidad
    lista_observaciones_colocadas=[[] for i in range (n_bins)]

    for i,prob in enumerate(y_probs):


        found=False
        intervalo_index=0

        # bucle que busca en que intervalo encaja la observacion
        while not found:
            
        
            # accedemos a los limites de cada intervalo
            inf=intervalos[intervalo_index][0]
            sup=intervalos[intervalo_index][1]
        
            # comprobamos si la osbervacion esta en el rango actual
            if inf<=prob<sup:
                
                # si lo esta añadimos si indice a la lista de observaciones colocadas
                lista_observaciones_colocadas[intervalo_index].append(i)

                # salimos de bucle
                found=True
            elif prob==1:
                
                lista_observaciones_colocadas[-1].append(i)
                found=True

            intervalo_index+=1


    # calculamos la frecuencia para cada bin
    true_proportions=[]   

    # iteramos por cada intervalo
    for lista in lista_observaciones_colocadas:
        
        suma=0
        count=0
      
        # iteramos por cada observacion del intervalo
        for i in lista:
            count+=1
            # comprobamos si la observacion es 1 en y_true
            if y_true[i]==positive_label:
                # si lo es lo contamos
                suma+=1
                
        # añadimos la frecuencia al array
        if len(lista)==0:
            true_proportions.append(0)
        else:
            true_proportions.append(suma/len(lista))
    true_proportions=np.array(true_proportions)
    bin_centers=np.array(bin_centers)

    plt.figure(figsize=(6,4)) 
    sns.lineplot(x=bin_centers, y=true_proportions, marker="o", linewidth=2, markersize=8,label="model predictions")
    x_range = np.linspace(0, 1, 100) 
    sns.lineplot(x=x_range, y=x_range, linestyle="--", color="r", linewidth=2, label="ideal predictor")

    plt.xlabel("avg predicted probability in each bin")
    plt.ylabel("fraction of positives")

    plt.show()

    return {"bin_centers": bin_centers, "true_proportions": true_proportions}



def plot_probability_histograms(y_true, y_probs, positive_label, n_bins=10):
    """
    Plot probability histograms for the positive and negative classes separately.

    This function creates two histograms showing the distribution of predicted
    probabilities for each class. This helps in understanding how the model
    differentiates between the classes.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.
        n_bins (int, optional): Number of bins for the histograms. Defaults to 10. 
                                Bins are equally spaced in the range [0, 1].

    Returns:
        dict: A dictionary with the following keys:
            - "array_passed_to_histogram_of_positive_class": 
                Array of predicted probabilities for the positive class.
            - "array_passed_to_histogram_of_negative_class": 
                Array of predicted probabilities for the negative class.

    """

    y_true_mapped = np.array([1 if label == positive_label else 0 for label in y_true])
    
    
    positivos = y_probs[y_true_mapped==1]  
    negativos = y_probs[y_true_mapped==0]  

  

    # Histograma de la clase positiva
    plt.hist(positivos, bins=n_bins)
    plt.title("positivos Positiva")
    plt.xlabel("predicted probability")
    plt.ylabel("frequency")
    plt.show()
    # Histograma de la clase negativa
    plt.hist(negativos, bins=n_bins)
    plt.title("Clase Negativa")
    plt.xlabel("predicted probability")
    plt.ylabel("fFrequency")

    plt.show()
    
   


    return {
        "array_passed_to_histogram_of_positive_class": y_probs[y_true_mapped == 1],
        "array_passed_to_histogram_of_negative_class": y_probs[y_true_mapped == 0],
    }



def plot_roc_curve(y_true, y_probs, positive_label):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    The ROC curve is a graphical representation of the diagnostic ability of a binary
    classifier system as its discrimination threshold is varied. It plots the True Positive
    Rate (TPR) against the False Positive Rate (FPR) at various threshold settings.

    Args:
        y_true (array-like): True labels of the data. Can be binary or categorical.
        y_probs (array-like): Predicted probabilities for the positive class. 
                            Expected values are in the range [0, 1].
        positive_label (int or str): The label considered as the positive class.
                                    Used to map categorical labels to binary outcomes.

    Returns:
        dict: A dictionary containing the following:
            - "fpr": Array of False Positive Rates for each threshold.
            - "tpr": Array of True Positive Rates for each threshold.

    """


    y_true = np.array([1 if label == positive_label else 0 for label in y_true])

    tpr = []  
    fpr = []  
    
    # esto sera una lista con los umbrales del 0 al 1 de 0.1 a 0.1
    umbrales = np.arange(0, 1.1, 0.1) 
    for umbral in umbrales:
        # clasificamos la predicciones segun el umbral
        y_pred = []
        for prob in y_probs:
            if prob >= umbral:
                y_pred.append(1)
            else:
                y_pred.append(0)
        # calculamos los valores de tp,fp,tn,fn 
        tp =0
        fn = 0
        fp = 0
        tn = 0
        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                tp += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                fn += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                fp += 1
            elif y_true[i] == 0 and y_pred[i] == 0:
                tn += 1
        # calculamos los ratios TPR y FPR
        if (tp + fn) > 0:
            tpr.append(tp / (tp + fn) )
        else:
            tpr.append(0)

        if (fp + tn) > 0:
            fpr.append(fp / (fp + tn))
        else:
            fpr.append(0)
    
    # Convertimocpnvertimos en arrays
    tpr =np.array(tpr)
    fpr=np.array(fpr)
    
    # ploteamos la curva
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], 'k--', label="random classifier")
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.title("ROC Curve")
    plt.show()
    
    
    return {"fpr": np.array(fpr), "tpr": np.array(tpr)}
