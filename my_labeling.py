__authors__ = ['1706732', '1707531', '1704706']
__group__ = '14'

from Kmeans import *
from KNN import *
from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_k_means, visualize_retrieval
import time
import matplotlib.pyplot as plt

    # You can start coding your functions here

def Retrieval_by_color(images, labels, question):
    
    """
    Funció que cerca imatges que tenen unes etiquetes de color
    
    Entrada: una llista d’imatges i una llista d’strings amb els colors que volem buscar. 
    Retorna: totes les imatges de la llista que contenen les etiquetes de la pregunta que fem. 
    Possible millora: Aquesta funció pot ser millorada afegint un paràmetre d’entrada que contingui el 
    percentatge de cada color que conté la imatge, i retorni les imatges ordenades segons el percentatge. 

    """
    
    resultat = []
    for img, labels_img in zip(images, labels):
        if all(color in labels_img for color in question):
            resultat.append(img)
    return resultat


def Retrieval_by_shape(images, labels, question):
    
    """
    Funció que cerca imatges que tenen unes etiquetes de forma
    
    Entrada: una llista d’imatges i una llista d’strings amb les formes que volem buscar. 
    Retorna: totes les imatges de la llista que contenen les etiquetes de la pregunta que fem. 
    Possible millora: Aquesta funció pot ser millorada afegint un paràmetre d’entrada que contingui el 
    percentatge de K-neighbors amb l’etiqueta que busquem i retorni les imatges ordenades segons el
    percentatge.

    """
    
    resultat = []
    for img, labels_img in zip(images, labels):
        if any(shape in labels_img for shape in question):
            resultat.append(img)
    return resultat

def Retrieval_combined(images, color_labels, shape_labels, color_question, shape_question):
    
    """
    Funció que cerca imatges que tenen etiquetes de forma i color
    Entrada: una llista d’imatges, una llista d’strings de color i una llista d’strings de forma
    Retorna: totes les imatges de la llista que contenen les etiquetes de les dues preguntes que fem. 
    Possible millora: Aquesta funció pot ser millorada introduint les dades de percentatge de color i forma de 
    les etiquetes
    
    """
    
    #Tornarem a programar una de les funcions anteriors, però amb la diferència
    #de que aquest cop haurem de tenir constància de les labels de les imatges filtrades
    
    retrieval_color = []
    filtered_shape_labels = []
    
    
    for img, color_labels_img, shape_labels_img in zip(images, color_labels, shape_labels):
        if all(color in color_labels_img for color in color_question):
            retrieval_color.append(img)
            filtered_shape_labels.append(shape_labels_img)
            
    result = Retrieval_by_shape(retrieval_color, filtered_shape_labels, shape_question)  #Ara podem reutilitzar la funció anterior
    return result

def Kmean_statistics(llista_imgs, Kmax, color_labels, inits):
    
    """
    Funció que genera un set d'estadístiques d'execució del Kmeans amb 
    la corresponent visualització d'aquestes.
    Entrada: una instància de la classe Kmeans, un conjunt d'imatges i Kmax
    Retorna: la visualització de les estadístiques en format gràfic

    """
    
    acc_k = []
    wcds_init = {init: [] for init in inits}
    iters_init = {init: [] for init in inits}
    
    ks = list(range(2, Kmax+1))
    
    options = {'fitting':'WCD', 'km_init':'first'}
    for k in ks:
        n = len(llista_imgs)
        labels = []
        for i in range(n):
            kmeans = KMeans(cropped_images[i], k, options=options)
            kmeans.fit()
            labels.append(get_colors(kmeans.centroids))
        
        acc = Get_color_accuracy(labels, color_labels)
        acc_k.append(acc)
        
    for init in inits:
        options = {'fitting':'WCD', 'km_init':init}
        for k in ks:
            wcds = []
            iters = []
            for img in llista_imgs:
                kmeans = KMeans(img, k, options=options)
                kmeans.fit()
                wcds.append(kmeans.withinClassDistance())
                iters.append(kmeans.num_iter)
            wcds_init[init].append(np.mean(wcds))
            iters_init[init].append(np.mean(iters))
    
    # ACC
    plt.figure()
    plt.plot(ks, acc_k)
    plt.title("K vs Color Accuracy")
    plt.xlabel("K")
    plt.ylabel("Color Accuracy")
    plt.grid(True)
    plt.show()
    
    # WCD
    plt.figure()
    for init in inits:
        plt.plot(ks, wcds_init[init], label=f"Init:{init}")
    plt.title("K vs WCD")
    plt.xlabel("K")
    plt.ylabel("Within-Class Distance")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # ITERS
    plt.figure()
    for init in inits:
        plt.plot(ks, iters_init[init], label=f"Init:{init}")
    plt.title("K vs Nombre d'iteracions")
    plt.xlabel("K")
    plt.ylabel("Iteracions")
    plt.legend()
    plt.grid(True)
    plt.show()


    
def Get_shape_accuracy(result_labels, gt_labels):
    
    """
    Funció que valida les etiquetes obtingudes a una execució del KNN
    Entrada: una llista d'etiquetes resultat del KNN, Ground-Truth per al set d'imatges
    Retorna: el percentatge d'etiquetes correctes
    Possible millora: Aquesta funció pot ser millorada retornant la correspondència d'etiquetes errònies amb 
    les etiquetes correctes en base al Ground-Truth

    """
    result = 0 #Nombre d'etiquetes correctes
    for r_label, gt_label in zip(result_labels, gt_labels):
        if r_label == gt_label:
            result += 1
    return result / len(result_labels)

def Get_color_accuracy(result_labels, gt_labels):
    """
    Funció que valida les etiquetes obtingudes resultat del Kmeans
    Entrada: una llista d'etiquetes resultat del Kmeans, Ground-Truth per al set d'imatges
    Retorna: el percentatge d'etiquetes correctes (revisar sessió de teoria idees per a mesurar la similitud)
    Possible millora: Aquesta funció pot ser millorada retornant la correspondència d'etiquetes errònies 
    amb les etiquetes correctes en base al Ground-Truth
    
    
    En el nostre cas, hem decidit que cal tenir dues mètriques diferents, amb objectiu de medir exactament
    el que volem.
    color_accuracy: Ens indica el percentetge de prediccions encertades amb el color.
    n_colors_accuracy: Ens indica el percentetge de prediccions encertades amb nombre de colors (K).

    
    """
    result = 0
    
    for r_label, gt_label in zip(result_labels, gt_labels):
        r_set = set(r_label)
        gt_set = set(gt_label)
        result += len(r_set & gt_set) / len(r_set | gt_set)

            
    return result/len(result_labels)



def real_vs_predicted_K(cropped_images, color_labels, max_K=10, fitting='WCD'):
    K_real_list = []
    K_pred_list = []

    for i, img in enumerate(cropped_images):
        X = img
        k_real = len(set(color_labels[i]))
        K_real_list.append(k_real)

        options = {'fitting': fitting, 'km_init': 'random'}
        kmeans = KMeans(X, options=options)
        k_pred = kmeans.find_bestK(max_K)
        K_pred_list.append(k_pred)

    plt.figure()
    plt.scatter(K_real_list, K_pred_list)
    plt.plot(range(1, max(K_real_list)+1), range(1, max(K_real_list)+1), 'r--', label='K_pred = K_real')
    plt.xlabel("K real")
    plt.ylabel("K predit")
    plt.title(f"Comparació K real vs predit - Fitting: {fitting}")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    
if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)
    
    #Aqui començen les diverses proves que hem fet:
    
    # === TEST 1: Evaluació Qualitativa Kmeans i KNN ===
    
    # TEST 2.1: Test de filtratge per color

    Kmeans_options = {'fitting':'WCD', 'km_init':'maxdist'}  #Opcions pel fitting de bestK
    n = len(cropped_images)                  #Nombre d'imatges de test que volem analitzar
    
    labels = []
    for i in range(n):
        kmeans = KMeans(cropped_images[i], K = 1, options=Kmeans_options)
        kmeans.find_bestK(10)
        kmeans.fit()
        labels.append(get_colors(kmeans.centroids))
    
    filtered_imgs = Retrieval_by_color(imgs, labels, ['Green', 'Red' ])
    visualize_retrieval(filtered_imgs, 8)
    
    # TEST 2.2: Test de filtratge per forma
    
    knn = KNN(train_imgs, train_class_labels)
    knn_predicts = knn.predict(test_imgs, 3)    #Podem ajustar la K aqui
    
    filtered_imgs = Retrieval_by_shape(test_imgs, knn_predicts, ['Jeans', 'Dress'])
    visualize_retrieval(filtered_imgs, 8)
        
    # TEST 2.3: Test de filtratge per forma i color

    filtered_imgs = Retrieval_combined(test_imgs, labels, knn_predicts, ['Grey'], ['Shirt'])
    visualize_retrieval(filtered_imgs, 4)

    
    # # === TEST 2: Evaluació Quantitativa KNN ===
    # knn = KNN(train_imgs, train_class_labels)
    # knn_predicts = knn.predict(test_imgs, 3)    #Podem ajustar la K aqui
    
    # shape_acc = Get_shape_accuracy(knn_predicts, test_class_labels) #% d'encerts de KNN
    # print (f"[KNN] Encerts en prediccions de forma: {shape_acc * 100:.2f}")
    
    # # === TEST 3: Evaluació Kmeans ===
    
    # # TEST 2.1: Test d'accuracy amb bestK i opcions
    


    # color_acc = Get_color_accuracy(labels, color_labels[0:n])
    # print (f"[Kmeans] Encerts en prediccions de color: {color_acc * 100:.2f}")

    # # TEST 2.2: Recopilació de dades i gràfics sobre Kmeans        
    
    # Kmean_statistics(cropped_images, 11, color_labels, ['first', 'random', 'maxdist'])
    
    # plot_real_vs_predicted_K(cropped_images, color_labels, fitting="WCD")


