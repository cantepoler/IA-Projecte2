__authors__ = '1706732, , '
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

def Kmean_statistics(llista_imgs, Kmax):
    
    """
    Funció que genera un set d'estadístiques d'execució del Kmeans amb 
    la corresponent visualització d'aquestes.
    Entrada: una instància de la classe Kmeans, un conjunt d'imatges i Kmax
    Retorna: la visualització de les estadístiques en format gràfic

    """
    
    glob_wcds = []
    glob_iteracions = []     #Seran llistes de llistes, per a cada kmeans
    glob_temps = []
    ks = list(range(2, Kmax+1))
    for img in llista_imgs:        
        wcds = []
        iteracions = []
        temps = []
        
        for k in ks:
            kmeans = KMeans(img, k, options=options)
            time_start = time.time()
            kmeans.fit()
            time_end = time.time()
            
            temps.append(time_end-time_start)
            wcds.append(kmeans.withinClassDistance())
            iteracions.append(kmeans.num_iter)
        glob_wcds.append(wcds)
        glob_iteracions.append(iteracions)
        glob_temps.append(temps)
            
    mitjana_wcds = np.mean(glob_wcds, axis=0)
    mitjana_iteracions = np.mean(glob_iteracions, axis= 0)
    mitjana_temps = np.mean(glob_temps, axis = 0)
    
    # WCD
    plt.figure()
    plt.plot(ks, mitjana_wcds)
    plt.title('K vs WCD')
    plt.xlabel('K')
    plt.ylabel('Distància Intra-Class (WCD)')
    plt.grid(True)
    plt.show()

    # Iteracions
    plt.figure()
    plt.plot(ks, mitjana_iteracions)
    plt.title('K vs Iteracions')
    plt.xlabel('K')
    plt.ylabel('Nombre iteracions')
    plt.grid(True)
    plt.show()

    # Temps
    plt.figure()
    plt.plot(ks, mitjana_temps)
    plt.title('K vs Temps')
    plt.xlabel('K')
    plt.ylabel('Temps')
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
    color_accuracy = 0
    n_colors_accuracy = 0
    
    for r_label, gt_label in zip(result_labels, gt_labels):
        r_set = set(r_label)
        gt_set = set(gt_label)
        if r_set == gt_set: color_accuracy +=1 #Si exactament els conjunts de colors coincideixen,sumem.
        
        n_colors_accuracy += min(len(r_label), len(gt_label)) / max(len(r_label), len(gt_label))
        
        
    return color_accuracy/len(result_labels), n_colors_accuracy/len(result_labels)

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
    
    # knn = KNN(train_imgs, train_class_labels)
    # knn_predicts = knn.predict(test_imgs, 1)
    
    # print (Get_shape_accuracy(knn_predicts, test_class_labels)) #% d'encerts de KNN
    
    #Test_kmeans:Fisher
    
    Kmeans_options = {'fitting':'Fisher'}  #Opcions pel Kmeans
    n = 1                  #Nombre d'imatges de test que volem analitzar
    
    labels = []
    for i in range(n):
        kmeans = KMeans(train_imgs[i], K = 2, options=Kmeans_options)
        kmeans.find_bestK(5)
        kmeans.fit()
        labels.append(get_colors(kmeans.centroids))

    print(Get_color_accuracy(labels, train_color_labels[0:n]))         

    imatges = Retrieval_by_color(train_imgs[0:n], labels, ['Blue'])
    visualize_retrieval(imatges, 5)
    
    
    options = {'km_init':'random'}
    
    Kmean_statistics(train_imgs[0:2], 4)

