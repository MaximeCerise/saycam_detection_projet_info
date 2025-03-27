Objectif du projet : Il s'agit d'un projet de traitement vidéo qui semble se concentrer sur le traitement d'images et potentiellement la détection d'objets.
Structure du projet :
vid/ : Contient probablement les vidéos source
extracted_frames/ : Stocke les images extraites des vidéos
preprocessed_frames/ : Contient les images après prétraitement
Plusieurs scripts Python pour le traitement
Fonctionnalités principales :
Extraction d'images depuis des vidéos (extract_frames.py)
Prétraitement des images avec plusieurs opérations :
Rotation d'images
Amélioration du contraste
Réduction du bruit
Super-résolution simple
Application de flou gaussien
Traitement par lots (batch_preprocessing.py)
Technologies utilisées :
OpenCV pour le traitement d'images
PyTorch et TorchVision pour le deep learning
Ultralytics (probablement pour YOLO)
Keras pour le deep learning
Matplotlib et Plotly pour la visualisation
Pandas pour la manipulation de données
Approche méthodologique :
Le projet suit une approche en pipeline :
Extraction des frames depuis les vidéos
Prétraitement des images
Potentiellement une étape de détection d'objets (vu l'utilisation d'Ultralytics)
Il semble que ce projet soit lié à la vision par ordinateur, possiblement pour la détection d'objets ou l'analyse vidéo, avec un accent particulier sur le prétraitement des images pour améliorer leur qualité ou les préparer pour une analyse ultérieure.





Structure orientée objet :
Création d'une classe YOLODetector pour une meilleure organisation
Encapsulation des paramètres et méthodes
Paramètres de détection optimisés :
conf_threshold = 0.25 : Seuil de confiance adapté aux images prétraitées
iou_threshold = 0.45 : Seuil IoU pour la suppression des non-maximum
agnostic_nms = True : NMS agnostique aux classes pour une meilleure gestion des chevauchements
max_det = 100 : Limite le nombre de détections par image
Traitement par lots :
Ajout du traitement par lots (batch_size = 8)
Optimisation des performances avec le traitement en parallèle
Meilleure gestion de la mémoire
Statistiques de détection :
Nouvelle méthode get_detection_stats
Suivi du nombre total de détections
Calcul de la confiance moyenne
Comptage des détections par classe
Gestion des erreurs et logging :
Logging structuré pour un meilleur suivi
Gestion des exceptions pour chaque image
Continuation du traitement même en cas d'échec
Optimisations techniques :
Utilisation de pathlib pour une meilleure gestion des chemins
Support des images numpy et des chemins de fichiers
Typage statique pour une meilleure maintenabilité
Pour utiliser ce code amélioré :



Pour utiliser le script de détection :

python run_detection.py \
    --visualize \
    --model yolov8n.pt \
    --input preprocessed_frames/vid1 \
    --output results/vid1 \
    --conf 0.25 \
    --iou 0.45 \
    --batch-size 8