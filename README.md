# Pipeline de Traitement Vidéo et Détection d'Objets

Ce projet implémente une pipeline complète de traitement vidéo et de détection d'objets, allant de l'extraction des frames jusqu'à la détection d'objets avec YOLO.

## 🚀 Fonctionnalités

- Extraction de frames depuis des vidéos
- Prétraitement avancé des images :
  - Rotation automatique
  - Amélioration du contraste
  - Réduction du bruit
  - Super-résolution
  - Amélioration des détails
- Détection d'objets avec YOLOv8
- Traitement par lots optimisé
- Visualisation des résultats

## 📋 Prérequis

- Python 3.8 ou supérieur
- CUDA compatible GPU (recommandé pour la détection YOLO)

## 🛠️ Installation

1. Cloner le repository :
```bash
git clone https://github.com/MaximeCerise/saycam_detection_projet_info.git
cd saycam_detection_projet_info
```

2. Créer et activer un environnement virtuel :
```bash
# Création de l'environnement virtuel
python -m venv venv

# Activation de l'environnement virtuel
# Sur Windows :
venv\Scripts\activate
# Sur macOS/Linux :
source venv/bin/activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

Mettre les vidéos dans 
## 📁 Structure du Projet

Structure du projet :
```
├── vid/ # Dossier contenant les vidéos source
├── extracted_frames/ # Frames extraites des vidéos
├── preprocessed_frames/ # Frames après prétraitement
├── results/ # Résultats de la détection
├── scripts/
│ ├── extract_frames.py # Extraction des frames
│ ├── preprocessing.py # Prétraitement des images
│ ├── batch_preprocessing.py # Traitement par lots
│ ├── run_detection.py # Détection YOLO
│ └── run.py # Pipeline complet
└── requirements.txt # Dépendances du projet
```

## 🎯 Utilisation

### Pipeline Complet

Pour exécuter la pipeline complète (extraction, prétraitement et détection) :

```bash
python run.py \
    --video-dir chemin/vers/videos \
    --output output \
    --model yolov8x.pt \
    --conf 0.3 \
    --iou 0.5 \
    --batch-size 8 \
    --visualize \
    --device cuda \
    --fps 10
```

### Extraction et Prétraitement Seuls

Pour extraire et prétraiter les frames sans faire la détection :

```bash
python batch_preprocessing.py \
    chemin/vers/videos \
    output \
    --fps 10
```

### Détection Seule

Pour exécuter uniquement la détection sur des frames déjà prétraitées :

```bash
python run_detection.py \
    --input output/preprocessed_frames/vid* \
    --output results/vid* \
    --model yolov8x.pt \
    --conf 0.3 \
    --iou 0.5 \
    --batch-size 8 \
    --device cuda
```

## ⚙️ Paramètres

### Pipeline Complète (run.py)
- `--video-dir` : Dossier contenant les vidéos à traiter
- `--output` : Dossier de sortie principal
- `--model` : Chemin vers le modèle YOLO (défaut: yolov8x.pt)
- `--conf` : Seuil de confiance pour la détection (défaut: 0.3)
- `--iou` : Seuil IoU pour la suppression des non-maximum (défaut: 0.5)
- `--batch-size` : Taille du lot pour le traitement (défaut: 8)
- `--visualize` : Visualiser les détections
- `--device` : Device à utiliser (cuda, cpu, ou None pour auto-détection)
- `--fps` : Nombre de frames par seconde à extraire (défaut: 30)

### Prétraitement (batch_preprocessing.py)
- `input_dir` : Dossier contenant les vidéos ou les frames
- `output_dir` : Dossier de sortie principal
- `--fps` : Nombre de frames par seconde à extraire

### Détection (run_detection.py)
- `--input` : Dossier contenant les frames à traiter
- `--output` : Dossier de sortie pour les résultats
- `--model` : Chemin vers le modèle YOLO
- `--conf` : Seuil de confiance
- `--iou` : Seuil IoU
- `--batch-size` : Taille du lot
- `--visualize` : Visualiser les détections
- `--device` : Device à utiliser

## 🔍 Résultats

Les résultats sont organisés dans la structure suivante :
```
output/
├── extracted_frames/ # Frames extraites
│ └── video1/
├── preprocessed_frames/ # Frames prétraitées
│ └── video1/
└── results/ # Résultats de la détection
└── video1/
├── detected_frame_1.jpg
├── detected_frame_2.jpg
└── detections.csv
```
### Visualisation des Résultats

Pour visualiser les résultats de la détection de manière interactive :

```bash
python run_viz.py \
    --data output/results/vid4/predictions_XXXXXXX_XXXXXX.csv \
    --video output/result_viz/vid4/
```

Les arguments sont :
- `--data` ou `-d` : Chemin vers le fichier CSV contenant les résultats de détection (généralement dans `output/results/vid*/predictions_XXXXXXX_XXXXXX.csv.csv`)
- `--video` ou `-v` : Chemin vers le dossier contenant les images de visualisation (généralement dans `output/result_viz/vid*/`)

La visualisation s'ouvrira automatiquement dans votre navigateur par défaut. Elle permet de :
- Voir les trajectoires des objets détectés
- Naviguer dans le temps avec un slider
- Cliquer sur les points pour voir l'image correspondante
- Jouer/pause l'animation

Note : Assurez-vous que les chemins spécifiés correspondent bien à la même vidéo (même numéro dans `vid*`).

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## 👥 Auteurs

- MOREAU Maxime - [@MaximeCerise](https://github.com/MaximeCerise)
- JANATI Mehdi -
- TSAGUE Yannick -
