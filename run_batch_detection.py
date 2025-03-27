import os
from pathlib import Path
from utilities_yolo11 import YOLODetector
import logging
import json
from datetime import datetime
import argparse
from run_detection import save_results
import pandas as pd

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_video(video_dir: str, args) -> None:
    """
    Traite une vidéo avec YOLO.
    
    Args:
        video_dir (str): Dossier contenant les frames à traiter
        args: Arguments de la ligne de commande
    """
    # Initialiser le détecteur YOLO
    detector = YOLODetector(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device
    )
    
    # Créer le dossier de sortie
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Exécution de la détection
        predictions = detector.predict_batch(
            frames_dir=video_dir,
            batch_size=args.batch_size,
            visualize=args.visualize
        )
        
        # Calcul des statistiques
        stats = detector.get_detection_stats(predictions)
        
        # Affichage des résultats
        logger.info(f"Résultats pour {video_dir}:")
        logger.info(f"Nombre total d'images traitées: {stats['total_images']}")
        logger.info(f"Nombre total de détections: {stats['total_detections']}")
        logger.info(f"Confiance moyenne: {stats['average_confidence']:.2f}")
        logger.info("Détections par classe:")
        for cls, count in stats['class_counts'].items():
            logger.info(f"  Classe {cls}: {count} détections")
        
        # Sauvegarde des résultats
        save_results(predictions, stats, output_dir)
        
        # Sauvegarde des prédictions en CSV
        detector.save_predictions(predictions, output_dir)
        
        logger.info(f"Traitement terminé avec succès pour {video_dir}")
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de {video_dir}: {str(e)}")
        raise

def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(description='Exécute la détection YOLO sur plusieurs vidéos')
    parser.add_argument('--model', type=str, default="yolov8n.pt", help='Chemin vers le modèle YOLO')
    parser.add_argument('--input', type=str, default="preprocessed_frames", help='Dossier parent contenant les dossiers vid1, vid2, etc.')
    parser.add_argument('--output', type=str, default="results", help='Dossier de sortie pour les résultats')
    parser.add_argument('--conf', type=float, default=0.25, help='Seuil de confiance pour la détection')
    parser.add_argument('--iou', type=float, default=0.45, help='Seuil IoU pour la suppression des non-maximum')
    parser.add_argument('--batch-size', type=int, default=8, help='Taille du lot pour le traitement')
    parser.add_argument('--visualize', action='store_true', help='Visualiser les détections sur les frames')
    parser.add_argument('--device', type=str, default=None, help='Device à utiliser (cuda, cpu, ou None pour auto-détection)')
    
    args = parser.parse_args()
    
    # Vérification du répertoire d'entrée
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Le répertoire {args.input} n'existe pas")
        return
    
    # Création du dossier de sortie principal
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Récupération des dossiers de vidéos
    video_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    video_dirs.sort()  # Tri alphabétique pour un traitement ordonné
    
    if not video_dirs:
        logger.error(f"Aucun dossier vidéo trouvé dans {args.input}")
        return
    
    logger.info(f"Trouvé {len(video_dirs)} vidéos à traiter")
    
    # Traitement de chaque vidéo
    for video_dir in video_dirs:
        try:
            video_name = video_dir.name
            output_subdir = output_dir / video_name
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            # Créer une copie des arguments pour ce traitement spécifique
            video_args = argparse.Namespace(**vars(args))
            video_args.output = str(output_subdir)
            
            process_video(str(video_dir), video_args)
            logger.info(f"Traitement terminé pour {video_name}")
        except Exception as e:
            logger.error(f"Échec du traitement de {video_dir}: {str(e)}")
            continue
    
    logger.info("Traitement de toutes les vidéos terminé")

if __name__ == "__main__":
    main() 