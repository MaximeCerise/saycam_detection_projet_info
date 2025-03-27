import os
from pathlib import Path
import logging
import argparse
from datetime import datetime
import subprocess
from utilities_yolo11 import YOLODetector
from run_batch_detection import process_video

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def preprocess_frames(input_dir: str, output_dir: str, fps: int = 30) -> None:
    """
    Prétraite les frames extraites.
    
    Args:
        input_dir (str): Dossier contenant les frames brutes
        output_dir (str): Dossier de sortie pour les frames prétraitées
        fps (int): Nombre de frames par seconde à extraire
    """
    logger.info(f"Prétraitement des frames de {input_dir}")
    try:
        subprocess.run([
            'python', 'batch_preprocessing.py',
            input_dir,
            output_dir,
            '--fps', str(fps)
        ], check=True)
        logger.info(f"Prétraitement des frames terminé dans {output_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors du prétraitement des frames: {str(e)}")
        raise

def process_pipeline(args) -> None:
    """
    Exécute la pipeline complète de traitement.
    
    Args:
        args: Arguments de la ligne de commande
    """
    # Création des dossiers nécessaires
    base_dir = Path(args.output)
    preprocessed_dir = base_dir / "preprocessed_frames"
    results_dir = base_dir / "results"
    
    for dir_path in [preprocessed_dir, results_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Prétraitement des frames (inclut l'extraction)
    preprocess_frames(
        input_dir=args.video_dir,
        output_dir=str(base_dir),  # On passe le dossier de base pour que batch_preprocessing.py crée extracted_frames et preprocessed_frames
        fps=args.fps
    )
    
    # 2. Détection YOLO pour chaque sous-dossier de frames prétraitées
    for video_dir in preprocessed_dir.iterdir():
        if video_dir.is_dir():
            video_name = video_dir.name
            output_subdir = results_dir / video_name
            output_subdir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Traitement de la vidéo : {video_name}")
            try:
                # Créer une copie des arguments pour ce traitement spécifique
                video_args = argparse.Namespace(**vars(args))
                video_args.output = str(output_subdir)
                
                # Traiter la vidéo
                process_video(
                    video_dir=str(video_dir),
                    args=video_args
                )
                logger.info(f"Traitement terminé pour {video_name}")
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {video_name}: {e}")
    
    logger.info("Pipeline terminée avec succès")

def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(description='Pipeline complète de traitement vidéo')
    
    # Arguments pour les vidéos
    parser.add_argument('--video-dir', type=str, required=True, help='Dossier contenant les vidéos à traiter')
    parser.add_argument('--fps', type=int, default=30, help='Nombre de frames par seconde à extraire')
    
    # Arguments pour la détection YOLO
    parser.add_argument('--model', type=str, default="yolov8x.pt", help='Chemin vers le modèle YOLO')
    parser.add_argument('--conf', type=float, default=0.3, help='Seuil de confiance pour la détection')
    parser.add_argument('--iou', type=float, default=0.5, help='Seuil IoU pour la suppression des non-maximum')
    parser.add_argument('--batch-size', type=int, default=8, help='Taille du lot pour le traitement')
    parser.add_argument('--visualize', action='store_true', help='Visualiser les détections sur les frames')
    parser.add_argument('--device', type=str, default=None, help='Device à utiliser (cuda, cpu, ou None pour auto-détection)')
    
    # Arguments pour les chemins
    parser.add_argument('--output', type=str, default="output", help='Dossier de sortie principal')
    
    args = parser.parse_args()
    
    # Vérification du dossier de vidéos
    if not os.path.exists(args.video_dir):
        logger.error(f"Le dossier {args.video_dir} n'existe pas")
        return
    
    # Exécution de la pipeline
    process_pipeline(args)
    
    logger.info("Pipeline terminée avec succès")

if __name__ == "__main__":
    main() 