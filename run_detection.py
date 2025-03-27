import os
from pathlib import Path
from utilities_yolo11 import YOLODetector
import logging
import json
from datetime import datetime
import argparse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def save_results(predictions, stats, output_dir):
    """
    Sauvegarde les résultats de la détection.
    
    Args:
        predictions: Liste des prédictions
        stats: Statistiques des détections
        output_dir: Répertoire de sortie
    """
    # Création du répertoire de sortie
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarde des statistiques
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_file = output_dir / f"detection_stats_{timestamp}.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    logger.info(f"Statistiques sauvegardées dans {stats_file}")

def main():
    # Configuration des arguments
    parser = argparse.ArgumentParser(description='Exécute la détection YOLO sur un dossier de frames')
    parser.add_argument('--model', type=str, default="yolov8n.pt", help='Chemin vers le modèle YOLO')
    parser.add_argument('--input', type=str, default="preprocessed_frames/vid1", help='Dossier d\'entrée contenant les frames')
    parser.add_argument('--output', type=str, default="results/vid1", help='Dossier de sortie pour les résultats')
    parser.add_argument('--conf', type=float, default=0.25, help='Seuil de confiance pour la détection')
    parser.add_argument('--iou', type=float, default=0.45, help='Seuil IoU pour la suppression des non-maximum')
    parser.add_argument('--batch-size', type=int, default=8, help='Taille du lot pour le traitement')
    parser.add_argument('--visualize', action='store_true', help='Visualiser les détections sur les frames')
    
    args = parser.parse_args()
    
    # Vérification du répertoire d'entrée
    if not os.path.exists(args.input):
        logger.error(f"Le répertoire {args.input} n'existe pas")
        return
    
    try:
        # Initialisation du détecteur avec des paramètres optimisés
        detector = YOLODetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou
        )
        
        logger.info(f"Démarrage de la détection sur {args.input}")
        logger.info(f"Paramètres: conf={args.conf}, iou={args.iou}, batch_size={args.batch_size}")
        
        # Exécution de la détection
        predictions = detector.predict_batch(
            frames_dir=args.input,
            batch_size=args.batch_size,
            visualize=args.visualize
        )
        
        # Calcul des statistiques
        stats = detector.get_detection_stats(predictions)
        
        # Affichage des résultats
        logger.info("Résultats de la détection:")
        logger.info(f"Nombre total d'images traitées: {stats['total_images']}")
        logger.info(f"Nombre total de détections: {stats['total_detections']}")
        logger.info(f"Confiance moyenne: {stats['average_confidence']:.2f}")
        logger.info("Détections par classe:")
        for cls, count in stats['class_counts'].items():
            logger.info(f"  Classe {cls}: {count} détections")
        
        # Sauvegarde des résultats
        save_results(predictions, stats, args.output)
        
        # Sauvegarde des prédictions en CSV
        detector.save_predictions(predictions, args.output)
        
        logger.info("Traitement terminé avec succès")
        
    except Exception as e:
        logger.error(f"Une erreur est survenue: {str(e)}")
        raise

if __name__ == "__main__":
    main() 