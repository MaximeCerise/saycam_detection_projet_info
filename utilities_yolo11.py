import torch
import os
from glob import glob
from typing import List, Union, Dict, Any
import pandas as pd
from pathlib import Path
import logging
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import cv2
import shutil

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25, iou_threshold: float = 0.45, device: str = None):
        """
        Initialise le détecteur YOLO avec des paramètres optimisés.
        
        Args:
            model_path (str): Chemin vers le modèle YOLO
            conf_threshold (float): Seuil de confiance pour la détection (0.25 par défaut)
            iou_threshold (float): Seuil IoU pour la suppression des non-maximum (0.45 par défaut)
            device (str): Device à utiliser ('cuda', 'cpu', ou None pour auto-détection)
        """
        # Détection automatique du device si non spécifié
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = device
        logger.info(f"Utilisation du device: {self.device}")
        
        # Initialisation du modèle avec le device spécifié
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Configuration optimisée pour les images prétraitées
        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        self.model.agnostic_nms = True  # NMS agnostique aux classes
        self.model.max_det = 100  # Nombre maximum de détections par image
        
    def predict_single(self, img: Union[str, np.ndarray]) -> pd.DataFrame:
        """
        Effectue une prédiction YOLO sur une image avec des paramètres optimisés.
        
        Args:
            img (Union[str, np.ndarray]): Chemin de l'image ou image numpy
            
        Returns:
            pd.DataFrame: Résultats de la prédiction
        """
        try:
            results = self.model(img, verbose=False)[0]  # On prend le premier résultat car on traite une seule image
            if len(results.boxes) == 0:
                return pd.DataFrame()  # Retourne un DataFrame vide si aucune détection
            
            # Extraction du numéro de frame depuis le nom du fichier
            frame_number = 0
            if isinstance(img, str):
                try:
                    frame_number = int(Path(img).stem.split('_')[1])
                except:
                    frame_number = 0
            
            # Conversion des résultats en DataFrame
            boxes = results.boxes
            data = []
            for box in boxes:
                # Conversion des tenseurs en numpy avec gestion du device
                x1, y1, x2, y2 = map(float, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())
                # Récupération du nom de la classe
                cls_name = results.names[cls]
                data.append({
                    'frame': frame_number,
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2,
                    'confidence': conf,
                    'class': cls,
                    'name': cls_name
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction YOLO: {str(e)}")
            raise

    def get_path_frames(self, frames_dir: Union[str, Path]) -> List[str]:
        """
        Récupère tous les chemins d'images dans un répertoire.
        
        Args:
            frames_dir (Union[str, Path]): Chemin du répertoire
            
        Returns:
            List[str]: Liste des chemins d'images
        """
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Le répertoire {frames_dir} n'existe pas")
            
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif")
        images = []
        for ext in extensions:
            images.extend(glob(str(frames_dir / ext)))
        
        if not images:
            logger.warning(f"Aucune image trouvée dans {frames_dir}")
        else:
            logger.info(f"Trouvé {len(images)} images dans {frames_dir}")
            
        return images

    def visualize_detection(self, img_path: str, predictions: pd.DataFrame, output_path: str) -> None:
        """
        Visualise les détections sur une image.
        
        Args:
            img_path (str): Chemin de l'image source
            predictions (pd.DataFrame): Prédictions pour cette image
            output_path (str): Chemin de sortie pour l'image annotée
        """
        # Lire l'image
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Impossible de lire l'image: {img_path}")
            return

        # Dessiner chaque détection
        for _, row in predictions.iterrows():
            # Coordonnées de la boîte
            x1, y1, x2, y2 = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            
            # Couleur de la boîte (rouge par défaut)
            color = (0, 0, 255)
            
            # Dessiner la boîte
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Texte à afficher (classe et confiance)
            label = f"{row['name']} {row['confidence']:.2f}"
            
            # Taille du texte
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Position du texte
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            text_x = x1
            text_y = y1 - 5
            
            # Dessiner le fond du texte
            cv2.rectangle(img, (text_x, text_y - text_height - 5),
                         (text_x + text_width, text_y + 5), color, -1)
            
            # Dessiner le texte
            cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Sauvegarder l'image
        cv2.imwrite(output_path, img)
        logger.info(f"Image annotée sauvegardée: {output_path}")

    def predict_batch(self, frames_dir: Union[str, Path], batch_size: int = 8, visualize: bool = False) -> List[pd.DataFrame]:
        """
        Effectue des prédictions YOLO sur un lot d'images.
        
        Args:
            frames_dir (Union[str, Path]): Répertoire des images
            batch_size (int): Taille du lot pour le traitement
            visualize (bool): Si True, visualise les détections
            
        Returns:
            List[pd.DataFrame]: Liste des prédictions
        """
        frames = self.get_path_frames(frames_dir)
        predictions = []
        
        # Créer le dossier de visualisation si nécessaire
        if visualize:
            # Créer le dossier result_viz/vidX
            frames_path = Path(frames_dir)
            vid_name = frames_path.name  # Récupère le nom de la vidéo (vid1, vid2, etc.)
            viz_dir = frames_path.parent.parent / "result_viz" / vid_name
            viz_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Dossier de visualisation créé: {viz_dir}")
        
        # Ajuster la taille du batch en fonction du device
        if self.device == 'cuda':
            # Utiliser une taille de batch plus grande sur GPU
            batch_size = min(batch_size * 2, 32)
            logger.info(f"Taille de batch ajustée pour GPU: {batch_size}")
        
        # Traitement par lots pour optimiser les performances
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            logger.info(f"Traitement du lot {i//batch_size + 1}/{len(frames)//batch_size + 1}")
            
            for frame in batch:
                try:
                    pred = self.predict_single(frame)
                    predictions.append(pred)
                    
                    # Visualiser les détections si demandé
                    if visualize and not pred.empty:
                        frame_name = Path(frame).name
                        output_path = viz_dir / f"viz_{frame_name}"
                        self.visualize_detection(frame, pred, str(output_path))
                    
                    logger.info(f"Prédiction réussie pour: {frame}")
                except Exception as e:
                    logger.error(f"Échec de la prédiction pour {frame}: {str(e)}")
                    continue
                    
        return predictions

    def get_detection_stats(self, predictions: List[pd.DataFrame]) -> Dict[str, Any]:
        """
        Calcule des statistiques sur les détections.
        
        Args:
            predictions (List[pd.DataFrame]): Liste des prédictions
            
        Returns:
            Dict[str, Any]: Statistiques des détections
        """
        stats = {
            'total_detections': 0,
            'average_confidence': 0.0,
            'class_counts': {},
            'total_images': len(predictions)
        }
        
        confidences = []
        for pred in predictions:
            if not pred.empty:
                stats['total_detections'] += len(pred)
                confidences.extend(pred['confidence'].values)
                for cls in pred['class'].unique():
                    # Conversion du type numpy en int Python natif
                    cls_int = int(cls)
                    stats['class_counts'][str(cls_int)] = stats['class_counts'].get(str(cls_int), 0) + 1
        
        if confidences:
            stats['average_confidence'] = float(np.mean(confidences))
            
        return stats

    def save_predictions(self, predictions: List[pd.DataFrame], output_dir: Union[str, Path]) -> None:
        """
        Sauvegarde les prédictions dans un fichier CSV.
        
        Args:
            predictions (List[pd.DataFrame]): Liste des prédictions
            output_dir (Union[str, Path]): Répertoire de sortie
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Concaténer toutes les prédictions
        all_predictions = pd.concat(predictions, ignore_index=True)
        
        # Trier par numéro de frame
        all_predictions = all_predictions.sort_values('frame')
        
        # Sauvegarder en CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"predictions_{timestamp}.csv"
        all_predictions.to_csv(output_file, index=False)
        logger.info(f"Prédictions sauvegardées dans {output_file}")

def main():
    # Exemple d'utilisation
    model_path = "yolov8n.pt"  # Remplacer par le chemin de votre modèle
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=0.25,  # Seuil de confiance adapté aux images prétraitées
        iou_threshold=0.45    # Seuil IoU pour la suppression des non-maximum
    )
    
    frames_dir = "preprocessed_frames"  # Répertoire des images prétraitées
    predictions = detector.predict_batch(frames_dir)
    stats = detector.get_detection_stats(predictions)
    
    logger.info(f"Statistiques des détections: {stats}")

if __name__ == "__main__":
    main()
