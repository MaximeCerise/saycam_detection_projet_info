import os
import cv2
import numpy as np
from pathlib import Path
import argparse

class ImagePreprocessor:
    def __init__(self, input_dir, output_dir):
        """
        Initialise le preprocesseur d'images.
        
        Args:
            input_dir (str): Répertoire contenant les images d'entrée
            output_dir (str): Répertoire de sortie pour les images préprocessées
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Créer le répertoire de sortie s'il n'existe pas
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def rotate_image(self, image):
        """
        Fait pivoter l'image de 180 degrés.
        
        Args:
            image (numpy.ndarray): Image d'entrée
        
        Returns:
            numpy.ndarray: Image pivotée
        """
        return cv2.rotate(image, cv2.ROTATE_180)
    
    def enhance_contrast(self, image):
        """
        Améliore le contraste de l'image.
        
        Args:
            image (numpy.ndarray): Image d'entrée
        
        Returns:
            numpy.ndarray: Image avec contraste amélioré
        """
        # Utiliser l'égalisation d'histogramme adaptatif
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        
        # S'assurer que l'image est en niveaux de gris si nécessaire
        if len(image.shape) > 2:
            image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            image_yuv[:,:,0] = clahe.apply(image_yuv[:,:,0])
            image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)
        else:
            image = clahe.apply(image)
        
        return image
    
    def denoise(self, image):
        """
        Réduit le bruit de l'image.
        
        Args:
            image (numpy.ndarray): Image d'entrée
        
        Returns:
            numpy.ndarray: Image débruitée
        """
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def simple_super_resolve(self, image):
        """
        Applique une super-résolution simple en utilisant le redimensionnement bicubique.
        
        Args:
            image (numpy.ndarray): Image d'entrée
        
        Returns:
            numpy.ndarray: Image agrandie
        """
        # Augmenter la résolution d'un facteur 2
        try:
            super_resolved = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            return super_resolved
        except Exception as e:
            print(f"Erreur lors de la super-résolution : {e}")
            return image
    
    def apply_gaussian_blur(self, image):
        """
        Applique un flou gaussien à une région spécifique de l'image.
        
        Args:
            image (numpy.ndarray): Image d'entrée
        
        Returns:
            numpy.ndarray: Image avec région floue
        """
        # Copie de l'image pour ne pas modifier l'originale
        blurred_image = image.copy()
        
        # Région à flouter (x de 40 à 540, y de 30 à 100)
        region = blurred_image[15:100, 40:590]
        
        # Appliquer un flou gaussien à la région
        # Le noyau doit être impair (ici 21x21)
        blurred_region = cv2.GaussianBlur(region, (101, 101), 0)
        
        # Remplacer la région originale par la version floutée
        blurred_image[15:100, 40:590] = blurred_region
        
        return blurred_image
    
    def preprocess_image(self, image_path):
        """
        Effectue le preprocessing complet sur une image.
        
        Args:
            image_path (Path): Chemin vers l'image
        
        Returns:
            numpy.ndarray: Image préprocessée
        """
        # Charger l'image
        image = cv2.imread(str(image_path))
        
        # Rotation
        image = self.rotate_image(image)
        
        # Amélioration du contraste
        image = self.enhance_contrast(image)
        
        # Réduction du bruit
        image = self.denoise(image)
        
        # Super-résolution simple
        #image = self.simple_super_resolve(image)
        
        # Flou gaussien sur la région spécifiée
        #image = self.apply_gaussian_blur(image)
        
        return image
    
    def process_directory(self):
        """
        Traite toutes les images du répertoire d'entrée.
        """
        # Parcourir tous les fichiers image
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        processed_count = 0
        
        for image_file in sorted(self.input_dir.glob('*')):
            if image_file.suffix.lower() in image_extensions:
                try:
                    # Préprocesser l'image
                    processed_image = self.preprocess_image(image_file)
                    
                    # Sauvegarder l'image préprocessée
                    output_path = self.output_dir / image_file.name
                    cv2.imwrite(str(output_path), processed_image)
                    
                    processed_count += 1
                    print(f"Traitement de {image_file.name} terminé")
                
                except Exception as e:
                    print(f"Erreur lors du traitement de {image_file.name}: {e}")
        
        print(f"\nTotal d'images préprocessées : {processed_count}")

def main():
    # Configuration de l'argument parser
    parser = argparse.ArgumentParser(description='Preprocessing d\'images pour détection d\'objets')
    parser.add_argument('input_dir', type=str, help='Répertoire contenant les images d\'entrée')
    parser.add_argument('output_dir', type=str, help='Répertoire de sortie pour les images préprocessées')
    
    args = parser.parse_args()
    
    # Créer et exécuter le preprocesseur
    preprocessor = ImagePreprocessor(args.input_dir, args.output_dir)
    preprocessor.process_directory()

if __name__ == '__main__':
    main()