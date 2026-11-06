from src.image import HistogramImage, LBTImage, LCVBPImage
from src.metrics import bhattacharyya, pearson_correlation, chi_square
from pathlib import Path

def main():
    folder = Path('images')
    image = folder / '1.jpg'
    base_image = LBTImage(image, chi_square)
    print(image.name)
    for other in folder.iterdir():
        if other.suffix == '.jpg':
            other_image = LBTImage(other, chi_square)
            value = base_image.compare(other_image)
            print(f'{other.name} : {value}')


if __name__ == "__main__":
    main()
