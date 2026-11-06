from __future__ import annotations
import cv2
import numpy as np

class Image:
    def __init__(self, path):
        self.image = cv2.imread(path)
        self.path = path
        if self.image is None:
            raise ValueError(f"Nie udało się wczytać obrazu ze ścieżki: {path}")

    def compare(self, other):
        raise NotImplementedError()


class HistogramImage(Image):
    def __init__(self, path, comparator):
        super().__init__(path)
        self.hist = None
        self.comparator = comparator

    def histogram(self):
        if self.hist is None:
            hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            self.hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256]).flatten()
            cv2.normalize(self.hist, self.hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
        return self.hist

    def compare(self, other):
        other_histogram = other.histogram()
        score = self.comparator(self.histogram(), other_histogram)
        return score


class LBTImage(Image):
    def __init__(self, path, comparator, R=1, P=8, resize=(256, 256), field=(16, 16)):
        super().__init__(path)
        self.lbt_hist = None
        self.comparator = comparator
        self.R = R
        self.P = P
        if (resize[0] % field[0] != 0) or (resize[1] % field[1] != 0):
            raise ValueError()
        self.resize = resize
        self.field = field

    def lbt(self):
        if self.lbt_hist is None:
            # 1. Konwersja do odcieni szarości i skalowanie do stałego wymiaru
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, self.resize)
            H, W = gray.shape

            # 2. Wyznaczenie mapy LBP dla całego obrazu
            center = gray.astype(np.float32)
            lbp_map = np.zeros((H, W), dtype=np.uint64)

            xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            for p in range(self.P):
                angle = 2.0 * np.pi * p / self.P
                map_x = xx + self.R * np.cos(angle)
                map_y = yy - self.R * np.sin(angle)
                
                # Połączenie współrzędnych w jedną mapę typu CV_32FC2 wymaganą przez OpenCV
                map_xy = np.stack([map_x, map_y], axis=-1).astype(np.float32)
                
                # Próbkowanie z uwzględnieniem podpikseli (interpolacja dwuliniowa)
                sampled = cv2.remap(center, map_xy, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                
                bit = (sampled >= center).astype(np.uint64)
                lbp_map |= (bit << p)

            # 3 & 4. Podział na segmenty i liczenie histogramów lokalnych
            fh, fw = self.field
            histograms = []
            num_bins = 2 ** self.P
            
            for y in range(0, H, fh):
                for x in range(0, W, fw):
                    block = lbp_map[y:y+fh, x:x+fw]
                    hist, _ = np.histogram(block.ravel(), bins=num_bins, range=(0, num_bins))
                    histograms.append(hist)

            # 5. Konkatenacja histogramów oraz normalizacja
            self.lbt_hist = np.concatenate(histograms).astype(np.float32)
            norm_sum = np.sum(self.lbt_hist)
            if norm_sum > 0:
                self.lbt_hist /= norm_sum

        return self.lbt_hist

    def compare(self, other):
        other_histogram = other.lbt()
        score = self.comparator(self.lbt(), other_histogram)
        return score

        
class LCVBPImage(Image):
    def __init__(self, path, comparator, R=1, P=8, resize=(512, 512), field=(16, 16)):
        super().__init__(path)
        self.lcvbp_hist = None
        self.comparator = comparator
        self.R = R
        self.P = P
        if (resize[0] % field[0] != 0) or (resize[1] % field[1] != 0):
            raise ValueError("Wymiary obrazu muszą być podzielne przez wymiary pola (segmentu).")
        self.resize = resize
        self.field = field

    def _compute_lbp_map(self, channel_map, H, W):
        """Pomocnicza metoda wyznaczająca mapę LBP dla danej mapy cech (np. normy lub stosunku kanałów)"""
        center = channel_map
        lbp_map = np.zeros((H, W), dtype=np.uint64)
        xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        
        for p in range(self.P):
            angle = 2.0 * np.pi * p / self.P
            map_x = xx + self.R * np.cos(angle)
            map_y = yy - self.R * np.sin(angle)
            
            # Pakowanie współrzędnych do formatu CV_32FC2 wymaganego przez cv2.remap
            map_xy = np.stack([map_x, map_y], axis=-1).astype(np.float32)
            
            sampled = cv2.remap(center, map_xy, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            bit = (sampled >= center).astype(np.uint64)
            lbp_map |= (bit << p)
            
        return lbp_map

    def lcvbp(self):
        if self.lcvbp_hist is None:
            # 1. Wczytanie i skalowanie obrazu kolorowego
            img_resized = cv2.resize(self.image, self.resize).astype(np.float32)
            H, W, _ = img_resized.shape
            B, G, R = cv2.split(img_resized)

            # 2. Color Norm Pattern (CNP) - norma wektora koloru
            norm_map = np.sqrt(B**2 + G**2 + R**2)
            cnp_lbp = self._compute_lbp_map(norm_map, H, W)

            # 3. Color Angular / Ratio Pattern (CAP) - relacje między kanałami (odporne na oświetlenie)
            eps = 1e-5
            ratio1 = R / (G + eps)
            ratio2 = G / (B + eps)
            ratio3 = B / (R + eps)
            
            cap_lbp1 = self._compute_lbp_map(ratio1, H, W)
            cap_lbp2 = self._compute_lbp_map(ratio2, H, W)
            cap_lbp3 = self._compute_lbp_map(ratio3, H, W)

            # Wszystkie wygenerowane mapy LBP (CNP + 3x CAP)
            all_maps = [cnp_lbp, cap_lbp1, cap_lbp2, cap_lbp3]

            # 4 & 5. Podział na segmenty, liczenie histogramów i konkatenacja
            fh, fw = self.field
            num_bins = 2 ** self.P
            histograms = []

            for lbp_map in all_maps:
                for y in range(0, H, fh):
                    for x in range(0, W, fw):
                        block = lbp_map[y:y+fh, x:x+fw]
                        hist, _ = np.histogram(block.ravel(), bins=num_bins, range=(0, num_bins))
                        histograms.append(hist)

            # Połączenie wszystkich histogramów w jeden spójny wektor cech
            self.lcvbp_hist = np.concatenate(histograms).astype(np.float32)
            
            # Normalizacja wektora
            norm = np.linalg.norm(self.lcvbp_hist)
            if norm > 0:
                self.lcvbp_hist /= norm

        return self.lcvbp_hist

    def compare(self, other):
        other_histogram = other.lcvbp()
        score = self.comparator(self.lcvbp(), other_histogram)
        return score