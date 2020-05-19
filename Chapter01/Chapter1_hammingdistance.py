from PIL import Image
import imagehash
from scipy.spatial import distance
hash1 = imagehash.phash(Image.open('car2.png'))
hash2 = imagehash.phash(Image.open('car.png'))
print(distance.hamming(hash1, hash2))
