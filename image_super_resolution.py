import numpy as np
from PIL import Image
from ISR.models import RDN


img = Image.open('data/input/test_images/sample_image.jpg')
lr_img = np.array(img)


rdn = RDN(weights='psnr-small')
sr_img = rdn.predict(lr_img)
Image.fromarray(sr_img)