import matplotlib.image as mp_image

filename="21765246_118515978866530_3171170106726421284_n.jpg"

input_img = mp_image.imread(filename)

print ('input dim={}'.format(input_img.ndim))
print ('input shape={}'.format(input_img.shape))

import matplotlib.pyplot as plt

plt.imshow(input_img)
plt.show()