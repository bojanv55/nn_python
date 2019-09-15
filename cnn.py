import matplotlib.pyplot as plt
import imageio

im = imageio.imread("./Figures/chapel.png")
print(im.shape)
plt.imshow(im)

im_red = im[:,:,0]
print(im_red.shape)
plt.imshow(im_red, cmap='gray')

im_blue = im[:,:,2]
plt.imshow(im_blue, cmap='gray')

plt.show()