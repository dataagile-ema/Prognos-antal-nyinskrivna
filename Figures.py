import glob
import imageio
import os

def delete_all_figures():
    for filename in glob.glob('Figures model/*.png'):
        os.remove(filename)
    for filename in glob.glob('Figures naive/*.png'):
        os.remove(filename)

def save_gif():
    images = []
    for filename in glob.glob('figures model/*.png'):
        images.append(imageio.imread(filename))
    # add a delay between each image
    imageio.mimsave('Animations/animation model.gif', images, duration=0.5)

    images = []
    for filename in glob.glob('figures naive/*.png'):
        images.append(imageio.imread(filename))
    # add a delay between each image
    imageio.mimsave('Animations/animation naive.gif', images, duration=0.5)




