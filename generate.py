#%%imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json

#%% load json and create model
json_file = open('models\\discriminator_CGAN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models\\discriminator_CGAN_model.h5")
print("Loaded model from disk")

json_file = open('models\\generator_CGAN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models\\generator_CGAN_model.h5")
print("Model is loaded")

#%%generate new objects
class_map = {0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal", 6:"Shirt", 7:"Sneaker",8:"Bag",9:"Ankle boot"}

row, column = 1, 10
size = 20

while(True):
    noise = np.random.normal(0, 1, (row * column, 100))
    sampled_labels = np.arange(0, 10).reshape(-1, 1)
    generator_images = loaded_model.predict([noise, sampled_labels]) 
    generator_images = 0.5 * generator_images + 0.5
    
    fig, axs = plt.subplots(row, column,figsize=(row*size,column*size))
    cnt = 0
    for i in range(row):
        for j in range(column):
            axs[j].imshow(generator_images[cnt, :,:,0], cmap='gray')
            axs[j].set_title("Class: %s" % class_map[sampled_labels[cnt][0]])
            axs[j].axis('off')
            cnt += 1
    #fig.savefig("images/%d.png" % epoch)
    plt.show()
    plt.close()
    userInput = input("Press enter to generate new objects")