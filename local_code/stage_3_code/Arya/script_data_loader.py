from local_code.base_class.dataset import dataset
import pickle
from matplotlib import pyplot as plt
import numpy as np

# loading ORL dataset
class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        if 1:
            f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
            data = pickle.load(f)
            f.close()

            X_train = []
            y_train = []

            for instance in data['train']:
                image_matrix = instance['image'][:, :, 0]
                image_label = instance['label']
                X_train.append(image_matrix)
                y_train.append(image_label - 1)
                #plt.imshow(image_matrix)
                #plt.show()
                #print(image_matrix)
                #print(image_label)
                # remove the following "break" code if you would like to see more image in the training set
                # break

            X_test = []
            y_test = []

            for instance in data['test']:
                image_matrix = instance['image'][:, :, 0]
                image_label = instance['label']
                X_test.append(image_matrix)
                y_test.append(image_label - 1)
                #plt.imshow(image_matrix)
                #plt.show()
                #print(image_matrix)
                #print(image_label)
                # remove the following "break" code if you would like to see more image in the testing set
                # break

            X_train = np.array(X_train) / 255.0
            y_train = np.array(y_train)
            X_test = np.array(X_test) / 255.0
            y_test = np.array(y_test)

            return {'X_train': X_train, 'y_train': y_train,
                    'X_test': X_test, 'y_test': y_test}

        return None

        # loading CIFAR-10 dataset
        if 0:
            f = open(self.dataset_source_folder_path + 'CIFAR', 'rb')
            data = pickle.load(f)
            f.close()
            for instance in data['train']:
                image_matrix = instance['image']
                image_label = instance['label']
                plt.imshow(image_matrix)
                plt.show()
                print(image_matrix)
                print(image_label)
                # remove the following "break" code if you would like to see more image in the training set
                break

            for instance in data['test']:
                image_matrix = instance['image']
                image_label = instance['label']
                plt.imshow(image_matrix)
                plt.show()
                print(image_matrix)
                print(image_label)
                # remove the following "break" code if you would like to see more image in the testing set
                break

        # loading MNIST dataset
        if 0:
            f = open(self.dataset_source_folder_path + 'MNIST', 'rb')
            data = pickle.load(f)
            f.close()
            for instance in data['train']:
                image_matrix = instance['image']
                image_label = instance['label']
                plt.imshow(image_matrix, cmap='gray')
                plt.show()
                print(image_matrix)
                print(image_label)
                # remove the following "break" code if you would like to see more image in the training set
                break

            for instance in data['test']:
                image_matrix = instance['image']
                image_label = instance['label']
                plt.imshow(image_matrix, cmap='gray')
                plt.show()
                print(image_matrix)
                print(image_label)
                # remove the following "break" code if you would like to see more image in the testing set
                break
	
	
	