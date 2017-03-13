# author: Mohammad Minhazul Haq

import csv
from scipy import misc
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from scipy.sparse import coo_matrix
from config import Config


class DataGrabber:

    def __init__(self):
        self.resized_width = Config.original_size
        self.resized_height = Config.original_size

        self.total_data_class_1 = Config.total_positive_samples
        self.total_data_class_2 = Config.total_negative_samples

        self.total_data = self.total_data_class_1 + self.total_data_class_2

        self.total_train_data = self.total_data // 2
        self.total_test_data = self.total_data // 2

        self.train_batch_size = Config.batch_size
        self.test_batch_size = Config.eval_batch_size

        self.train_batch_index = 0
        self.test_batch_index = 0

        self.max_train_batch_index = (self.total_train_data // self.train_batch_size) - 1
        self.max_test_batch_index = (self.total_test_data // self.test_batch_size) - 1

        self.data = np.zeros((self.total_data, self.resized_width * self.resized_height), dtype=np.int)
        self.labels = np.zeros(self.total_data, dtype=np.int)

        self.unannotated_dict = {}

        index = 0

        with open('all_annotations.csv') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                patient_id_dcm = row['PATIENT_ID_DCM']
                total_slices = int(row['SLICES_TOTAL_NUM'])
                x_min = int(row['X_MIN'])
                x_max = int(row['X_MAX'])
                y_min = int(row['Y_MIN'])
                y_max = int(row['Y_MAX'])
                z = row['Z']

                if patient_id_dcm in self.unannotated_dict:
                    try:
                        self.unannotated_dict[patient_id_dcm].remove(int(z))
                    except ValueError:
                        pass #do nothing
                else:
                    self.unannotated_dict[patient_id_dcm] = list(range(1, total_slices + 1))
                    self.unannotated_dict[patient_id_dcm].remove(int(z))

                file_path = '/smile/nfs/share/DSB2017/dicom_image/' + patient_id_dcm + '/instance' + z + '.png'

                image = misc.imread(file_path)
                cropped_image = image[y_min:y_max, x_min:x_max, :]

                image_2 = Image.fromarray(cropped_image)

                resized_image = image_2.resize((self.resized_width, self.resized_height), Image.ANTIALIAS)

                image_array = np.array(resized_image)
                resized_image_array = np.resize(image_array, (1, self.resized_width * self.resized_height))

                self.data[index] = resized_image_array
                self.labels[index] = 1

                index += 1

        # negative samples from xinliang zhu's noncancer_nodules.csv file
        with open('noncancer_nodules.csv') as csvfile:
            reader = csv.DictReader(csvfile)

            for row in reader:
                file_name = row['image']
                x_min = int(float(row['xmin']))
                x_max = int(float(row['xmax']))
                y_min = int(float(row['ymin']))
                y_max = int(float(row['ymax']))

                directory_name = file_name[0:8]

                file_path = '/smile/nfs/KaggleBowlCTimages/Image/Train_image/' + directory_name + '/' + file_name + '.png'

                image = misc.imread(file_path)
                cropped_image = image[y_min:y_max, x_min:x_max, :]

                image_2 = Image.fromarray(cropped_image)
                resized_image = image_2.resize((self.resized_width, self.resized_height), Image.ANTIALIAS)

                image_array = np.array(resized_image)
                resized_image_array = np.resize(image_array, (1, self.resized_width * self.resized_height))

                self.data[index] = resized_image_array
                self.labels[index] = 0

                index += 1

        print('index is: ' + str(index))

        # shuffle data and labels
        data_sparse = coo_matrix(self.data)
        self.data, data_sparse, self.labels = shuffle(self.data, data_sparse, self.labels, random_state=0)

    def get_train_data_labels_next_batch(self):
        data = self.data[self.train_batch_index*self.train_batch_size : ((self.train_batch_index+1)*self.train_batch_size)-1, :]
        labels = self.labels[self.train_batch_index*self.train_batch_size : ((self.train_batch_index+1)*self.train_batch_size)-1]

        self.train_batch_index += 1

        if self.train_batch_index > self.max_train_batch_index:
            self.train_batch_index = 0

        return data, labels

    def get_test_data_labels_next_batch(self):
        data = self.data[self.total_train_data + (self.test_batch_index*self.test_batch_size) :
            self.total_train_data + (((self.test_batch_index+1)*self.test_batch_size)-1), :]
        labels = self.labels[self.total_train_data + (self.test_batch_index*self.test_batch_size) :
            self.total_train_data + (((self.test_batch_index+1)*self.test_batch_size)-1)]

        self.test_batch_index += 1

        if self.test_batch_index > self.max_test_batch_index:
            self.test_batch_index = 0

        return data, labels
