import os
import tensorflow as tf
import zipfile

class DataManager:

    def __init__(self,dataset_folder = None, dataset_link = None):
        self.dataset_link = dataset_link
        self.dataset_folder = dataset_folder
        self.datas = None

    def dataset_unzip(self, zip_files_path):
        files_and_directories = os.listdir(zip_files_path)
        files = [f for f in files_and_directories if os.path.isfile(os.path.join(zip_files_path, f))]
        zip_files=[]
        for f in files:
            if f.endswith('.zip'):
                zip_files.append(f)
        for zip_file in zip_files:
            with zipfile.ZipFile(os.path.join(zip_files_path, zip_file), 'r') as zip_ref:
                zip_ref.extractall(zip_files_path)
        return

    def kaggle_dataset_gather(self, extract=True):
        #save_path = os.path.join(os.getcwd(), save_path)
        if not self.dataset_link == None:
            os.system(f'kaggle datasets download -d {self.dataset_link} -p {self.dataset_folder}')
            
            files_and_directories = os.listdir(self.dataset_folder)
            files = [f for f in files_and_directories if os.path.isfile(os.path.join(self.dataset_folder, f))]

            print(f'Files in {self.dataset_folder}:')
            for file in files:
                print(file)

            zip_file_paths=[]
            for f in files:
                if f.endswith('.zip'):
                    zip_file_paths.append(f)
            if extract == True and len(zip_file_paths) == 1:
                self.dataset_unzip(self.zip_file_paths)
        return

    def read_dataset(self, normalize=True, batch_size=32, image_size=(64,64)):
        datas=tf.keras.utils.image_dataset_from_directory(os.path.join(self.dataset_folder,'traffic_Data','DATA'), batch_size=batch_size, image_size=image_size)
        if normalize == True:
            datas=datas.map(lambda x,y:(x/255,y))
        self.datas = datas
        return datas
    
    def data_split(self, train_percent, test_percent):
        datas = self.read_dataset()
        
        train_size=int(train_percent*len(datas))
        test_size=int(test_percent*len(datas))
        validation_size=len(datas)-train_size-test_size

        train_datas=datas.take(train_size)
        test_datas=datas.skip(train_size).take(test_size)
        validation_datas=datas.skip(train_size + test_size).take(validation_size)

        return train_datas, test_datas, validation_datas