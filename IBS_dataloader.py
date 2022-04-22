from Subfigures_cropper import  *
from tensorflow.keras.preprocessing import  image
from tensorflow.keras.applications.imagenet_utils import  preprocess_input
from sklearn.utils import shuffle
import shutil

def labelencoding_onehot(type):
    if type == 'Positive':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    elif type =='BarCharts':
        return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    elif type =='BoxPlots':
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    elif type =='Chemical':
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    elif type =='CircosPlots':
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    elif type =='HeatMap':
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    elif type =='Histogram':
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    elif type =='PieCharts':
        return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    elif type =='LinePlots':
        return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    elif type =='ScatterPlots':
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
def labelencoding(type):
    if type == 'Biological Sequences':
        return 0
    elif type =='BarCharts':
        return 1
    elif type =='BoxPlots':
        return 2
    elif type =='Chemical':
        return 3
    elif type =='CircosPlots':
        return 4
    elif type =='FlowCharts':
        return 5
    elif type =='HeatMap':
        return 6
    elif type =='Histogram':
        return 7
    elif type =='InterNetworks':
        return 8
    elif type =='LinePlots':
        return 9
    elif type =='MicroImages':
        return 10
    elif type =='PieCharts':
        return 11
    elif type =='ScatterPlots':
        return 12
    elif type =='Structure':
        return 13
    elif type =='Western':
        return 14
def labelencoding_reverse(index):
    if index == 0:
        return 'Biological Sequences'
    elif index == 1:
        return 'BarCharts'
    elif index == 2:
        return 'BoxPlots'
    elif index == 3:
        return 'Chemical'
    elif index == 4:
        return 'CircosPlots'
    elif index == 5:
        return 'FlowCharts'
    elif index == 6:
        return 'HeatMap'
    elif index == 7:
        return 'Histogram'
    elif index == 8:
        return 'InterNetworks'
    elif index == 9:
        return 'LinePlots'
    elif index == 10:
        return 'MicroImages'
    elif index == 11:
        return 'PieCharts'
    elif index == 12:
        return 'ScatterPlots'
    elif index == 13:
        return 'Structure'
    elif index == 15:
        return 'Western'
def findAllFile(base):
	for root, ds, fs in os.walk(base):
		for f in fs:
			if f.endswith('.jpg'):
				fullname = os.path.join(root, f)
				yield fullname
def findAllFile_returnName(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.jpg') or f.endswith('.png'):
                fullname = os.path.join(root, f)
                yield fullname,f

def load_data():

    type_list = ['BarCharts', 'BoxPlots', 'Chemical', 'CircosPlots',  'FlowCharts','HeatMap', 'Histogram','InterNetworks',
                 'LinePlots','MicroImages','PieCharts','ScatterPlots','Structure','Western']
    img_data_list = []
    labels = []
    root_path = "F:\\IBS_data\\TrainingData\\Negative_resized\\"
    for temp_type in type_list:
        path = root_path+temp_type
        current_label = labelencoding(temp_type)
        for temp_path in findAllFile(path):
            img = image.load_img(temp_path)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
        # print('Input image shape:', x.shape)
            img_data_list.append(x)
            labels.append(current_label)

    root_path_p = "F:\\IBS_data\\TrainingData\\Positive_resized\\"
    pos_label = labelencoding("Positive")
    for temp_path in findAllFile(root_path_p):
        img = image.load_img(temp_path)
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_data_list.append(x)
        labels.append(pos_label)
    img_data = np.array(img_data_list)
    img_data = np.rollaxis(img_data, 1, 0)
    img_data = img_data[0]
    return img_data,labels



if __name__ == '__main__':
    load_data()



