from IBS_dataloader import *
from Model import *
import pathlib
import shutil

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
    elif index == 14:
        return 'Western'

#     predict pipeline
def predict():
    # load model
    model = IBS_Model_ResNet152()
    model.load_weights("F:\\IBS_model\\all15class\\resnet152\\4fold\\0\\")
    # create save directory
    for j in range(15):
        label_type = labelencoding_reverse(j)
        make_path = "F:\\IBS_data\\Science\\CroppedData2\\predicted_resnet152\\" + label_type + "\\"
        pathlib.Path(make_path).mkdir(parents=True, exist_ok=True)
    dir_name="test"
    predict_data, file_name_list = load_data(dir_name)
    # save_path
    save_path_root = "F:\\IBS_data\\Science\\CroppedData\\predicted_resnet152\\" + dir_name + "\\"
    # source path
    source_path = "F:\\IBS_data\\Science\\CroppedData\\resized\\" + dir_name + "\\"
    # # predict
    y_pred = model.predict(predict_data)
    for m in range(len(file_name_list)):
        temp_label_index = np.argmax(y_pred[m], axis=None, out=None)
        temp_label = labelencoding_reverse(temp_label_index)
        dst_path = save_path_root + temp_label + "\\"
        shutil.copyfile(source_path + file_name_list[m], dst_path + file_name_list[m])


if __name__ == '__main__':
    predict()

