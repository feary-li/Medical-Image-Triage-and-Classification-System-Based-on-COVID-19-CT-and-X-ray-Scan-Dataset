# Medical-Image-Triage-and-Classification-System-Based-on-COVID-19-CT-and-X-ray-Scan-Dataset

Here shows the code and the dataset from the paper Medical Image Triage and Classification System——Based on COVID-19 CT and X ray Scan Dataset

## Dataset

All of the datasets are in the folder called 'dataset', there are three different folders with datasets corresponding to the datasets used in each step of the experiment. Please note that the name of the dataset folder is the same as the name of the code or code folder.

After you download the dataset, please put it under the same folder where the dataset from.

## Requirement

Before you run the code, you must confirm that your python environment has such packages:

skimage;

os;

sklearn;

seaborn;

matplotlib;

keras;

numpy;

scrpy;

time;

toch;

IPython;



## Run the code

If you want to test the step of first OOD, you can run the code under the folder which is called 'first_OOD', the name of the python code is 'first_OOD.py'.
If you want to test the step of our integrated criticism system， you can run the code name 'main.py' which is under the folder called 'second_OOD'. The results from both the GLCM_SVM and our VMDD will be shown when you run the code.
If you want to test the classification model, you can run the code, you can run the code name 'DenseNet169.py' which is under the folder called 'classification_model'.

 //文件的下载
    downloadFile(item) {
      var x = new XMLHttpRequest();
      x.open("GET", item.url, true);
      x.responseType = 'blob';
      x.onload = function (e) {
        var url = window.URL.createObjectURL(x.response)
        var a = document.createElement('a');
        a.href = url
        a.download = item.name ; //下载后的文件名
        a.click()
      }
      x.send();
      // this.$message.success('下载成功!')
    }



