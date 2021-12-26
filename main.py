import first_OOD.first_ood as fs
import second_OOD.GLCM_SVM as GS
import second_OOD.Mahalanobis_distance as Ma
import classification_model.DenseNet169 as Classify
if __name__ == '__main__':
    fs.first_OOD()
    GS.detect()
    Ma.detect()
    Classify.final_classify()
