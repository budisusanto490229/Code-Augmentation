import numpy as np
import scipy.stats


def delta(list1, list2):
    len1 = len(list1)
    len2 = len(list2)
    large_num = 0
    small_num = 0
    for item1 in list1:
        for item2 in list2:
            if item1 > item2:
                large_num += 1
            elif item1 < item2:
                small_num += 1
    return (large_num - small_num) / (len1 * len2)


if __name__ == '__main__':
    resNet_result1 = [0.5, 0.59523809, 0.547619045, 0.547619045, 0.535714269, 0.571428597,
                      0.535714269, 0.607142866, 0.476190478, 0.547619045]
    resNet_result2 = [0.65476191, 0.797619045, 0.738095224, 0.702380955, 0.630952358,
                      0.607142866, 0.511904776, 0.690476179, 0.702380955, 0.678571403]

    w_resNet, p_resNet = scipy.stats.brunnermunzel(resNet_result1, resNet_result2)
    print('resNet P:', p_resNet)
    d_resNet = delta(resNet_result1, resNet_result2)
    print('resNet Delta', d_resNet)

    rf_result1 = [0.548, 0.5, 0.56, 0.56, 0.571, 0.548, 0.5, 0.56, 0.56, 0.571]
    rf_result2 = [0.595, 0.595, 0.56, 0.583, 0.607, 0.595, 0.595, 0.56, 0.583, 0.607]

    w_rf, p_rf = scipy.stats.brunnermunzel(rf_result1, rf_result2)
    print('RF P:', p_rf)
    d_rf = delta(rf_result1, rf_result2)
    print('RF Delta', d_rf)

    knn_result1 = [0.56, 0.536, 0.536, 0.536, 0.571, 0.56, 0.536, 0.536, 0.536, 0.571]
    knn_result2 = [0.571, 0.607, 0.56, 0.536, 0.595, 0.571, 0.607, 0.56, 0.536, 0.595]

    w_knn, p_knn = scipy.stats.brunnermunzel(knn_result1, knn_result2)
    print('knn P:', p_knn)
    d_knn = delta(knn_result1, knn_result2)
    print('knn Delta', d_knn)

    mobile_result1 = [0.523809552, 0.559523821, 0.40476191, 0.702380955, 0.583333313, 0.65476191, 0.607142866,
                      0.59523809, 0.547619045, 0.583333313]
    mobile_result2 = [0.702380955, 0.607142866, 0.642857134, 0.59523809, 0.511904776, 0.583333313, 0.619047642,
                      0.619047642, 0.59523809, 0.666666687]

    w_mobile, p_mobile = scipy.stats.brunnermunzel(mobile_result1, mobile_result2)
    print('mobile P:', p_mobile)
    d_mobile = delta(mobile_result1, mobile_result2)
    print('mobile Delta', d_mobile)

    dense_result1 = [0.619047642, 0.523809552, 0.571428597, 0.65476191, 0.476190478, 0.583333313, 0.559523821,
                     0.523809552,
                     0.571428597, 0.642857134]
    dense_result2 = [0.702380955, 0.666666687, 0.583333313, 0.65476191, 0.630952358, 0.65476191, 0.690476179,
                     0.607142866,
                     0.65476191, 0.511904776]

    w_dense, p_dense = scipy.stats.brunnermunzel(dense_result1, dense_result2)
    print('dense P:', p_dense)
    d_dense = delta(dense_result1, dense_result2)
    print('dense Delta', d_dense)

    model_result1 = [0.536, 0.583, 0.643, 0.607, 0.679, 0.607, 0.619, 0.679, 0.595, 0.619]
    model_result2 = [0.667, 0.643, 0.714, 0.643, 0.738, 0.631, 0.655, 0.619, 0.762, 0.726]

    w_model, p_model = scipy.stats.brunnermunzel(model_result1, model_result2)
    print('model P:', p_model)
    d_model = delta(model_result1, model_result2)
    print('model Delta', d_model)
