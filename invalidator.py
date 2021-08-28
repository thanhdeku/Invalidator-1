from classifier_ase88 import classifier as cf1 
from classifier_patchsim139 import classifier as cf2

def classifier(threshold):
    tp1, fp1, x, rs4project_tp_1, rs4project_fp_1, total_inc_1, total_corr_1 = cf1(threshold, softrules = True, hardrules = True, learning= True, only_learning = False)
    tp2, fp2, y, rs4project_tp_2, rs4project_fp_2, total_inc_2, total_corr_2= cf2(threshold, softrules = True, hardrules = True, learning= True, only_learning = False)
    # print(x)
    # print(y)
    tp = tp1 + tp2
    fp = fp1 + fp2
    recall = tp/160
    precision = tp/(tp + fp)
    f1 = 2*recall*precision/(recall + precision)
    print(" ========= Final Result ==========")
    print(" === Per projects ===>")
    for project in {"Chart", "Time", "Lang", "Math"}:
        print("   {}: TP = {}/{} ({}), FP = {}/{} ({})".format(project, rs4project_tp_1[project] + rs4project_tp_2[project], total_inc_1[project] + total_inc_2[project], (rs4project_tp_1[project] + rs4project_tp_2[project])/(total_inc_1[project] + total_inc_2[project]), rs4project_fp_1[project] + rs4project_fp_2[project], total_corr_1[project] + total_corr_2[project], (rs4project_fp_1[project] + rs4project_fp_2[project])/(total_corr_1[project] + total_corr_2[project])))
    print(" === Total ===>")
    print("   True Positve: {}/{}".format(tp, 163))
    print("   False Positve: {}/{}".format(fp, 64))
    print("   Recall: {}".format(recall))
    print("   Precision: {}".format(precision))
    print("   F1: {}".format(f1))
    return tp, fp, recall, precision, f1
if __name__ == "__main__":
    # print("***========= Invalidator-1 =========***")
    # classifier(0.03)
    print("***========= Invalidator-2 =========***")
    classifier(0.36)


