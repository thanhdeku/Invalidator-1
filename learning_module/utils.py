import os
import re
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

data_folder = "data/"
projects_ASE21 = ["CapGen", "Jaid", "SOFix", "SequenceR", "SketchFix"]
ASE21_TEST_DATA = ['104', '168', '157', '150', '159', '166', '161', '105', '160', '158', '167', '151', '169', '156', '174', '180', '187', '18', '173', '145', '11', '142', '16', '189', '129', '196', '111', '127', '143', '17', '188', '144', '197', '10', '172', '186', '19', '181', '175', '121', '126', '110', '128', '117', '153', '154', '162', '165', '131', '109', '100', '107', '164', '163', '155', '152', '106', '139', '108', '130', '112', '115', '123', '124', '184', '170', '177', '183', '148', '141', '15', '146', '12', '179', '125', '114', '113', '147', '178', '13', '140', '14', '182', '176', '149', '171', '185', "157", "190"]
DATA_139 = ['Patch1','Patch2','Patch4','Patch5','Patch6','Patch7','Patch8','Patch9','Patch10','Patch11','Patch12','Patch13','Patch14','Patch15','Patch16','Patch17','Patch18','Patch19','Patch20','Patch21','Patch22','Patch23','Patch24','Patch25','Patch26','Patch27','Patch28','Patch29','Patch30','Patch31','Patch32','Patch33','Patch34','Patch36','Patch37','Patch38','Patch44','Patch45','Patch46','Patch47','Patch48','Patch49','Patch51','Patch53','Patch54','Patch55','Patch58','Patch59','Patch62','Patch63','Patch64','Patch65','Patch66','Patch67','Patch68','Patch69','Patch72','Patch73','Patch74','Patch75','Patch76','Patch77','Patch78','Patch79','Patch80','Patch81','Patch82','Patch83','Patch84','Patch88','Patch89','Patch90','Patch91','Patch92','Patch93','Patch150','Patch151','Patch152','Patch153','Patch154','Patch155','Patch157','Patch158','Patch159','Patch160','Patch161','Patch162','Patch163','Patch165','Patch166','Patch167','Patch168','Patch169','Patch170','Patch171','Patch172','Patch173','Patch174','Patch175','Patch176','Patch177','Patch180','Patch181','Patch182','Patch183','Patch184','Patch185','Patch186','Patch187','Patch188','Patch189','Patch191','Patch192','Patch193','Patch194','Patch195','Patch196','Patch197','Patch198','Patch199','Patch201','Patch202','Patch203','Patch204','Patch205','Patch206','Patch207','Patch208','Patch209','Patch210','PatchHDRepair1','PatchHDRepair3','PatchHDRepair4','PatchHDRepair5','PatchHDRepair6','PatchHDRepair7','PatchHDRepair8','PatchHDRepair9','PatchHDRepair10']
_ASE21_INFO_PATH= data_folder + "ASE20_Patches/patches.json"
_ICSE18_INFO_PATH = data_folder + "ICSE18_Patches/INFO/{}.json"
_PATCH_ICSE18_PATH = "data/ICSE18_Patches/{}"
_PATCH_ICSE20_PATH = "data/ASE20_Patches/Patches_ICSE/{}/{}/{}/{}.patch"
_PATCH_ASE20_PATH = "data/ASE20_Patches/Patches_others/{}/{}/{}/{}.patch"
_DEFECTS4J_CORRECT_PATCH = "data/defects4j-developer/"

def read_info(dataset, path = None):
    data = {}
    if dataset == "ICSE18":
        list_patches = []
        for i in range(210):
            list_patches.append("Patch"+ str(i+1))
        for i in range(10):
            list_patches.append("PatchHDRepair"+ str(i+1))
        for patch in list_patches:
            with open(_ICSE18_INFO_PATH.format(patch), "r") as f:
                tmp = eval(f.read())
                data[tmp["ID"]] = tmp       
    if dataset == "ASE21":    
        with open(path, "r") as f:
            tmp = eval(f.read())
            for patch in tmp:
                data[str(patch["id"])] = patch

    return data

def read_patch(path, type):
    with open(path, "r") as f:
        code = ''
        p = r"([^\w_])"
        flag = True
        for line in f:
            line = line.strip()
            if '*/' in line:
                flag = True
                continue
            if not flag:
                continue
            if line != '':
                if line.startswith('@@') or line.startswith('diff') or line.startswith('index'):
                    continue
                if line.startswith('Index') or line.startswith('==='):
                        continue
                elif '/*' in line:
                    flag = False
                    continue
                elif type == "buggy":
                    if line.startswith('---') or line.startswith('PATCH_DIFF_ORIG=---'):
                        continue
                    elif line.startswith('-'):
                        if line[1:].strip().startswith('//'):
                            continue
                        line = re.split(pattern=p, string=line[1:].strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('') 
                        line = ' '.join(line)                    
                        code += line.strip() + ' '
                    elif line.startswith('+'):
                        pass
                    else:
                        line = re.split(pattern=p, string=line.strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                            line.remove('')
                        line = ' '.join(line)
                        code += line.strip() + ' '
                
                elif type == 'fixed':
                    if line.startswith('+++'):
                         continue
                    elif line.startswith('+'):
                         if line[1:].strip().startswith('//'):
                             continue
                         line = re.split(pattern=p, string=line[1:].strip())
                         line = [x.strip() for x in line]
                         while '' in line:
                             line.remove('')
                         line = ' '.join(line)
                         code += line.strip() + ' '
                    elif line.startswith('-'):
                         pass
                    else:
                        line = re.split(pattern=p, string=line.strip())
                        line = [x.strip() for x in line]
                        while '' in line:
                             line.remove('')
                        line = ' '.join(line)
                        code += line.strip() + ' '
    if len(code) > 512:
        code = code[:512]
    return code
def tokenize(code):
    inputs = tokenizer(code, padding='max_length', max_length=512, truncation=True, return_tensors="pt")
    return inputs.data['input_ids'][0], inputs.data['attention_mask'][0]

def prepare_test_data():
    data = []
    count = 0
    patches_ASE = read_info("ASE21", _ASE21_INFO_PATH) 
    for id, patch_info in patches_ASE.items():
        patch_data = {}
        label = 0
        if id in ASE21_TEST_DATA:
            if patch_info["correctness"] != "Error" and patch_info["project"] in {"Chart", "Time", "Lang", "Math"}:
                count += 1
                if patch_info["tool"] in projects_ASE21:
                    path = _PATCH_ASE20_PATH.format(patch_info["correctness"], patch_info["tool"], patch_info["project"], patch_info["bug_id"])
                else:
                    path = _PATCH_ICSE20_PATH.format(patch_info["correctness"], patch_info["tool"], patch_info["project"], patch_info["bug_id"])

                if patch_info["correctness"] in ["Ddifferent", "Dcorrect", "Dsame"]:
                    label = 0
                else:
                    label = 1
                bug_code = read_patch(path, "buggy")
                fixed_code = read_patch(path, "fixed")
                patch_data["label"] = label 
                patch_data["origin_patch"] = "ASE21_" + str(id)
                patch_data["buggy"], patch_data["att_buggy"] = tokenize(bug_code)
                patch_data["fixed"],  patch_data["att_fixed"]= tokenize(fixed_code)
                data.append(patch_data)
            else:
                print(patch_info)
    patches_ICSE = read_info("ICSE18")
    for id in DATA_139: 
        patch_info = patches_ICSE[id]
        patch_data = {}
        label = 0
        path = _PATCH_ICSE18_PATH.format(patch_info["ID"])
        if patch_info["correctness"] == "Correct":
            label = 0
        else:
            label = 1
        bug_code = read_patch(path, "buggy")
        fixed_code = read_patch(path, "fixed")
        patch_data["label"] = label 
        patch_data["origin_patch"] = "ICSE18_" + str(id)
        patch_data["buggy"], patch_data["att_buggy"] = tokenize(bug_code)
        patch_data["fixed"],  patch_data["att_fixed"]= tokenize(fixed_code)
        data.append(patch_data)
    
    return data 

def prepare_train_data():
    data = []
    count_corr = 0
    count_incorr = 0
    patches_ASE = read_info("ASE21", _ASE21_INFO_PATH) 
    for id, patch_info in patches_ASE.items():
        patch_data = {}
        label = 0
        if id not in ASE21_TEST_DATA:
            #print("Preprocess ASE Patch {}".format(id))    
            if patch_info["correctness"] != "Error" and patch_info["project"] in {"Chart", "Time", "Lang", "Math"}:
                
                if patch_info["tool"] in projects_ASE21:
                    path = _PATCH_ASE20_PATH.format(patch_info["correctness"], patch_info["tool"], patch_info["project"], patch_info["bug_id"])
                else:
                    path = _PATCH_ICSE20_PATH.format(patch_info["correctness"], patch_info["tool"], patch_info["project"], patch_info["bug_id"])
                
                if patch_info["correctness"] in ["Ddifferent", "Dcorrect", "Dsame"]:
                    label = 0
                    count_corr += 1
                else:
                    label = 1
                    count_incorr += 1
                bug_code = read_patch(path, "buggy")
                fixed_code = read_patch(path, "fixed")
                patch_data["label"] = label 
                patch_data["origin_patch"] = "ASE21_" + str(id)
                patch_data["buggy"], patch_data["att_buggy"] = tokenize(bug_code)
                #print(len(patch_data["buggy"]))
                patch_data["fixed"],  patch_data["att_fixed"]= tokenize(fixed_code)
                #print(len(patch_data["fixed"]))
                data.append(patch_data)

    for bug in ["Math", "Chart", "Time", "Lang"]:
        bug_path = os.path.join(_DEFECTS4J_CORRECT_PATCH, bug)
        correct_patches = os.path.join(bug_path,'patches')
        for patch in os.listdir(correct_patches):
            path = correct_patches + "/" + patch
            # #print(correct_patches)
            if not patch.endswith('src.patch'):
                continue
            patch_data = {}
            try:
                bug_code = read_patch(path, "buggy")
                fixed_code = read_patch(path, "fixed")
            except:
                continue
            patch_data["label"] = 0
            # count_corr += 1
            patch_data["origin_patch"] = "Developers_" + bug.replace("src.patch", "") + str(patch)
            patch_data["buggy"], patch_data["att_buggy"] = tokenize(bug_code)
            patch_data["fixed"],  patch_data["att_fixed"]= tokenize(fixed_code)
            data.append(patch_data)
    print(count_incorr)
    print(count_corr)
    # exit()
    return data


if __name__ == "__main__":
    data = prepare_train_data()
    print(len(data))

