import os
import numpy as np
from tqdm import tqdm
import subprocess

_INFO_PATH = "ICSE22_data/data/patch_info/ASE_INFO/Patch{}.json"
_IDX2ID_PATH = "ICSE22_data/data/patch_info/139_patches.json"
_INVARIANT_PATCH_PASS = "ICSE22_data/data/raw_invariants/ase_88/patches/{}/result_passing.txt"
_INVARIANT_BUG_PASS = "ICSE22_data/data/raw_invariants/ase_88/b/{}/{}/result_passing.txt"
_INVARIANT_FIX_PASS = "ICSE22_data/data/raw_invariants/ase_88/f/{}/{}/result_passing.txt"
_INVARIANT_PATCH_FAIL = "ICSE22_data/data/raw_invariants/ase_88/patches/{}/result_failing.txt"
_INVARIANT_BUG_FAIL = "ICSE22_data/data/raw_invariants/ase_88/b/{}/{}/result_failing.txt"
_INVARIANT_FIX_FAIL = "ICSE22_data/data/raw_invariants/ase_88/f/{}/{}/result_failing.txt"
_CF_PATCH_PASS = "ICSE22_data/data/raw_invariants/ase_88/patches/{}/result_passing_confidence_score.txt"
_CF_BUG_PASS = "ICSE22_data/data/raw_invariants/ase_88/b/{}/{}/result_passing_confidence_score.txt"
_CF_FIX_PASS = "ICSE22_data/data/raw_invariants/ase_88/f/{}/{}/result_passing_confidence_score.txt"
_CF_PATCH_FAIL = "ICSE22_data/data/raw_invariants/ase_88/patches/{}/result_failing_confidence_score.txt"
_CF_BUG_FAIL = "ICSE22_data/data/raw_invariants/ase_88/b/{}/{}/result_failing_confidence_score.txt"
_CF_FIX_FAIL = "ICSE22_data/data/raw_invariants/ase_88/f/{}/{}/result_failing_confidence_score.txt"

def read_learning_distance(path = "ICSE22_data/data/syntactic_distance/learning_cf.txt"):
    data = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("ASE21"):
                imp = line.split("\t")
                data[imp[0].split("_")[1]] = [float(imp[1]), float(imp[2])]
    return data

def read_idx2id(path=_IDX2ID_PATH):
    idx2id = {}
    with open(path, "r") as f:
            data = eval(f.read())
            for item in data:
                idx2id[item["id"]] = item["patch_file"]
    return idx2id

def read_info_patch(patch):
    path = _INFO_PATH.format(patch)
    try:
        with open(path, "r") as f:
            data = eval(f.read())
            return data["project"], data["bug_id"], data["correctness"], data[
                "tool"], data["path"]
    except FileNotFoundError:
        print(path)
        print(
            "Invalid patch !!! \nPlease use available patch: \n==> Patch1, ..., Patch210; \n==> HDRepair1, ..., HDRepair10"
        )

def read_invariant_with_cf(project, bug_id, patch):
    p_inv_pass = read_invariant_with_confidence_score(
        _CF_PATCH_PASS.format(patch))
    p_inv_fail = read_invariant_with_confidence_score(
        _CF_PATCH_FAIL.format(patch))
    b_inv_pass = read_invariant_with_confidence_score(
        _CF_BUG_PASS.format(project, bug_id))
    b_inv_fail = read_invariant_with_confidence_score(
        _CF_BUG_FAIL.format(project, bug_id))
    f_inv_pass = read_invariant_with_confidence_score(
        _CF_FIX_PASS.format(project, bug_id))
    f_inv_fail = read_invariant_with_confidence_score(
        _CF_FIX_FAIL.format(project, bug_id))
    return p_inv_pass, p_inv_fail, b_inv_pass, b_inv_fail, f_inv_pass, f_inv_fail

def read_invariant(project, bug_id, patch):
    p_inv_pass = read_invariant_with_path(
        _INVARIANT_PATCH_PASS.format(patch))
    p_inv_fail = read_invariant_with_path(
        _INVARIANT_PATCH_FAIL.format(patch))
    b_inv_pass = read_invariant_with_path(
        _INVARIANT_BUG_PASS.format(project, bug_id))
    b_inv_fail = read_invariant_with_path(
        _INVARIANT_BUG_FAIL.format(project, bug_id))
    f_inv_pass = read_invariant_with_path(
        _INVARIANT_FIX_PASS.format(project, bug_id))
    f_inv_fail = read_invariant_with_path(
        _INVARIANT_FIX_FAIL.format(project, bug_id))
    return p_inv_pass, p_inv_fail, b_inv_pass, b_inv_fail, f_inv_pass, f_inv_fail

def read_invariant_with_confidence_score(path):
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                current_key = 0
                last_key = 0
                exit_count = 0
                is_start = False
                for line in f:
                    if line[0:3] == "===":
                        is_start = True
                        current_key = f.readline().strip('\n')
                        # print(f.readline())
                        if "EXIT" in current_key:
                            if last_key != current_key:
                                exit_count = 0
                            current_key = get_method_name_with_exit(current_key)
                            last_key = current_key
                            if exit_count == 0:
                                data[current_key] = {}
                            # current_key = current_key + str(exit_count)
                            exit_count += 1
                        else:
                            last_key = current_key
                            data[current_key] = {}
                            exit_count = 0
                    else:
                        if is_start:
                            conf_line = line
                            confidence_score = float(conf_line.split()[-1].replace("]", ""))
                            inv_line = f.readline().strip()
                            data[current_key][inv_line] = confidence_score
        except:
            pass
    return data

def read_invariant_with_path(path):
    data = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            current_key = 0
            last_key = 0
            exit_count = 0
            is_start = False
            for line in f:
                if line[0:3] == "===":
                    is_start = True
                    current_key = f.readline().strip('\n')
                    if "EXIT" in current_key:
                        if last_key != current_key:
                            exit_count = 0
                        current_key = get_method_name_with_exit(current_key)
                        last_key = current_key
                        if exit_count == 0:
                            data[current_key] = []
                        # current_key = current_key + str(exit_count)
                        exit_count += 1
                    else:
                        last_key = current_key
                        data[current_key] = []
                        exit_count = 0
                else:
                    if is_start:
                        data[current_key].append(line.strip())
    # print(data)
    return data

def get_error_behaviour_syntax(bugprog, devprog):
    print("Error behaviours ...")
    error_beha = {}
    keys = list(bugprog.keys())
    for i in tqdm(range(len(keys))):
        program_point = keys[i]
        if program_point in devprog:
            bug_inv = bugprog[program_point]
            dev_inv = devprog[program_point]
            error_beha[program_point] = []
            for inv_b in bug_inv:
                if inv_b not in dev_inv:
                    error_beha[program_point].append(inv_b)
        else:
            error_beha[program_point] = []
            for inv in bugprog[program_point]:
                error_beha[program_point].append(inv)
        
    return error_beha


def get_correct_spec_syntax(bugprog, devprog):
    print("Correct specifications ...")
    correct_spec = {}
    keys = list(bugprog.keys())
    for i in tqdm(range(len(keys))):
        program_point = keys[i]
        if program_point in devprog:
            bug_inv = bugprog[program_point]
            dev_inv = devprog[program_point]
            correct_spec[program_point] = []
            for inv_b in bug_inv:
                if inv_b in dev_inv:
                    correct_spec[program_point].append(inv_b)
    return correct_spec
 
def get_method_name_with_exit(string):
    index = 0
    for i in range(len(string)):
        if string[i] == ":":
            index = i
    return string[:index + 5]

def is_array(s):
    if len(s) > 0:
        tmp = s.strip()
        if tmp[0] == "[" and tmp[-1] == "]":
            return True

    return False


def get_element_of_array(expr):
    ele = []
    if len(expr) > 3:
        ele = expr.replace(" ", "")[1:-1].split(",")
    return ele

def get_method_name(line):
    start = -1
    end = 9999999
    for j in range(len(line)):
        if line[j] == "(":
            end = j
            break
    if end != 9999999:
        for j in range(end, 0, -1):
            if (line[j] == " "):
                start = j + 1
                break
        return line[start:end].strip()
    else:
        return None

   