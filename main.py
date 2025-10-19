import subprocess
import torch

# dataset/original을 gt/train, gt/test로 분리
def split_original():
    subprocess.run(['python', 'Part0/dataset_train_test_txt.py'])

# gt/train, gt/test -> lq/train, lq/test
def degradation():
    subprocess.run(['python', 'Part0/degradation_folder_for_super_resolution.py'])

# meta info pairdata 만들기
def generate_meta_info_pairdata():
    subprocess.run(['python', 'external/Real-ESRGAN/scripts/generate_meta_info_pairdata.py', '--input', 'dataset/gt/train', 'dataset/lq/train', '--meta_info', 'dataset/meta_info/meta_info_RE_pair.txt'])

# Real-ESRGAN 학습
def train_SR():
    subprocess.run(['python', 'external/Real-ESRGAN/realesrgan/train.py', '-opt', 'external/Real-ESRGAN/options/finetune_realesrgan_x4plus_pairdata.yml', '--auto_resume'])

# SR 테스트
def test_SR():
    subprocess.run(['python', 'Part1/Real-ESRGAN/test.py'])

def split_SR_output():
    subprocess.run(['python', 'Part0/dataset_train_test_for_classification.py'])

# CLS
def classification():
    subprocess.run(['python', 'Part2/ResNet50nAdaptivePooling.py'])

# e2e
def e2e_train_test():
    subprocess.run(['python', 'Part3/E2E_inference.py'])

if __name__ == "__main__":
    # print("-----split original-----")
    # split_original()
    # print("-----degradation--------")
    # degradation()
    # print("-----pairdata-----------")
    # generate_meta_info_pairdata()
    # torch.cuda.empty_cache()
    # print("-----train--------------")
    # train_SR()
    # print("-----test---------------")
    # test_SR()
    # print("-----split SR output----")
    # split_SR_output()
    # print("-----classification-----")
    # classification()
    print("-----e2e train test-----")
    e2e_train_test()
    # subprocess.run(['python', 'Part2/classification.py'])