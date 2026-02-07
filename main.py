import subprocess
import torch

# dataset/originals의 파일 수를 줄임
def reduce_originals():
    subprocess.run(['python', 'Part0/reduce_dataset.py'])

# dataset/original을 gt/train, gt/test로 분리
def split_original():
    subprocess.run(['python', 'Part0/dataset_train_test_for_e2e.py'])

# gt/train, gt/test -> lq/train, lq/test
def degradation():
    subprocess.run(['python', 'Part0/degradation_folder_for_e2e.py'])

# meta info pairdata 만들기
def generate_meta_info_pairdata():
    subprocess.run(['python', 'external/Real-ESRGAN/scripts/generate_meta_info_pairdata_with_label.py', '--input', 'dataset/gt/train', 'dataset/lq/train', '--meta_info', 'dataset/meta_info/meta_info_RE_pair.txt'])

# Real-ESRGAN 학습
def train_E2E():
    subprocess.run(['python', 'external/Real-ESRGAN/realesrgan/train.py', '-opt', 'external/Real-ESRGAN/options/finetune_realesrgan_x4plus_pairdata.yml', '--auto_resume'])

# SR 테스트
def test_E2E():
    subprocess.run(['python', 'Part3/test.py'])


if __name__ == "__main__":
    # print("-----reduce dataset-----")
    # reduce_originals()
    # print("-----split original-----")
    # split_original()
    # print("-----degradation--------")
    # degradation()
    # print("-----pairdata-----------")
    # generate_meta_info_pairdata()
    print("-----train--------------")
    train_E2E()
    print("-----test---------------")
    test_E2E()
