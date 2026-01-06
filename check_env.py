import os
# 1. OpenMP 충돌 방지 설정 확인
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    import numpy as np
    import cv2
    import platform

    print("=" * 50)
    # 2. Python 버전 확인
    print(f"[1] Python 버전: {platform.python_version()} (3.9.x 권장)")
    
    # 3. NumPy 버전 확인
    print(f"[2] NumPy 버전: {np.__version__} (1.26.4 권장)")
    
    # 4. PyTorch 및 CUDA 인식 확인
    print(f"[3] PyTorch 버전: {torch.__version__}")
    print(f"[4] CUDA 사용 가능: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"    - GPU 이름: {torch.cuda.get_device_name(0)}")
        print(f"    - CUDA 빌드 버전: {torch.version.cuda}")
        
        # 5. RTX 5070 실제 연산 테스트 (sm_120 호환성 체크)
        test_tensor = torch.ones(1).cuda()
        result = test_tensor + 1
        print(f"[5] GPU 연산 테스트: 성공 (결과: {result.item()})")
    else:
        print("[!] 경고: GPU를 인식하지 못하고 있습니다.")

    print("=" * 50)
    print("모든 설정이 완료되었습니다. 이제 작업을 시작하셔도 좋습니다!")

except Exception as e:
    print(f"\n[!] 오류 발생: {e}")