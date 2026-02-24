import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def parse_log(file_path):
    data = []
    
    # 1. 파일 경로 확인
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame()

    print(f"📂 로그 파일 분석 중: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue  # 빈 줄 건너뛰기

            # 2. 정규표현식 수정 (음수, 소수점, 긴 숫자 완벽 대응)
            # 예: "out_d_real: -0.0375..." 같은 음수도 잡아냄
            matches = re.findall(r'([a-zA-Z0-9_]+):\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', line)
            
            if not matches:
                continue

            row = {}
            gan_count = 0
            
            for key, value in matches:
                try:
                    val = float(value)
                except ValueError:
                    continue

                # 3. gan_loss 중복 처리 로직
                if key == 'gan_loss':
                    gan_count += 1
                    if gan_count == 1:
                        # 첫 번째 gan_loss는 이름 그대로 유지 (우리가 원하는 값)
                        k = 'gan_loss'
                    else:
                        # 두 번째, 세 번째는 _2, _3 등을 붙여서 구분
                        k = f'gan_loss_{gan_count}'
                else:
                    k = key
                
                row[k] = val
            
            # 4. Iteration 계산 (줄 번호 * 100)
            if row:
                row['iteration'] = (i + 1) * 100
                data.append(row)

    df = pd.DataFrame(data)
    
    # 5. 데이터 파싱 결과 확인
    if df.empty:
        print("⚠️ 경고: 데이터를 추출하지 못했습니다. 정규표현식이 맞지 않거나 파일이 비어있습니다.")
    else:
        print(f"✅ 파싱 성공! 총 {len(df)}개의 Iteration 데이터를 읽었습니다.")
        print(f"   [감지된 컬럼]: {list(df.columns)}")
        
    return df

def plot_loss_analysis(df, max_iter=None, selected_losses=None, smoothing_window=100):
    # 데이터가 비어있거나 iteration 컬럼이 없으면 중단 (에러 방지)
    if df.empty or 'iteration' not in df.columns:
        print("❌ 그래프를 그릴 데이터가 없습니다.")
        return

    # Iteration 범위 필터링
    if max_iter:
        df = df[df['iteration'] <= max_iter]
    
    # 보여줄 Loss 선택
    if not selected_losses:
        selected_losses = [c for c in df.columns if c != 'iteration']
    
    # 실제 존재하는 컬럼만 남기기
    valid_losses = [l for l in selected_losses if l in df.columns]

    plt.figure(figsize=(12, 6))
    
    for loss in valid_losses:
        # 이동 평균(Smoothing) 적용하여 부드러운 선 그리기
        smoothed = df[loss].rolling(window=smoothing_window, min_periods=1).mean()
        plt.plot(df['iteration'], smoothed, label=loss)
        
        # 최솟값 정보 출력
        min_val = df[loss].min()
        # 최솟값일 때의 iteration 찾기
        min_iter = df.loc[df[loss] == min_val, 'iteration'].iloc[0]
        print(f"   🔹 [{loss}] 최솟값: {min_val:.5f} (iter: {min_iter:,})")

    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# --- 실행 부분 ---
# 실제 파일 경로를 여기에 입력하세요
# file_path = 'models/e2e/wo/e2e_60k_wo cls.txt' 
file_path = 'models/e2e/e2e_120k_ver2.txt' 

df = parse_log(file_path)

# 데이터가 정상적으로 읽혔을 때만 그래프 그리기
if not df.empty:
    # 보고 싶은 loss만 골라서 리스트에 넣으세요
    # 첫 번째 gan_loss는 그냥 'gan_loss'로 저장되어 있습니다.
    TARGET_LOSSES = ['l1_loss'] 
    TARGET_LOSSES2 = ['percep_loss'] 
    TARGET_LOSSES3 = ['gan_loss'] 
    TARGET_LOSSES4 = ['cross_entropy_loss'] 
    TARGET_LOSSES5 = ['total_g_loss'] 
    TARGET_LOSSES6 = ['l_d_real', 'l_d_fake'] 
    
    plot_loss_analysis(df, max_iter=120000, selected_losses=TARGET_LOSSES)
    plot_loss_analysis(df, max_iter=120000, selected_losses=TARGET_LOSSES2)
    plot_loss_analysis(df, max_iter=120000, selected_losses=TARGET_LOSSES3)
    plot_loss_analysis(df, max_iter=120000, selected_losses=TARGET_LOSSES4)
    plot_loss_analysis(df, max_iter=120000, selected_losses=TARGET_LOSSES5)
    plot_loss_analysis(df, max_iter=120000, selected_losses=TARGET_LOSSES6)