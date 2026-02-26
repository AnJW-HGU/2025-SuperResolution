import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def parse_log(file_path):
    data = []
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return pd.DataFrame()

    print(f"📂 로그 파일 분석 중: {file_path}")

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line: continue 

            # 더 강력한 정규표현식: 지수 표기법(e+15) 및 음수를 더 명확히 매칭
            # (?:...)는 비캡처 그룹, [eE][+-]?\d+는 지수 부분입니다.
            pattern = r'([a-zA-Z0-9_]+):\s*([-+]?\d*\.?\d+(?:[eE][+-]?\d+)?)'
            matches = re.findall(pattern, line)
            
            if not matches: continue

            row = {}
            gan_count = 0
            for key, value in matches:
                try:
                    val = float(value)
                    # 만약 값이 너무 커서 무한대(inf)가 되면 매우 큰 숫자로 치환하거나 체크
                    row[f"{key}_{gan_count}" if key == 'gan_loss' and (gan_count := gan_count + 1) > 1 else key] = val
                except ValueError:
                    continue
            
            if row:
                row['iteration'] = (i + 1) * 100
                data.append(row)

    df = pd.DataFrame(data)
    if not df.empty:
        print(f"✅ 파싱 성공! 총 {len(df)}개 행 추출.")
        print(f"📊 데이터 샘플 (첫 5행):\n{df.head()}") # 실제로 값을 읽었는지 여기서 확인 가능
    return df

def plot_loss_analysis(df, selected_losses=None, smoothing_window=10, use_log_scale=True):
    if df.empty: return

    valid_losses = [l for l in (selected_losses or df.columns) if l in df.columns and l != 'iteration']

    plt.figure(figsize=(12, 6))
    for loss in valid_losses:
        # 이동 평균 적용 (폭주하는 경우 윈도우를 작게 잡는 것이 좋습니다)
        smoothed = df[loss].rolling(window=smoothing_window, min_periods=1).mean()
        plt.plot(df['iteration'], smoothed, label=loss, alpha=0.8)
        
        print(f"   🔹 [{loss}] 최댓값: {df[loss].max():.2e} / 최솟값: {df[loss].min():.2e}")

    plt.xlabel('Iteration')
    plt.ylabel('Loss Value' + (' (Log Scale)' if use_log_scale else ''))
    
    # 💡 핵심: Y축을 로그 스케일로 설정 (0.1과 10^15를 동시에 보기 위함)
    if use_log_scale:
        plt.yscale('log')
        
    plt.title('Training Loss Analysis')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tight_layout()
    plt.show()

# # --- 실행 ---
# file_path = 'log/loss/only_L1_Loss.txt' # 파일명에 맞게 수정하세요
file_path = 'log/loss/basic_loss.txt' # 파일명에 맞게 수정하세요
df = parse_log(file_path)

if not df.empty:
    # 1. 전체적인 흐름 확인 (로그 스케일 필수)
    plot_loss_analysis(df, selected_losses=['l1_loss'], use_log_scale=True)
    plot_loss_analysis(df, selected_losses=['percep_loss'], use_log_scale=True)
    plot_loss_analysis(df, selected_losses=['gan_loss'], use_log_scale=True)
    plot_loss_analysis(df, selected_losses=['total_g_loss'], use_log_scale=True)
    plot_loss_analysis(df, selected_losses=['l_d_real', 'l_d_fake'], use_log_scale=True)