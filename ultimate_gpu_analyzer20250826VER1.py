#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
究極のPC環境・GPUベンチマークツール

システム情報、Python環境の整合性、GPU性能、主要ライブラリの動作を
一つのGUIで診断・分析します。

主な機能:
- システム情報 (OS, CPU, メモリ)
- GPU詳細情報 (nvidia-smiを使用)
- PyTorchとCUDAの連携テスト
- CPU vs GPUの行列計算ベンチマーク (100x100, 10000x10000)
- VSCodeとターミナルのPython環境整合性チェック
- 主要な機械学習ライブラリのインストール確認

必要なライブラリ:
pip install FreeSimpleGUI torch psutil cupy-cuda12x nvidia-ml-py3
(cupy-cuda12xの部分はご自身のCUDAバージョンに合わせてください)
"""
import sys
import os
import platform
import subprocess
import time
import importlib
import json

# --- ライブラリのインポートとGUI利用可否の判定 ---
try:
    import FreeSimpleGUI as sg
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("警告: FreeSimpleGUIがインストールされていません。GUIは利用できません。")
    print("インストールコマンド: pip install FreeSimpleGUI")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# --- 各種診断機能 ---

def get_system_info():
    """OS、CPU、メモリなどの基本システム情報を取得します。"""
    report = ["\n--- 💻 システム情報 ---"]
    if not PSUTIL_AVAILABLE:
        report.append("❌ psutilライブラリが見つかりません。詳細なシステム情報を取得できません。")
        report.append("   インストールコマンド: pip install psutil")
        return report

    try:
        report.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
        report.append(f"プロセッサ: {platform.processor()}")
        
        # CPU情報
        cpu_cores_physical = psutil.cpu_count(logical=False)
        cpu_cores_logical = psutil.cpu_count(logical=True)
        report.append(f"CPUコア数: {cpu_cores_physical} (物理) / {cpu_cores_logical} (論理)")
        
        # メモリ情報
        memory = psutil.virtual_memory()
        total_mem_gb = memory.total / (1024**3)
        available_mem_gb = memory.available / (1024**3)
        report.append(f"メモリ(RAM): 合計 {total_mem_gb:.2f} GB / 空き {available_mem_gb:.2f} GB ({memory.percent} % 使用中)")

    except Exception as e:
        report.append(f"❌ システム情報の取得中にエラーが発生しました: {e}")
    return report

def get_gpu_info():
    """nvidia-smiコマンドを使用してGPUの詳細情報を取得します。"""
    report = ["\n--- 🎮 GPU情報 (nvidia-smi) ---"]
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        )
        gpus = result.stdout.strip().split('\n')
        report.append(f"✅ NVIDIA GPUを {len(gpus)} 台検出しました。")
        for i, line in enumerate(gpus):
            name, driver, mem_total, mem_used, util, temp = [p.strip() for p in line.split(',')]
            report.append(f"  [GPU {i}]")
            report.append(f"    モデル名: {name}")
            report.append(f"    ドライバVer: {driver}")
            report.append(f"    メモリ: {mem_used} MB / {mem_total} MB 使用中")
            report.append(f"    使用率: {util} %")
            report.append(f"    温度: {temp} °C")

    except FileNotFoundError:
        report.append("⚠️ nvidia-smiコマンドが見つかりません。NVIDIAドライバがインストールされていない可能性があります。")
    except subprocess.CalledProcessError as e:
        report.append(f"❌ nvidia-smiの実行に失敗しました: {e.stderr}")
    except Exception as e:
        report.append(f"❌ GPU情報の取得中に予期せぬエラーが発生しました: {e}")
    return report

def get_pytorch_info():
    """PyTorchとCUDAの連携状況を確認します。"""
    report = ["\n--- 🔥 PyTorch & CUDA連携 ---"]
    if not TORCH_AVAILABLE:
        report.append("❌ PyTorchがインストールされていません。")
        report.append("   公式サイトを参考にインストールしてください: https://pytorch.org/")
        return report

    report.append(f"✅ PyTorch バージョン: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    report.append(f"CUDA利用可能か: {'✅ はい' if cuda_available else '❌ いいえ'}")

    if cuda_available:
        report.append(f"PyTorchビルド時CUDA Ver: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        report.append(f"検出されたCUDAデバイス数: {device_count}")
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            total_mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            report.append(f"  GPU {i}: {device_name} ({total_mem_gb:.2f} GB)")
        
        cudnn_ver = torch.backends.cudnn.version()
        report.append(f"cuDNN バージョン: {cudnn_ver}")
    else:
        report.append("⚠️ GPUはシステムに存在しますが、PyTorchがCUDAを利用できません。")
        report.append("   PyTorchがCPU版としてインストールされている可能性があります。")
        report.append("   GPU対応版のPyTorchを再インストールしてください。")
        
    return report

def check_environment_consistency():
    """VSCodeとターミナルのPython環境の整合性を診断します。"""
    report = ["\n--- 🐍 Python環境 整合性診断 ---"]
    try:
        # 1. 現在のスクリプトが使用しているPythonインタプリタ
        script_interpreter = sys.executable
        report.append(f"📜 スクリプト実行中のPython: {script_interpreter}")

        # 2. 環境変数から仮想環境パスを取得
        venv_path = os.environ.get('VIRTUAL_ENV')
        conda_env_path = os.environ.get('CONDA_PREFIX')
        report.append(f"📦 VIRTUAL_ENV (venv/virtualenv): {venv_path or '未設定'}")
        report.append(f"📦 CONDA_PREFIX (Conda): {conda_env_path or '未設定'}")

        # 3. ターミナルが認識する `python` コマンドのパス
        shell_command = 'where' if platform.system() == "Windows" else 'which'
        result = subprocess.run([shell_command, 'python'], capture_output=True, text=True, encoding='utf-8')
        terminal_interpreter = result.stdout.strip().split('\n')[0]
        report.append(f"🖥️ ターミナルの'python'コマンド: {terminal_interpreter}")
        
        # --- 診断ロジック ---
        report.append("\n[診断結果]")
        match = True
        
        # VSCode(スクリプト)とターミナルが一致しているか
        if os.path.normcase(script_interpreter) != os.path.normcase(terminal_interpreter):
            report.append("⚠️ 不一致: スクリプト実行中のPythonとターミナルのPythonが異なります。")
            match = False

        # 仮想環境が有効なのに、スクリプトがその中のPythonを使っていない場合
        active_env_path = venv_path or conda_env_path
        if active_env_path and not script_interpreter.startswith(active_env_path):
            report.append("⚠️ 不一致: 仮想環境が有効化されていますが、VSCodeが別のPythonを見ています。")
            report.append(f"   (有効な環境: {active_env_path})")
            match = False
            
        if match:
            report.append("✅ 良好: VSCodeとターミナルは同じPython環境を認識しています。")
        else:
            report.append("\n[対策]")
            report.append("  - VSCodeの右下でPythonインタプリタを再選択してください。")
            report.append(f"  - ターミナルで '{active_env_path}' が正しく有効化されているか確認してください。")

    except Exception as e:
        report.append(f"❌ 整合性診断中にエラーが発生しました: {e}")
    return report

def run_performance_benchmark():
    """CPUとGPUの行列計算速度を比較するベンチマークを実行します。"""
    report = ["\n--- 🚀 CPU vs GPU ベンチマーク (行列計算) ---"]
    if not TORCH_AVAILABLE:
        report.append("❌ PyTorchがないため、ベンチマークを実行できません。")
        return report

    def measure_time(device, size):
        try:
            tensor_a = torch.randn(size, size, device=device)
            tensor_b = torch.randn(size, size, device=device)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            for _ in range(5):  # 複数回実行して平均化
                torch.matmul(tensor_a, tensor_b)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            # メモリ解放
            del tensor_a, tensor_b
            if device == 'cuda':
                torch.cuda.empty_cache()

            return (end_time - start_time) / 5
        except torch.cuda.OutOfMemoryError:
            return "メモリ不足"
        except Exception as e:
            return f"エラー: {str(e)[:50]}"

    # --- 100x100 (軽量) ベンチマーク ---
    report.append("\n[軽量テスト: 100x100 行列]")
    cpu_time_light = measure_time('cpu', 100)
    report.append(f"  CPU 実行時間: {cpu_time_light * 1000:.4f} ms")

    if torch.cuda.is_available():
        gpu_time_light = measure_time('cuda', 100)
        report.append(f"  GPU 実行時間: {gpu_time_light * 1000:.4f} ms")
        if isinstance(cpu_time_light, float) and isinstance(gpu_time_light, float) and gpu_time_light > 0:
            speedup = cpu_time_light / gpu_time_light
            report.append(f"  🚀 速度比: GPUはCPUの {speedup:.2f} 倍高速")
    
    # --- 10000x10000 (重量) ベンチマーク ---
    report.append("\n[重量テスト: 10000x10000 行列]")
    cpu_time_heavy = measure_time('cpu', 10000)
    if isinstance(cpu_time_heavy, float):
        report.append(f"  CPU 実行時間: {cpu_time_heavy:.4f} 秒")
    else:
        report.append(f"  CPU 実行結果: {cpu_time_heavy}")


    if torch.cuda.is_available():
        gpu_time_heavy = measure_time('cuda', 10000)
        if isinstance(gpu_time_heavy, float):
            report.append(f"  GPU 実行時間: {gpu_time_heavy:.4f} 秒")
            if isinstance(cpu_time_heavy, float) and gpu_time_heavy > 0:
                speedup = cpu_time_heavy / gpu_time_heavy
                report.append(f"  🚀🚀 速度比: GPUはCPUの {speedup:.2f} 倍高速 (大規模計算)")
        else:
            report.append(f"  GPU 実行結果: {gpu_time_heavy}")
            
    return report

def check_key_packages():
    """主要な機械学習・データサイエンスライブラリのインストール状況を確認します。"""
    report = ["\n--- 📚 主要ライブラリ確認 ---"]
    packages = [
        'numpy', 'pandas', 'matplotlib', 'scipy', 
        'sklearn', 'cupy', 'psutil', 'requests'
    ]
    
    for package in packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', '不明')
            report.append(f"  ✅ {package.ljust(12)}: v{version}")
        except ImportError:
            report.append(f"  ❌ {package.ljust(12)}: 未インストール")
        except Exception as e:
            report.append(f"  ⚠️ {package.ljust(12)}: インポートエラー ({e})")
            
    return report

def run_all_diagnostics():
    """全ての診断機能を順番に実行し、結果を一つの文字列にまとめます。"""
    full_report = []
    full_report.append("究極のPC環境・GPUベンチマークツール - 診断レポート")
    full_report.append(f"診断日時: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 各診断を実行
    full_report.extend(get_system_info())
    full_report.extend(get_gpu_info())
    full_report.extend(get_pytorch_info())
    full_report.extend(check_environment_consistency())
    full_report.extend(check_key_packages())
    full_report.extend(run_performance_benchmark())
    
    full_report.append("\n--- ✅ 診断完了 ---")
    
    return "\n".join(full_report)

# --- GUIメイン処理 ---
def main_gui():
    """FreeSimpleGUIを使用してGUIを構築・実行します。"""
    sg.theme("SystemDefault")

    layout = [
        [sg.Text("究極のPC環境・GPUベンチマークツール", font=("Meiryo", 16, "bold"))],
        [sg.Multiline(
            "「解析実行」ボタンを押して、システム診断とベンチマークを開始してください。",
            size=(100, 30),
            key="-OUTPUT-",
            font=("Consolas", 10),
            disabled=True
        )],
        [
            sg.Button("解析実行", size=(12, 1), key="-RUN-"),
            sg.Button("レポート保存", size=(12, 1), key="-SAVE-", disabled=True),
            sg.Button("終了", size=(10, 1), key="-EXIT-")
        ],
        [sg.StatusBar("", size=(80, 1), key="-STATUS-")]
    ]

    window = sg.Window("究極の環境アナライザー", layout, finalize=True)
    
    report_content = ""

    while True:
        event, values = window.read()

        if event in (sg.WINDOW_CLOSED, "-EXIT-"):
            break
        
        elif event == "-RUN-":
            window["-RUN-"].update(disabled=True)
            window["-SAVE-"].update(disabled=True)
            window["-STATUS-"].update("解析を実行中... これには数分かかることがあります。")
            window["-OUTPUT-"].update("解析を実行中...\n\nCPUとGPUの重量ベンチマーク（10000x10000行列計算）には時間がかかりますので、しばらくお待ちください。")
            window.refresh() # UIを即時更新

            report_content = run_all_diagnostics()
            
            window["-OUTPUT-"].update(report_content)
            window["-STATUS-"].update("解析が完了しました。")
            window["-RUN-"].update(disabled=False)
            window["-SAVE-"].update(disabled=False)

        elif event == "-SAVE-":
            try:
                save_path = sg.popup_get_file(
                    "レポートをテキストファイルとして保存",
                    save_as=True,
                    default_extension=".txt",
                    file_types=(("Text Files", "*.txt"), ("All Files", "*.*")),
                    no_window=True
                )
                if save_path:
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(report_content)
                    sg.popup("成功", f"レポートを {save_path} に保存しました。")
            except Exception as e:
                sg.popup_error("エラー", f"ファイルの保存に失敗しました: {e}")

    window.close()

if __name__ == "__main__":
    if GUI_AVAILABLE:
        main_gui()
    else:
        # GUIが利用できない場合のフォールバック
        print("--- CUIモードで診断を実行します ---")
        report = run_all_diagnostics()
        print(report)