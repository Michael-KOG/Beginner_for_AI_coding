import sys, os, json

EXPECTED_ENV_NAME = "rag_llm_GPU"
EXPECTED_PYTHON_PATH = "C:\\Users\\Shimada\\anaconda3\\envs\\rag_llm_GPU\\python.exe"
VSCODE_SETTINGS_PATH = ".vscode/settings.json"

def check_python_path():
    actual_path = sys.executable
    print(f"🔍 Python path: {actual_path}")
    return actual_path == EXPECTED_PYTHON_PATH

def check_conda_env():
    env = os.environ.get("CONDA_DEFAULT_ENV", "Unknown")
    print(f"📦 Conda environment: {env}")
    return env == EXPECTED_ENV_NAME

def fix_vscode_settings():
    print("🛠️ 修正中: VS Code の settings.json")
    os.makedirs(".vscode", exist_ok=True)
    settings = {}
    if os.path.exists(VSCODE_SETTINGS_PATH):
        with open(VSCODE_SETTINGS_PATH, "r", encoding="utf-8") as f:
            try:
                settings = json.load(f)
            except json.JSONDecodeError:
                print("⚠️ settings.json が壊れている可能性があります。新規作成します。")
    settings["python.defaultInterpreterPath"] = EXPECTED_PYTHON_PATH
    with open(VSCODE_SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)
    print("✅ settings.json を修正しました。")

if __name__ == "__main__":
    python_ok = check_python_path()
    conda_ok = check_conda_env()
    if not python_ok:
        print("⚠️ Python path mismatch. 自動修正を試みます。")
        fix_vscode_settings()
    else:
        print("✅ Python path OK")
    if not conda_ok:
        print("⚠️ Conda environment mismatch. 手動確認を推奨します。")
    else:
        print("✅ Conda environment OK")