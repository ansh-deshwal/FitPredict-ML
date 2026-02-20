"""Master runner script"""
import sys
import subprocess
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

STEPS = [
    ("1_prepare_data.py", "Data Preparation"),
    ("2_train_autoencoder.py", "Train Autoencoder"),
]

def run_step(step_num):
    if step_num < 1 or step_num > len(STEPS):
        print(f"✗ Invalid step: {step_num}")
        return False
    
    script_name, description = STEPS[step_num - 1]
    script_path = SCRIPTS_DIR / script_name
    
    print(f"\n{'='*70}")
    print(f"RUNNING STEP {step_num}: {description}")
    print(f"{'='*70}\n")
    
    try:
        subprocess.run([sys.executable, str(script_path)], check=True, cwd=str(SCRIPTS_DIR))
        print(f"\n✓ Step {step_num} complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Step {step_num} failed with code {e.returncode}")
        return False

def main():
    for step in range(1, len(STEPS) + 1):
        if not run_step(step):
            print(f"\n✗ Pipeline failed at step {step}")
            sys.exit(1)
    
    print(f"\n{'='*70}")
    print("✓ PIPELINE COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()