#!/usr/bin/env python3
import os
import shutil
import site

# Find fairseq in site-packages
site_packages = site.getsitepackages()[0]
fairseq_path = os.path.join(site_packages, "fairseq")

if os.path.exists(fairseq_path):
    problematic_files = [
        "tasks/speech_dlm_task.py",
        "tasks/online_backtranslation.py",
        "models/speech_to_speech/__init__.py",
        "models/speech_to_speech/s2s_conformer.py",
        "models/speech_to_speech/s2s_transformer.py",
    ]
    
    for rel_path in problematic_files:
        full_path = os.path.join(fairseq_path, rel_path)
        backup_path = full_path + ".bak"
        
        if os.path.exists(full_path) and not os.path.exists(backup_path):
            shutil.move(full_path, backup_path)
            with open(full_path, 'w') as f:
                f.write("# Disabled due to missing dependencies\n")
            print(f"Patched: {rel_path}")
    
    print("Fairseq patched successfully!")
else:
    print(f"Warning: Could not find fairseq at {fairseq_path}")
