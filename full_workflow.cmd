@echo off
REM ============================================
REM XAIDroid Full Workflow Script
REM ============================================

echo Starting XAIDroid Full Workflow...
echo --------------------------------------------

REM 1. Go to project directory
cd C:\rigzen\AIDroid

REM 2. Activate virtual environment
echo Activating venv...
call venv\Scripts\activate.bat


echo ============================================
echo STEP 1: Preprocessing APKs (Decompile, Extract APIs, Build Graphs)
echo ============================================
pause

python scripts\preprocess_apks.py ^
    --malware_dir data\raw\malware ^
    --benign_dir data\raw\benign ^
    --output_dir data\graphs ^
    --sensitive_apis config\sensitive_apis.json ^
    --n_workers 4 ^
    --train_ratio 0.7 ^
    --val_ratio 0.15 ^
    --test_ratio 0.15 ^
    --seed 42 ^
    --log_dir logs ^
    --aggressive_matching


echo ============================================
echo STEP 2: Train GAT Model
echo ============================================
pause

python scripts\train_gat.py ^
    --config config\config.yaml ^
    --device cuda ^
    --log_dir logs\gat

echo ============================================
echo STEP 3: Train GAM Model
echo ============================================
pause

python scripts\train_gam.py ^
    --config config\config.yaml ^
    --device cuda ^
    --log_dir logs\gam

echo ============================================
echo STEP 4: Evaluate Models (GAT + GAM + Ensemble)
echo ============================================
pause

python scripts\evaluate.py ^
    --test_data data\graphs\test ^
    --gat_model models\gat_best.pt ^
    --gam_model models\gam_best.pt ^
    --output results\evaluation ^
    --device cuda

echo ============================================
echo STEP 5: Inference on New APK
echo ============================================
pause

python scripts\inference.py ^
    --apk_path data\test\unknown.apk ^
    --gat_model models\gat_best.pt ^
    --gam_model models\gam_best.pt ^
    --output results ^
    --device cuda

echo ============================================
echo STEP 6: Visualize Attention
echo ============================================
pause

python scripts\visualize_attention.py ^
    --apk_path data\test\unknown.apk ^
    --gat_model models\gat_best.pt ^
    --gam_model models\gam_best.pt ^
    --output results\attention ^
    --top_k 20 ^
    --device cuda

echo ============================================
echo WORKFLOW COMPLETE!
echo ============================================

pause
