# 12_test Q-table 확인 후 11_test 실측(UE) 실험 준비
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$src = Join-Path (Split-Path $PSScriptRoot -Parent) "12_test\outputs\rl_collision_qtable.json"
if (-not (Test-Path $src)) {
    Write-Host "[ERR] 12_test policy not found. Run first:" -ForegroundColor Red
    Write-Host "  cd ..\12_test" -ForegroundColor Yellow
    Write-Host "  python run_fast.py --fresh --episodes 1500" -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path "outputs")) {
    New-Item -ItemType Directory -Path "outputs" | Out-Null
}

Write-Host "[OK] 12_test policy: $src"
python -c @"
from rl_collision_learner import RLCollisionLearner
from config import CFG
l = RLCollisionLearner()
print(f'[OK] Loaded states={len(l.q)} epsilon={CFG.RL_EPSILON}')
print('[OK] Start UE/AirSim then: python main.py')
"@
