# [변경 이력 시작]
#   2026-10-03  최초 생성(복사한 날짜 기준): 미니 데이터로 2 데이터셋 x 프리셋 4개의 전 단계를 돌리는 스크립트
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
$ErrorActionPreference='Continue'; $env:PYTHONIOENCODING='utf8'
Set-Location "G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset\baseline\seld_challenge"
$common = (Get-Content C:\asrwork\mini_common.txt)
function RunStage($st,$ds,$pr){ "=== $ds $pr $st"; conda run -n seld_test --no-capture-output python run.py $st --dataset $ds --preset $pr @common 2>&1 | Select -Last 6; "(exit $LASTEXITCODE)" }
foreach($ds in 'main20','ctrl5'){ foreach($pr in 'B1','B2','B2-0','B3'){
  foreach($st in 'prepare','features','train','eval','swaptest','predict','score'){
    if($pr -eq 'B1' -and $st -eq 'swaptest'){ continue }
    RunStage $st $ds $pr
  } } }
'=== summary'; conda run -n seld_test --no-capture-output python run.py summary --dataset main20 --preset B2 @common 2>&1 | Select -Last 40
'ALLDONE'
