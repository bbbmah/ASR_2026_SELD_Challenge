# [변경 이력 시작]
#   2026-10-03  최초 생성(복사한 날짜 기준): 접미사 없는 실행 이름으로 gaps 계산, fix20 평가, 제외 목록 시험
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
$ErrorActionPreference='Continue'; $env:PYTHONIOENCODING='utf8'
Set-Location "G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset\baseline\seld_challenge"
$common = (Get-Content C:\asrwork\mini_common.txt) + @('--config','C:/asrwork/mini_base.yaml')
function RunStage($st,$ds,$pr,$more=@()){ "=== $ds $pr $st $more"; conda run -n seld_test --no-capture-output python run.py $st --dataset $ds --preset $pr @common @more 2>&1 | Select -Last 5; "(exit $LASTEXITCODE)" }
foreach($ds in 'main20','ctrl5'){ foreach($pr in 'B1','B2','B2-0','B3'){
  foreach($st in 'train','eval','swaptest'){ if($pr -eq 'B1' -and $st -eq 'swaptest'){ continue }; RunStage $st $ds $pr } } }
# 다른 데이터셋(fix20)으로 평가 + 제외 목록
foreach($ds in 'main20','ctrl5'){
  RunStage 'prepare' $ds 'B2' @('--set','eval_dataset=fix20')
  RunStage 'features' $ds 'B2' @('--set','eval_dataset=fix20')
  RunStage 'eval' $ds 'B2' @('--set','eval_dataset=fix20')
  RunStage 'eval' $ds 'B2' @('--set','exclude_list=configs/exclude_val_fix20.txt')
}
'=== summary'; conda run -n seld_test --no-capture-output python run.py summary --dataset main20 --preset B2 @common 2>&1 | Select -Last 40
'ALLDONE'
