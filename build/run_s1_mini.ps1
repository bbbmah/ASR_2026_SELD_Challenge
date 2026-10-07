# [변경 이력 시작]
#   2026-10-07  최초 생성: 미니 S1 학습 연동 시험(prepare, features, datainfo, train, eval, swaptest, summary) 실행 스크립트
#   2026-10-07  변경 이력 주석 추가
# [변경 이력 끝]
$ErrorActionPreference='Continue'; $env:PYTHONIOENCODING='utf8'
Set-Location "G:\내 드라이브\Colab Notebooks\ASR_2026-2\for_dataset\baseline\seld_challenge"
$common = (Get-Content C:\asrwork\mini_common.txt) + @('--config','C:/asrwork/mini_base.yaml','--set','generated_dir=C:/asrwork/mini/generated')
function St($st,$extra){ "=== $st $extra"; conda run -n seld_test --no-capture-output python run.py $st --dataset main20 --preset B2 @common @extra 2>&1 | Select -Last 14; "(exit $LASTEXITCODE)" }
$s01 = @('--set','train_stages=0,1','--set','updates=12')
St 'prepare' $s01
St 'features' $s01
St 'datainfo' $s01
St 'train' $s01
St 'eval' $s01
St 'swaptest' $s01
$f1 = @('--set','train_stages=0,1','--set','train_family=F1','--set','train_mirror=0','--set','epochs=1')
St 'datainfo' $f1
St 'train' $f1
St 'eval' $f1
'=== summary'; conda run -n seld_test --no-capture-output python run.py summary --dataset main20 --preset B2 @common 2>&1 | Select -Last 30
'ALLDONE'
