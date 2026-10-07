<!-- [변경 이력 시작]
  2026-10-02  최초 생성: 데이터셋 형식 설명
  2026-10-03  ctrl5 v2 설명 절 추가
  2026-10-07  변경 이력 주석 추가
[변경 이력 끝] -->

# 카메라 이동 AV-SELD 챌린지 데이터셋

STARSS23 개발 데이터(FOA + 360° 영상)로 만든 스테레오 소리 + 투시 영상 데이터. 두 가지가 있다.

| 이름 | 클립 | 카메라 |
|---|---|---|
| `main20` | 20초 | 훑기 궤적: 요(좌우)가 클립 하나에 한 바퀴, 피치가 −25°~0°에서 천천히 오르내림 |
| `ctrl5` | 5초 | main20 창을 4등분, 조각마다 무작위 방향으로 고정, 피치 0 (DCASE 2025 Task 3와 같은 방식, 단 방위각을 접지 않음). 내부용, 비공개 |

## ctrl5 (v2) 에 관해
`ctrl5`는 `main20`의 20초 창을 그대로 가져와 5초씩 4등분하고, 조각마다 카메라를 무작위 방향으로 고정(피치 0)해 원본 360° 영상과 FOA에서 다시 렌더링한 데이터다. 그래서 `main20`과 `ctrl5`는 같은 소리 구간을 쓰고, 다른 것은 카메라와 클립 길이뿐이다. **조각 단위로는 거르지 않았기 때문에 라벨이 하나도 없는 조각이 있다**(헤더만 있는 csv). 소리가 너무 커서 한 조각이라도 피크가 0.9999를 넘으면 그 20초 창을 `main20`과 `ctrl5` 양쪽에서 뺐다(옮겨 둠: `_dropped/`). 이 데이터는 대조 실험용 내부 데이터이며 공개하지 않는다(공개 zip 없음). 이전 버전은 `ctrl5_v1_old/`에 있다(폐기).
부모-조각 대응은 `ctrl5/private/parent_map.csv`(열: `clip, parent_clip, j, start_frame, recording, psi`)에 있다.

## 폴더
```
{main20,ctrl5}/
  train/  video/*.mp4  audio/*.wav  labels/*.csv  meta/*.csv   (공개)
  val/    같은 구조                                            (공개)
  test/   video/*.mp4  audio/*.wav                             (입력만 공개)
  private/ test_labels  test_meta                              (비공개)
  manifest.csv  split_info.json
```
클립 이름은 `{split}_{번호5자리}`; 같은 이름으로 video/audio/labels가 대응한다.

## 형식
- **오디오**: 스테레오(L,R) 24 kHz 16비트 wav. FOA(W,Y,Z,X)에서 `newY = sinθ·X + cosθ·Y`, `L = W + newY`, `R = W − newY`. θ = −ψ(t)이고 샘플마다 바뀐다.
- **영상**: 640×360, **10 fps**, mp4(OpenCV `mp4v`). 수평 시야 100°, 수직 시야 약 67.7°. 영상 프레임 k는 라벨 프레임 k와 같은 시각이다.
- **프레임 시각 규칙**: 프레임 k의 기준 시각은 클립 시작 후 `0.1k + 0.05`초(그 100 ms 구간의 가운데). 카메라 자세도 이 시각의 값. 원본 영상은 이 시각에 가장 가까운 프레임을 썼다.
- **라벨** `labels/{clip}.csv`: `frame,class,source,azimuth,distance,onscreen`
  - `frame`: 0부터, 100 ms 단위. 한 프레임에 여러 행(동시 발생 소리)이 있을 수 있다.
  - `class`: 0~12 (0 여성 말소리, 1 남성 말소리, 2 박수, 3 전화, 4 웃음, 5 생활소음, 6 발걸음, 7 문, 8 음악, 9 악기, 10 수도꼭지, 11 종, 12 노크).
  - `source`: 말하는/연주하는 사람 번호(클립 안에서가 아니라 원 녹음 기준). 0은 사람이 아닌 소리.
  - `azimuth`: **카메라 기준** 방위각(도), 정면 0, **반시계(왼쪽)가 +**, 범위 −180~180. 접지 않음. 같은 소리라도 카메라가 돌면 값이 바뀐다.
  - `distance`: 원본 단위(cm).
  - `onscreen`: 소리 방향이 그 프레임 시야 안이면 1.
- 고도(elevation)는 공개 라벨에 없다(베이스라인 형식과 같음). 피치는 스테레오 소리에 영향이 없다.
- **분석용(공개 split만)** `meta/{clip}_ext.csv`: 위 열 + `az_world, el_world, cam_yaw, cam_pitch`. `meta/{clip}_traj.csv`: 프레임별 카메라 `psi, phi`. 참가자 입력(소리, 영상)에는 카메라 자세가 들어 있지 않다.

## 좌표 규약
- 세계 방향 벡터 `(cos el·cos az, cos el·sin az, sin el)`, x 앞, y 왼쪽, z 위.
- 카메라는 세계 방위각 ψ, 고도 φ를 본다(φ<0은 아래). 라벨 방위각 = wrap(az_world − ψ).
- 시야 안 조건: 카메라 좌표 (x,y,z)에서 x>0, |y| ≤ x·tan 50°, |z| ≤ x·tan 33.84°.

## 분할
- test: 원래 시험 fold의 방 전부. val: 학습 방 중 Tampere 1개와 Sony 1개(`split_info.json`). train: 나머지. 방은 split끼리 섞이지 않는다.
- train 클립은 같은 녹음에서 겹치게 뽑힌다. val/test는 겹치지 않는 창이다.
- 라벨된 프레임이 30개 미만이거나 오디오 피크가 0.9999를 넘는 창은 제외했다.

## 파일
`train.zip`, `val.zip`, `test_inputs.zip`(test 영상·오디오만) 이 공개용이다. `private_test.zip`은 채점용이고 공개하면 안 된다.
