# AirSim 다기체 자율 탐색 · 충돌 회피 RL

Microsoft AirSim(Unreal) 환경에서 **4대 멀티로터**가 협력해 미지 영역을 탐색하고, **redball_01** 타겟을 탐지·추적하는 실험 코드입니다.  
탭형 Q-learning으로 **전방 장애물 회피**를 온라인 학습하며, belief map·에피소드 로그·TensorBoard로 결과를 기록합니다.

> **저장소:** [Capstone-Project](https://github.com/fkrlssh/Capstone-Project.git)  
> **UE 맵·AirSim 실행 파일**은 이 repository에 포함되지 않을 수 있습니다. (예: AirsimForest)

---

## 주요 기능

| 기능 | 설명 |
|------|------|
| **4기체 자율 탐색** | Agent1~4, 소프트웨어 `CommandHub`로 목표·예약·보상 조율 |
| **6방향 그리드 플래너** | `planner_6grid.py` — 미방문·프론티어 기반 목표 선택 |
| **Belief map** | ±100 m 그리드, VISITED / OBSTACLE / TARGET 표시 |
| **타겟** | `redball_01` segmentation 탐지 → Agent1 추적, 맵에 TARGET 마킹 |
| **충돌 회피 RL** | `rl_collision_learner.py` — depth 기반 회피 + Q-table 저장 |
| **RPC 안정화** | `airsim_rpc.py` — 공유 클라이언트·이미지 동시 호출 제한 |
| **백그라운드 학습** | `train_background.py` — `main.py` 감시 + 지표 주기 export |

---

## 요구 사항

- **Windows** (권장), Python 3.9+
- **Unreal Engine + AirSim** (Play 모드에서 시뮬레이션 실행)
- Python: `airsim`, `numpy`, `opencv-python` 등
- **선택** (학습 곡선·TensorBoard): `pip install -r requirements-tensorflow.txt`

---

## 디렉터리 구조

```
11_test/
├── main.py                      # 미션 루프 (에피소드·이륙·충돌 모니터)
├── config.py                    # 전역 설정
├── settings.json                # AirSim 차량·카메라 → Documents/AirSim 복사
├── agent_worker.py              # 기체별 perception + search/track
├── command_hub.py               # 공유 상태·타겟·보상·RL
├── belief_map.py                # 그리드 맵·발표용 union
├── airsim_rpc.py                # AirSim RPC 직렬화
├── camera_utils.py              # depth·장애물 감지
├── target_detector.py           # redball segmentation
├── rl_collision_learner.py
├── plot_collision_learning_tf.py
├── train_background.py
├── export_normalized_indicators.py
├── merge_presentation_grid.py
├── normalized_metrics.py
├── safe_io.py
├── run_background.ps1
├── prepare_12_policy.ps1
└── outputs/                     # 학습 산출물 (.gitignore)
```

---

## 빠른 시작

### 1. AirSim 설정

1. `settings.json`을 **`%USERPROFILE%\Documents\AirSim\settings.json`** 에 복사합니다.
2. UE에서 AirSim 맵을 **Play** 합니다.
3. **`ClockSpeed`는 `1.0` 권장** — `config.py`의 `SIM_CLOCK_SPEED`와 맞추세요.  
   1.5 이상이면 `API call was not received` 호버가 잦아질 수 있습니다.

### 2. Python 실행

```powershell
cd 11_test
python main.py
```

시작 시 확인할 로그:

- `[ENV] agents=4 ['Agent1', 'Agent2', 'Agent3', 'Agent4']`
- `[RPC] shared client + serialized calls...`
- `[MAP] belief grid ... x=[-100,100] ...`

### 3. 학습 곡선 / TensorBoard (선택)

```powershell
pip install -r requirements-tensorflow.txt
python plot_collision_learning_tf.py
python plot_collision_learning_tf.py --last-n 100
tensorboard --logdir outputs/tensorboard/rl_collision
```

### 4. 장시간 백그라운드 실행 (선택)

```powershell
python train_background.py
python train_background.py --metrics-interval 120
.\run_background.ps1
```

---

## 설정 (`config.py`)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `VERSION` | `four_agents_rpc_stable_v26_...` | 설정 스냅샷 라벨 |
| `AGENTS` | Agent1~4 | `settings.json` Vehicles와 일치 |
| `MAP_X/Y_MIN/MAX` | ±100 m | Belief map·지오펜스 |
| `TARGET_OBJECT_NAME` | `redball_01` | UE 씬 객체 이름 |
| `TRACKER_AGENT` | `Agent1` | 타겟 추적·FPV |
| `MAX_EPISODES` | 500 | 세션당 최대 에피소드 |
| `FAST_AGENT_SPEED` | 5.0 m/s | 순항 속도 (테스트 프리셋) |
| `SIM_CLOCK_SPEED` | 1.0 | settings.json `ClockSpeed`와 동기화 |
| `RPC_SHARED_CLIENT_ENABLED` | True | API 부하 완화 |
| `OUTPUT_DIR` | `outputs` | 로그·맵·Q-table |

---

## 산출물 (`outputs/`)

| 파일 | 설명 |
|------|------|
| `episode_log_*.csv` | 에피소드별 충돌·커버리지·보상·shaping |
| `rl_collision_qtable.json` | Q-table (재실행 시 이어 학습) |
| `rl_training_history.jsonl` | Q 상태 수 추이 |
| `belief_grid.*` / `episodes/` | 맵 스냅샷 |
| `belief_presentation_union.*` | 발표용 에피소드 union 맵 |
| `targets.csv` | 타겟 탐지 기록 |
| `normalized_indicators.*` | 정규화 지표 export |

`.gitignore`로 대용량 `outputs`는 기본적으로 Git에 올리지 않습니다.  
팀 공유가 필요하면 Q-table·요약 CSV만 선택해 추가하세요.

---

## API 오류 대응

`API call was not received, entering hover mode` 발생 시:

1. **`ClockSpeed` → 1.0** 후 UE 완전 재시작
2. **카메라 해상도** 낮추기 (`settings.json`의 Width/Height, 기체당 ImageType 0·1·5 동일)
3. `config.py`: `RPC_IMAGE_MAX_CONCURRENT = 1`, `OBSTACLE_CHECK_DT` 증가
4. UE에서 **그림자·반사·Lumen** 비활성화
5. `FAST_AGENT_SPEED` 완화, 기체 수 유지(4대)

---

## UE / 맵 참고

- 장애물(원통·폴리지)은 **UE 에디터**에서 배치합니다. (`OBSTACLE_DENSITY`는 CSV 로그용 라벨)
- 타겟 `redball_01`이 맵 가장자리(x≈89 m 등)에 있으면 belief map **±100 m** 범위가 필요합니다.

---

## 라이선스 · 기여

캡스톤 프로젝트용 코드입니다. 이슈·PR은 [Capstone-Project](https://github.com/fkrlssh/Capstone-Project.git) 저장소를 이용하세요.
