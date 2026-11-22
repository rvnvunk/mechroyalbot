# Mech Royale Survival Bot

Builders Buildathon, Mech Royale 트랙용 Python 봇. 표준 라이브러리만 사용해서 별도 패키지 설치가 필요 없음.
트랙 규칙 정리는 `docs/MECH_ROYALE_PROJECT.md` 참고.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `Dockerfile` | 제출용 이미지 (저장소 최상단에 있어야 함) |
| `bot.py` | 봇 두뇌 (이동, 조준, 사격, 업그레이드 선택) |
| `mechio.py` | mechfight/1 입출력 루프 (밀린 관측은 최신 것만 답함) |
| `tools/smoke_test.py` | 로컬 간단 테스트 (공식 check 아님) |
| `docs/MECH_ROYALE_PROJECT.md` | 트랙 규칙 요약 |

## 전략 요약

1. **자기장 우선**: 다음 원 안쪽 65% 지점까지 미리 이동. 남은 시간이 부족하면 부스트.
2. **정면 유지**: 위협 쪽으로 몸통을 돌리고(뒤에서 맞으면 1.5배) 다리로는 뒤로 물러남.
3. **HP 40% 미만이면 도망**: 150m 거리 확보까지 부스트로 후퇴, 5초 지나면 HP 재생.
4. **전투 중 좌우 무빙** + 날아오는 총알/포탄 회피.
5. **안전할 때만 파밍**: 주변에 적이 없을 때 큐브, 상자, NPC 드론 사냥 (큐브 = 최대 HP 증가).
6. **두 번째 무기 줍기**: 빈 슬롯이 있으면 가까운 무기로 이동해서 `take`.
7. **보급(supply drop)**: 아무도 없을 때만 점령 (HP 풀회복 + 큐브 3개).
8. **업그레이드**: 방어구 > HP/재생 > 속도/에너지 > 나머지 순.
9. **조용히 이동**: 할 일 없을 때는 5 m/s 미만으로 걸어서 발소리 안 냄.
10. **끼임 탈출**: 1초 가까이 안 움직이면 방향 틀고 점프+제트.

## 로컬 테스트

```sh
python3 tools/smoke_test.py -- python3 -u bot.py
```

## 공식 체크 (킷 필요)

`mechfight-kit.tar.gz`를 트랙 페이지에서 받아서 `check/` 폴더와 `arena/` 폴더(맵)를 이 저장소 옆이나 안에 풀어 둔 뒤:

```sh
python3 check/check.py -- python3 -u bot.py

docker build -t my-bot .
python3 check/check.py --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" my-bot
```

> `check/`, `arena/`는 `.gitignore`에 넣어 두었음 (저장소 20MB 제한).

## 제출

1. GitHub에 **public** 저장소를 만들고 이 폴더 내용을 그대로 push (Dockerfile이 최상단)
2. 트랙 페이지 Submit 칸에 저장소 URL 입력 (커밋 비우면 기본 브랜치 최신 커밋)
3. 1시간에 4번까지 제출 가능. 하나 바꿀 때마다 점수 비교 (기준선 1.15)

## 다음 개선 아이디어

- `/arena/v2/world.json`의 건물(prism)과 나무 데이터를 읽어서 엄폐물 뒤로 숨기, 나무 숲에 숨기 (시야 차단)
- 시야 체크로 벽에 총 쏘는 낭비 줄이기
- 소리(`sounds`)로 적 위치 추적해서 미리 피하기
- 상대가 든 무기에 맞춘 거리 조절 (번개는 45m, 박격포는 30m 이내가 약점)
- `hello.config` 숫자를 더 많이 활용 (밸런스가 바뀔 수 있음)
