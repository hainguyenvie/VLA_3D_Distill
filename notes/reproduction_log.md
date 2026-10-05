# Reproduction log

Mỗi mục ghi theo mẫu ở plan §16. Số của paper chỉ là mốc neo; chỉ gọi là "reproduced" khi setup khớp.

## Protocol đánh giá cố định của project (LIBERO, policy token rời rạc)

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Model | OpenVLA-OFT biến thể token rời rạc: 1 ảnh third-person 224, không wrist, không proprio, chunk 8, 56 token × 256 bin | SimpleVLA-RL |
| Tiền xử lý ảnh | agentview 256 xoay 180°, JPEG encode/decode, resize lanczos3 về 224, center crop 0.9 | openvla-oft `openvla_utils.py` |
| Số học | trọng số bf16, forward dưới `autocast(bf16)` | SimpleVLA-RL rollout |
| Episode | init state benchmark thứ 0..49 của mỗi task, 10 bước chờ, thực thi open-loop cả chunk 8 | openvla-oft / SimpleVLA-RL |
| Horizon | **512 bước cho mọi suite** (SimpleVLA-RL); openvla-oft dùng 280 cho Object. Log `env_steps` cho phép tính lại success ở horizon ngắn hơn | SimpleVLA-RL `rob_rollout.py` |
| Decode | greedy khi đánh giá; rollout huấn luyện lấy mẫu T=1.6 | SimpleVLA-RL |
| Chuẩn hoá action | q01/q99 của `libero_object_no_noops` trong `dataset_statistics.json` của từng checkpoint | checkpoint |
| Phần mềm | python 3.10, torch 2.2.0+cu121, transformers fork openvla-oft 4.40.1, robosuite 1.4.1, **mujoco 3.2.3**, LIBERO @ 8f1084e, openvla-oft @ e4287e9 | `scripts/server/build_env.sh` |
| Phần cứng | NVIDIA L40 | — |

Sai khác đã biết so với upstream:
- Suy luận theo batch nhiều env (upstream batch 1). Trong bf16, cùng một state nhưng khác thành phần batch làm
  logit lệch trung bình 0.18 và lật khoảng 1% token argmax (đo bằng `scripts/check_policy.py`); padding không
  gây thêm sai số. Hệ quả: kết quả từng episode không tái lập tuyệt đối giữa hai lần chạy có batch khác nhau.
- mujoco 3.2.3 thay vì bản mới nhất: robosuite 1.4.1 lỗi `AssertionError` khi dựng env với mujoco 3.14.
- Teacher phải dùng **cùng thống kê chuẩn hoá action** với student thì KL theo token mới có nghĩa. Teacher
  GRPO của RLinf và student traj1 dùng chung thống kê (10 trajectory); checkpoint full-SFT (454 trajectory)
  thì không, nên không dùng làm teacher mức token được.

## Gate kỹ thuật

| Gate | Kết quả | Lệnh |
|---|---|---|
| Policy wrapper = upstream `predict_action` | action trùng khớp trên 12/12 state (batch 1); nạp checkpoint không thiếu/thừa key | `scripts/check_policy.py` |
| Camera + depth oracle | sai số depth chéo view trung vị 1.2 mm (agent→wrist) và 2.2 mm (wrist→agent); điểm eef chiếu đúng vào gripper | `scripts/check_geometry.py` |

## Kết quả mốc (LIBERO-Object, greedy, horizon 512, seed 7)

| Model | Máy / renderer | Episode | Của ta | Đã công bố | Nguồn số công bố |
|---|---|---|---|---|---|
| Student 1-traj SFT (`Haozhan72/…-object-traj1`) | L40 / GPU EGL | 500 | **51.4** | 54.9 | SimpleVLA-RL Table 5 |
| Full-SFT 454 demo (`Haozhan72/…-object-trajall`) | L40 / GPU EGL | 500 | **95.8** | 95.3 | SimpleVLA-RL Table 2 |
| RLinf GRPO (`RLinf/…-GRPO-LIBERO-object`) | L40 / GPU EGL | 500 | **66.4** | 97.7 | model card, pipeline RLinf |
| Student 1-traj | H200 / CPU OSMesa | 100 | 53 | — | cùng 100 episode trên L40: 52 |
| Full-SFT | H200 / CPU OSMesa | 100 | 93 | — | cùng 100 episode trên L40: 98 |
| RLinf GRPO | H200 / CPU OSMesa | 100 | 75 | — | cùng 100 episode trên L40: 70 |

Sai số chuẩn nhị thức: khoảng 2.2 điểm cho 500 episode ở mức 50%, khoảng 5 điểm cho 100 episode.

Nhận xét:
- Student và full-SFT tái lập được số của SimpleVLA-RL trong sai số → pipeline đánh giá đáng tin.
- Horizon gần như không ảnh hưởng tới student: 50.8% ở 280 bước, 51.4% ở 512 bước (episode thành công có
  trung vị 138 bước).
- **Checkpoint RLinf không thuộc pipeline này.** Theo code RLinf: ảnh không center-crop, không JPEG, resize
  bicubic bằng torchvision; reset bằng action toàn 0 nên gripper nửa mở; rollout và eval đều lấy mẫu T=1.6;
  model được fine-tune toàn bộ bằng RL trong đúng điều kiện đó. Số 28.83 của họ cho student cũng đến từ
  pipeline ấy. Vì vậy 66.4% ở đây không mâu thuẫn với 97.7% trên card, và ta **không dùng nó làm teacher**.
- **Teacher của project: full-SFT**, đưa về bin của student bằng chuyển khối xác suất theo phần giao của các
  khoảng action (`src/policy/rebin.py`, gate `scripts/check_rebin.py`: kỳ vọng action lệch trung vị 0.001 bin,
  phân vị 99 là 0.026 bin; 0.23% giá trị nằm ngoài dải của student). Đây là lựa chọn số 2 trong plan §4
  ("checkpoint mạnh nhất tái lập được, huấn luyện với nhiều demo hơn"). Sai khác so với VLA-OPD: teacher của
  họ là policy RL (96.1 trên Object), không công bố.

## Renderer

- Trên cùng state, render GPU (NVIDIA EGL) và render CPU (OSMesa llvmpipe) lệch trung bình 1.7/255 mỗi pixel,
  khoảng 4% pixel lệch hơn 8 mức (viền texture mảnh, khử răng cưa). Logits của student lệch KL 0.31,
  argmax trùng 84%; sàn nhiễu (batch 1 so với 2) là KL 0.009, trùng 97%.
- Cùng frame, hai máy khác nhau: KL 0.011 → số học hai máy tương đương, khác biệt là do renderer.
- Ở mức success rate, khác biệt nằm trong nhiễu của 100 episode (bảng trên). Cần 500 episode để biết mức lệch
  5 điểm của full-SFT có thật không.
- Frame của `EnvRunner` (render 1 lần mỗi chunk) trùng từng bit với cách step upstream trên 4 episode liên
  tiếp, kể cả sau episode kết thúc giữa chunk (`scripts/check_env_stepping.py`).

## Các lần chạy đã loại bỏ

- Mọi kết quả trên H200 trước commit `821d567` (thư mục `outputs/week1/_bug_stale_frames` trên H200): bug
  timer camera làm frame trễ hoặc đen từ episode thứ hai của mỗi worker. Con số "render CPU làm tụt 20–35 điểm"
  suy ra từ các lần chạy đó là sai.

## Chẩn đoán trên rollout của student (L40, 500 episode, 20 152 state, teacher gán nhãn = RLinf GRPO)

| Trên state student đi qua | Episode thành công | Episode hỏng |
|---|---|---|
| KL(student‖teacher) mỗi token | 1.09 | 2.08 |
| Tỉ lệ token argmax trùng teacher | 76% | 52% |
| Entropy của student | 0.12 | 0.38 |
| Entropy của teacher | 0.22 | 2.34 |

- KL theo tiến độ episode: thành công 1.00 → 0.87 → 1.58 (cuối); hỏng 1.56 → 2.41, tăng đều.
- AUC dự đoán "episode sẽ hỏng" từ KL: 0.75 chỉ với 5 query đầu, 0.99 với cả episode.
- Lưu ý: teacher gán nhãn ở đây là checkpoint RLinf, vốn lệch pipeline; cần lặp lại với teacher full-SFT
  trước khi dùng các số này làm luận cứ.
