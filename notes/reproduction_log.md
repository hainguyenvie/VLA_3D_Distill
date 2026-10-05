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

## Kết quả

(điền khi job xong; mọi dòng cũng nằm trong `results/week1_summary.csv`)
