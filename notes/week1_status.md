# Tuần 1 — trạng thái và các quyết định đã chốt

Cập nhật 05/10/2026. Số liệu chi tiết: [reproduction_log.md](reproduction_log.md),
[../results/week1_summary.csv](../results/week1_summary.csv). Bối cảnh related work: [related_work.md](related_work.md).

## Thiết lập thí nghiệm đã chốt

| Hạng mục | Lựa chọn | Lý do / sai khác so với plan |
|---|---|---|
| Benchmark | LIBERO-Object trước (Long sau) | đường cong VLA-OPD ngắn nhất; plan §4 |
| Student | `Haozhan72/Openvla-oft-SFT-libero-object-traj1` (OpenVLA-OFT token rời rạc, 1 demo mỗi task) | đúng student-init của SimpleVLA-RL / VLA-OPD; ta đo 51.4% (paper 54.9) |
| Teacher | `Haozhan72/…-object-trajall` (full-SFT 454 demo, 95.8%), phân phối token được chuyển sang bin của student | VLA-OPD không có code/checkpoint; teacher RL duy nhất công khai cho Object (RLinf) thuộc pipeline khác và chỉ đạt 66% ở đây. Đây là lựa chọn số 2 của plan §4 |
| Loss distill | **forward KL** (teacher‖student) trên 56 token, dạng đóng | reverse-KL của VLA-OPD không hội tụ với teacher khác dòng dõi (xem dưới); bản reverse-KL vẫn có trong code (`--mode rkl`, `rkl_pg`) |
| Cập nhật | LoRA r=32 (mọi lớp linear), AdamW lr 1e-4, batch 8, clip 1.0, 1 epoch trên dữ liệu mới mỗi vòng | VLA-OPD: full-parameter, siêu tham số không công bố |
| Ngân sách | 20 vòng × 2048 state được giám sát (cố định số state, không cố định số episode) | để nhánh state-teacher (episode ngắn) và state-student (episode hỏng dài) có cùng số update |
| Rollout huấn luyện | lấy mẫu T=1.6 từ 50 init state benchmark của mỗi task | như SimpleVLA-RL; train và eval dùng chung init state (cảnh báo rò rỉ đã ghi nhận) |
| Eval trong lúc train | greedy, 100 episode (10 init state đầu × 10 task), mỗi 2 vòng | sai số chuẩn khoảng 4–5 điểm; eval cuối 500 episode |
| Máy | H200, render CPU (OSMesa) cho mọi nhánh của ma trận; L40 (render GPU) để xác nhận checkpoint cuối | render GPU trên H200 bị driver kill khi có CUDA; hai renderer cho success rate khớp nhau (95.2 / 95.8 cho teacher, 500 episode) |

## Ma trận 2×2 (thay cho so sánh B3 và B4 mơ hồ trong plan §7)

Plan định nghĩa Δ_E = SR(B3) − SR(B0/B2 "matched"). Ta làm rõ thành thiết kế 2×2 với cùng teacher, cùng loss,
cùng số state và số update; chỉ khác **ai lái rollout** và **có loss depth hay không**:

| | Không 3D | + depth đặc quyền (oracle của simulator) |
|---|---|---|
| State do **student** đi tới (on-policy) | B2 | B4 |
| State do **teacher** đi tới | B2′ | B3 |

Δ_π = SR(B4) − SR(B2); Δ_E = SR(B3) − SR(B2′). Giả thuyết trung tâm: Δ_π > Δ_E.

Loss depth: head MLP nhẹ đọc 256 token ảnh ở lớp 24 của LLM, dự đoán log-depth 64×64 của chính ảnh agentview
(cùng center crop với input của policy), L1, trọng số 0.5; bỏ head lúc inference.

## Những gì đã biết (LIBERO-Object, một seed)

1. **Mốc tái lập được**: student 51.4% (paper 54.9), full-SFT 95.8% (paper 95.3) trên 500 episode.
2. **Reverse-KL không dùng được với teacher khác dòng dõi.** 6 vòng đầu với reverse-KL: nhánh state-teacher có
   loss đứng yên ở khoảng 6.4 nat/token và eval nhảy 69 → 88 → 70; nhánh on-policy 39 → 61 → 69. Student và
   teacher đều nhọn nhưng lệch nhau trung bình 13–21 bin, nên reverse-KL gần như không có gradient kéo khối
   xác suất về bin của teacher. Hai run này đã dừng ở vòng 7 (thư mục `*_rkl_s7_stopped_iter7` trên H200).
3. **Feature của student giải mã depth tốt, và gần như không kém đi trên state lỗi.** Probe depth trên feature
   đóng băng (sai số tương đối, state held-out): lớp 8: 3.0% (episode thành công) so với 3.4% (episode hỏng);
   lớp 24: 4.1% so với 4.2%; lớp 32: 4.6% so với 4.7%. Thông tin depth giảm dần theo độ sâu của LLM. Đây là
   bằng chứng ban đầu **không ủng hộ** tiền đề "nhận thức 3D của student sụp ở state lệch phân phối".
4. **Teacher full-SFT chỉ kém tự tin hơn một chút trên state lỗi của student** (entropy 1.00 so với 0.79;
   xác suất top-1 0.61 so với 0.67). Mức chênh lớn từng thấy (2.34 so với 0.22) là đặc thù của checkpoint RLinf
   lệch pipeline, không phải hiện tượng chung.
5. **Student lệch teacher nhiều hơn ở episode hỏng**: khoảng cách kỳ vọng action 21 bin so với 13.5 bin.

## Việc đang chạy / tiếp theo

- H200: ba nhánh forward-KL (B2, B2′, B4), B3 nối sau khi biết VRAM mỗi job.
- L40: lặp lại B0 với code mới (kiểm độ ổn định), rồi "teacher tiếp quản" từ state của student
  (`scripts/takeover_eval.py`): đo trực tiếp teacher có cứu được episode từ state student đi tới hay không.
- Sau ma trận: eval 500 episode cho checkpoint cuối trên cả hai renderer; nếu Gate B/C của plan qua thì thêm seed.
- Chuẩn bị sẵn: LIBERO-Plus (repo + assets đã tải về H200) để đo ở chiều Robot-init / Camera / Layout.
