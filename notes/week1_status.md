# Tuần 1 — trạng thái và các quyết định đã chốt

Cập nhật 05/10/2026. Số liệu chi tiết: [reproduction_log.md](reproduction_log.md),
[../results/week1_summary.csv](../results/week1_summary.csv). Bối cảnh related work: [related_work.md](related_work.md).

## Thiết lập thí nghiệm đã chốt

| Hạng mục | Lựa chọn | Lý do / sai khác so với plan |
|---|---|---|
| Benchmark | LIBERO-Object trước (Long sau) | đường cong VLA-OPD ngắn nhất; plan §4 |
| Student | `Haozhan72/Openvla-oft-SFT-libero-object-traj1` (OpenVLA-OFT token rời rạc, 1 demo mỗi task) | đúng student-init của SimpleVLA-RL / VLA-OPD; ta đo 51.4% (paper 54.9) |
| Teacher | `Haozhan72/…-object-trajall` (full-SFT 454 demo, 95.8%), phân phối token được chuyển sang bin của student | VLA-OPD không có code/checkpoint; teacher RL duy nhất công khai cho Object (RLinf) thuộc pipeline khác và chỉ đạt 66% ở đây. Đây là lựa chọn số 2 của plan §4 |
| Loss distill | **reverse KL** (student‖teacher) trên 56 token, dạng đóng | như VLA-OPD (họ dùng estimator một mẫu; ta lấy kỳ vọng chính xác). Forward-KL và cross-entropy đã thử và sụp, xem dưới |
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

## Trần và headroom (đo trước khi thử phương pháp)

**LIBERO-Object chuẩn** (greedy, 500 episode): student 51.4% → teacher 95.8%. Trần của distill thuần là teacher.
Nhánh distill không-3D trên state của teacher từng chạm 88% chỉ sau 4 vòng (reverse-KL, chưa ổn định), nên phần
còn lại cho "3D thêm vào" trên benchmark chuẩn nhiều nhất chỉ vài điểm, ngang mức nhiễu của eval. **Không nên
lấy LIBERO chuẩn làm thước đo chính cho đóng góp 3D.**

**LIBERO-Plus, suite Object** (zero-shot, 60 task mỗi loại nhiễu, 1 trial, render CPU trên H200):

| Loại nhiễu | Teacher full-SFT (95.8% trên bản chuẩn) | OpenVLA-OFT đã công bố (2 camera + proprio) |
|---|---|---|
| Camera Viewpoints | **13.3** | 38.9 |
| Light Conditions | **30.0** | 73.7 |
| Sensor Noise | 43.3 | 72.3 |
| Robot Initial States | 55.0 | 25.4 |
| Objects Layout | 58.3 | 71.8 |
| Background Textures | 73.3 | 97.6 |
| Language Instructions | 75.0 | 99.0 |
| **Tổng** | **49.8** | 66.5 |

Theo mức khó: L1 79% → L5 27%.

Student 1-traj trên cùng 420 task: **tổng 21.2%** (Camera 3.3, Noise 6.7, Light 13.3, Robot-init 18.3, Layout
33.3, Background 35.0, Language 38.3).

Đọc bảng này:
- Headroom thật nằm ở đây: ngay cả teacher cũng chỉ đạt một nửa, và trần **không** bị teacher chặn, vì giám sát
  3D có thể cho student độ bền mà teacher không có.
- Camera và Light là hai chiều yếu nhất của model 1-camera, cũng là hai chiều mà các paper distill geometry
  (GaussianWAM, MVUCF) tăng mạnh nhất trên backbone khác.
- Cần xác nhận lại trên L40 (render GPU) trước khi so với số đã công bố: nhiễu ánh sáng và camera có thể nhạy
  với renderer hơn bản chuẩn. LIBERO-Plus đã cài xong trên L40.

## Những gì đã biết (LIBERO-Object, một seed)

1. **Mốc tái lập được**: student 51.4% (paper 54.9), full-SFT 95.8% (paper 95.3) trên 500 episode.
2. **Chọn loss: reverse-KL là công thức duy nhất không sụp** (eval greedy 100 episode, mốc 52%, teacher 95.8%):

   | Loss | State của student | State của teacher |
   |---|---|---|
   | reverse-KL | 39 → 61 → 69 (vòng 2, 4, 6) | 69 → 88 → 70 |
   | forward-KL | 33 → 39 → 28 | 37 → 48 → 50 |
   | cross-entropy theo argmax của teacher | — | 17 (vòng 2) |

   Teacher khác dòng dõi với student (lệch trung bình 13–21 bin), nên loss nào kéo mạnh khối xác suất sang bin
   của teacher (forward-KL, CE) đều đưa student qua một vùng "nửa nọ nửa kia" nơi action lẫn lộn; entropy của
   student phình lên 2.5–2.9 nat. Reverse-KL dịch chuyển chậm và giữ student nhọn. Điều này khớp với ablation
   của VLA-OPD. Tôi đã dừng nhầm hai run reverse-KL ở vòng 7 rồi chạy tiếp từ checkpoint; đường cong của nhánh
   state-teacher còn nhảy (88 → 70), cần xem tới vòng 20.
3. **Feature của student giải mã depth tốt, và gần như không kém đi trên state lỗi.** Probe depth trên feature
   đóng băng (sai số tương đối, state held-out): lớp 8: 3.0% (episode thành công) so với 3.4% (episode hỏng);
   lớp 24: 4.1% so với 4.2%; lớp 32: 4.6% so với 4.7%. Thông tin depth giảm dần theo độ sâu của LLM. Đây là
   bằng chứng ban đầu **không ủng hộ** tiền đề "nhận thức 3D của student sụp ở state lệch phân phối".
4. **Teacher full-SFT chỉ kém tự tin hơn một chút trên state lỗi của student** (entropy 1.00 so với 0.79;
   xác suất top-1 0.61 so với 0.67). Mức chênh lớn từng thấy (2.34 so với 0.22) là đặc thù của checkpoint RLinf
   lệch pipeline, không phải hiện tượng chung.
5. **Student lệch teacher nhiều hơn ở episode hỏng**: khoảng cách kỳ vọng action 21 bin so với 13.5 bin.

## Việc đang chạy / tiếp theo

- H200: reverse-KL trên state student (B2) và state teacher (B2′) chạy tiếp tới vòng 20; B4 (state student +
  depth) chạy mới; eval LIBERO-Plus của checkpoint student đã distill (reverse-KL, state teacher, vòng 4).
- L40: "teacher tiếp quản" từ state của student (`scripts/takeover_eval.py`) với công cụ đã sửa; nhánh đối chứng
  student tự tiếp quản đã qua (91% ở mốc 25% trên các episode nó từng thành công).
- Sau đó: B3 (state teacher + depth); eval LIBERO-Plus cho checkpoint cuối của cả bốn nhánh; xác nhận bảng
  LIBERO-Plus trên L40 (render GPU).
- Hướng đề xuất: lấy LIBERO-Plus làm thước đo chính; so {không 3D, 3D trên state teacher, 3D trên state student}
  theo từng loại nhiễu.

## Lỗi của chính mình đã gặp (để không lặp lại)

- Frame trễ/đen từ episode thứ hai của mỗi worker sau khi tối ưu render: gate chỉ kiểm episode đầu. Đã thêm gate
  nhiều episode liên tiếp. Hậu quả: một kết luận sai về renderer đã được báo rồi phải rút lại.
- Công cụ takeover khôi phục state của simulator nhưng bỏ sót trạng thái của robosuite nằm ngoài nó: mục tiêu
  tích luỹ của gripper và pose end-effector mà bộ điều khiển tay máy cache từ lúc reset. Lộ ra nhờ nhánh đối
  chứng (student chỉ còn 11% ở mốc 25% trên chính episode nó từng thành công; sau khi sửa: 91%). Mọi công cụ can
  thiệp vào simulator cần một nhánh đối chứng "không đổi gì thì phải ra như cũ" trước khi đọc kết quả.
- Dừng hai run reverse-KL ở vòng 7 vì đọc đường cong quá sớm, rồi mất hai giờ thử forward-KL và CE. Khi thay một
  thành phần của baseline đã công bố, chạy bản gốc tới hết ngân sách trước.
- Tín hiệu "teacher rất bất định trên state lỗi của student" đến từ checkpoint RLinf lệch pipeline; với teacher
  đúng pipeline thì hiệu ứng gần như biến mất.
