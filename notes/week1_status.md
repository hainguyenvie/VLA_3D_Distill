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

OFT chuẩn (checkpoint chính thức, 2 camera + proprio) đo trong harness của ta trên cùng 420 task: **70.2%**
(Camera 48, Robot-init 27, Noise 65, Layout 67, Light 90, Language 97, Background 98), so với 66.5% đã công bố cho
toàn bộ suite Object; trên LIBERO-Object chuẩn nó đạt 96.8% (paper 98.4). Harness LIBERO-Plus vì vậy tái lập được
dạng kết quả đã công bố.

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

## Teacher tiếp quản từ state của student (L40, render GPU)

`scripts/takeover_eval.py`: khôi phục simulator về state mà student đã đi tới (kèm mục tiêu gripper và bộ điều
khiển tay máy), rồi để policy khác chạy greedy tới hết horizon. Lấy mẫu phân tầng 6 episode cho mỗi (task, kết
cục) từ 500 episode của B0.

| Trao quyền tại (% độ dài episode) | 0% | 25% | 50% | 75% |
|---|---|---|---|---|
| Teacher, từ episode student **thành công** (n=53) | 96 | 92 | 98 | 100 |
| Teacher, từ episode student **hỏng** (n=58) | **100** | **7** | 3 | 0 |
| Đối chứng: student tự tiếp quản, episode thành công | 92 | 91 | 98 | 100 |
| Đối chứng: student tự tiếp quản, episode hỏng | 17 | 0 | 0 | 0 |

- Đối chứng qua: khôi phục state không làm hỏng episode (student lặp lại được thành công của chính nó).
- State ban đầu của các episode hỏng không khó (teacher 100%), nhưng sau 25% episode thì teacher gần như không
  cứu được. Episode hỏng trở thành "không thể cứu" (với teacher này) rất sớm; 77% state on-policy của student
  nằm trong các episode hỏng, phần lớn sau điểm đó, nên nhãn action của teacher ở đó không dẫn tới thành công.
- Đang đo mịn hơn ở 5/10/15/20% để định vị điểm không thể cứu.

## Phân loại failure của student (B0, 500 episode, 243 hỏng; `scripts/analyze_failures.py`)

Đọc thẳng quỹ đạo vật thể và gripper từ state của simulator đã log, không cần chạy lại.

| Kiểu lỗi | Số episode | Tỉ lệ | Lệch ngang lúc kẹp lần đầu (trung vị) |
|---|---|---|---|
| Kẹp trượt sát vật (near miss): kẹp cạnh vật đích, tay không | 113 | 46% | 3.5 cm |
| Làm đổ / đẩy lệch vật đích | 54 | 22% | 5.5 cm |
| Kẹp ở chỗ khác (gần vật khác hơn vật đích) | 34 | 14% | 18 cm |
| Nhấc nhầm vật khác | 32 | 13% | 20 cm |
| Đã nhấc đúng vật nhưng rơi / đặt sai | 10 | 4% | 2.5 cm |

- Sai lệch ngang giữa gripper và vật đích lúc kẹp lần đầu: episode thành công 1.1 / 1.6 / 2.1 cm (tứ phân vị);
  episode hỏng 3.2 / 5.2 / 12.1 cm. **Khoảng 69% failure là lỗi chính xác không gian cỡ vài cm ở pha gắp**
  (F2/F5 trong plan), khoảng 27% là lỗi ngữ nghĩa nhầm vật / nhầm chỗ (F1), 4% xảy ra muộn.
- Dung sai gắp khoảng 2.5 cm, và success rate rơi rất dốc quanh đó (500 episode, theo sai lệch ngang lúc kẹp
  lần đầu):

  | Lệch ngang (cm) | <1 | 1–1.5 | 1.5–2 | 2–2.5 | 2.5–3 | 3–4 | 4–6 | 6–10 | >10 |
  |---|---|---|---|---|---|---|---|---|---|
  | Số episode | 46 | 77 | 71 | 55 | 43 | 56 | 38 | 40 | 74 |
  | Success rate (%) | 96 | 94 | 82 | 76 | 44 | 29 | 11 | 5 | 0 |

  Lúc kẹp, episode hỏng cũng ở cao hơn vật đích (trung vị 2.9 cm so với 0.8 cm ở episode thành công).
- Lần kẹp đầu tiên xảy ra quanh bước 48 ở cả episode thành công lẫn hỏng: số phận episode được quyết định trong
  khoảng 10% đầu của horizon 512 bước.
- Lỗi phân bố rất khác nhau theo task: task 0 (alphabet soup) gần như toàn near miss; task 3 (bbq sauce) chủ yếu
  làm đổ vật; task 4 (ketchup) chủ yếu nhầm vật / nhầm chỗ.

Ghép với takeover của teacher (cùng các episode):

| Trao quyền cho teacher tại | 0% | 5% | 10% | 15% | 20% | 25% | 50% |
|---|---|---|---|---|---|---|---|
| Episode near miss | 100 | 82 | 79 | 63 | 20 | 8 | 5 |

(bản đầy đủ, 37 episode mỗi ô: 86 / 78 / 65 / 24 ở các mốc 5 / 10 / 15 / 20%; trên episode student thành công
teacher đạt 92–94% ở cả bốn mốc.) Gộp mọi kiểu lỗi ở các mốc 0/25/50/75%: trao quyền **trước** lần kẹp đầu tiên
của student thì teacher thành công 95% (n=62); **sau** lần kẹp đó chỉ 3% (n=170).

Đọc kết quả:
- Điểm không thể cứu chính là **lần kẹp hỏng đầu tiên**. Ngay cả near miss, khi vật đích còn nguyên chỗ, teacher
  cũng không gắp lại được: teacher học từ demo nên không có hành vi "thử lại" từ trạng thái tay kẹp rỗng cạnh vật.
- Vì vậy tín hiệu sửa lỗi hữu ích của on-policy distillation nằm gần hết ở **pha tiếp cận trước lần kẹp đầu**
  (khoảng 40–50 bước), còn phần lớn state on-policy (sau lần kẹp hỏng) mang nhãn action không dẫn tới thành công.
- Lỗi quyết định là lệch 3–5 cm giữa gripper và vật ở pha tiếp cận, đúng loại lỗi mà giám sát hình học có cơ
  hội giúp. Gợi ý cho phương pháp (chưa kiểm chứng): đặt giám sát 3D và trọng số distill vào các state tiền
  tiếp xúc mà student thực sự đi qua, thay vì các state "recovery" sau khi đã hỏng.

## LIBERO-Plus của student sau distill (Object, cùng 420 task, render CPU)

| | Tổng | Camera | Light | Noise | Robot | Layout | BG | Lang |
|---|---|---|---|---|---|---|---|---|
| Student 1-traj (trước distill) | 21.2 | 3.3 | 13.3 | 6.7 | 18.3 | 33.3 | 35.0 | 38.3 |
| Student sau distill không-3D (reverse-KL, state teacher, vòng 4; 88% trên bản chuẩn) | 42.9 | 5.0 | **60.0** | 21.7 | 48.3 | 46.7 | 53.3 | 65.0 |
| Teacher full-SFT | 49.8 | 13.3 | 30.0 | 43.3 | 55.0 | 58.3 | 73.3 | 75.0 |

Distill không-3D đã kéo student từ 21% lên 43% trên LIBERO-Plus, gần teacher (50%). Riêng Light student vượt
teacher (60 so với 30; mỗi ô 60 task, sai số chuẩn khoảng 6 điểm), còn Camera và Noise vẫn rất thấp.

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
