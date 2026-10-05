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

**Spatial Forcing** (cùng backbone OFT, căn feature theo VGGT lúc train; checkpoint đã công bố) trên cùng 420 task:
**71.9%**, chỉ hơn OFT 1.7 điểm.

| | Tổng | Camera | Robot | Noise | Layout | Light | BG | Lang |
|---|---|---|---|---|---|---|---|---|
| OFT chuẩn | 70.2 | 48.3 | 26.7 | 65.0 | 66.7 | 90.0 | 98.3 | 96.7 |
| Spatial Forcing | 71.9 | 56.7 | 38.3 | **41.7** | 70.0 | 100 | 96.7 | 100 |
| Chỉ SF đúng / chỉ OFT đúng / cả hai hỏng (số task) | 34 / 27 / 91 | 11 / 6 / 20 | 10 / 3 / 34 | 1 / 15 / 20 | 3 / 1 / 17 | 6 / 0 / 0 | 1 / 2 / 0 | 2 / 0 / 0 |

- So cặp trên cùng task: SF hơn rõ ở Light (6–0), có xu hướng hơn ở Robot-init (10–3) và Camera (11–6), nhưng
  **kém hẳn ở Sensor Noise (1–15)**. Tổng thể 34–27 là chưa phân biệt được với nhiễu.
- **91/420 task (22%) cả hai model SOTA đều hỏng**, dồn vào Robot-init (34/60), Camera (20/60), Noise (20/60),
  Layout (17/60). Đây là headroom còn nguyên ngay cả với phương pháp 3D mạnh nhất có checkpoint công khai.
- Theo mức khó, SF: 96 / 88 / 73 / 63 / 47 (L1→L5); OFT: 92 / 85 / 74 / 70 / 40.

Đọc bảng này:
- Headroom thật nằm ở đây: ngay cả teacher cũng chỉ đạt một nửa, và trần **không** bị teacher chặn, vì giám sát
  3D có thể cho student độ bền mà teacher không có.
- Camera và Light là hai chiều yếu nhất của model 1-camera, cũng là hai chiều mà các paper distill geometry
  (GaussianWAM, MVUCF) tăng mạnh nhất trên backbone khác.
- Đã xác nhận trên L40 (render GPU), cùng 420 task: teacher **50.0%** (render CPU: 49.8%). Theo loại nhiễu
  (GPU / CPU): Camera 20 / 13, Light 30 / 30, Noise 40 / 43, Robot 45 / 55, Layout 62 / 58, Background 82 / 73,
  Language 72 / 75; mọi chênh lệch đều trong sai số của 60 task. Bảng LIBERO-Plus không phụ thuộc renderer.

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

## Probe offset tay kẹp → vật đích trên feature đóng băng của student (`scripts/probe_offset.py`)

Câu hỏi: lúc student kẹp lệch vài cm, representation của nó có "biết" vật ở đâu so với tay kẹp không? Probe nhỏ
(attention pooling + MLP) hồi quy vector tay kẹp → vật đích từ feature đóng băng, chỉ trên state **trước lần kẹp
đầu tiên** của 500 episode B0; fit trên trial 0–39, đo trên trial 40–49 (100 episode, 748 state). "Thuộc lòng" là
mốc không nhìn ảnh: offset trung bình của cùng task và cùng thứ tự query.

| Sai số ngang trung vị (cm) | Token ảnh lớp 24 | Hidden state ở 56 vị trí action (lớp cuối) | Thuộc lòng |
|---|---|---|---|
| Mọi state tiếp cận, episode thành công | 0.61 | 0.86 | 1.77 |
| Mọi state tiếp cận, episode hỏng | 0.78 | 1.02 | 2.15 |
| Hai query cuối trước khi kẹp, thành công | 0.50 | 0.67 | 1.84 |
| Hai query cuối trước khi kẹp, hỏng | 0.81 | 1.05 | 2.75 |
| … riêng near miss | 0.68 | 0.95 | 2.73 |
| … riêng nhầm vật / nhầm chỗ | 0.85 | 1.69 | 4.32 |

- **Thông tin hình học có sẵn và đủ chính xác**: ngay trên episode hỏng, feature vẫn cho vị trí vật so với tay
  kẹp với sai số khoảng 1 cm, trong khi cú kẹp thật lệch trung vị 7.2 cm (near miss 3.5 cm) và dung sai gắp là
  2.5 cm. Thông tin này còn nguyên ở chính các vị trí mà action được giải mã ra.
- **Cú kẹp trượt đi theo quỹ đạo thuộc lòng, không theo vị trí vật**: trên episode hỏng, vector lệch của cú kẹp
  tương quan r = 0.79 với độ lệch của cảnh hiện tại so với offset trung bình lúc train (mốc "thuộc lòng"), và chỉ
  r = 0.11 với sai số của probe trên token ảnh. Episode thành công: r = 0.28.
- Cùng probe trên feature của **teacher** (cùng các state của student): token ảnh 0.37 cm, vị trí action 0.61 cm
  (student: 0.69 và 0.91); trên hai query cuối của episode hỏng 0.43 / 0.64 (student 0.81 / 1.05). Feature của
  teacher định vị chính xác gấp khoảng hai lần, nhưng cả hai đều nằm sâu dưới dung sai gắp 2.5 cm.
- Kết luận tạm (một seed, 100 episode held-out, probe phi tuyến): lỗi chính của student **không phải thiếu nhận
  thức 3D**, mà là action không dùng thông tin hình học đã có; student phát lại quỹ đạo của 1 demo thay vì bám
  theo vật. Khớp với probe depth (mục 3 ở trên) và với việc nhánh có loss depth chưa cho khác biệt.

## Failure của OFT chuẩn trên LIBERO-Plus theo loại nhiễu (H200, cùng 420 task, có lưu state)

Chạy lại có lưu state: 71.4% (lần trước 70.2%). 120 episode hỏng:

| Loại nhiễu | Hỏng / 60 | Nhầm vật hoặc nhầm chỗ | Làm đổ vật | Nhấc được rồi rơi / đặt sai | Near miss | Lệch ngang lúc kẹp (trung vị, episode hỏng) |
|---|---|---|---|---|---|---|
| Robot-init | 43 | **31** | 7 | 4 | 1 | 18.6 cm |
| Camera | 31 | 4 | 12 | **14** | 1 | 4.6 cm |
| Layout | 20 | 11 | 4 | 1 | 4 | 13.7 cm |
| Noise | 17 | 6 | 2 | 5 | 4 | 5.1 cm |
| Light | 6 | 3 | 0 | 2 | 1 | 12.9 cm |
| Language / Background | 3 | 0 | 0 | 3 | 0 | — |
| **Tổng** | 120 | 55 (46%) | 25 (21%) | 29 (24%) | 11 (9%) | 10.2 cm |

- Model SOTA hỏng theo kiểu khác student: không còn là lệch vài cm, mà **đưa tay tới sai chỗ** (lệch 10–20 cm).
  Riêng Robot-init, 31/43 failure là kẹp ở chỗ khác hẳn vật đích: khi tư thế xuất phát bị dời, policy vẫn thực
  hiện chuyển động quen thuộc thay vì hướng tới vật.
- Camera hỏng muộn hơn: gắp được rồi làm rơi / đặt trượt giỏ (14), hoặc làm đổ vật (12).
- Lần kẹp đầu vẫn ở bước 40–48 cho cả thành công lẫn hỏng: lỗi vẫn được quyết định ở pha tiếp cận.
- Hai phân tích cùng chỉ một hướng: **policy phát lại chuyển động đã học thay vì đóng vòng điều khiển trên vị trí
  tương đối tay kẹp – vật**. Ở student là lệch vài cm trong phân phối; ở OFT là lệch hàng chục cm khi trạng thái
  robot bị dời.

## Teacher 1-camera trên LIBERO-Plus: kiểu lỗi và probe offset chuyển sang cảnh có nhiễu (L40, render GPU)

Chạy lại có lưu state: 49.3% (213 hỏng / 420). Kiểu lỗi: kẹp ở sai chỗ 116 (54%), near miss 36, làm đổ 24, nhầm
vật 20, rơi / đặt sai 16. "Kẹp ở sai chỗ" chiếm đa số ở Camera (30/51), Light (34/42), Noise (27/35), Robot-init
(17/32); lệch ngang lúc kẹp của episode hỏng 14–22 cm.

Probe offset fit trên cảnh chuẩn (feature của teacher; sai số 0.4–0.6 cm ở đó) áp nguyên sang các state tiếp cận
của teacher trên LIBERO-Plus, sai số ngang trung vị (cm), token ảnh / vị trí action:

| Loại nhiễu | Episode thành công | Episode hỏng |
|---|---|---|
| Background | 1.1 / 1.4 | 1.6 / 1.7 |
| Language | 0.7 / 1.4 | 2.0 / **11.7** |
| Layout | 1.4 / 1.9 | 6.6 / 6.7 |
| Light | 2.3 / 3.6 | 3.0 / 5.5 |
| Robot-init | 2.8 / 3.5 | 4.6 / 6.6 |
| Sensor Noise | 2.3 / 2.7 | 7.8 / 8.0 |
| Camera | 3.1 / 2.8 | 7.2 / 9.2 |

- Khác hẳn trong phân phối: dưới nhiễu thị giác, hình học đọc được từ feature **hỏng theo** (từ dưới 1 cm lên
  6–10 cm ở episode hỏng của Camera / Noise / Layout), và episode hỏng có sai số gấp 2–4 lần episode thành công
  cùng loại nhiễu. Ở đây nhận thức là nút thắt thật, đúng chỗ thước đo chính (LIBERO-Plus) đang đo.
- Language: token ảnh vẫn định vị đúng vật (2 cm) nhưng feature ở vị trí action chỉ sang chỗ khác (11.7 cm): lỗi
  hiểu câu lệnh, không phải lỗi nhìn.
- Dè dặt: probe chỉ được fit trên cảnh chuẩn, nên sai số lớn có thể do probe không bất biến chứ chưa chắc feature
  mất thông tin; nhưng head action của policy cũng ở đúng hoàn cảnh đó (chỉ học trên cảnh chuẩn).

**Phép thử oracle tiếp cận (chạy thử 24 episode, L40):** thay chuyển động ngang của student bằng servo đặc quyền
tới tâm vật cho đến lần kẹp đầu. Task 4 (chủ yếu nhầm chỗ): 7/8, so với 6–8% của student. Task 0 và 3 (near miss /
làm đổ): 0/8, tệ hơn student (22% và 2%), tức servo tới tâm vật chưa phải cận trên sạch cho các task này (cần
đúng cả độ cao / điểm kẹp). Chưa chạy bản đầy đủ.

## LIBERO-Long: student trên hai renderer

Student 1-traj của Long, 100 episode đầu, greedy: **27%** trên L40 (render GPU) so với **16–17%** trên H200 (render
CPU); paper 17.3. Ở Object hai renderer khớp nhau (51.4 / 52.6), ở Long thì lệch khoảng 10 điểm (cỡ 2 lần sai số
chuẩn của hiệu). Mọi run Long đang chạy trên H200 nên so sánh nội bộ vẫn cùng renderer; cần đo lại 500 episode ở
cả hai máy trước khi so với số đã công bố.

## LIBERO-Plus của student sau distill (Object, cùng 420 task, render CPU)

| | Tổng | Camera | Light | Noise | Robot | Layout | BG | Lang |
|---|---|---|---|---|---|---|---|---|
| Student 1-traj (trước distill) | 21.2 | 3.3 | 13.3 | 6.7 | 18.3 | 33.3 | 35.0 | 38.3 |
| Student sau distill không-3D (reverse-KL, state teacher, vòng 4; 88% trên bản chuẩn) | 42.9 | 5.0 | **60.0** | 21.7 | 48.3 | 46.7 | 53.3 | 65.0 |
| Student sau distill không-3D (reverse-KL, **state student**, vòng 10; 89% trên bản chuẩn) | 46.9 | 11.7 | 45.0 | 26.7 | 43.3 | 48.3 | 78.3 | 75.0 |
| Student sau distill không-3D (reverse-KL, state teacher, vòng 8; 90% trên bản chuẩn) | 45.7 | 6.7 | 56.7 | 18.3 | 55.0 | 45.0 | 73.3 | 65.0 |
| Student sau distill không-3D (reverse-KL, state student, **vòng 20**; 86.2% trên 500 episode chuẩn) | 48.1 | 13.3 | 61.7 | 21.7 | 46.7 | 48.3 | 80.0 | 65.0 |
| Student sau distill không-3D (reverse-KL, state teacher, vòng 20; 73.2% trên 500 episode chuẩn) | 39.0 | 5.0 | 50.0 | 15.0 | 48.3 | 38.3 | 63.3 | 53.3 |
| Teacher full-SFT | 49.8 | 13.3 | 30.0 | 43.3 | 55.0 | 58.3 | 73.3 | 75.0 |

Distill không-3D đã kéo student từ 21% lên 43% trên LIBERO-Plus, gần teacher (50%). Riêng Light student vượt
teacher (60 so với 30; mỗi ô 60 task, sai số chuẩn khoảng 6 điểm), còn Camera và Noise vẫn rất thấp.

## Đường cong baseline không-3D tới hết ngân sách (Object, seed 7, eval greedy 100 episode)

| Vòng | 0 | 2 | 4 | 6 | 8 | 10 | 12 | 14 | 16 | 18 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 (state student) | 52 | 39 | 61 | 69 | 88 | 89 | 85 | 90 | 84 | 82 | 88 |
| B2′ (state teacher) | 52 | 69 | 88 | 70 | 90 | 68 | 83 | 84 | 81 | 78 | 71 |

Từ vòng 8 trở đi B2 ổn định quanh 82–90 (trung bình 86.6), còn B2′ dao động 68–90 và đi xuống ở cuối (trung bình
79.4). VLA-OPD báo 93.8 cho Object.

**Adapter cuối (vòng 20) trên 500 episode chuẩn, greedy:**

| Task | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Tổng |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Student ban đầu (H200) | 22 | 66 | 46 | 2 | 6 | 74 | 100 | 62 | 86 | 62 | 52.6 |
| Teacher full-SFT (H200) | 90 | 100 | 100 | 80 | 94 | 100 | 100 | 92 | 100 | 96 | 95.2 |
| **B2** (state student, không 3D) | 72 | 98 | 96 | 74 | 96 | **30** | 100 | 96 | 100 | 100 | **86.2** |
| **B2′** (state teacher, không 3D) | 76 | 74 | 84 | 52 | 100 | 60 | 86 | 96 | **4** | 100 | **73.2** |
| **B4** (state student + depth) | 42 | 92 | 98 | 80 | 88 | 76 | 98 | 98 | 100 | 90 | **86.2** |

- B2 thấp hơn VLA-OPD 7.6 điểm (86.2 so với 93.8): **chưa đạt mốc "trong 3 điểm"**. Bảy task đã ở 96–100; khoảng
  hụt nằm ở task 5 (30%, trong khi student ban đầu đạt 74% và teacher 100%: distill làm *hỏng* task này) và hai
  task khó nhất 0 và 3 (72–74%).
- B2′ có kiểu sụp tương tự ở task khác (task 8: 86% → 4%). Sụp theo từng task trên những task student vốn làm
  được là dấu hiệu bất ổn của quá trình distill (teacher khác dòng dõi, lr cố định), không phải thiếu năng lực.
- **Idea hiện có (loss depth trên state on-policy) không hơn baseline**: B4 = 86.2%, bằng đúng B2. Đường cong train
  cũng trùng nhau (vòng 10–20: B4 88 / 81 / 80 / 81 / 82 / 87; B2 89 / 85 / 90 / 84 / 82 / 88). Khác nhau chỉ ở
  task nào bị sụp (B4: task 0 còn 42%; B2: task 5 còn 30%), tức là nhiễu giữa các run, không phải hiệu ứng của depth.
- Task 5 của B2 theo checkpoint (50 trial): vòng 10: 60%, vòng 14: 64%, vòng 20: 30%; task 0: 76 / 72 / 72; task 3:
  84 / 70 / 74. Task 5 chưa bao giờ về lại mức 74% của student ban đầu và còn tụt tiếp ở cuối: quá trình distill
  dao động chứ không hội tụ (lr cố định 1e-4, khoảng 2000 state mỗi vòng, teacher khác dòng dõi).
- Sai khác so với VLA-OPD cần nhớ khi so: LoRA thay vì full-parameter, khoảng 41k state so với khoảng 300k của
  họ, teacher full-SFT chuyển bin thay vì teacher RL cùng dòng dõi.

## LIBERO-Long: mốc teacher (H200, render CPU)

Teacher RL của SimpleVLA-RL (`Haozhan72/openvla-oft-libero10-traj1-rl`, cùng bin action với student): **85.5%** trên
200 episode (paper 91.7; sai số chuẩn khoảng 2.5 điểm). Theo task: 60 / 95 / 100 / 95 / 50 / 85 / 85 / 100 / 90 /
95. Thấp hơn paper khoảng 6 điểm, cùng chiều với student Object (51.4 so với 54.9). Vừa qua ngưỡng 85% đã đặt nên
dùng làm teacher cho Long.

Student 1-traj của Long, eval đầu của run distill (100 episode, greedy): **17%** (paper 17.3); theo task 0 / 20 / 60 /
10 / 0 / 30 / 30 / 0 / 0 / 20.

## Bước sửa theo failure analysis: distill dưới nhiễu thị giác, teacher nhìn cảnh sạch (đang chạy)

Chuỗi lập luận: (1) loss depth trên cảnh chuẩn không đổi kết quả (B4 = B2), vì trong phân phối feature đã đủ hình
học; (2) trên LIBERO-Plus, thứ hỏng là việc đọc hình học khi góc nhìn / ánh sáng / nhiễu ảnh đổi; (3) teacher chỉ
dạy được ở những chỗ nó nhìn được. Vì vậy: giữ nguyên vòng lặp distill, nhưng **render lại chính các state đó dưới
góc nhìn / ánh sáng / nhiễu ngẫu nhiên cho student**, còn teacher gán nhãn (và lái, khi tới lượt) từ ảnh chuẩn của
cùng state mô phỏng. Đây là chỗ 3D thật sự cần: muốn có ảnh của cùng một state từ góc nhìn khác thì phải có cảnh 3D
(simulator ở đây; 3DGS ngoài đời thật).

- Cài đặt: `--view_aug` (`src/rollout/vec_env.py::VIEW_AUG`). Mỗi episode một lần rút: 25% giữ nguyên; còn lại mỗi
  yếu tố camera / ánh sáng / cảm biến bật với xác suất 0.6. Camera quay quanh điểm nó đang nhìn trên mặt bàn
  (phương vị ±75°, nâng 0–15°, khoảng cách ×1–2, lệch hướng ngắm ±10°); ánh sáng ×0.3–1.7, dời đèn ±1 m, ám màu
  ±15%; nhiễu Gauss tới 0.08 và blur tới 2 px. Ảnh nhiễu được render ngay tại thời điểm sensor lấy mẫu; cổng
  `scripts/check_view_aug.py` xác nhận biên độ 0 cho đúng từng pixel ảnh chuẩn và vật lý không đổi.
- `--state_source mixed`: các lô episode luân phiên do student lái (trên ảnh nhiễu) và teacher lái (trên ảnh
  chuẩn), vì dưới nhiễu mạnh student hỏng sớm và phần lớn state của nó nằm sau lần kẹp hỏng (nhãn vô dụng).
- Hai nhánh, cùng siêu tham số với B2 (seed 7, 20 vòng × 2048 state): **V1** = mixed + view_aug (H200); **V2** =
  V1 + loss depth trên ảnh nhiễu (L40). Đánh giá: 500 episode chuẩn và cùng 420 task LIBERO-Plus.
- Điều phải nói rõ khi báo cáo: dải nhiễu được chọn rộng cỡ LIBERO-Plus (họ quay camera tới ±75° và kéo xa tới
  2 lần), nên đây **không còn là zero-shot theo loại nhiễu**; không dùng demo mới và không dùng task / file nhiễu
  của LIBERO-Plus. Mốc so công bằng là student / teacher cùng backbone và OFT train trên data nhiễu (79.5).

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

- Thêm tham số `obs` vào giao diện `act` cho policy OFT mà không cập nhật `RebinnedPolicy`: nhánh state-teacher (B3)
  crash ngay vòng đầu và nằm im một giờ vì hàng đợi không kiểm tra run có qua được vòng 1 hay không. Sau khi khởi
  động một run từ hàng đợi phải có bước xác nhận nó đã ghi vòng 1.
- Tính ngân sách bộ nhớ GPU bằng tay, lẫn GiB với "GB" và không tính process của dự án khác trên cùng card: run
  Long bị OOM ở vòng 1. Đã thay bằng cổng bộ nhớ GPU có khoá trong `run_py.sh`.
- Dùng `pkill -f <mẫu>` qua ssh, mẫu nằm trong chính dòng lệnh nên tự giết phiên của mình (lần thứ hai). Chỉ kill
  theo PID sau khi loại `$$`.

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
