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
| **B4**: state student + loss depth, vòng 20 (86.2% trên 500 episode chuẩn) | 45.0 | 8.3 | 50.0 | 16.7 | 41.7 | 51.7 | 68.3 | 78.3 |
| **B3**: state teacher + loss depth, vòng 20 (85.0% trên 500 episode chuẩn) | 49.3 | 10.0 | 53.3 | 21.7 | 48.3 | 48.3 | 85.0 | 78.3 |
| **V1**: state hỗn hợp + nhiễu thị giác, vòng 20 (87.8% chuẩn) | **56.4** | **41.7** | **75.0** | **53.3** | 45.0 | 45.0 | 73.3 | 61.7 |
| **V2**: V1 + loss depth trên ảnh nhiễu, vòng 20 (77.0% chuẩn; checkpoint cuối sụp, vòng 16–18 đạt 92% / 100 ep) | 50.2 | 23.3 | 75.0 | 43.3 | 36.7 | 46.7 | 66.7 | 60.0 |
| **B2 với lr cosine**, vòng 20 (93.0% chuẩn) | 51.4 | 5.0 | 63.3 | 25.0 | 48.3 | 51.7 | 83.3 | 83.3 |
| Teacher full-SFT | 49.8 | 13.3 | 30.0 | 43.3 | 55.0 | 58.3 | 73.3 | 75.0 |

Loss depth trên cảnh chuẩn cũng không giúp trên LIBERO-Plus: B4 45.0% so với 48.1% của B2 cùng vòng (chênh trong
mức nhiễu của 420 task, và không có loại nhiễu nào B4 hơn rõ). Kết luận cho idea ban đầu, một seed: **không hơn
baseline trên cả hai thước đo**.

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
| **B3** (state teacher + depth) | 94 | 38 | 96 | 62 | 100 | 68 | 100 | 92 | 100 | 100 | **85.0** |

**Ma trận 2×2 đã đủ (LIBERO-Object chuẩn, 500 episode, adapter vòng 20, seed 7):**

| | Không 3D | + loss depth | Hiệu |
|---|---|---|---|
| State của student | 86.2 (B2) | 86.2 (B4) | Δ_π = 0.0 |
| State của teacher | 73.2 (B2′) | 85.0 (B3) | Δ_E = +11.8 |

Giả thuyết ban đầu là Δ_π > Δ_E; số liệu một seed cho chiều ngược lại. Phần lớn Δ_E đến từ việc B2′ sụp task 8
ở vòng cuối (4%), còn B3 sụp nhẹ hơn ở task 1 (38%); trung bình eval 100 episode các vòng 8–20 là 79.3 (B2′) so với
83.1 (B3). Đọc thận trọng: loss depth có thể đóng vai trò điều hoà khi học trên state của teacher, chứ không có dấu
hiệu nào cho thấy nó giúp riêng trên state on-policy. Cần thêm seed mới nói được gì về hiệu ứng cỡ vài điểm.

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

**Distill không-3D trên Long (state student, reverse-KL, lr cố định 1e-4, 40 vòng × 2048 state, H200):** eval 100
episode mỗi 4 vòng: 16 → 31 → 42 → 40 → 48 → 59 → 51 → 55 → 51 → 62 → 57; adapter cuối trên **200 episode: 59.0%**
(theo task 40 / 45 / 75 / 95 / 25 / 70 / 60 / 75 / 30 / 75). VLA-OPD công bố 78.9 sau 50 bước (teacher của họ 90.7, của
ta 85.5). Khoảng cách 20 điểm; cùng kiểu dao động như Object ở lr cố định (59 → 51 → 55 → 51 → 62 → 57). Bản lr cosine
đang xếp hàng.

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

## Kết quả bước sửa (06/10, một seed)

| | LIBERO-Object chuẩn (500 ep) | LIBERO-Plus (420 task) | Camera | Light | Noise |
|---|---|---|---|---|---|
| Teacher | 95.2 | 50.0 | 13–20 | 30 | 40–43 |
| B2 (baseline distill) | 86.2 | 48.1 | 13 | 62 | 22 |
| **V1** (nhiễu thị giác, teacher nhìn sạch) | **87.8** | **56.4** | **42** | **75** | **53** |
| V2 (V1 + depth) | 77.0 | 50.2 | 23 | 75 | 43 |

- V1 là kết quả đầu tiên đạt mốc 2 đã đặt (student vượt chính teacher của nó trên LIBERO-Plus, ≥ 55), và không
  tụt trên bản chuẩn. Mức tăng nằm đúng ba loại nhiễu được render lại lúc train (Camera +29, Noise +31, Light +13);
  Robot-init, Layout, Language, Background không đổi trong mức nhiễu.
- V2 không hơn V1: checkpoint cuối của V2 là một checkpoint sụp (76% / 100 ep ở vòng 20, trong khi vòng 16–18 đạt
  92%), nên số này chủ yếu đo sự bất ổn cuối run chứ chưa đo được hiệu ứng của depth dưới nhiễu. Cần chạy lại với
  lr giảm dần (và thêm seed) trước khi kết luận.
- **Baseline với lr cosine (1e-4 → 1e-5): 93.0% trên 500 episode** (theo task 86 / 98 / 98 / 72 / 94 / 92 / 100 / 96 /
  100 / 94; không task nào sụp), so với 93.8 của VLA-OPD: **mốc "trong 3 điểm" đã đạt**. Đường cong eval 100 episode:
  49 → 45 → 77 → 86 → 88 → 92 → 91 → 91 → 96 → 95. Khoảng hụt 86.2 của bản lr cố định là do dao động cuối run, không
  phải do ngân sách hay LoRA. Từ đây mọi nhánh mới dùng lr cosine.
- Baseline lr cosine trên LIBERO-Plus: **51.4** (teacher 50.0): distill không-3D ổn định thì chạm teacher, không hơn;
  Camera vẫn 5%. V1 (56.4) hơn baseline ổn định 5 điểm, toàn bộ ở Camera / Noise / Light. So công bằng hơn cho V1 là
  bản V1 + lr cosine đang chạy trên L40.
- V1 seed 8 (H200, lr cố định): eval 100 ep 83 → 81 → 75 ở ba vòng cuối, adapter cuối **77.2** trên 500 episode
  (seed 7: 87.8). Hai seed của V1 cho 87.8 và 77.2: nhiễu giữa seed lớn, phần lớn do cú tụt ở vòng cuối.
- V1 + lr cosine (L40) đã train xong: eval 100 episode 52 → 33 → 56 → 66 → 45 → 65 → 77 → 82 → 87 → 89 → **64 ở vòng
  20**. Khác với cos_b2 (ổn định tới cuối), nhánh có nhiễu thị giác vẫn sụp ở vòng cuối dù lr đã về 1e-5; nguyên nhân
  chưa rõ (một lô episode xấu ở vòng cuối? lr cuối vẫn quá cao cho nhánh này?). Số 500 episode và LIBERO-Plus của
  adapter cuối đang chấm; cần xem thêm adapter vòng 18.

## Nhánh OFT chuẩn: tự distill dưới nhiễu thị giác (A2), kết quả giữa chừng (06/10)

A2 = checkpoint OFT chính thức, teacher là chính nó (đóng băng, nhìn ảnh sạch), student là nó + LoRA nhìn ảnh render
lại dưới nhiễu, state hỗn hợp, loss L1, lr cosine. **Adapter vòng 4/20 trên cùng 420 task LIBERO-Plus: 82.1%** (OFT
gốc 70.2, Spatial Forcing 71.9). So cặp với OFT gốc: 58 task được sửa, 8 task hỏng thêm, 67 cả hai cùng hỏng.

| | Tổng | Camera | Noise | Light | Background | Language | Layout | Robot-init |
|---|---|---|---|---|---|---|---|---|
| OFT gốc | 70.2 | 48 | 65 | 90 | 98 | 97 | 67 | 27 |
| Spatial Forcing | 71.9 | 57 | 42 | 100 | 97 | 100 | 70 | 38 |
| **A2 vòng 4** | **82.1** | **92** | **98** | **100** | 98 | 98 | 65 | 23 |
| OFT train trên 20k demo nhiễu (công bố, 4 suite) | 79.5 | 93 | 89 | 95 | 94 | 86 | 78 | 30 |

Theo mức khó: 89 / 89 / 82 / 83 / 71 (L1 → L5). Eval cảnh chuẩn giữ 98% (100 episode) ở mọi vòng. Đọc: toàn bộ
headroom *thị giác* của backbone OFT đã lấy được sau 4 vòng (khoảng 8k state, không demo mới), ngang mức OFT train trên
20.000 demo nhiễu; phần còn lại nằm đúng ở **Robot-init và Layout** — không đổi, vì render lại ảnh không đổi trạng thái.
Đối chứng A1 (cùng vòng lặp, không nhiễu) đang xếp hàng trên L40; adapter vòng 20 sẽ được chấm 500 episode + LIBERO-Plus.

## D2: che proprio của OFT lúc test (06/10)

| Proprio | LIBERO chuẩn (100 ep) | Robot-init (60 task) | Layout (60 task) |
|---|---|---|---|
| bình thường | 96.8 (500 ep) | 27–28 | 67 |
| nhiễu Gauss σ = 0.3 (chuẩn hoá) | — | 32 | 67 |
| về 0 | 95 | **17** | 62 |

OFT gần như không cần proprio trong phân phối (95 khi che hẳn), nhưng che proprio làm Robot-init *tệ hơn*, nhiễu
không đổi gì rõ (sai số chuẩn 6 điểm). Giả thuyết "proprio là lối tắt gây hỏng Robot-init" **không được ủng hộ** với
OFT; lỗi nằm ở cách action gắn với cảnh nhìn thấy. (Model 1 camera không proprio đạt 45–55 ở Robot-init có lẽ vì lý do
khác: nó nhìn tay kẹp trong ảnh và không có đầu vào nào "nhắc" tư thế quen.)

## D1: probe phản thực — dời vật thì action có đi theo không? (06/10, L40)

`scripts/probe_counterfactual.py`: 231 state tiếp cận (trước lần kẹp đầu) từ 77 episode của student, mỗi state khôi
phục lại trong simulator rồi dời vật đích ±3 / ±6 cm theo x / y, render lại, hỏi policy; p = phần chuyển động ngang
của chunk đi theo vật (1 = theo hết, 0 = không phản ứng).

| Student 1-traj | n | p trung vị | p trung bình | tỉ lệ p > 0.25 |
|---|---|---|---|---|
| mọi state | 1848 | **0.00** | 0.001 | 15% |
| episode thành công / hỏng | 888 / 960 | 0.00 / 0.00 | 0.013 / −0.010 | 14% / 16% |
| sớm (≥ 3 query trước khi kẹp) / muộn | 1152 / 696 | 0.00 / 0.03 | −0.011 / 0.022 | 8% / 26% |
| dời 3 cm / 6 cm | 924 / 924 | 0.00 / 0.00 | | 15% / 15% |

| Teacher full-SFT (95%) | n | p trung vị | p trung bình | tỉ lệ p > 0.25 |
|---|---|---|---|---|
| mọi state | 1848 | 0.03 | 0.12 | 27% |
| sớm (≥ 3 query trước khi kẹp) | 1152 | 0.00 | 0.01 | 12% |
| **muộn (< 3 query trước khi kẹp)** | 696 | **0.27** | **0.31** | **52%** |
| dời 3 cm / 6 cm | 924 / 924 | 0.02 / 0.03 | 0.15 / 0.10 | 31% / 24% |

Teacher cũng không phản ứng khi còn xa vật (chunk 8 bước đầu là chuyển động chung "đi vào vùng làm việc"), nhưng
**khi đã gần vật nó bám theo**: trung vị 27% độ dời được bù ngay trong một chunk, 52% state có phản ứng rõ; student ở
cùng pha chỉ 3% / 26%. Tức "đi theo vật" ở policy 95% là một hành vi pha cuối, và đó đúng là pha quyết định cú kẹp
(dung sai 2.5 cm). Hệ quả cho phương pháp: distill hiệu action từ teacher có tín hiệu thật ở các state gần vật; ở
state xa, cặp phản thực nên có trọng số thấp hoặc dùng độ dời lớn hơn.

Student **không phản ứng** với vật bị dời: chuyển động của chunk gần như y nguyên dù vật đã ở chỗ khác 6 cm, kể cả
trên các episode nó sẽ thành công. Đây là bằng chứng trực tiếp nhất cho "phát lại quỹ đạo": cùng với probe offset
(feature biết vật ở đâu, sai < 1 cm), nó nói rõ thông tin có nhưng không đi vào action. Teacher đang được đo cùng probe
để biết "đi theo vật" trông như thế nào ở một policy 95%.

## Nhánh π0.5: cổng kiểm tra đã qua (06/10)

`lerobot/pi05_libero_finetuned` qua wrapper của ta (`src/policy/pi05_policy.py`, tokenizer bản không gated, thực thi
10/50 bước mỗi chunk, horizon 280): **100/100 episode LIBERO-Object** (10 init state × 10 task; LeRobot báo 99.0), trung
bình 135 bước. Mốc π0.5 trên LIBERO-PRO Object (swap / task / lan / object / position) và LIBERO-Plus Robot-init /
Layout đang được đo trong cùng harness (`scripts/server/pi05_anchors.sh`).

## Mốc π0.5 trong harness của ta (06/10, 10 init state × 10 task, horizon 280)

| | Object chuẩn | PRO lan (đổi câu lệnh) | PRO object (đổi vật) | **PRO swap (đổi chỗ vật)** | **PRO task (đổi mục tiêu)** | PRO position |
|---|---|---|---|---|---|---|
| π0.5 (LeRobot, fine-tune LIBERO) | 100 | 100 | 94 | **17** | **0** | **10** (dời x 0.3 theo file temp_x0.3 của LIBERO-PRO) |

LIBERO-Plus Object (60 task mỗi loại): **Robot-init 83, Layout 82** — π0.5 bản fine-tune LIBERO không yếu ở Robot-init
trên Object (OFT: 27), nên trục còn trống của π0.5 là **vị trí vật của LIBERO-PRO** (position 10, swap 17, task 0).

Đúng như LIBERO-PRO và ECT mô tả: π0.5 miễn nhiễm với câu lệnh / vật đổi màu, nhưng sụp khi **vị trí** vật đổi (swap
17%, trong đó 7/10 task bằng 0) và khi mục tiêu đổi (0%). ECT công bố swap Object 38 → 71 với protocol riêng của họ;
số 17 của ta là theo file swap chính thức của LIBERO-PRO. Đây là headroom cho hướng cặp thế giới trên π0.5.

V1 + lr cosine, adapter cuối trên 500 episode chuẩn: **73.6** (task 0: 2%, task 5: 40%) — xác nhận sụp ở vòng cuối
(eval 100 ep: 89 ở vòng 18 → 64 ở vòng 20). LIBERO-Plus của adapter sụp này: **41.2** (Camera 12, Noise 40): cú sụp cuối
xoá luôn phần robustness đã học. Adapter vòng 18 đang được chấm LIBERO-Plus để biết mức thật của V1 cosine.

## Cập nhật 06/10 chiều: seed 2 của V1, ma trận 2×2 ở lr cosine, V1 cosine

| Run | LIBERO-Object chuẩn (500 ep) | LIBERO-Plus (420) | Camera | Light | Noise | Robot | Layout |
|---|---|---|---|---|---|---|---|
| V1 seed 7 (lr cố định) | 87.8 | 56.4 | 42 | 75 | 53 | 45 | 45 |
| **V1 seed 8** (lr cố định) | 77.2 | **60.2** | 40 | 75 | 53 | 52 | 55 |
| V1 + lr cosine, adapter 18 / 20 | — / 73.6 | 47.9 / 41.2 | 25 / 12 | 53 / 55 | 43 / 40 | 42 / 32 | 43 / 37 |
| cos_b2 (baseline ổn định) | 93.0 | 51.4 | 5 | 63 | 25 | 48 | 52 |
| cos_b3 (state teacher + depth, cosine) | 83.6 | 41.2 | 8 | 30 | 18 | 43 | 52 |

- Hai seed của V1 cho 56.4 và 60.2 trên LIBERO-Plus (baseline ổn định 51.4, teacher 50): mức tăng +5 đến +9 lặp lại
  được, dù số trên bản chuẩn dao động mạnh (87.8 / 77.2) vì cú tụt ở vòng cuối.
- V1 với lr cosine lại **kém hơn** bản lr cố định trên LIBERO-Plus (47.9 ở adapter 18, Camera 25): lr nhỏ ở các vòng
  cuối có vẻ không đủ để tiếp thu các state nhiễu; cần hiểu thêm trước khi chuẩn hoá lr cosine cho nhánh có nhiễu.
- Ở chế độ lr ổn định, **state teacher + depth (cos_b3) thua rõ baseline on-policy** (83.6 / 41.2 so với 93.0 / 51.4),
  nên "depth giúp trên state teacher" ở lr cố định (85.0 so với 73.2) phần lớn là tác dụng ổn định hoá, không phải
  hình học; cos_b2p (state teacher, không depth, cosine) đang ở vòng 10 (85%) để chốt.

## Chẩn đoán π0.5 trên máy 8×H200 (06/10 tối)

Rollout greedy có lưu state (20 init state × 10 task, horizon 280), phân loại failure và probe phản thực:

| | Object chuẩn | LIBERO-PRO swap (đổi chỗ vật) | LIBERO-PRO position (dời x) |
|---|---|---|---|
| Thành công | 98.5 | **18.5** | **10.5** |
| Kiểu hỏng chính | rơi khi mang (3) | **nhặt nhầm vật 148/163** (lệch 21 cm) | rơi / đặt sai 86, nhầm vật 50, near miss 28 |
| Probe p (chunk bù độ dời của vật đích) | **0.91** (sát vật 1.43) | **0.00** (tập hỏng −0.04) | — |

- π0.5 có vòng điều khiển kín tốt trên cảnh quen (dời vật 3–6 cm thì chunk dời theo, khác student 1-traj có p = 0).
- Khi đổi chỗ vật, nó đi tới **vị trí quen** và gắp vật đang ở đó, bỏ qua tên vật trong câu lệnh: lỗi nằm ở **chọn
  đích theo vị trí đã thuộc**, không ở điều khiển. Khớp với chẩn đoán "instructions retrieve trajectories" (ECT).
- Hệ quả: π0.5 đóng băng không làm teacher cho trường hợp này được (nó mắc đúng lỗi cần sửa).

**Phương pháp chỉnh lại (đang chạy):** on-policy counterfactual distillation với thế giới **đổi chỗ**: ở mỗi state
trước khi kẹp mà student tự đi tới, đổi chỗ vật đích với một vật khác (chỉ đổi pose khi render, vật lý không đổi),
đổi câu lệnh sang tên vật đang đứng ở vị trí cũ của đích; hành động đúng là chunk gốc (nhãn chính xác, không cần
teacher giỏi hơn). Loss flow matching ghép cặp (chung noise / time) + số hạng nhất quán giữa hai thế giới. π0.5 gốc
chỉ làm neo cho cảnh chuẩn. Ba nhánh cùng ngân sách (20 vòng × 1024 state): swap, shift (dời vật, teacher gán nhãn),
và distill on-policy không phản thực; đánh giá trên Object, năm ô LIBERO-PRO Object, LIBERO-Plus Robot / Layout.

## Kết quả cuối đợt đầu trên máy 8×H200 (07/10 tối)

**Nhánh OFT, cùng máy, cùng vòng lặp, chỉ khác có render lại dưới nhiễu (adapter vòng 20):**

| | Chuẩn (500 ep) | LIBERO-Plus | Camera | Noise | Light | Robot | Layout | BG | Lang |
|---|---|---|---|---|---|---|---|---|---|
| OFT gốc | 96.8 | 70.2 | 48 | 65 | 90 | 27 | 67 | 98 | 97 |
| A1 (đối chứng: tự distill, ảnh chuẩn) | 97.0 | 68.8 | 45 | 68 | 85 | 27 | 63 | 95 | 98 |
| **A2 (tự distill, ảnh render lại dưới nhiễu)** | **97.0** | **81.9** | **97** | **95** | 83 | 33 | 72 | 93 | 100 |

A2 − A1 = **+13.1** trên LIBERO-Plus, cảnh chuẩn không đổi; A1 ≈ OFT gốc nên mức tăng không đến từ việc fine-tune.
Vượt Spatial Forcing (71.9) và OFT train trên 20k demo nhiễu (79.5, công bố, 4 suite). Một seed.

**Seed 8 (07/10):** A1 97.0 / **68.6** (Camera 43, Robot 25, Light 83, Sensor 68); A2 98.4 / **84.3** (Camera 98, Robot 30,
Light 98, Sensor 100) → **+15.7**. Hai seed: A1 68.7, A2 **83.1**, chênh **+14.4** (13.1 / 15.7), cảnh chuẩn 97.0 / 97.7.

**Nhánh π0.5 (adapter vòng 20, 200 episode mỗi ô LIBERO-PRO; Robot / Layout của LIBERO-Plus 120 task):**

| | Object | PRO swap | PRO position | PRO object | PRO lan | PRO task | Plus Robot / Layout |
|---|---|---|---|---|---|---|---|
| π0.5 gốc | 98.5 | 18.5 | 10.5 | 94 | 100 | 0 | 83 / 82 |
| base (distill on-policy) | 99.0 | 15.5 | 11.0 | 93 | 99.5 | 0 | 83 / 85 |
| shift (dời vật, teacher gán nhãn) | 99.0 | 18.0 | 15.5 | 93 | 99 | 0 | 87 / 83 |
| swap (đổi chỗ, nhãn chính xác) | 96.5 | **23.0** | 0.5 | 83 | 97 | 0 | 68 / 67 |
| swap nhẹ tay (lr 2e-5, encoder đóng băng) | 86.0 | 26.0 | 3.5 | 78 | 87.5 | 0 | — |
| swap off-policy (state của π0.5 gốc) | 86.0 | 25.5 | 0.0 | 71.5 | 85.5 | 0 | — |

- Phản thực đổi chỗ như đã thiết kế **có hại**: +5–8 trên ô swap nhưng dời vị trí sụp về 0, Robot / Layout −15. Nguyên
  nhân (phân tích của tôi): trong cặp đổi chỗ, vật được gọi tên luôn đứng ở vị trí quỹ đạo gốc đi tới, nên cặp không
  bao giờ đòi đi tới chỗ mới; policy học "đi tới vị trí quen bất kể tên" còn chặt hơn. Thiết kế tiếp theo phải có cặp
  mà hành động đúng đi tới vị trí khác với nhãn biết chắc (ví dụ phản chiếu 3D cả cảnh và quỹ đạo).
- shift trung tính (position +4.5, trong sai số).

**Nhánh token trên máy mới:** cos_b2 87.4 (500 ep) / 48.8 LIBERO-Plus; cos_b2p 84.6 / 45.2 (máy cũ, render CPU: 93.0 /
51.4 cho cos_b2). LIBERO-Long lr cosine: eval 100 episode 62% ở vòng 40 (lr cố định: 57–62; VLA-OPD 78.9).

## Failure analysis sâu của nhánh π0.5 (07/10 sáng)

**Hai ô LIBERO-PRO thực ra đo gì** (`scripts/analyze_swap_layout.py`, so init state của ô với LIBERO-Object chuẩn):
cả 10 task Object dùng chung **6 ô vị trí cố định** + giỏ cố định ở (0, 0.26); mỗi task xếp bộ vật khác vào các ô.
- **swap**: vật đích đổi chỗ chính xác với một vật khác (18–38 cm), giỏ đứng yên (lệch 1.5 cm).
- **position (temp)**: vật đích dời 21 cm theo x sang một **chỗ trống**, một vật khác bị đưa ra khỏi cảnh, giỏ đứng yên.
→ π0.5 học "câu lệnh → ô quen trong bố cục". Phản thực biến đổi cả cảnh (xoay, soi gương) giữ nguyên quan hệ giữa
các vật nên **có thể không phá được lối tắt này**; cần phản thực dời riêng vật đích so với phần còn lại.

**Thời điểm cam kết** (`scripts/analyze_commit.py`, 200 tập mỗi ô):
- swap: π0.5 cam kết đi tới vật sai rất sớm (trung vị query 2; 42% ngay query 0; 86% trong 2 query đầu). Adapter
  swap còn sớm hơn (56% ở query 0). Base on-policy giống π0.5 (+5 / −13 tập so với π0.5).
- Adapter swap vòng 20 so với π0.5: +20 / −13 tập, gain dồn vào task 6 (+13) và 7 (+5), mất ở task 0 và 3. Theo task,
  ô swap rất "rời rạc": task 2, 4, 5, 8, 9 bằng 0 với mọi model.
- position: cam kết muộn hơn (query 3), và **phần lớn thất bại không phải nhầm vật**.

**Ô position: π0.5 tìm được vật nhưng không xong việc.** Trong 179 tập hỏng của π0.5: lạc khi chuyển 86, nhầm vật 50,
near miss 28. Trong 86 tập "lạc khi chuyển", 63% **đến hết giờ vẫn đang cầm vật**, cách giỏ trung vị 10 cm, đã nhấc
28 cm; 55% lần kẹp đầu của tập hỏng bị trượt (phải kẹp lại). Giả thuyết: kẹp ở vị trí lạ kém chính xác → kẹp lại
nhiều lần → hết 280 bước. Đang kiểm bằng eval cùng ô với 400 bước (`pi05_pro_temp_ms400_steps`). Base on-policy
giống hệt (lạc khi chuyển 84, 73% còn cầm vật lúc hết giờ). Adapter swap biến phần lớn thành nhầm vật (113).

**Chỗ headroom nằm:** (1) swap: chọn đích sai từ những query đầu (~75% tập); (2) position: độ chính xác kẹp ở chỗ lạ
và thời gian, sau đó mới đến chọn đích.

**Vật đích của LIBERO-Object chỉ nằm ở 2 ô** (07/10, `scripts/analyze_target_slots.py`): A = (−0.12, −0.24) cho task 0, 4, 6, 7, 8 và
B = (+0.05, −0.10) cho task 1, 2, 3, 5, 9. Các ô C (−0.15, +0.06), D (+0.10, −0.20), E (+0.15, +0.03), F (−0.20, −0.08)
chỉ chứa vật phụ. Ô swap đưa vật đích tới:

| Task | Vật đích | Ô gốc → ô swap | π0.5 | xoay s8 (vòng 8) |
|---|---|---|---|---|
| 0 | alphabet soup | A → F | 0.35 | **0.95** |
| 1 | cream cheese | B → A | 0.90 | **1.00** |
| 3 | bbq sauce | B → A | 0.30 | **1.00** |
| 6 | butter | A → B | 0.30 | **0.90** |
| 2 | salad dressing | B → F | 0 | 0 |
| 4 | ketchup | A → E | 0 | 0 |
| 5 | tomato sauce | B → A | 0 | 0 |
| 7 | milk | A → B | 0 | 0 |
| 8 | chocolate pudding | A → D | 0 | 0 |
| 9 | orange juice | B → C | 0 | 0 |

- Vật đích tới một ô **chưa từng là ô đích** (C, D, E: task 4, 8, 9) → 0 với mọi model. Phép xoay chỉ dời vật dọc cung
  tròn có bán kính của A (0.54 m) và B (0.66 m) quanh trục khớp 1; C, D, E nằm ở bán kính 0.45, 0.73, 0.75 → không phủ.
- **Dự đoán (ghi trước khi có số):** coshift dời vật đích tự do 8–30 cm trong vùng bày vật nên sẽ cải thiện task 4, 8, 9
  nhiều hơn xoay. Nếu không → lỗi ở đó không phải do vị trí.
- **Kết quả dự đoán (coshift s7, vòng 8, swap 31.0):** theo task 0.80 0.95 0 0.60 0 0 0.65 0 0 0.10 → gain vẫn dồn vào
  task 0, 1, 3, 6; task 9 nhích 0.10, task 4, 8 vẫn 0. **Dự đoán sai phần lớn**: dạy vật đích ở chỗ khác so với bố cục
  (tại vòng 8) chưa phá được lỗi ở 6 task này. Ở cả 6 task, π0.5 kẹp đúng vật mồi đứng ở ô quen 20/20 lần (tất định).
  Rollout có state trên 6 task này (vòng 8, 10 tập mỗi task):

  | Task (vật đích → ô) | π0.5: kẹp gần | xoay s8: kẹp gần | coshift s7: kẹp gần, kết cục |
  |---|---|---|---|
  | 2 salad dressing → F | mồi 20/20 | mồi 6, vật khác 4 | mồi 5, **đích 3** |
  | 4 ketchup → E | mồi 20/20 | mồi 5, vật khác 3 | vật khác (bbq) 9, mồi 1 |
  | 5 tomato sauce → A | mồi 20/20 | mồi 9 | mồi 8 |
  | 7 milk → B | mồi 20/20 | mồi 10 | mồi 7, **đích 3** |
  | 8 chocolate pudding → D | mồi 20/20 | mồi 10 | mồi 10 |
  | 9 orange juice → C | mồi 19/20 | mồi 9 | **đích 10/10**: làm đổ 6, rơi khi mang 3, thành công 1 |

  Xoay không đổi hành vi trên các task này (vẫn đi tới mồi). Coshift **đổi được việc chọn đích** ở task 9 (10/10 tới
  đúng vật) và một phần task 2, 7, nhưng hỏng ở khâu thực thi tại chỗ lạ (làm đổ hộp nước cam cao, rơi khi mang) —
  cùng loại lỗi với ô position. Tức là headroom còn lại tách thành hai tầng: chọn đích (coshift chạm tới được) và
  thao tác ở vị trí chưa từng thấy (chưa phương pháp nào chạm tới; teacher π0.5 cũng kém ở đó).
- Task 5, 7 (đích tới A / B nhưng vẫn 0) chưa giải thích được bằng hình học; có thể là nhận dạng vật (tomato sauce /
  milk) — cần xem state.
- Ô position: vật đích tới chỗ trống, ô quen để trống → π0.5 vẫn đi tới vật (nhặt được ~50%). Tức lỗi swap là **bị hút
  về vật đang đứng ở ô quen**, cộng với thực thi kém ở chỗ lạ.

## Phản thực mới: tổng quát hơn và nhắm đúng lỗi (07/10)

Nguyên lý chung: dựng thế giới phản thực bằng một phép biến đổi 3D mà ta biết chính xác action phải biến đổi theo,
trên chính các state student đi qua; nhãn = phép biến đổi áp lên chunk của teacher (π0.5 gốc) ở thế giới gốc.

| Phản thực | Biến đổi | Nhãn | Cần gì | Cổng kiểm tra |
|---|---|---|---|---|
| soi gương | cả cảnh phản chiếu qua mặt phẳng đứng qua đế robot | đảo dấu dy, rx, rz | robot đối xứng, cảnh gần đối xứng | tay lệch 0.21 mm / 0.06°, phát lại 0.0 mm |
| **xoay** | mọi vật, bàn và đồ cố định quay θ quanh trục khớp 1, q1 += θ (camera, đế, sàn đứng yên) | chunk xoay θ (dx dy, rx ry) | chỉ cần khớp 1 trục đứng | tay lệch 0.21 mm / 0.03°, phát lại < 0.5 mm / 0.17°; phát lại vòng hở các tập thành công của π0.5 trong thế giới xoay 0.3 rad: Object 18/20, Spatial 16/19 (bản gốc 20/20, 19/20) |
| **coshift** | vật đích và bàn tay dời cùng một vector (tay bằng IK giữ hướng); vật khác và giỏ đứng yên; 50% là chiếm chỗ một vật khác | chunk gốc, mask sau lần kẹp đầu + 10 bước | danh tính vật đích | tay lệch 0.17 mm / 0.04°; sai lệch mỗi chunk trước khi kẹp trung vị 2.6 mm, 90% 6.6 mm; phát lại vòng hở vẫn nhấc vật 10/12 |

- Xoay và soi gương phá lối tắt "vị trí tuyệt đối"; coshift phá lối tắt "ô quen trong bố cục" (vật đích ở chỗ khác so
  với mọi thứ policy có thể đã thuộc), đúng loại lỗi của hai ô PRO. Sai lệch nhỏ của coshift đến từ bộ điều khiển
  OSC bám chunk hơi khác ở cấu hình tay khác; nhỏ so với 8–30 cm cần dạy, và policy lập lại kế hoạch mỗi chunk.
- Độ lệch của chính π0.5 khi được hỏi trong thế giới coshift (so với nhãn đúng): 0.32 (đơn vị chuẩn hoá) — teacher
  không bám theo vật bị dời, khớp với chẩn đoán.

**Đang chạy (07/10 02:00–):** đợt 3 (soi gương s7, soi gương chỉ 4 query đầu s7, base s8), đợt 4 (xoay s7, s8, soi
gương s8), đợt 5 (coshift s7, s8); cùng ngân sách 20 vòng × 1024 state, cùng bộ đánh giá (Object, năm ô PRO, Plus
Robot / Layout). Đo trần π0.5 gốc trên Spatial / Goal / Long (chuẩn + ô swap). OFT seed 8 (A1 / A2) trên Object.
07/10 02:51: người dùng cần 2 card trống → hai run xoay (s7, s8) dừng sau vòng 3 và chạy tiếp từ `state.pt` trên
card 4 và 6 (`scripts/server/pi05_arm.sh`); khi chạy tiếp, bộ sinh số ngẫu nhiên được seed lại theo (seed, vòng 3),
tức vẫn tất định nhưng không trùng từng bit với một lần chạy liền mạch. Card 2 và 3 để trống cho người dùng.

Đo trần (π0.5 gốc, 200 tập): Spatial chuẩn 98.5, **Spatial ô swap 42.0**; Goal chuẩn 98.5, **Goal ô swap 26.0**; Long
chuẩn 94.0, **Long ô swap 9.0** → lỗi đi theo vị trí quen không chỉ có ở Object.

Ô position với 400 bước thay vì 280 (π0.5 gốc): **21.5** (từ 10.5). Khoảng 11 điểm là do hết giờ, nhưng vẫn còn 74 tập
lạc khi chuyển, 46% trong số đó đến bước 400 vẫn cầm vật, lơ lửng cách giỏ ~11 cm: không chỉ là thiếu thời gian, mà
policy không hoàn tất được việc đặt vật khi vật được nhặt từ chỗ lạ. Còn lại: nhầm vật 52, near miss 23.

**Đọc sớm ở vòng 8, ô PRO swap (200 tập, π0.5 gốc 18.5):**

| | s7 | s8 | trung bình |
|---|---|---|---|
| base (không phản thực) | 15.5 (vòng 4–20: 22.0 / 15.5 / 16.5 / 14.5) | 15.5 | 15.5 |
| soi gương | 19.5 | 24.0 | 21.8 |
| soi gương, 4 query đầu | 23.5 | — | — |
| xoay | 24.0 | **38.5** | **31.3** |
| coshift | **31.0** | 23.5 | 27.3 |

Cả ba phản thực hơn base ở cả hai seed; phương sai giữa seed lớn (xoay 24 / 38.5). Gain dồn vào task 0, 1, 3, 6.

**Kết quả cuối (vòng 20, 07/10 08:00; … = đang chạy; 200 tập mỗi ô PRO, 120 tập Plus):**

| | Object | swap | position | object | lan | task | Plus Robot / Layout | swap / position ở vòng 8 |
|---|---|---|---|---|---|---|---|---|
| π0.5 gốc | 98.5 | 18.5 | 10.5 | 94 | 100 | 0 | 83 / 82 | |
| base s7 | 99.0 | 15.5 | 11.0 | 93.0 | 99.5 | 0 | 83 / 85 | |
| base s8 | 99.0 | 20.5 | 12.0 | 93.0 | 99.0 | 0 | 83 / 83 | 15.5 / 14.0 |
| soi gương s7 | 98.0 | 20.0 | 12.5 | 90.5 | 99.0 | 0 | 78 / 78 | 19.5 / 13.5 |
| soi gương s8 | 98.0 | 21.5 | 13.0 | … | 98.0 | 0 | … | 24.0 / 13.5 |
| soi gương 4 query đầu s7 | 97.5 | 25.5 | 11.5 | 92.5 | 99.0 | 0 | 75 / 85 | 23.5 / 15.5 |
| xoay s7 | 99.5 | 27.0 | 14.5 | … | … | 0 | … | 24.0 / 12.5 |
| xoay s8 | 99.0 | **34.5** | 16.0 | 92.0 | 99.5 | 0 | … | 38.5 / 15.0 |
| coshift s7 | 99.0 | 24.0 | **22.5** | 92.5 | 100 | 0 | **90** / 80 | 31.0 / 23.0 |
| coshift s8 | 99.0 | 24.5 | **18.0** | 92.0 | 99.5 | 0 | **90** / 83 | 23.5 / 20.5 |

**Theo checkpoint (vòng 8 / 12 / 16 / 20; 200 tập mỗi số):**

| | swap s7 | swap s8 | position s7 | position s8 |
|---|---|---|---|---|
| base | 22.0 (v4) / 16.5 / … / 15.5 | 15.5 / … / … / 20.5 | … / … / … / 11.0 | 14.0 / … / … / 12.0 |
| xoay | 24.0 / 30.5 / 26.5 / 27.0 | 38.5 / 27.0 / 34.0 / 34.5 | 12.5 / 15.5 / 15.0 / 14.5 | 15.0 / 15.5 / 15.5 / 16.0 |
| coshift | 31.0 / 31.0 / 27.0 / 24.0 | 23.5 / 27.0 / 27.0 / 24.5 | 23.0 / 25.5 / 24.5 / 22.5 | 20.5 / 24.0 / 23.5 / 18.0 |

Trung bình 4 checkpoint × 2 seed: **xoay swap 30.3, position 15.0; coshift swap 26.9, position 22.7**; base swap ~18,
position ~12. Mỗi ô swap chỉ có 10 bố cục nên một số đơn lẻ dao động ±5–10; coshift ở position ổn định ~2× base ở mọi
checkpoint; coshift swap giảm dần sau vòng 12.

Trung bình 2 seed so với base (18.0 swap, 11.5 position, 83 Robot):
- **xoay**: swap **30.8** (+12.8), position 15.3 (+3.8); cảnh chuẩn 99.3.
- **coshift**: swap 24.3 (+6.3), position **20.3** (+8.8), Plus Robot **90** (+7, cả hai seed); cảnh chuẩn 99.0.
- **soi gương**: swap 20.8 (+2.8), position 12.8; Plus Robot / Layout giảm (78 / 78 ở s7).
- Ô task (đổi vật đích bằng câu lệnh mới) = 0 với mọi model: ngoài tầm của các phản thực này.
- Gain bị bào mòn sau vòng 8 ở coshift s7 (31.0 → 24.0) và xoay s8 (38.5 → 34.5).
- Chưa so được với SOTA: ECT công bố Object swap 38.3 → 70.8 (protocol khác: 50 trial, baseline 38.3 so với 18.5 /
  23.5 của ta ở 280 / 520 bước); phải cài lại ECT trong harness (xem related_work §10).

Vòng 8 trên ô position: base s8 14.0, soi gương s7 13.5, soi gương 4 query đầu 15.5, xoay s7 12.5 — không sụp (khác
phản thực đổi chỗ: 0.5–1.0), nhưng cũng chưa tăng.

## Ablation coshift và bản gộp (đợt 6–7, 07/10 chiều; Object, kết quả cuối vòng 20)

| | Object | swap | position | object | lan | task | Plus Robot / Layout | swap / position vòng 8 |
|---|---|---|---|---|---|---|---|---|
| coshift s7 (tham chiếu) | 99.0 | 24.0 | 22.5 | 92.5 | 100 | 0 | 90 / 80 | 31.0 / 23.0 |
| coshift trên state π0.5 gốc s7 (gần ECT) | 98.0 | 22.5 | 23.5 | 89.0 | 97.0 | 0 | 93 / 80 | 31.0 / 21.0 |
| coshift chỉ chỗ trống s7 | 98.5 | **31.0** | 15.5 | 92.0 | 99.0 | 0 | 85 / 88 | 32.0 / 17.0 |
| coshift bỏ số hạng nhất quán s7 | 97.0 | 22.5 | 22.0 | 92.0 | 97.5 | 0 | 80 / 80 | 21.0 / 21.5 |
| gộp xoay + coshift s7 | 98.0 | 28.0 | 21.5 | 93.0 | 97.5 | 0 | 75 / 88 | 28.5 / … |
| gộp xoay + coshift s8 | 98.5 | 26.5 | 22.0 | 91.0 | 97.5 | 0 | 75 / 85 | 31.0 / … |

Đọc (một seed cho ablation, cẩn thận với nhiễu ±5):
- **State on-policy không tạo ra gain**: dùng state của π0.5 gốc cho swap 22.5 / position 23.5, ngang on-policy. Đóng góp
  không thể là "on-policy"; nó là **phản thực 3D có nhãn chính xác** (dựng lại cảnh + IK, không cần replay như ECT).
- **Số hạng nhất quán** gần như không đóng góp ở vòng 20 (swap 22.5 vs 24.0, position 22.0 vs 22.5); ở vòng 8 có chênh
  (21 vs 31) → có thể giúp học nhanh hơn, chưa chắc.
- **Chỉ chỗ trống** (bỏ việc chiếm chỗ vật khác): swap tốt hơn (31.0) nhưng position mất gain (15.5) → đánh đổi; đang chạy
  seed 8 (đợt 9).
- **Gộp xoay + coshift** cân bằng nhất: swap 27.3 (+9 so với base), position 21.8 (+10), cả hai seed, cảnh chuẩn 98.
- **LIBERO-Plus Robot init (60 tập, nhiễu ±6):** coshift 90 / 90 (s7 / s8), coshift trên state π0.5 gốc 93, chỉ chỗ trống 85,
  bỏ số hạng nhất quán 80, base 83 / 83; **xoay s8 78, gộp 75 / 75**. Giả thuyết: phản thực xoay luôn quay tay cùng cảnh
  → policy học "tay lệch góc ⇒ xoay action"; ô Robot init của LIBERO-Plus chỉ đổi tư thế tay → lối tắt đó hại. Coshift
  ngược lại: tay ở cấu hình khác (IK) mà action giữ nguyên → dạy bất biến với tư thế tay → Robot init tăng.

**Spatial (đợt 8):** base s7 chuẩn 98.0, ô swap **42.0** (= π0.5 gốc) — distill không phản thực không thay đổi gì.
coshift s7: chuẩn 97.5, ô swap **37.0 (−5)**, task 0. Ô swap của Spatial **khác Object**: bát cần nhặt đứng yên (lệch
< 2 cm), **đĩa (nơi đặt) bị dời 17–47 cm** (`analyze_swap_layout.py --cell libero_spatial_swap --base libero_spatial`);
theo task, π0.5 / base / coshift: 1.0 1.0 1.0 | 0.3 0.3 0.1 | 1.0 0.95 0.95 | 0 0 0 | 0.4 0.4 0.3 | 0.6 0.55 0.35 | 0 0 0 |
0 0 0 | 0.9 1.0 1.0 | 0 0 0. Vì sao coshift không giúp: (1) nó chỉ tác động trước khi kẹp và mask phần sau kẹp, còn lỗi
Spatial ở pha đặt; (2) câu lệnh Spatial gọi vật đích bằng quan hệ ("bát giữa đĩa và ramekin"), dời riêng bát làm câu lệnh
sai trong thế giới phản thực → nhãn mâu thuẫn với ngôn ngữ. **Giới hạn của coshift: chỉ hợp lệ khi vật đích được gọi
bằng tên / ngoại hình.**
Hướng sửa có nguyên tắc: dời tay cùng **vật mà tay đang thao tác theo** — trước khi kẹp: vật đích; sau khi kẹp: vật đang
cầm + nơi đặt (đĩa / giỏ). Ở pha sau, quan hệ tay–nơi đặt giữ nguyên nên nhãn là chunk gốc, chính xác cho pha mang /
đặt, và không đụng tới quan hệ dùng để chọn vật đích.

**Kết quả Spatial đủ (s7, vòng 20):**

| | Spatial chuẩn | ô swap | object | lan | task | Plus Robot / Layout |
|---|---|---|---|---|---|---|
| π0.5 gốc | 98.5 | 42.0 | … | … | … | … |
| base | 98.0 | 42.0 | 98.5 | 96.0 | 0.5 | 83 / 98 |
| xoay | **99.5** | 39.5 | 97.0 | 96.5 | 0 | 82 / 97 |
| coshift | 97.5 | 37.0 | 98.0 | 96.5 | 0 | 80 / 95 |

**Cả hai phản thực không sang được Spatial** (ô swap −2.5 / −5, trong nhiễu nhưng không có dấu hiệu tăng). Lý do theo
phân tích bố cục: lỗi Spatial nằm ở **pha đặt** (đĩa bị dời), mà xoay giữ nguyên quan hệ giữa đĩa và các vật khác, còn
coshift chỉ tác động trước khi kẹp. Tức là gain trên Object gắn với loại lỗi "chọn vật đích theo ô quen" — đúng loại lỗi
mà các phản thực này nhắm tới — chứ không phải một cải thiện chung.

**Coshift pha sau kẹp** (`--coshift_phase post|both`): khi vật đích đang được cầm, dời tay (IK) + vật đang cầm + nơi đặt
cùng một vector 8–40 cm, các vật khác đứng yên; nhãn = chunk gốc, không mask. Cổng (`check_coshift.py --phase post`):
từ state đầu tiên vật được cầm của các tập thành công, dời rồi phát lại phần còn lại của tập. Tính trên các ca IK với tới
(ca không với tới bị loại, như lúc huấn luyện): Object **8/8** thành công (gốc 12/12), Spatial **9/11** (gốc 12/12).
(Lần chạy đầu báo FAILED vì script giữ handle của simulator cũ — LIBERO dựng lại simulator ở mỗi reset.)

**Đợt 10 — kết quả:** coshift sau kẹp trên Spatial s7: chuẩn 98.0, **ô swap 40.0** (base 42.0) — không tăng (chỉ ~9% state
có cặp hợp lệ: đang cầm bát + IK với tới + có chỗ cho đĩa). Coshift hai pha trên Object s7: chuẩn 99.5, swap 20.0 (coshift
trước kẹp 24.0).

**Lỗi Spatial swap là nhớ vị trí đặt** (`analyze_failures` trên rollout π0.5, `scripts/analyze_place_location.py`): π0.5 nhặt đúng bát
(kẹp gần bát đích ở 9/10 task), hỏng chủ yếu "lạc khi chuyển" (91 / 200); trong các tập đó bát được đặt xuống **cách chỗ
quen của đĩa 3 cm** (trung vị), cách đĩa thật (đã dời 26 cm) 25 cm.

**Vì sao coshift sau kẹp không giúp — giới hạn có tính nguyên lý:** dời tay + vật cầm + đĩa cùng một vector giữ nguyên
vector tay → đĩa, nên policy "đi một đoạn đã thuộc từ chỗ nhặt tới chỗ đặt" vẫn khớp nhãn; phản thực không phạt lối tắt đó.
Tổng quát: mọi phản thực có nhãn chính xác ta dựng được (xoay cả cảnh; dời tay cùng vật) đều **giữ nguyên quan hệ tay–đích**
→ chúng dạy bất biến với bố cục / tư thế và cải thiện thao tác ở chỗ lạ (ô position của Object), nhưng **không dạy suy ra
đích ở vị trí mới so với tay** — đúng loại lỗi của Spatial swap (và phần "nhầm vật" của Object swap). Muốn dạy điều đó cần
nhãn cho những thế giới mà quan hệ tay–đích đổi: teacher π0.5 không làm được (nó mắc chính lỗi này); ứng viên: teacher có
đặc quyền (biết vị trí đĩa) cho pha mang / đặt — bộ điều khiển kịch bản hoặc lập kế hoạch đi tới vị trí đĩa mới, như cách
ECT dùng replay controller cho quỹ đạo đã biến đổi.

**Phản thực "dời nơi đặt" (relocate) + teacher kịch bản có đặc quyền** (07/10 tối, `src/rollout/scripted.py`):
- Khi đang mang vật (vật được cầm), chỉ dời nơi đặt (đĩa / giỏ) tới chỗ trống — tránh vật khác và đồ cố định (tủ, bếp);
  tay và vật đang cầm giữ nguyên → **quan hệ tay–đích đổi thật**. Nhãn = chunk của bộ điều khiển kịch bản biết vị trí đĩa
  (mang ở độ cao +15 cm trên đĩa, hạ xuống, thả khi vật cách mặt đĩa < 4.5 cm), sinh trên mô hình động học của OSC
  (dịch chuyển mỗi bước ≈ g · a · 5 cm, g = 0.19 / 0.25 / 0.13 cho x / y / z, khớp từ log π0.5).
- Cổng (`check_scripted_place.py`, 20 tập thành công, đĩa dời 10–40 cm): bộ điều khiển chạy vòng kín 19/20 (đĩa không dời
  19/20); chạy theo chunk như student sẽ học (10 bước của chunk 50 bước, rồi lập lại) **18/20**. (Lần đầu 8/20: đĩa bị dời
  vào chỗ tủ / bếp — đã thêm điều kiện tránh đồ cố định.)
- Chạy thử: ~11% state có cặp; độ lệch giữa chunk của π0.5 và nhãn trong thế giới dời đĩa **0.70** (cao nhất từ trước tới
  giờ — π0.5 vẫn đi tới chỗ đĩa cũ).
- Đợt 12 (đang chạy): relocate trên Spatial s7, s8, lambda nhất quán 0 (nhãn không phải biến đổi của chunk gốc).
- Lưu ý cho paper: đây là teacher có đặc quyền dạng kịch bản cho một pha của task — khác về bản chất với các phản thực nhãn
  chính xác ở trên; cần nói rõ.

**Kết quả sớm (vòng 8, ô swap Spatial, 200 tập; base s7 42.0, π0.5 gốc 42.0, xoay 39.5, coshift 37.0, coshift sau kẹp
40.0): relocate s7 **62.0**, s8 **56.0** (+14 / +20).** Cảnh chuẩn khi train: 1.0 ở vòng 8. Theo task (π0.5 → relocate s7):
t1 0.30 → 0.55, t4 0.40 → 0.55, t5 0.60 → 0.95, t6 0 → 0.10, **t7 0 → 0.80**, t9 0 → 0.25 — đúng các task hỏng vì đặt bát
vào chỗ đĩa quen. Đây là bằng chứng đầu tiên cho chẩn đoán "phải đổi quan hệ tay–đích mới dạy được suy ra đích".

**Relocate trên Spatial — kết quả cuối (vòng 20, 2 seed):** s7 chuẩn 98.5, **swap 64.5**, object 98.0, lan 97.0, task 0.5,
Plus R/L 88 / 98; s8 chuẩn 98.0, **swap 63.0**, object 97.0, lan 96.5, task 0.5, Plus 93 / 100 (base s7: 98.0, 42.0, 98.5,
96.0, 0.5, 83 / 98). Ô swap **+22**, các ô khác và cảnh chuẩn giữ nguyên. Theo checkpoint (vòng 8 / 12 / 20): s7 62.0 /
61.5 / 64.5, s8 56.0 / 57.5 / 63.0.

**Relocate trên LIBERO-Long, bản đầu (s7) — có hại:** eval chuẩn khi train 0.98 → 0.72 / 0.80 / 0.74 (vòng 4 / 12 / 16; base
0.91–0.98), hỏng ở các task hai vật vào giỏ (t0 0.1, t7 0.6, t4 0.5, t8 0.6). Hai lỗi của chính mình khi mở rộng sang task
nhiều vật: (1) vật vừa thả vào giỏ vẫn cao và gần tay → bị coi là "đang cầm" → dạy mang lại vật đã đặt; (2) sau lúc thả, nhãn
của bộ điều khiển là "đi thẳng lên", mâu thuẫn với việc đi lấy vật thứ hai. Sửa: chỉ coi là đang cầm khi gripper đang được
lệnh đóng và hai ngón không khép rỗng, vật trong 8 cm quanh tay; mask chunk sau lúc thả + 5 bước. Run cũ dừng ở vòng 16;
chạy lại `p05lg_ocd_relocate2_s7`. Base Long s7: chuẩn 95.0, swap 8.5, task 8.0.

**Relocate trên Object và bản sửa trên Long cũng sụp (đợt 14, 08/10 01:45):** Object relocate s7 eval chuẩn 0.72 ở vòng 4
(rollout 0.47–0.73), Object coreloc s7 0.55, Long relocate2 rollout 0.68 ở vòng 3; Spatial relocate s9 vẫn 0.99. Khác biệt:
**tỉ lệ state mang nhãn kịch bản** — Object 60% (pha mang tới giỏ dài), Long 27%, Spatial ~12%. Nhãn kịch bản khác phong cách
π0.5 (đi thẳng ở độ cao cố định); khi chiếm đa số, nhất là với các lần dời nhỏ (ảnh gần như cảnh gốc), policy nhận nhãn mâu
thuẫn → hỏng. Sửa (đợt 15): giữ cặp phản thực trên tối đa 15% state (`--cf_frac 0.15`) và dời nơi đặt ít nhất 15 cm
(`--cf_coshift_post 0.15,0.4`); chạy lại Object relocate3, Object coreloc3, Long relocate3 (s7). Ba run sụp đã dừng.

**Nguyên nhân thật: bộ điều khiển đặt vật hỏng với giỏ** (08/10 02:30). Đợt 15 vẫn sụp dù đã giới hạn 15% (Object relocate3
rollout 0.47 ở vòng 3, Long 0.78). Cổng kiểm tra bộ điều khiển chỉ mới chạy trên Spatial (đĩa phẳng) — lỗi của mình: trên
Object, ngay cả khi giỏ đứng yên nó chỉ thành công **7/20**: mang vật ở +15 cm trên gốc toạ độ của giỏ (dưới miệng giỏ) nên
vật đâm vào thành và đẩy giỏ đi (có ca giỏ bị đẩy từ y 0.18 tới 0.83). Các run relocate trên Object / Long đã học từ nhãn
sai. Sửa: tính độ cao miệng nơi đặt từ hình học (`rim_height`); nơi đặt có thành (≥ 3 cm) thì mang ở miệng + 15 cm và thả
khi vật cách miệng 6 cm; đĩa phẳng giữ như cũ. Cổng lại: Object **20/20** (giỏ dời **18/20**), Spatial 19/20 / 18/20 như cũ.
Đợt 16: Object relocate4 s7 có / không giới hạn 15%. Long: đang sinh rollout có state của π0.5 để chạy cổng trước khi train.
Bài học: cổng kiểm tra phải chạy trên **mọi** bộ trước khi train, không chỉ bộ đầu tiên.
Cổng trên Long: chấm theo env (cần cả hai vật) cho 0/15 — tiêu chí sai cho task nhiều vật; chấm theo từng vật (vật mang
nằm trong nơi đặt, tay đã rời): Long **15/15**, giỏ dời **13/15**; Object 20/20, 18/20. Chạy lại `p05lg_ocd_relocate4_s7`.

**Giữa chừng đợt 16 (08/10 03:46, vòng 4):** Object relocate4 (bộ điều khiển đã sửa) **không sụp nữa**: eval chuẩn 0.94
(không giới hạn, ~46% state có cặp) và 0.98 (giới hạn 15%), rollout 0.90–0.99 → lỗi đâm vào thành giỏ là nguyên nhân chính
ở Object. Long relocate4 vẫn tụt (eval 0.85, rollout 0.65–0.78; base ~0.93) dù cặp chỉ 12–16% → Long còn vấn đề khác chưa
tìm ra; dừng ở vòng 5 (chạy tiếp được từ `state.pt`). Người dùng yêu cầu tạm dừng sau khi các run hiện tại xong.

**Kết quả (08/10 06:40):**
- **Spatial relocate 3 seed:** swap 64.5 / 63.0 / **66.5** (TB **64.7**); base s7 / s8 42.0 / 38.0 (TB 40.0) → **+24.7**. Chuẩn
  98.5 / 98.0 / 98.5 (base 98.0 / 97.0); object / lan giữ nguyên. Plus Robot 88 / 93 / 80 vs base 83 / 83 (trong nhiễu).
- **Object relocate4 (bộ điều khiển đã sửa) s7:** **position 32.0** (không giới hạn) / **34.0** (giới hạn 15%), swap 23.0 /
  20.0, chuẩn 99.0 / 99.0, lan 100 / 98. Base 3 seed position 12.2, coshift 20.7 → **relocate +20 trên ô position**. Khớp chẩn
  đoán: ở ô position π0.5 nhặt được vật ở chỗ lạ rồi mang theo đoạn đã thuộc tới giỏ → trượt giỏ; relocate sửa đúng lỗi đó.
  Swap không đổi (lỗi swap ở bước chọn vật).
- **Goal base s7:** chuẩn 97.5, swap 29.0 (π0.5 26.0), task 10.0, object 87.5, lan 96.5, Plus R/L 73 / 68.
- Tiếp: Object relocate4cap s8 (card 2), Object coreloc4 s7 (card 3; coshift trước kẹp + relocate khi mang); ablation
  relocate trên Spatial (card 1); chẩn đoán ô swap Goal (card 0).
- Object relocate4 Plus R/L: 73 / 88 (không giới hạn), 83 / 92 (giới hạn) — trong nhiễu; object 91.5 / 90.0.
- **Ô swap Goal giống Object, không giống Spatial** (`analyze_swap_layout.py --cell libero_goal_swap`): vật đích đổi chỗ với
  một vật khác 12–37 cm; nơi đặt hầu hết là đồ cố định (bếp, nóc tủ, giá) và không dời. π0.5 / base theo task: t1 0.70 / 0.80,
  t3 1.0 / 1.0, t7 0.95 / 0.90, t8 0.05 / 0.20, còn lại 0 → lỗi chọn nhầm vật; relocate không chạm tới. Đang chạy Goal xoay
  và Goal augmentation (s7, card 0). Hướng tiếp: phản thực "retarget" cho pha tiếp cận — dời riêng vật đích (tay giữ nguyên),
  nhãn từ bộ điều khiển kịch bản biết vị trí vật, đưa tay tới trên vật — đối xứng với relocate ở pha đặt.

**Cận trên "tiếp cận oracle" (08/10 07:40, `scripts/oracle_approach.py`):** trên ô swap Object, bộ điều khiển kịch bản đưa tay
tới trên đúng vật đích (2–4 query; tới nơi ở 100% tập), rồi π0.5 gốc làm phần còn lại: **59.0** (π0.5 tự làm 18.5; 520 bước
23.5). Theo task: 0.3 0.8 0.6 1.0 0.0 0.9 1.0 0.5 0.8 0.0 — task 2, 5, 7, 8 (trước 0) lên 0.5–0.9; task 4 (ketchup → ô E),
9 (orange juice → ô C) vẫn 0. → Phần lớn lỗi ô swap nằm ở pha tiếp cận; khi đã đứng trên đúng vật π0.5 kẹp và đặt được.
**Phản thực retarget** (`--cf_mode retarget`): trước khi kẹp, dời riêng vật đích (chỗ trống, hoặc chiếm chỗ vật khác với xác
suất p_swap), tay giữ nguyên; nhãn = `approach_chunk` (tới 10 cm trên vật, gripper mở, giữ hướng) trên mô hình động học,
mask sau khi tới nơi + 3 bước; giới hạn 15% state. Đợt 18: retarget Object s7 (card 2), Goal s7 (card 3).

**Ablation relocate trên Spatial (đợt 17, s7, ô swap):**

| | thế giới phản thực | teacher có đặc quyền | swap | chuẩn |
|---|---|---|---|---|
| base (TB s7 / s8) | — | — | 40.0 | 97.5 |
| dời đĩa, nhãn = π0.5 (`--relocate_label teacher`) | có | — | **40.0** | 98.0 |
| nhãn kịch bản ở thế giới gốc (`--cf_coshift_post 0,0`) | — | có | **35.5** | 98.5 |
| relocate đầy đủ (TB 3 seed) | có | có | **64.7** | 98.3 |

→ **Không thành phần nào tự nó có tác dụng; chỉ khi kết hợp mới +25.** Thế giới phản thực tạo ra tình huống quan hệ tay–đích
đổi; teacher có đặc quyền cho nhãn đúng ở đó (π0.5 không làm được vì mắc chính lỗi này). Trả lời trực tiếp phản biện "chỉ
là dùng expert kịch bản": expert kịch bản một mình còn giảm 4.5.

**Retarget giữa chừng (08/10 10:15):** Object s7 vòng 8: **swap 46.0** (base 18.7, augmentation 24.2, xoay 29.2, cận trên 59.0);
eval chuẩn khi train 0.72 (v4) → 0.82 (v8) → **0.98 (v12)**. Goal s7 vòng 8: swap **33.0** (base 29.0, π0.5 26.0), chuẩn 0.81 →
0.93 → 0.97. **Cận trên tiếp cận oracle trên Goal: 23.0** (π0.5 tự làm 26.0; theo task 0 0.2 0 0 0 0 0.5 0.9 0.7 0; t1 "bát lên
bếp" 0.70 → 0.2 — tới trên tâm bát không phải tư thế kẹp tốt cho bát) → lỗi Goal không nằm ở pha tiếp cận như bộ điều khiển
hiện tại hiểu; retarget không thể tăng nhiều ở đó. Object retarget s8 đang chạy.

**08/10 trưa:**
- **Coshift + relocate (coreloc4, bộ điều khiển đã sửa) Object s7: position 42.5** (relocate 32 / 34, coshift 20.7, base 12.2),
  swap 24.0, chuẩn 97.0 → hai phản thực cộng dồn theo pha (coshift: nhặt ở chỗ lạ; relocate: mang tới giỏ từ chỗ lạ).
- Relocate4cap Object s8: position **31.5** (s7 34.0) → relocate trên ô position lặp lại qua 2 seed; swap 25.0, chuẩn 98.5.
- Goal augmentation s7: chuẩn 97.0, swap 29.0 (= base).
- **Teacher chuẩn hoá thất bại** (`scripts/canon_teacher_eval.py`): hỏi π0.5 trong thế giới đã đưa vật đích về chỗ quen và dời
  tay ngược −δ (IK), thực thi trong thế giới thật → ô swap Object **22.0** (π0.5 18.5, oracle viết tay 59.0). π0.5 nhớ cả quỹ đạo
  từ tư thế tay quen, không chỉ vị trí vật (khớp chẩn đoán của ECT) → π0.5 không làm teacher được ở bất kỳ thế giới lệch nào;
  teacher tổng quát phải là thứ khác (lập kế hoạch chuyển động, policy RL có đặc quyền).
- Mới: chế độ `rr` (retarget trước kẹp + relocate khi mang) Object s7; coreloc4 s8.

**08/10 chiều:**
- **Retarget Object s7 kết quả cuối: swap 51.0** (vòng 8: 46.0; base 18.7, aug 24.2, xoay 29.2, oracle 59.0), chuẩn 99.0,
  position 18.0, lan 99.5, task 0. **Nhưng ô object (đổi ngoại hình vật) 72.5** (base 93.0, coshift 92.5; theo task 0.40 0.60
  0.70 0.70 **0.00** 1.0 1.0 1.0 0.85 1.0). Nghi lối tắt "nhặt vật nằm lệch chỗ" (trong thế giới retarget vật đích luôn là vật
  bị dời; ở ô swap lối tắt này trùng đáp án đúng). Đang kiểm bằng phép thử "vật lệch chỗ" (`--displace_distractor 0.2`: một
  vật phụ bắt đầu lệch ~20 cm ở Object chuẩn) cho base / coshift / retarget / relocate.
- Goal retarget s7: swap **32.5** (base / aug 29.0), chuẩn 94.5, lan 91.0, task 11.0. Goal xoay s7: chuẩn 92.0, swap 27.0.
  Goal aug s7: chuẩn 97.0, swap 29.0, object 91.0, lan 97.0, task 11.5, Plus R/L 75 / 65.
- **Lỗi ô swap Goal là "lạc khi chuyển"** (π0.5, `analyze_failures` sau khi bỏ qua task có vật đích là đồ cố định): t2 chai lên
  nóc tủ 17/20, t4 bát lên nóc tủ 20/20, t8 bát lên đĩa 19/20 lạc khi chuyển; t6 phô mai vào bát 20/20 tới vật mồi; t9 chai lên
  giá 15/20 làm đổ. → cùng loại lỗi relocate đã sửa ở Spatial / Object; giải thích vì sao retarget ít tác dụng trên Goal.
- **Relocate cho nơi đặt là vùng trên đồ cố định** (LIBERO khai báo vùng đặt là site: `wooden_cabinet_1_top_side`,
  `flat_stove_1_cook_region`, `wine_rack_1_top_region`): dời cả khối đồ cố định khi render (`m.body_pos` của gốc), đích của bộ
  điều khiển = vị trí site. Cổng trên Goal: 19/30 không dời, 20/30 dời — đạt ở task 1 (bếp), 6 (bát), 8 (đĩa), 9 (giá), hỏng ở
  2, 4 (nóc tủ), 3 (ngăn kéo) ngay cả khi không dời. Chạy Goal relocate s7 chỉ với cặp ở task qua cổng (`--cf_tasks 1,6,8,9`)
  — như ECT chỉ giữ demo phát lại thành công.
- **Phép thử "vật lệch chỗ"** (Object chuẩn, một vật phụ bắt đầu lệch ~20 cm, 100 tập): base 99 (0 nhầm vật), coshift 99 (1),
  **retarget 95 (0 nhầm vật**; hỏng: rơi khi mang 3, đổ 2), relocate 97 (2) → retarget **không** học lối tắt "nhặt vật lệch chỗ";
  gain swap là thật.
- **Vì sao ô object tụt:** ô "object" của LIBERO-PRO **đổi màu vật đích** (red_cream_cheese, green_bbq_sauce, blue_ketchup). Task
  1–4, 10 tập: base 97.5 (kẹp đúng vật, lệch < 2 cm); **retarget 42.5** — t1 không dám kẹp 4, t3 nhặt nhầm 4, t4 (ketchup xanh)
  0/10 (không kẹp 5, kẹp lệch 16–27 cm). Base làm tốt ô này vì tìm vật **theo vị trí quen** (đổi màu không ảnh hưởng); retarget
  dạy tìm vật **theo ngoại hình** — cần cho ô swap, nhưng màu đổi thì không nhận ra. → Hai cách nhận diện vật đích, mỗi ô của
  benchmark thưởng một cách; dữ liệu train mỗi vật chỉ có một ngoại hình. Hướng sửa: ngẫu nhiên hoá màu vật khi render trong
  lúc train retarget, để policy nhận vật theo hình dạng / tên chứ không theo màu cụ thể.
- Goal: xoay Plus 73 / 67, object 81.0, lan 92.0, task 12.5; retarget Plus 85 / 60, object 79.5 (cũng tụt — cùng nguyên nhân).
- **Retarget Object s8: swap 52.5** (s7 51.0) → TB 2 seed **51.8** (+33 so với base 18.7), chuẩn 98.5.
- Ngẫu nhiên hoá màu vật (`--obj_tint p`, mỗi vật với xác suất p được nhân màu vật liệu với màu ngẫu nhiên, mô hình dựng lại
  mỗi reset nên không cộng dồn): đang chạy retarget + màu (s7) và base + màu (s7) trên Object.
- Retarget s8: position 22.0, object **76.0** (s7 72.5) → tradeoff swap ↔ object lặp lại; lan 100, Plus 83 / 82.
- **Coreloc4 s8: position 42.0** (s7 42.5) → TB **42.3** (+30 so với base 12.2); swap 27.5, chuẩn 96.5, lan 98.0.
- **Retarget trên state π0.5 gốc (s7, gần ECT về nguồn state): swap 50.5, position 20.5** vs on-policy 51.0 / 52.5, 18.0 / 22.0 →
  nguồn state không tạo khác biệt (như ablation coshift trước); khác biệt so với ECT phải nằm ở teacher theo pha / phân tích /
  không cần demo hay bộ phát lại, không phải on-policy.
- **rr (retarget trước kẹp + relocate khi mang) Object s7: swap 63.5** (retarget 51.8, cận trên tiếp cận 59.0), chuẩn 97.5 → hai
  teacher theo pha cộng dồn, vượt cận trên của riêng pha tiếp cận. rr s8 đang chạy.

**08/10 tối:**
- **Retarget Object 3 seed: swap 51.0 / 52.5 / 55.5 (TB 53.0, +34)**, position 18.0 / 22.0 / 21.5, chuẩn 99.0 / 98.5 / 99.5;
  object 72.5 / 76.0 (tradeoff, xem trên). Retarget trên state π0.5 gốc: object 72.0, Plus 85 / 92.
- rr s7: position 29.5, object **80.5**, lan 94.5, task 0 (swap 63.5). Coreloc4 s8 Plus 87 / 87, object 88.5.
- Base + màu ngẫu nhiên s7: chuẩn 99.0, swap 18.0 (= base) — riêng đổi màu không giúp swap.
- Chế độ `full` (trước kẹp luân phiên retarget / coshift, khi mang relocate) Object s7 đang chạy.
- Goal relocate s7 (bản lọc task cũ 1,6,8,9, ~4% state có cặp): chuẩn 98.0, swap **28.5** (= base) — các task lỗi chính (nóc tủ)
  bị loại.
- **Relocate theo mục tiêu BDDL** (08/10 16:30): nơi đặt của vật đang cầm lấy từ vị từ `on` / `in` của mục tiêu task (vật di
  động, vùng trên vật di động như lòng giỏ / ngăn khay, vùng trên đồ cố định như bếp / nóc tủ / lò vi sóng; vùng trên mặt bàn bỏ
  qua); vật đang cầm = bất kỳ vật nào trong mục tiêu được nhấc gần tay. Cổng (`scripts/check_relocate_gate.py`, dùng chính
  cơ chế của env, chấm theo từng vật, nơi đặt dời 15–40 cm): Long task 0 / 1 / 7 (giỏ) 5/5, 4–5/5; 3 (ngăn kéo) 2/5; 9 (lò vi
  sóng) 1/3; 2, 8 (ấm moka lên bếp), 6 (cốc lên đĩa) 0. Goal 1 (bếp) 4/6, 4 (nóc tủ) 4/6, 6 (bát) 3/4, 8 (đĩa) 4/5, 3 (ngăn
  kéo) 5/6. Object 14/20 (không dời 20/20). Chạy Goal relocate2 (`--cf_tasks 1,2,3,4,6,8,9`) và Long relocate5
  (`--cf_tasks 0,1,7`), s7.
- **Đổi màu khi train gỡ phần lớn tradeoff:** retarget + màu (p 0.5) s7: chuẩn 96.5, **swap 55.0**, position 19.0, **object 85.5**
  (retarget đơn lẻ 72.5 / 76.0 / 75.0; base 93), lan 97.0. Base + màu s7: chuẩn 99.0, swap 18.0, position 11.0, object 92.5, Plus
  83 / 83 (= base: đổi màu một mình không làm gì). Tiếp: rr + màu s7, retarget + màu p 0.9 s7.
- Retarget s9 đủ: object 75.0, lan 99.0, Plus 85 / 82. Goal relocate (bản cũ) object 86.5, lan 98.0, task 10.0, Plus 78 / 68.
- rr s8 đủ: chuẩn 98.0, swap 54.5, position 25.5, lan 98.5, task 0.

**09/10 (đêm UTC 08/10): bỏ danh sách task chọn tay — ba kiểm tra tự động cho teacher có đặc quyền.** Mục tiêu: một cấu hình
chạy y nguyên trên cả 4 bộ. Các lựa chọn theo bộ trước đây (retarget chỉ cho Object / Goal, `--cf_tasks` theo cổng) thay bằng:
1. **Vật có bản sao không được dời** (`--cf_unique`): nếu cảnh có vật khác cùng loại (cùng lớp đối tượng LIBERO), câu lệnh
   chỉ có thể chỉ ra nó bằng vị trí ("bát cạnh ramekin", "đĩa bên trái") → dời nó là trái câu lệnh. `scripts/check_unique.py`:
   Spatial: vật đích (2 bát giống hệt) bị khoá ở cả 10 task, đĩa vẫn dời được → retarget tự tắt trên Spatial, relocate giữ;
   Object, Goal: không vật nào có bản sao; Long: task 4 (đĩa trái / phải) và 8 (hai ấm moka) bị khoá.
2. **Teacher phải đồng ý với chuyên gia ở thế giới thật** (`--cf_agree c`): env tính thêm nhãn của chính teacher kịch bản ở
   thế giới không dời (`cf_label_nom`); chỉ giữ cặp nếu cos(tổng xyz 10 bước đầu của teacher, của π0.5) ≥ c. Lý do: kiểm tra
   check_unique lộ ra vật đích lấy theo `obj_of_interest[0]` sai ở vài task (Long 6: chọn cái đĩa thay vì cái cốc; Goal 3,
   Long 2: task bắt đầu bằng mở ngăn kéo / bật bếp, không phải tiếp cận vật) — teacher nói "tới vật" trong khi chuyên gia
   (đúng ở thế giới thật) làm việc khác. Lọc theo từng cặp, không theo task.
   Chọn tiêu chí (`scripts/analyze_agree.py` trên cặp của 4 run thử 1 vòng, chế độ full + unique + màu): 10 bước nhầm cú nhấc
   lên của π0.5 với hướng đi (Goal 8 khi mang chỉ giữ 0.2 dù placer qua cổng); **chốt xyz 25 bước, cos ≥ 0.5**. Tỉ lệ cặp giữ:

   | | đúng (teacher hợp lệ) | sai đã biết |
   |---|---|---|
   | Object | trước kẹp 0.81–1.0, khi mang 0.76–0.98 | — |
   | Spatial | khi mang 1.0 (mọi task) | — |
   | Goal | 1 / 4 / 6 / 8: 0.86–1.0 | 5 (đẩy đĩa) 0.33; **3 (mở ngăn kéo trước) 0.76 — không lọc được** (ngăn kéo cùng hướng với bát) |
   | Long | 0 / 1 / 3 / 7 / 9: 0.75–1.0 | **2 (bật bếp trước) 0.0; 6 (vật đích nhầm là đĩa) 0.0** |

   **Cổng relocate, 20 tập / task của π0.5 gốc** (đặt thành công trong thế giới đã dời / số lần thử; không dời gần như luôn
   thành công):
   - Object: 0 16/18, 1 12/20, 3 17/20, 4 18/20, 5 16/20, 6 16/20, 8 12/20, 9 13/20 → qua; **2 (salad dressing) 10/19, 7
     (sữa) 5/20 → loại** (vật cao, chạm thành giỏ).
   - Spatial: qua 0, 1, 2, 3, 5, 6, 8, 9 (14–20 / 17–20); **loại 4 (bát trong ngăn kéo) 9/16, 7 (bát trên bếp) 10/19**.
   - Goal: qua 3 (ngăn kéo) 13/19, 4 (nóc tủ) 16/19, 6 (bát) 12/20, 8 (đĩa) 18/20; **loại 1 (bếp) 8/19, 9 (giá) 2/6, 2 (n = 1)**.

**Đợt 19 — một cấu hình cho cả 4 bộ** (`scripts/server/pi05_round19.sh`): chế độ `full` (Object s7: swap 53.0, position 39.0,
chuẩn 96.0, lan 96.5 — cân bằng nhất: rr 59.0 / 27.5, coreloc 27.5 / 42.3) + `--cf_unique --cf_agree 0.5` + danh sách cổng
ở trên + đổi màu p 0.5, cap 15%, s7. Object / Spatial (card 0), Goal (card 2) chạy từ 21:15 UTC 08/10; Long (card 3) từ 21:20.
Cổng Long: qua 0 18/19, 1 20/20, 7 19/20 (giỏ); loại 3 (ngăn kéo) 6/18, 9 (lò vi sóng) 3/17, 6 2/19; 2, 8 (ấm moka lên bếp)
0/16, 0/13 cả khi không dời; 4 bị khoá bởi `unique`.

**Kết quả xong trong đêm 08/10:**
- `full` s7 (Object, không màu, không 3 kiểm tra): chuẩn 96.0, swap 53.0, position **39.0**, object 80.0, lan 96.5, Plus 88 / 80.
- rr 3 seed: swap 63.5 / 54.5 / **51.0** (TB 56.3), position 29.5 / 25.5 / 16.0 (TB 23.7); s9 chuẩn 95.0.
- **Goal relocate theo mục tiêu BDDL** (task 1, 2, 3, 4, 6, 8, 9; 11% state có cặp): chuẩn 97.0, swap **28.5** (base 29.0), task
  10.0 → relocate không giúp Goal, kể cả khi đã gồm nóc tủ / ngăn kéo.
- **Long relocate5** (task giỏ 0, 1, 7): chuẩn 91.0 (base 95.0), swap **11.0** (base 8.5) → không giúp.

- **Đổi màu không nhất quán**: rr + màu s7 object **74.0** (rr không màu 80.5 / 82.0 / 81.5), swap 58.5, position 27.5, chuẩn
  98.0, lan 96.0; retarget + màu p 0.9 s7: chuẩn 98.5, swap 49.0, position 20.5, lan 96.0. Cùng lúc retarget + màu p 0.5 cho
  object 85.5 (retarget 72.5–76). → hiệu ứng của đổi màu lên ô object chưa ổn định (1 seed mỗi cấu hình); rút lại nhận định
  "đổi màu gỡ phần lớn tradeoff" cho tới khi có thêm seed. Đợt 19 vẫn dùng màu p 0.5. Retarget + màu p 0.9: object **81.5**,
  Plus 83 / 87 → với riêng retarget, cả hai mức màu (85.5, 81.5) > mọi seed không màu (72.5–76); với rr thì không (74.0).

**ECT trong harness (đợt 20, 08/10 22:20 UTC).** Đọc kỹ ECT (related_work §11): biến đổi cả cảnh (gương / dịch), **robot và tư thế
đầu giữ nguyên**, nhãn = bộ điều khiển bám đường đi EEF đã biến đổi, chỉ giữ replay thành công — tức cũng đổi quan hệ tay–đích
với nhãn có đặc quyền. Cài lại (`EnvRunner.ect_replay`, `--cf_mode ect`): mỗi vòng 16 tập thành công của vòng đó được chạy lại
(lấy đường đi EEF), biến đổi cảnh (vật tự do + đồ nội thất trên bàn; bàn, robot đứng yên), bám đường đi đã biến đổi bằng action
gốc đã biến đổi (dấu tịnh tiến theo trục bị phản chiếu, xoay theo luật giả vectơ −Mω, gripper chép) + ½ sai số vị trí qua mô
hình động học; ghép query t của tập gốc với query t của replay (cùng câu lệnh, cùng noise / thời điểm flow như loss ECT); nhãn =
lệnh của replay từ t, mask sau khi hết. Cùng ngân sách với ta (cap 15%, không số hạng nhất quán). Biến đổi theo Bảng 20 của họ:
Spatial / Goal gương y; Object gương y, gương x + dịch, gương xy + dịch; Long dịch (cỡ dịch họ không nêu — ta chọn 8–10 cm).
- Lỗi tự phát hiện khi xem ảnh: `mirror_quat` (viết cho vật quay mặt về robot) làm tủ phản chiếu quay mặt ra ngoài → ngăn
  kéo hỏng. Sửa: đồ nội thất quay mặt về khu đặt vật (lật trục ngang vuông góc hướng đó).
- Cổng `scripts/check_ect.py` (3 tập / task, π0.5 gốc): đối chứng identity Spatial 0.97, Goal 0.97, Long 0.90, Object 1.00
  (bản đầu chỉ bám vị trí: Goal 0.77, ngăn kéo hỏng — đổi sang feedforward + hiệu chỉnh); biến đổi: Object 1.00 (cả 3),
  Spatial gương 0.87, Goal gương 0.63 (đẩy đĩa / bật bếp / giá rượu hỏng), Long dịch 0.70.
3. **Cổng thực thi với ngưỡng chốt trước** (`check_relocate_gate.py`): placer chạy vòng kín trong thế giới đã dời, task qua
   nếu thành công ≥ 60% (n ≥ 5); chỉ áp cho cặp relocate (`--relocate_tasks`), cặp trước kẹp giữ ở mọi task. Đang sinh 20
   tập / task của π0.5 gốc trên Goal, Long, Spatial để chạy lại cổng với mẫu đủ lớn (bản trước 2–6 tập / task).

**Seed 9 (Object, đợt 11) — bảng 3 seed (s7 / s8 / s9, kết quả cuối vòng 20, 200 tập mỗi ô):**

| | Object | swap | TB | position | TB |
|---|---|---|---|---|---|
| base | 99.0 / 99.0 / 100 | 15.5 / 20.5 / 20.0 | 18.7 | 11.0 / 12.0 / 13.5 | 12.2 |
| base + aug 2D | 99.5 / 98.5 / 98.5 | 23.0 / 25.0 / 24.5 | 24.2 | 12.0 / 12.0 / 15.0 | 13.0 |
| **coshift** | 99.0 / 99.0 / 98.0 | 24.0 / 24.5 / 24.5 | 24.3 | **22.5 / 18.0 / 21.5** | **20.7** |
| xoay | 99.5 / 99.0 / 99.0 | 27.0 / 34.5 / … | … | 14.5 / 16.0 / … | … |

→ Ô position: coshift +8.5 so với base, +7.7 so với augmentation; mọi seed coshift > mọi seed đối chứng. Ô swap: coshift =
augmentation. LIBERO-Plus (60 tập, nhiễu ±8): augmentation 82 / 82 / 80 Robot.

## Đối chứng augmentation ảnh 2D thông thường (đợt 9, Object; trả lời "có phải chỉ là augmentation?")

Base (distill, không phản thực) + augmentation openpi (cắt 95% + resize, xoay ±5° ảnh agent view, đổi màu cả hai view),
cùng ngân sách:

| | Object | swap s7 / s8 (TB) | position s7 / s8 (TB) |
|---|---|---|---|
| base | 99.0 / 99.0 | 15.5 / 20.5 (18.0) | 11.0 / 12.0 (11.5) |
| **base + aug 2D** | 99.5 / 98.5 | 23.0 / 25.0 (**24.0**) | 12.0 / 12.0 (12.0) |
| coshift | 99.0 / 99.0 | 24.0 / 24.5 (24.3) | 22.5 / 18.0 (**20.3**) |
| xoay | 99.5 / 99.0 | 27.0 / 34.5 (**30.8**) | 14.5 / 16.0 (15.3) |
| gộp | 98.0 / 98.5 | 28.0 / 26.5 (27.3) | 21.5 / 22.0 (21.8) |

- **Trên ô swap, coshift = augmentation thông thường** (24.3 vs 24.0): phần gain swap của coshift không phải của phương pháp.
  Xoay hơn augmentation ~+7.
- **Trên ô position, augmentation = base** (12.0 / 12.0 ở cả hai seed) còn coshift 18–22.5 → gain position của coshift là thật.
- **LIBERO-Plus Robot init:** augmentation 82 / 82 (Layout 88 / 85), base 83 / 83, coshift 90 / 90 → gain Robot của coshift
  cũng không phải do augmentation.
- **Coshift chỉ chỗ trống, 2 seed:** swap 31.0 / 29.0 (TB **30.0**, +6 so với augmentation), position 15.5 / 16.0 (15.8);
  coshift gốc (p_swap 0.5): swap 24.3, position 20.3 → hai biến thể đánh đổi swap ↔ position, nhất quán qua seed.

**Phép thử lối tắt của xoay** (`--q1_offset`: tay bắt đầu quay quanh khớp 1, cảnh không quay; Object, 100 tập; chuẩn
98.5–99.5):

| | +0.25 rad | −0.25 rad | TB |
|---|---|---|---|
| π0.5 gốc | 90 | 61 | 75.5 |
| base s8 | 91 | 63 | 77.0 |
| xoay s8 | 95 | 55 | 75.0 |
| coshift s8 | 96 | 61 | 78.5 |
| gộp s8 | 91 | 55 | 73.0 |

Xoay / gộp kém base ở −0.25 (−8) nhưng hơn ở +0.25 (+4) → không thấy lối tắt nhất quán; trung bình mọi model trong ±3
(100 tập mỗi số, nhiễu ~±5). Giả thuyết "xoay học lối tắt tay lệch góc ⇒ xoay action" **không được xác nhận**. Robot
init của LIBERO-Plus đổi nhiều khớp, không chỉ khớp 1; lý do xoay giảm Robot (78, gộp 75) vẫn chưa rõ — cần phân loại
lỗi trên chính ô Robot (eval có lưu state).

**Chạy lại ô Robot init (60 tập, cùng 60 task, có lưu state, 07/10 18:10):** base s8 82 (lần trước 83), xoay s8 75 (78),
coshift s8 **80 (90)**, gộp s8 **83 (75)**. Cùng adapter, cùng task mà lệch tới ±10: π0.5 lấy nhiễu theo hàng của batch
(`fixed_noise`, hàng i chỉ phụ thuộc i), mà env nào nằm ở hàng nào phụ thuộc thứ tự env sẵn sàng → mỗi lần eval là một mẫu
khác của policy ngẫu nhiên; với 60 tập, độ lệch chuẩn nhị thức ~5 điểm. Phân loại lỗi (11–15 tập hỏng mỗi model) không
có kiểu lỗi riêng cho xoay. **Kết luận: các chênh lệch trên Plus Robot / Layout ở 60 tập (coshift +7, xoay −5) nằm trong
nhiễu; không dùng làm kết luận.** Muốn khẳng định cần ≥ 240 tập mỗi model. Các ô PRO (200 tập) cũng chịu nhiễu kiểu này
(~±3 điểm), nhỏ hơn các chênh lệch chính (position +8–10 qua 2–3 seed và 4 checkpoint).

## Câu hỏi tính tổng quát (người dùng, 07/10) và kế hoạch

Lo ngại: gain có phụ thuộc đặc thù dữ liệu không? Kế hoạch:
1. Chốt trên Object với π0.5 theo tiêu chí đã đặt (thắng base rõ, swap / position tăng đáng kể, cảnh chuẩn không sụp,
   ≥ 2 seed và nhiều checkpoint).
2. Tìm thành phần tạo gain bằng ablation bỏ bớt: loại phản thực (xoay / gương / coshift), state on-policy vs state
   của π0.5 gốc, có / không số hạng nhất quán, chỉ state đầu vs toàn tập, tỉ lệ swap trong coshift (0 = chỉ chỗ trống).
3. Không đổi cấu hình khi sang Spatial / Goal / Long (LIBERO-PRO + LIBERO-Plus); chỉ thắng trên Object thì báo là
   đặc thù dữ liệu. Ghi rõ yêu cầu của từng phản thực (xoay: không gì; coshift: danh tính vật đích, có sẵn trong mô tả
   task của mọi benchmark mô phỏng).
4. Gắn gain với intuition: probe phản thực p, thời điểm cam kết, độ lệch của teacher, trước / sau huấn luyện.

## Việc đang chạy / tiếp theo

Xem mục "Phản thực mới" ở trên. Các nhánh token (B2 / B2′ / B3 / B4) và takeover trên máy cũ đã kết thúc; máy cũ
đã xoá.

## Dọn ổ (07/10 16:30, theo yêu cầu người dùng)

Ổ chứa workspace trên máy 8×H200 (dùng chung) đầy 100% (32 GB trống) → xoá ~200 GB (`scripts/server/cleanup_workspace.sh`,
chạy thử trước, bỏ qua run còn đang train và eval còn đang ghi): còn 232 GB trống. Đã xoá:
- trọng số + optimizer của các hướng bỏ / đã thay: π0.5 swap, swap nhẹ tay, swap off-policy, shift, soi gương (3 run);
  OFT A1/A2 trên Spatial / Goal / Long (đã dừng giữa chừng); nhánh token (cos_b2, cos_b2p, long_cos_b2); smoke test;
- với run giữ lại: `state.pt`, `adapter_last` (trùng vòng cuối), adapter trung gian (π0.5 giữ vòng 8, 12 và vòng cuối;
  OFT giữ vòng cuối);
- `steps/` (state từng query) của các phân tích đã xong, trừ 4 thư mục mốc của π0.5 gốc (object, pro_swap, pro_temp,
  spatial_pro_swap) mà các cổng kiểm tra dùng;
- trọng số và `.npz` trong `outputs/old_h200`, `outputs/old_l40` (và trong bản relay trên laptop, 9 GB → 15 MB);
- checkpoint gốc của các hướng không đi tiếp (tải lại được từ HF bằng `scripts/server/fetch_checkpoints.sh`):
  Haozhan72 OFT-SFT object traj1 / trajall, libero10 traj1, libero10 traj1-rl; Spatial Forcing object.
Giữ nguyên: mọi log, `summary.json`, `episodes.jsonl`, `failures.*`, `train_log.jsonl`; π0.5 gốc; 4 checkpoint OFT của moojink.

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

## Mổ xẻ cơ chế (09/10 sáng)

- Cấu hình π0.5 LIBERO của ta đưa proprio vào prompt dạng ~120 token chữ số (`Task: … , State: 255 255 173 … ;\nAction: `),
  câu lệnh chỉ ~13 token.
- **Phép thử bỏ proprio lúc test** (`--state_blind`: proprio = trung bình dataset, không mang thông tin; π0.5 gốc, không train):
  Object chuẩn **100.0** (thật: 99), swap **14.0** (thật: 18.5) → π0.5 không cần proprio trên Object, và lối tắt quỹ đạo
  **không đi qua đường proprio**. Bác bỏ giả thuyết "cắt proprio ở state mù". Quỹ đạo được truy xuất từ câu lệnh + ảnh (bố
  cục, có thể cả hình ảnh cánh tay — khớp với teacher chuẩn hoá 22%). Run distill state_blind (Object, Spatial, s7) vẫn
  chạy như ablation.
- Đang chạy: E1 (`scripts/probe_mechanism.py`, độ nhạy theo thời điểm: dời / đổi chỗ vật, dời giỏ, lệch proprio, đổi tên vật
  trong câu lệnh), E2 / E3 (`scripts/probe_internals.py`, probe tuyến tính theo tầng + attention của token hành động theo
  đoạn). Bản thử E2 nhỏ: vị trí vật đích đọc ra được ở tầng sau (token ảnh R² ~0.5, token hành động ~0.67) — trùng phát hiện
  "mã hoá được nhưng không điều khiển" (2610.06235).
- **E1, Object, π0.5 gốc** (`mech_object_base`; follow = 1 nếu chunk đi theo đúng thứ đã dời, 0 nếu bỏ qua; trung vị):

  | thời điểm | dời 3 cm | dời 20 cm | đổi chỗ | đổi tên trong câu lệnh | lệch proprio 5 cm |
  |---|---|---|---|---|---|
  | đầu pha tiếp cận (> 15 cm) | 0.18 | 0.22 | 0.22 | 0.25 | 0.0 |
  | giữa (5–15 cm) | 0.61 | 0.31 | 0.03 | 0.02 | 0.0 |
  | sát (< 5 cm) | 0.51 | 0.29 | 0.01 | −0.02 | 0.0 |
  | đang mang (dời giỏ) | 0.09 | 0.06 | — | — | 0.0 |

  Goal: đầu pha 0.23–0.33 (đổi chỗ 0.42, đổi tên 0.41), khi mang 0.48 → 0.14 theo cỡ dời. Spatial: đầu pha 0.21–0.30, giữa
  0.47 (3 cm) → 0.25 (20 cm), đổi chỗ 0.08–0.14, khi mang (dời đĩa) 0.17–0.30, proprio ≈ 0. → (1) ở khoảnh khắc quyết định model
  gần như mù; (2) khi đã tiến vào là bộ bám cục bộ (dời ít bám nhiều hơn dời xa), lựa chọn đã khoá (đổi chỗ / đổi tên ≈ 0);
  (3) đặt vật vào giỏ hoàn toàn thuộc lòng; (4) proprio không có tác dụng ở mọi pha.
- **E2 / E3, Object** (`internals_object_base`, 2579 truy vấn, bước flow đầu): attention của token hành động gần như giống
  nhau giữa lúc mù và lúc bám (không phải lỗi "không nhìn"). **Probe trong cùng state** (`scripts/analyze_internals.py`: độ
  chênh trạng thái ẩn giữa biến thể và state gốc ~ độ dời): dời giỏ ≥ 12 cm khi mang → token hành động tầng 8–17 R² 0.74–0.82,
  token ảnh ~0.9; dời vật ≥ 12 cm trước kẹp → token hành động tầng sau ~0.58. Tức là **phần sinh hành động có mã hoá việc giỏ /
  vật bị dời, nhưng động tác đầu ra gần như không đổi (follow 0.06–0.2)** → chỗ nghẽn ở khâu đọc ra cuối của action expert.
  Lưu ý: R² cao không nói tín hiệu lớn (probe chuẩn hoá khuếch đại tín hiệu nhỏ nhất quán).
- Thí nghiệm kiểm chứng đang chạy: (a) E1 / E2 trên model đã train (full s7, ECT s7) — mã hoá có đổi không, hay chỉ khâu đọc
  ra đổi; (b) chỉ train khâu đọc ra (`--lora_scope readout`, 34 nghìn tham số) và 6 tầng cuối + đọc ra (`expert_late`, 4.65
  triệu) với đúng dữ liệu của full s7.

## Tái hiện đối thủ trong harness và kiểm tra setup tương xứng (09/10)

- **Mức tương xứng của setup.** Đánh giá: base π0.5 của ta khớp π0.5 chính thức mà ECT đo (Object 18.7 vs 18.2, Long 8.5 vs
  9.4; Spatial 40 vs 46.6, Goal 29 vs 34.2), 20 tập / task (họ 50). Huấn luyện: **không** tương xứng — ta LoRA, từng bộ, 20 ×
  1024 state; ECT 4 bộ cùng lúc, 30k bước × batch 64; 2606.27663 69 task, full FT 30k bước × batch 256 và **dùng điểm oracle
  của simulator cả lúc test**. → So công bằng = cài lại đối thủ trong setup của ta (ECT: xong; 2606.27663: đang chạy), hoặc
  đưa ta vào setup lớn của họ (chưa làm).
- **2606.27663 cài lại** (`--geo`, `Pi05Policy.enable_geo`, `EnvRunner._geo_d`): d = đích con − tay (3D, đặc quyền; đích =
  vật cho tới khi nó cao hơn lúc đầu 1 cm, sau đó nơi đặt theo mục tiêu BDDL); MLP 3 → 1024 → 1024 (ReLU, lớp cuối khởi tạo 0)
  cộng vào embedding thời gian = điều kiện adaRMS (AdaLN) của mọi tầng action expert; thầy π0.5 không thấy d. Lựa chọn của
  ta: learning rate của MLP × 10 (module mới, ngân sách nhỏ). Đang chạy `p05_geo_s7` (Object), `p05sp_geo_s7` (Spatial).
  Chẩn đoán sau đó: E1 trên model này (còn mù ở khoảnh khắc quyết định không), và các chỗ ta nghi nó yếu — Spatial khi train
  từng bộ (bài gốc chỉ có GR00T: 15.2), Goal / Long (bài không báo), ô đổi màu.
- Có code nhưng không ưu tiên: Spatial Forcing, ROCKET (theo tài liệu không tăng swap), GAM (mô hình riêng). Không có code:
  2606.27663 ("coming soon"), ECT, QuoVLA ("sẽ công bố").
