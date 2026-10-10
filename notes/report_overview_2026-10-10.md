# Tổng quan dự án: từ ý tưởng đến tình hình hiện tại (10/10/2026)

Cập nhật khoảng 15:30 giờ VN (08:30 UTC). Mọi số là của chúng tôi, đo trong cùng harness, trừ khi ghi "công bố".

Đọc thêm khi cần chi tiết:
- [report_mechanism_2026-10-09.md](report_mechanism_2026-10-09.md): mổ xẻ cơ chế, tái hiện đối thủ.
- [week1_report.md](week1_report.md): báo cáo tổng hợp tuần đầu.
- [week1_status.md](week1_status.md): nhật ký từng bước.
- [related_work.md](related_work.md): khảo sát tài liệu.

---

## 0. Một trang

**Vấn đề.** π0.5 (VLA mạnh nhất hiện có trên LIBERO) đạt khoảng 97–99% ở cảnh chuẩn. Nhưng khi **đổi chỗ hai vật** (LIBERO-PRO
swap) hay **dời vật sang chỗ khác** (position), nó tụt còn 12–40%: robot vẫn đi tới chỗ cũ, theo quỹ đạo đã thuộc lòng.

**Insight.** Mổ xẻ bên trong π0.5 cho thấy:
- Nó **không mù**: thông tin "vật đã bị dời" có trong biểu diễn bên trong (probe tuyến tính đọc ra được).
- Nhưng ở **khoảnh khắc quyết định**, thông tin đó không đi ra tới động tác: động tác chỉ bám theo khoảng 3–20%.
- Huấn luyện phản thực của ta **không dạy model nhìn tốt hơn mà dạy nó dùng điều đã biết**:
  - mức mã hoá như nhau ở base, ECT và ta;
  - mức bám theo thì ở ta tăng mạnh (0.03 → 0.53).
- Phần thay đổi cần thiết nằm ở **phía VLM, chủ yếu tầng 9–17**, không nằm ở đầu ra hay action expert.

**Kết quả chính (cùng setup, 2 seed mỗi bên; seed 9 sắp xong):**

| Ô | Base | ECT cài lại | **Ta** |
|---|---|---|---|
| Object swap | 18.7 | 23.8 | **56.3** (+32.5 so với ECT) |
| Object position | 12.2 | 16.5 | **40.3** (+23.8) |
| Spatial swap | 40.0 | 50.8 | **66.0** (+15.2) |
| Object / Spatial chuẩn | 99 / 97.5 | 99 / 99 | 98 / 98.5 |

**Chỗ yếu.**
- Goal: hai task mở ngăn kéo tụt.
- Long: cảnh chuẩn 95 → 84–87; ECT cũng tụt như vậy.
- Ô đổi màu Object: 92.5 → khoảng 82. Đây là tradeoff có cơ chế.

**Hôm nay.**
- Sửa hai lỗi nhãn thật trong Goal và Long, nhưng chúng **không phải nguyên nhân chính** của mức tụt Goal.
- Phát hiện LIBERO **đặt lại ngẫu nhiên vị trí tủ / bếp ở mỗi lần reset** (lệch tới khoảng 2 cm), và kết quả eval Goal phụ
  thuộc vào điều đó. Đang đo có kiểm soát để biết model của ta có nhạy với vị trí đồ cố định hơn base không.

---

## 1. Bài toán và benchmark

### 1.1 Benchmark

- **LIBERO** gồm 4 bộ, mỗi bộ 10 task:
  - Object: nhặt một món tạp hoá bỏ vào giỏ;
  - Spatial: nhặt cái bát được mô tả bằng vị trí, đặt lên đĩa;
  - Goal: cùng một cảnh, 10 mục tiêu khác nhau;
  - Long: nhiệm vụ hai bước.
- **π0.5 đã gần bão hoà LIBERO chuẩn**, nên các bài gần đây chuyển sang **LIBERO-PRO** (2510.03827). LIBERO-PRO giữ nguyên
  task nhưng thay đổi:
  - **swap**: đổi chỗ các vật;
  - **position**: dời vật đích, chỉ có ở bộ Object;
  - **object**: đổi màu / ngoại hình vật;
  - **lan**: diễn đạt lại câu lệnh;
  - **task**: đổi hẳn task.
- **LIBERO-Plus**: nhiễu vị trí khởi đầu của robot và bố cục.
- Số π0.5 gốc của ta khớp số công bố mà ECT đo: Object swap 18.7 so với 18.2, Long 8.5 so với 9.4.

### 1.2 Vì sao đây là bài toán thật

Model thuộc lòng "câu lệnh → quỹ đạo", bỏ qua vị trí thật của vật. Đây là thất bại tổng quát hoá cơ bản nhất của VLA.
- **ECT** (đối thủ gần nhất) nêu đúng vấn đề này.
- **2606.27663** đưa điểm 3D oracle vào action expert.
- Hơn 20 bài đưa depth / 3D vào VLA nhưng không chữa được swap ở nơi nào có đo.

---

## 2. Insight: "biết nhưng không dùng"

### 2.1 Công cụ đo riêng của ta

Ta có thể dừng simulator ở **bất kỳ khoảnh khắc nào** của một episode, sửa đúng **một** thứ (dời vật, đổi chỗ, dời giỏ, đổi
tên trong câu lệnh), chụp lại ảnh, rồi đo:
- **E1:** động tác có bám theo thay đổi không (1 = bám hoàn toàn, 0 = bỏ qua).
- **E2:** biểu diễn bên trong có mã hoá thay đổi không (probe trong cùng một state, theo từng tầng).
- **E3:** token động tác nhìn vào đâu (tỉ lệ attention).

### 2.2 Ba model, cùng một phép đo (Object)

| | Base | ECT | **Ta** |
|---|---|---|---|
| E1: giữa pha tiếp cận, đổi chỗ | 0.03 | 0.07 | **0.53** |
| E1: giữa pha, đổi tên vật trong câu lệnh | 0.02 | 0.01 | **0.54** |
| E1: đang mang, dời giỏ 12–20 cm | 0.06 | 0.11 | **0.44–0.69** |
| E1: đầu pha (tay còn xa), dời vật | 0.13–0.22 | 0.14–0.23 | 0.20–0.30 |
| E2: mã hoá "đổi chỗ" (R², token động tác tầng sau) | 0.55 | 0.56 | 0.70 |
| E2: mã hoá "dời giỏ" | 0.80 | 0.86 | 0.85 |

→ Cả ba model đều **biết** (E2 gần nhau). Chỉ model của ta **dùng** (E1). Chỗ mù còn lại chung cho cả ba là **đầu pha tiếp
cận**.

**Bằng chứng mạnh nhất.**
- Bản cài lại 2606.27663 đưa **vị trí đích chính xác** (oracle) vào mọi tầng action expert, nhưng swap vẫn bằng base (18.5)
  và E1 giống hệt base.
- Kênh mới học thành một độ lệch cố định, gần như không đổi khi vị trí đích đổi (d đổi 10 cm thì đầu ra chỉ đổi 0.05).

→ **Có thông tin là chưa đủ; phải có dữ liệu buộc model dùng nó.**

### 2.3 Chỗ cần thay đổi nằm ở đâu (ablation phạm vi train, Object, cùng dữ liệu)

| Phần được train | Swap | Position | Chuẩn |
|---|---|---|---|
| Không train (base) | 18.7 | 12.2 | 99 |
| Chỉ lớp đọc ra cuối | 23.5 | 20.5 | 100 |
| Chỉ action expert (2 seed) | 32.0 | 29.8 | 94 |
| Chỉ K/V của VLM (thứ action expert đọc) | 20.0 | 9.5 | **72** (phá hành vi) |
| 9 tầng sau của VLM | 41.0 | 28.5 | 92 |
| **Chỉ VLM**, action expert đóng băng (3 seed) | **49.3** | 29.8 | 96 |
| Toàn bộ | 53.0 | 39.0 | 96 |

**Đọc:**
- Việc **chọn đúng vật** (swap) gần như hoàn toàn do VLM (khoảng 90% gain).
- Việc **đặt đúng chỗ** (position) cần cả hai nửa.
- Chỉ đổi K/V thì không đủ: VLM phải **xử lý lại** bên trong, chủ yếu ở tầng 9–17.
- Tradeoff ô đổi màu cũng sinh ra ở phía VLM: chỉ VLM thì đổi màu khoảng 69, chỉ action expert khoảng 87.

### 2.4 Các giả thuyết đã bác

| Giả thuyết | Thí nghiệm | Kết quả |
|---|---|---|
| Lối tắt đi qua trạng thái tay | Bỏ trạng thái tay khi test / khi train | Không đổi (swap 14.0 / 40.5 = base) |
| Nghẽn ở lớp đọc ra cuối | Chỉ train lớp cuối | 23.5 |
| Thiếu thông tin hình học | Điểm 3D oracle (2606.27663) | 18.5 (= base) |
| Điểm 3D + dữ liệu phản thực | Kết hợp hai thứ | 43.0, không hơn khi không có điểm 3D; kênh 3D vẫn bị bỏ qua |

---

## 3. Phương pháp hiện tại: giám sát phản thực tại khoảnh khắc quyết định

**Khung.** On-policy distillation: học trò (π0.5 + LoRA r32) tự chạy rollout; thầy (π0.5 gốc) gắn nhãn ở các state học trò đi
qua. 20 vòng × 1024 state.

**Phần riêng.** Với khoảng 15% state, ta dựng một **thế giới phản thực** ngay tại state đó và gắn nhãn đúng cho thế giới ấy
bằng một "thầy" kịch bản biết vị trí thật (privileged):

| Loại | Khi nào | Sửa gì trong cảnh | Nhãn |
|---|---|---|---|
| retarget | trước khi kẹp | dời vật đích, hoặc đổi chỗ với vật khác | đi tới vật ở chỗ mới |
| coshift | trước khi kẹp | dời cả tay và vật cùng nhau | động tác gốc (quan hệ tay–vật không đổi) |
| relocate | đang mang vật | dời nơi đặt (giỏ, đĩa, hoặc cả tủ / bếp) | mang tới nơi đặt mới |

**Các bộ lọc "nhãn phải đúng"** — đây là phần chiếm nhiều công sức nhất, và là nơi tìm ra hầu hết lỗi:
- **không dời vật có bản sao** (`--cf_unique`): khi có hai cái bát giống nhau, câu lệnh chỉ phân biệt được bằng vị trí;
- **kiểm tra đồng thuận** (`--cf_agree`): thầy kịch bản trong thế giới thật phải đi cùng hướng với thầy π0.5;
- **chỉ gắn cặp trong đoạn thật sự tiến tới vật** (`--hindsight`), từ hôm nay thêm **cảnh không còn bị thay đổi**
  (`--scene_tol`, xem §5.1);
- **không đổi chỗ với vật mà nhiệm vụ cũng phải mang** (`--no_cotarget_swap`, từ hôm nay, xem §5.2);
- đổi màu vật ngẫu nhiên khi train (`--obj_tint 0.5`), để giảm tradeoff ở ô đổi màu.

**Về cách trình bày** (theo góp ý của anh): phần dựng thế giới phản thực **không claim là đóng góp**, nó là công cụ. Đóng góp
dự kiến:
1. Chẩn đoán "biết nhưng không dùng", có đo so sánh giữa ba model.
2. Định vị chỗ cần thay đổi (VLM tầng 9–17; K/V không đủ; action expert chỉ là phụ).
3. Bằng chứng rằng đưa thêm thông tin hình học không giúp nếu thiếu áp lực dùng.
4. Can thiệp đúng khoảnh khắc mù, nhắm đúng tầng (còn để mở, xem §7).

---

## 4. So với đối thủ, cùng setup

### 4.1 Setup có tương xứng không

| Mặt | Tương xứng? | Chi tiết |
|---|---|---|
| Đánh giá | có | Cùng harness; số π0.5 gốc khớp số công bố |
| Huấn luyện | không, nếu lấy số công bố | Ta: LoRA, từng bộ, ngân sách nhỏ. ECT: 4 bộ, 30k bước. 2606.27663: 69 task, full fine-tune |
| Thông tin lúc test | không | 2606.27663 dùng vị trí oracle cả lúc test |

→ Vì vậy ta **cài lại đối thủ vào cùng setup**. ECT, 2606.27663 và QuoVLA chưa công bố code. Các đối thủ có code (Spatial
Forcing, ROCKET, GAM) thì theo tài liệu của chính họ không chữa được swap.

### 4.2 Toàn bộ các ô (seed 7; Object và Spatial có seed 8)

| Bộ / ô | Base | ECT | 2606.27663 | **Ta** | Nhận xét |
|---|---|---|---|---|---|
| Object swap | 18.7 | 23.8 | 18.5 | **56.3** | hơn rất rõ |
| Object position | 12.2 | 16.5 | 12.5 | **40.3** | hơn rất rõ |
| Object đổi màu | 92.5 | 93.3 | 93.0 | 81.8 | kém (tradeoff) |
| Spatial swap | 40.0 | 50.8 | 40.0 | **66.0** | hơn rõ |
| Goal swap | 29.0 | 24.5 | — | 28.5–31.0 | ngang base, hơn ECT |
| Goal chuẩn | 97.5 | 96.5 | — | 89.5–94.0 | kém (xem §5) |
| Long swap | 8.5 | 14.0 | — | 11.0–17.0 | ngang / hơn nhẹ |
| Long chuẩn | 95.0 | 84.0 | — | 84.0–87.0 | ngang ECT, cả hai tụt |

ECT với 50% cặp không tốt hơn 15% (Spatial 48.0 so với 52.0; Object 25.0 so với 24.0), nên baseline không bị làm yếu.

---

## 5. Chẩn đoán các ô yếu (phần làm hôm nay)

### 5.1 Goal task 3: lỗi chia pha — đã sửa, nhưng không phải nguyên nhân chính

- **Lỗi.** Ghi từng bước cho thấy π0.5 mở ngăn kéo trên bằng cách **gạt tay nắm, không đóng tay kẹp**. Bộ chia pha cũ dựa vào
  lệnh đóng tay kẹp để tách bước, nên coi cả pha mở ngăn kéo là "đang đi tới cái bát". Kết quả là cặp retarget dạy "tới bát"
  ngay lúc đang mở ngăn kéo; ở task 3, cả 107 / 107 cặp đều được giữ.
- **Sửa (tổng quát, không theo task).** Env ghi lại tổng dịch chuyển của phần cảnh không phải robot hay vật đích (ngăn kéo, núm
  vặn, cửa, vật khác). Pha "tiến tới vật" chỉ bắt đầu sau lần cuối cảnh còn đổi ≥ 1 cm. Kiểm tra trên dữ liệu thật: pha bắt đầu
  đúng lúc ngăn kéo mở xong, số cặp ở task 3 giảm từ 127 xuống 50, các task khác không đổi.
- **Nhưng v3 (bản có sửa) vẫn tụt** ở task 0 / 3, đo giữa chừng với 10 lần thử mỗi task:

  | | Vòng 4 | Vòng 8 | Vòng 12 |
  |---|---|---|---|
  | v3 | 0.6 / 0.3 | 0.4 / 0.4 | 0.6 / 0.4 |
  | v2 | 0.4 / 0.2 | 0.6 / 0.3 | 0.5 / 0.4 |
  | ECT | 0.6 / 0.6 | 0.9 / 0.7 | 0.9 / 0.7 |
  | Distill thường (không phản thực) | 0.9 / 1.0 | 0.9 / 1.0 | 0.9 / 1.0 |

  Task 0 ("mở ngăn kéo giữa") không có cặp phản thực nào mà vẫn tụt. Bản chỉ retarget (không bao giờ dời tủ) cũng tụt.
  → Lỗi chia pha là có thật, nhưng **mức tụt chính đến từ chỗ khác**.

### 5.2 Long task 0: lưỡng lự khi gắp — đã sửa một nghi phạm

- Ghi từng bước bản cũ (20 lần, đạt 65%): các lần hỏng là robot **lơ lửng trên lon, đóng / mở tay kẹp xen kẽ**, hoặc gắp trượt
  và làm đổ lon. Báo cáo hôm qua ghi "rơi vật giữa đường" ở một lần đo khác; hai kiểu hỏng có thể cùng tồn tại.
- **Nghi phạm.** Retarget đổi chỗ lon súp (vật đích) với một vật khác, có khi chính là lon sốt — cũng là vật hợp lệ để gắp
  trong "bỏ **cả hai** vào giỏ". Cặp như vậy dạy "lon dưới tay thì đừng gắp".
- **Đã sửa** (`--no_cotarget_swap`). v3 Long vòng 8: 0.77 (các biến thể v2: 0.64–0.78). Còn sớm để kết luận.

### 5.3 Phát hiện lớn: LIBERO đặt lại tủ / bếp ở mỗi lần reset

- **Hiện tượng.** Cùng một adapter v2, Goal task 0 / 3 cho kết quả rất khác tuỳ cách xếp lịch eval:

  | Cách chạy | Task 0 | Task 3 |
  |---|---|---|
  | Eval cả bộ (mỗi worker chạy liên tiếp 20 lần thử của một task, giống giao thức chính thức) | 70 | 70 (chạy lại: 45) |
  | Mỗi task một worker, tuần tự | 65 | 80 |
  | Các lần thử chia cho nhiều env mới dựng | 95, 95 | 85, 65 |

  Ở các lần chạy tuần tự, model hỏng **đúng cùng các trial** [1, 2, 3, 4, 8, 14].
- **Nguyên nhân** (`scripts/check_env_reuse.py`, `scripts/check_fixture_draws.py`):
  - Env vừa chạy một tập và env mới dựng có trạng thái mô phỏng, bộ điều khiển, tay kẹp giống hệt, vật lý trùng khớp suốt
    120 bước, **nhưng ảnh khác**.
  - Lý do: **mỗi lần reset, LIBERO đặt lại ngẫu nhiên đồ cố định** (tủ, bếp, giá rượu). Vị trí của chúng nằm trong tham số
    model, không nằm trong initial state của benchmark.
  - Qua 20 lần reset, tủ dao động khoảng 2 cm theo x và 1.4 cm theo y. Env mới luôn đặt tủ ở cùng một chỗ, nằm ở **rìa** phân bố.
- **Ý nghĩa.**
  - Các số Goal / Long trong báo cáo đo theo đúng giao thức chuẩn (env dùng lại), nên **có thật**, và so sánh giữa các phương
    pháp vẫn công bằng.
  - Câu hỏi còn lại: các model phản thực có **nhạy với vị trí đồ cố định** hơn base không?
- **Đo có kiểm soát (đang chạy).** Ghim tủ ở vị trí của env mới, cộng thêm độ lệch, rồi đo task 0 / 3 với 20 lần thử mỗi task:

  | Độ lệch tủ | Base | Distill thường | ECT | v2 (ta) |
  |---|---|---|---|---|
  | 0 | 95 / 90 | 95 / 95 | 100 / 90 | 100 / 80 |
  | +2 cm x | 100 / 95 | 95 / 85 | 95 / 95 | 85 / 90 |
  | +2 cm y | 90 / **55** | 95 / **55** | 85 / **60** | 85 / **55** |
  | −2 cm y | 95 / 95 | 100 / 90 | 90 / 95 | 85 / 85 |
  | −1, −2 cm x; −1.4 cm y; góc (−2, −1.4) | đang chạy | | | |

  **Đọc sơ bộ:**
  - Lệch +2 cm theo y làm **mọi model** (cả base) hỏng task 3. Đây là ngoài phân bố, không phải lỗi riêng của ta.
  - v2 thấp hơn base một chút ở task 0 (85 so với 95–100), nhưng **chưa tái hiện được mức 65–70** của eval tuần tự.
  - Các độ lệch âm theo x — đúng vùng mà các lần reset thật rơi vào — sẽ có trong khoảng 30–40 phút.
  - Nếu v2 sụt mạnh ở đó trong khi base thì không, nguyên nhân là **nhạy với vị trí đồ cố định**. Khi đó cách sửa tự nhiên là
    cho đồ cố định xê dịch khi train (hoặc thêm cặp phản thực cho chúng). Nếu không sụt, mình sẽ tìm tiếp.

### 5.4 Ô đổi màu Object: tradeoff có cơ chế

- Base và ECT bền với đổi màu **vì chúng không nhìn vật**. Model của ta đã học nhìn, nhưng nhận vật một phần **theo màu**.
- Ảnh hưởng của đổi màu ngẫu nhiên khi train:

  | | Swap | Ô đổi màu |
  |---|---|---|
  | Chỉ VLM, không đổi màu | 49.3 | khoảng 69 |
  | Chỉ VLM, có đổi màu (v2) | 39.5 | 85 |

- Bản điểm 3D oracle + phản thực giữ được ô đổi màu (92.5), vì oracle chỉ đúng vật bất kể màu. Muốn claim điều này thì phải
  thay oracle bằng một nguồn không đặc quyền.

---

## 6. Đang chạy (card 0–3) và lúc có kết quả

| Run | Nội dung | Tiến độ | Có kết quả (UTC) |
|---|---|---|---|
| `p05_uninogate_full_s9`, `p05sp_…_s9` | **Ta, seed 9** (Object, Spatial) | vòng 19 / 18; đánh giá giữa chừng 0.96 / 0.99 | khoảng 10:30 |
| `p05_ect_s9`, `p05sp_ect_s9` | **ECT, seed 9** | vòng 15 / 14 | khoảng 11:30 |
| `p05gl_v3_s7`, `p05lg_v3_s7` | v3 (v2 + hai bản sửa nhãn) trên Goal, Long | vòng 12 / 11 | 12:00 / 13:30 |
| `p05_v2_s7`, `p05sp_v2_s7` | v2 train toàn bộ trên Object, Spatial (cấu hình chung cho 4 bộ) | vòng 16 / 14 | khoảng 11:00 |
| `p05_v2vlmnotint_s7` | v2, chỉ VLM, không đổi màu (Object) | đang đánh giá; chuẩn 97.0 | khoảng 10:00 |
| `fix_gl_*` | quét độ lệch tủ (base, distill thường, ECT, v2) | lượt 2 | khoảng 09:15 |

---

## 7. Hướng tiếp và các quyết định

**Trong hôm nay:**
1. Seed 9 xong: bảng kết quả chính đủ **3 seed mỗi bên**.
2. Quét độ lệch tủ xong: quyết định Goal / Long là **nhạy vị trí đồ cố định** (sửa được khi train) hay là giới hạn.
3. Chọn cấu hình cuối cho cả 4 bộ:
   - bỏ cổng, train toàn bộ (mạnh nhất ở Object và Spatial);
   - v2 / v3 (thêm các bộ lọc nhãn);
   - chỉ VLM (Spatial tốt nhất, nhưng Object kém hơn khi có đổi màu).

**Phần phương pháp còn mở** — đây là chỗ cần có novelty theo góp ý của anh:
- **Can thiệp nhắm đúng chỗ đã định vị:** ví dụ chỉ train VLM tầng 9–17 với mục tiêu ở khoảnh khắc quyết định, hoặc một đường
  nối từ phần mã hoá vị trí (đã có sẵn trong model) sang action expert, chỉ mở ở khoảnh khắc mù.
- **Chỗ mù chung chưa ai sửa:** đầu pha tiếp cận (E1 0.2–0.3 ở cả ba model).
- **Nguồn hình học không đặc quyền** (thay oracle), để giữ được ô đổi màu như bản điểm 3D oracle.

**Câu hỏi cần anh quyết định:**
1. Paper đứng trên insight "VLA biết nhưng không dùng; giám sát phản thực dạy dùng, không dạy nhìn; thay đổi nằm ở VLM tầng
   9–17" cộng kết quả hơn ECT cùng setup — đã đủ chưa, hay cần thêm một can thiệp kiến trúc mới?
2. Goal / Long: báo là giới hạn, hay dành thêm thời gian sửa (tuỳ kết quả quét độ lệch tủ)?
3. Benchmark thứ hai (RoboCasa / CALVIN): làm ngay hay sau khi chốt phương pháp?

---

## 8. Sai lầm của chính mình / nhận định đã rút lại

| Nhận định | Thực tế |
|---|---|
| Lối tắt đi qua trạng thái tay | Sai |
| Nghẽn ở lớp đọc ra cuối | Sai |
| Cổng thực thi có ích | Có hại ở Spatial; đã bỏ |
| "Quá nhiều cặp" gây tụt Long; "đổi màu" gây tụt Long | Sai (5% cặp, không đổi màu đều vẫn khoảng 84–87) |
| `--hindsight` sửa được Goal task 3 | Chỉ sửa một lỗi chia pha phụ; task 3 vẫn tụt |
| Số theo từng task với 20 lần thử đủ tin | Không: cùng adapter có thể lệch 25–30 điểm tuỳ vị trí tủ / lịch chạy |
