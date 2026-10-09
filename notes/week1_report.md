# Báo cáo tổng hợp — thế giới phản thực và teacher có đặc quyền cho VLA (π0.5, LIBERO)

Cập nhật 09/10/2026, 08:20 giờ VN (01:20 UTC). Mọi số là của chúng tôi, đo trong cùng một harness, trừ khi ghi "công bố".
Số từng run: [../results/pi05_runs.txt](../results/pi05_runs.txt). Nhật ký chi tiết theo ngày: [week1_status.md](week1_status.md).
Đối thủ: [related_work.md](related_work.md) §9–11.

---

## 0. Tóm tắt một trang

**Mục tiêu.** Một bài báo có một intuition tốt và kết quả vượt SOTA trên các bài kiểm tra tổng quát hoá vị trí của VLA
(LIBERO-PRO, LIBERO-Plus), xuất phát từ π0.5.

**Đã làm được.**
- Chẩn đoán: π0.5 fine-tune trên LIBERO không "nhìn" vật ở đâu mà **đi lại quỹ đạo đã thuộc** ứng với câu lệnh. Đổi chỗ
  vật thì nó vẫn đi tới chỗ cũ.
- Phương pháp: distill π0.5 vào chính nó, nhưng ở một phần nhỏ state (≤ 15%) **dựng thêm thế giới phản thực từ cùng state
  mô phỏng**: dời riêng vật đích (trước khi kẹp) hoặc riêng nơi đặt (khi đang mang). Nhãn do một **teacher kịch bản có đặc
  quyền** (biết vị trí thật) sinh ra, vì chính π0.5 làm sai ở đó.
- Kết quả trên Object và Spatial:

  | Ô (π0.5, 200 tập mỗi ô) | Base | Phương pháp | Seed |
  |---|---|---|---|
  | Object swap | 18.7 | **53.0–56.3** | 3 |
  | Object position | 12.2 | **39–42** | 1–2 |
  | Spatial swap | 40.0 | **64.7** | 3 |

  Cảnh chuẩn vẫn ở 95–99.
- Thay các lựa chọn chọn tay theo bộ / theo task bằng **ba kiểm tra tự động** cho teacher. Một cấu hình duy nhất đang chạy
  trên cả 4 bộ.

**Chưa làm được.**
- **Goal và Long không tăng.** Goal swap khoảng 28–32 so với base 29; Long swap 11 so với base 8.5, mà cảnh chuẩn Long còn
  tụt 4 điểm.
- **Tradeoff ở ô "object" của LIBERO-PRO** (đổi màu vật đích): base 92.5, phương pháp 74–81.

**Vấn đề lớn nhất: ECT** (arXiv 2609.39971, công bố 30/09/2026) **trùng ý tưởng cốt lõi**.
- Họ có cùng chẩn đoán: câu lệnh "truy xuất quỹ đạo".
- Họ cũng làm thế giới đổi quan hệ tay–đích kèm nhãn có đặc quyền: gương / dịch cả cảnh, giữ robot, phát lại đường đi.
- Base của ta khớp dòng π0.5 chính thức của họ, nên so được trực tiếp:
  - Object: ta thắng (53–56 so với 40.8).
  - Spatial: ta thua (64.7 so với 72.8).
  - Goal: ta thua (khoảng 30 so với 35.8).
  - Long: ta thua (11 so với 26.6).
- Đã cài lại ECT trong harness của ta với cùng ngân sách để so công bằng. Đang chạy, kết quả có trong vài giờ.

**Cần quyết định** (chi tiết ở §9): định vị lại đóng góp (A), kết hợp với ECT để vượt SOTA (B), thay teacher viết tay bằng
một cơ chế tổng quát (D), hoặc mở rộng sang model khác (E).

---

## 1. Thuật ngữ

| Thuật ngữ | Nghĩa |
|---|---|
| **Cảnh chuẩn** | bộ LIBERO gốc (Spatial / Object / Goal / Long), mỗi bộ 10 task |
| **swap** (LIBERO-PRO) | vị trí các vật bị hoán đổi; câu lệnh giữ nguyên |
| **position / temp** (LIBERO-PRO) | vật bị dời sang vị trí mới (chỉ có ở Object) |
| **object** (LIBERO-PRO) | vật đích bị đổi màu / ngoại hình |
| **lan** | câu lệnh diễn đạt lại |
| **task** | đổi hẳn task; mọi phương pháp ≈ 0 trên Object / Spatial, ≈ 10 trên Goal / Long |
| **LIBERO-Plus Robot / Layout** | robot bắt đầu ở tư thế lạ / bố cục lạ; chỉ đo 60 tập nên dao động ±8 |
| **base** | distill π0.5 vào chính nó với cùng ngân sách, không phản thực (đối chứng chính) |
| **thế giới phản thực** | dựng lại cảnh từ đúng state mô phỏng đang có, rồi dời một vật (chỉ để render và gán nhãn) |
| **teacher có đặc quyền** | bộ điều khiển kịch bản đọc vị trí thật của vật từ simulator để sinh chunk action |
| **retarget** | trước khi kẹp: dời riêng vật đích (tay giữ nguyên); nhãn = đi tới 10 cm trên vật, gripper mở |
| **relocate** | khi đang mang vật: dời riêng nơi đặt (giỏ, đĩa, vùng trên bếp / nóc tủ); nhãn = mang tới, hạ, thả |
| **coshift** | dời vật đích **cùng tay** (IK); nhãn = chunk gốc của π0.5 (chính xác, vì quan hệ tay–đích không đổi) |
| **rr / full** | rr = retarget trước kẹp + relocate khi mang; full = retarget hoặc coshift trước kẹp + relocate khi mang |
| **cổng** | kiểm tra trong mô phỏng trước khi train: chạy teacher trong thế giới đã dời, đo tỉ lệ thành công |

---

## 2. Thiết lập

- **Model:** π0.5 bản LeRobot (`lerobot/pi05_libero_finetuned`, đã fine-tune trên cả 4 bộ LIBERO). Học sinh là π0.5 + LoRA
  rank 32; teacher trên cảnh thật là π0.5 gốc (tắt LoRA).
- **Huấn luyện** (giống nhau cho mọi run):
  - 20 vòng × 1024 state; mỗi vòng rollout 20 tập rồi distill bằng flow matching;
  - cặp phản thực trên tối đa 15% state;
  - train riêng từng bộ, mỗi run một card H200, khoảng 4–5 giờ.
- **Đánh giá:**
  - adapter cuối, 200 tập mỗi ô (20 init state × 10 task);
  - horizon theo openpi: Spatial 220, Object 280, Goal 300, Long 520 bước;
  - LIBERO-Plus 60 tập mỗi loại.
- **Độ tin cậy:** sai số chuẩn của một ô 200 tập là khoảng ±3 điểm. Chênh lệch giữa các seed ở ô swap là 5–12 điểm, nên
  mọi kết luận chính đều cần ≥ 2–3 seed.

---

## 3. Chẩn đoán: π0.5 đi theo trí nhớ, không theo cảnh

1. **Khoảng trống lớn.** Cảnh chuẩn 94–99%, nhưng LIBERO-PRO swap chỉ 9–42%:

   | | Spatial | Object | Goal | Long |
   |---|---|---|---|---|
   | π0.5 cảnh chuẩn | 98 | 99 | 97.5 | 95 |
   | π0.5 swap | 40 | 18.7 | 29 | 8.5 |
   | π0.5 position | — | 12.2 | — | — |

   Cho thêm thời gian (520 bước) chỉ tăng Object swap lên 23.5, nên đây không phải lỗi hết giờ.
2. **Lỗi tách theo pha, pha nào cũng đi theo vị trí đã thuộc:**
   - *Pha tiếp cận* (Object swap): ở các task hỏng, 20/20 tập tay đi tới vật đang đứng ở **ô quen của vật đích**.
   - *Pha đặt* (Spatial swap, Object position, Goal): nhặt đúng vật rồi đặt xuống **cách chỗ nơi đặt quen ~3 cm**, tức cách
     nơi đặt thật ~25 cm.
3. **Nhớ cả quỹ đạo, không chỉ vị trí.** Dựng thế giới đưa vật về chỗ quen nhưng tay ở chỗ lạ ("teacher chuẩn hoá") thì
   π0.5 chỉ được 22.0 trên Object swap. Vậy π0.5 không thể làm teacher ở bất kỳ thế giới lệch nào; teacher phải là thứ khác.
4. **Biến đổi có nhãn chính xác không đủ.** Xoay cả cảnh lẫn tay, soi gương cả tay, coshift đều **giữ nguyên quan hệ
   tay–đích**, nên lối tắt "đi đoạn đã thuộc" vẫn khớp nhãn. Thực nghiệm (Object swap, 2–3 seed):

   | | Swap |
   |---|---|
   | Base | 18.7 |
   | Soi gương | 20.8 |
   | Coshift | 24.3 |
   | Augmentation ảnh 2D | 24.2 |
   | Xoay | 29.2 |

   Spatial: xoay 39.5, coshift 37.0 (base 40). Muốn học "nhìn đích" thì thế giới phải **đổi quan hệ tay–đích** và nhãn phải
   do một teacher biết đích ở đâu.

> **Lưu ý:** ECT có cùng chẩn đoán (tên bài "When Instructions Retrieve Trajectories"), nên phần này không còn là đóng góp
> riêng. Phần còn của ta: tách lỗi theo pha, thí nghiệm teacher chuẩn hoá, và chứng minh biến đổi nhãn chính xác không giúp.

---

## 4. Phương pháp

Ở mỗi vòng distill, với tối đa 15% state của rollout:

| Phản thực | Khi nào | Dời gì | Nhãn | Mask |
|---|---|---|---|---|
| **retarget** | trước khi kẹp | riêng vật đích: sang chỗ trống, hoặc chiếm chỗ vật khác (vật kia sang chỗ của nó) | đi tới 10 cm trên vật đích, gripper mở | sau khi tới |
| **relocate** | khi đang mang | riêng nơi đặt, lấy từ mục tiêu BDDL (`on` / `in`): vật di động, vùng trên vật di động, hoặc cả món đồ nội thất | mang tới trên nơi đặt, hạ (tính cả độ cao thành giỏ), thả | sau khi thả + 5 bước |
| coshift (phụ) | trước khi kẹp | vật đích + tay (IK) | chunk gốc của π0.5 | sau lệnh kẹp |

- Hai teacher kịch bản dùng mô hình động học khớp từ log: bộ điều khiển OSC dịch chuyển ≈ g · a · 5 cm mỗi bước.
- Thế giới phản thực chỉ dùng để render và gán nhãn: dời vật, render, rồi khôi phục chính xác.

**Ba kiểm tra tự động cho teacher** (thêm ngày 08–09/10, thay cho các lựa chọn chọn tay):
1. **Không dời vật có bản sao trong cảnh.** Nếu có vật khác cùng loại, câu lệnh chỉ có thể chỉ ra vật bằng vị trí ("bát cạnh
   ramekin", "đĩa bên trái"); dời nó là trái câu lệnh. Quy tắc này tự động:
   - tắt retarget trên Spatial (hai bát giống hệt);
   - khoá Long task 4 (đĩa trái / phải) và task 8 (hai ấm moka).
2. **Teacher phải đồng ý với chuyên gia ở thế giới thật.** Hỏi cùng teacher đó ở thế giới không dời; chỉ giữ cặp nếu hướng đi
   (tổng xyz 25 bước) của teacher và của π0.5 có cos ≥ 0.5. Kiểm tra này loại được các trường hợp teacher sai:
   - Long 2 (phải bật bếp trước) và Long 6 (nhận nhầm đĩa làm vật đích): 0% cặp được giữ;
   - Goal 5 (đẩy đĩa): giữ 33%.

   Ở chỗ teacher đúng, kiểm tra giữ 76–100% (Object, Spatial). Không lọc được Goal 3 (mở ngăn kéo trước), vì ngăn kéo và cái
   bát nằm cùng hướng.
3. **Cổng thực thi với ngưỡng chốt trước.** Placer chạy vòng kín trong thế giới đã dời; task qua nếu thành công ≥ 60%
   (n ≥ 5), đo trên 20 tập / task của π0.5 gốc.

   | Bộ | Qua cổng | Bị loại |
   |---|---|---|
   | Object | 8/10 task | sữa, salad dressing (vật cao) |
   | Spatial | 8/10 | — |
   | Goal | 4/10 (3, 4, 6, 8) | — |
   | Long | 3/10 (giỏ) | — |

---

## 5. Kết quả

### 5.1 Object (3 seed nếu không ghi khác)

| Cấu hình | Chuẩn | **Swap** | **Position** | Object (đổi màu) | Lan |
|---|---|---|---|---|---|
| base | 99.3 | 18.7 | 12.2 | 92.5 | 99.2 |
| base + augmentation 2D (openpi) | 98.8 | 24.2 | 13.0 | 91.3 | 97.8 |
| base + đổi màu vật (1 seed) | 99.0 | 18.0 | 11.0 | 92.5 | 99.5 |
| coshift | 98.7 | 24.3 | 20.7 | 92.5 | 99.7 |
| xoay | 99.2 | 29.2 | 16.2 | 92.7 | 99.2 |
| **retarget** | 99.0 | **53.0** | 20.5 | 74.5 | 99.5 |
| **rr** (retarget + relocate) | 96.8 | **56.3** (63.5 / 54.5 / 51.0) | 23.7 | 81.3 | 96.5 |
| relocate (2 seed) | 98.8 | 22.5 | 32.8 | 90.5 | 97.5 |
| **coreloc** (coshift + relocate, 2 seed) | 96.8 | 25.8 | **42.3** | 90.0 | 98.3 |
| **full** (1 seed) | 96.0 | **53.0** | **39.0** | 80.0 | 96.5 |

- Mỗi phản thực sửa đúng ô của pha nó nhắm: retarget → swap (lỗi tiếp cận); relocate → position (lỗi đặt).
- "full" lấy được cả hai ô trong một cấu hình.
- **Cận trên** (đi tiếp cận bằng kịch bản rồi để π0.5 tự làm tiếp): Object swap 59.0. Retarget đạt 90% mức đó; rr vượt nhờ
  thêm pha đặt.

### 5.2 Spatial

| Cấu hình | Chuẩn | **Swap** | Object | Seed |
|---|---|---|---|---|
| base | 97.5 | 40.0 | 98.5 | 2 |
| xoay | 99.5 | 39.5 | 97.0 | 1 |
| coshift (trước / sau kẹp) | 97.5 / 98.0 | 37.0 / 40.0 | 98 / 97 | 1 |
| **relocate** | 98.3 | **64.7** (64.5 / 63.0 / 66.5) | 97.8 | 3 |
| ablation: thế giới phản thực + nhãn π0.5 | 98.0 | 40.0 | 97.0 | 1 |
| ablation: nhãn kịch bản ở thế giới gốc | 98.5 | 35.5 | 97.5 | 1 |

→ **Cần cả hai thành phần:** thế giới phản thực và teacher có đặc quyền. Bỏ một trong hai là mất toàn bộ gain.

### 5.3 Goal và Long — chưa tăng

| Goal (1 seed) | Chuẩn | **Swap** | Object | Lan |
|---|---|---|---|---|
| base | 97.5 | 29.0 | 87.5 | 96.5 |
| augmentation 2D | 97.0 | 29.0 | 91.0 | 97.0 |
| xoay | 92.0 | 27.0 | 81.0 | 92.0 |
| retarget | 94.5 | 32.5 | 79.5 | 91.0 |
| relocate theo mục tiêu BDDL (7 task) | 97.0 | 28.5 | **74.0** | 97.0 |

| Long (1 seed) | Chuẩn | **Swap** | Object | Lan |
|---|---|---|---|---|
| base | 95.0 | 8.5 | 70.0 | 96.0 |
| relocate (3 task giỏ) | **91.0** | 11.0 | 65.0 | 89.5 |

- Goal: lỗi chính là tụt vật khi mang lên nóc tủ / vào ngăn kéo (tương tác với đồ nội thất). Cận trên tiếp cận kịch bản chỉ
  23.0, thấp hơn base. Relocate (kể cả khi đã gồm nóc tủ / ngăn kéo) không tăng swap mà còn làm tụt ô object 13 điểm.
- Long: task nhiều bước. Relocate chỉ dùng được ở 3/10 task, không tăng swap, và làm tụt chuẩn / lan.

### 5.4 Các ablation khác

| Câu hỏi | Kết quả | Kết luận |
|---|---|---|
| State on-policy có cần không? | dùng state của π0.5 gốc thay cho state của học sinh: retarget 50.5 vs 51.0; coshift 22.5 vs 24.0 | **không**; không claim on-policy |
| Số hạng nhất quán theo cặp? | không đóng góp ở vòng cuối | bỏ (khớp với ECT: loss ghép cặp ≈ 0) |
| Gain có đến từ "nhặt vật lệch chỗ"? | cho một vật phụ bắt đầu lệch 20 cm: retarget nhặt nhầm nó 0/100 lần | không, model không học lối tắt đó |
| LIBERO-Plus Robot / Layout | mọi cấu hình 70–93, dao động ±8 giữa hai lần đo cùng model | không dùng để kết luận |

### 5.5 Tradeoff vị trí ↔ ngoại hình (ô "object")

- Base làm tốt ô đổi màu (92.5) *chính vì* nó tìm vật theo vị trí. Retarget dạy tìm theo ngoại hình nên tụt: 74.5 (rr 81.3).
- Đổi màu ngẫu nhiên vật khi train cho kết quả không nhất quán (1 seed mỗi cấu hình):

  | Cấu hình | Ô object |
  |---|---|
  | retarget + màu p 0.5 | 85.5 |
  | retarget + màu p 0.9 | 81.5 |
  | rr + màu | 74.0 |

  Với riêng retarget có dấu hiệu giúp; với rr thì không. **Chưa claim được.**

---

## 6. So với SOTA

### 6.1 ECT (Chen et al., 2609.39971, 30/09/2026) — đối thủ trực tiếp

**Họ làm gì.**
- Lấy demo, biến đổi cả cảnh (gương x / y / xy, hoặc dịch), **giữ nguyên robot và tư thế đầu**.
- Một bộ điều khiển bám đường đi của tay đã biến đổi để sinh action; chỉ giữ lần phát lại thành công.
- Huấn luyện trên demo gốc + bản biến đổi, ghép cặp trong batch.
- Train 4 bộ cùng lúc, 30k bước, batch 64.
- Phần dữ liệu cho gần hết gain; loss ghép cặp ≈ 0.

**Số công bố (LIBERO-PRO swap, 500 tập mỗi ô) so với ta:**

| | Spatial | Object | Goal | Long |
|---|---|---|---|---|
| π0.5 chính thức (họ đo) | 46.6 | 18.2 | 34.2 | 9.4 |
| **π0.5 base của ta** | 40.0 | 18.7 | 29.0 | 8.5 |
| ECT từ checkpoint chính thức (Full FT) | **72.8** | 40.8 | **35.8** | **26.6** |
| ECT Frozen-LM (base riêng của họ: Object 38.3) | 74.4 | 70.8 | 55.5 | 34.7 |
| **Ta** (LoRA, từng bộ) | 64.7 | **53.0–56.3** | ~29–32 | 11.0 |

- Base của ta khớp π0.5 chính thức của họ, nên dòng so sánh đúng là "ECT Full FT".
- Ta **thắng ở Object** (+12 đến +16) và **thua ở ba bộ còn lại**.
- Chế độ huấn luyện khác nhau: họ train 4 bộ cùng lúc, full fine-tune, 30k bước; ta dùng LoRA, từng bộ, 20 × 1024 state.
- Điểm có lợi cho ta: trong **ablation train từng bộ bằng LoRA** của chính họ (Bảng 19), mọi biến đổi ECT cho Object swap chỉ
  0–12. Họ viết: biến đổi "does not by itself raise Object Swap". Ta được +34 trong chế độ đó.

### 6.2 ECT cài lại trong harness của ta (đang chạy)

- **Cài đặt** (`--cf_mode ect`):
  - mỗi vòng chọn 16 tập thành công;
  - chạy lại để lấy đường đi của tay;
  - biến đổi cảnh theo Bảng 20 của họ: Spatial / Goal gương y; Object gương y, gương x + dịch, gương xy + dịch; Long dịch;
  - bám đường đi đã biến đổi; ghép query t của tập gốc với query t của bản phát lại.
- **Ngân sách** giống hệt run của ta: cap 15%, dùng chung noise và thời điểm flow như loss ghép cặp của họ.
- **Cổng phát lại:**

  | | Object | Spatial | Goal | Long |
  |---|---|---|---|---|
  | Đối chứng không biến đổi | 1.00 | 0.97 | 0.97 | 0.90 |
  | Biến đổi | 1.00 | 0.87 | 0.63 | 0.70 |

- Hai lỗi đã tự phát hiện và sửa trước khi chạy: bộ bám chỉ dùng phản hồi vị trí làm hỏng thao tác ngăn kéo; tủ phản chiếu
  quay mặt ra ngoài.
- **Tiến độ:** cả 4 bộ đang ở vòng 12–15/20, cảnh chuẩn giữa chừng 0.83–0.99. Kết quả cuối dự kiến 03:00–05:00 UTC
  (10:00–12:00 giờ VN).

### 6.3 Đối thủ khác (chưa đọc kỹ / chưa so được)

Anchor-Align (22.6% "position swap", không rõ protocol), CounterAlign, CofactVLA, LIBERO-CF / CAG (phản thực ngôn ngữ,
chữa lúc inference), HABILIS Brain 0. Chưa bài nào ngoài ECT cho số cùng protocol với π0.5.

---

## 7. Đóng góp còn lại sau khi có ECT — đánh giá thẳng

| Nội dung | Còn mới? |
|---|---|
| Chẩn đoán "câu lệnh truy xuất quỹ đạo" | **không** (ECT) |
| Đổi quan hệ tay–đích + nhãn có đặc quyền | **không** về ý tưởng (ECT dùng phát lại; ta dùng teacher kịch bản) |
| Biến đổi nhãn chính xác (giữ quan hệ tay–đích) không giúp | một phần (ECT có ablation gương; ta có xoay / coshift / soi gương cả tay) |
| **Can thiệp theo pha tại state bất kỳ** (dời nơi đặt *khi đang mang*), một vật mỗi lần, không cần demo | **có** — ECT biến đổi cả cảnh từ trạng thái đầu của demo |
| **Gain lớn khi train từng bộ bằng LoRA**, chế độ mà ECT không tăng Object swap | **có**, nhưng cần ECT trong harness để khẳng định |
| **Ba kiểm tra tính hợp lệ** của teacher (bản sao, đồng ý với chuyên gia, cổng thực thi) | **có**, nhưng là đóng góp phụ |
| Tradeoff vị trí ↔ ngoại hình | có, nhưng chưa có cách gỡ vững |

**Giới hạn của chính phương pháp:**
- Teacher viết tay cho nhặt–đặt: không tổng quát cho mở ngăn kéo, xoay núm, task nhiều bước.
- Không tăng Goal / Long.
- Tụt ô đổi màu.

---

## 8. Đang chạy (card 0–3) và khi nào có kết quả

| Run | Nội dung | Vòng | Kết quả cuối (UTC) |
|---|---|---|---|
| đợt 19, s7, 4 bộ | **một cấu hình** (full + 3 kiểm tra + màu p 0.5) | Goal xong train, đang eval; Object 17, Spatial 15, Long 17 | ~02:30–04:30 |
| đợt 20, s7, 4 bộ | **ECT trong harness** | 12–15 | ~03:00–05:00 |
| đợt 19, s8 | một cấu hình, Object + Spatial | 4–5 | ~06:00 |

Theo dõi giữa chừng của đợt 19 (đánh giá trên cảnh có đổi màu):
- Object 0.96, Spatial 0.98.
- Goal 0.84–0.89.
- Long 0.64–0.79: đáng lo; ECT Long cùng lúc được 0.83.

---

## 9. Các hướng đi tiếp — để cân nhắc

**A. Định vị lại đóng góp, giữ phương pháp hiện tại.**
- Câu chuyện: can thiệp phản thực **theo pha, tại state bất kỳ**, với teacher có đặc quyền được **kiểm tra tính hợp lệ tự
  động**. Mạnh khi lỗi nằm ở một pha cụ thể (Object tiếp cận, Spatial đặt).
- Điều kiện: ECT trong harness phải thua ta rõ ở Object / Spatial.
- Rủi ro: thua ECT ở Goal / Long, và reviewer sẽ thấy gần ECT.
- Chi phí: thấp (chỉ thêm seed + viết).

**B. Kết hợp với ECT để vượt SOTA trên cả 4 bộ.**
- Hai phương pháp bổ sung nhau: ECT biến đổi toàn cảnh từ trạng thái đầu (mạnh ở Spatial / Goal theo số công bố), ta can
  thiệp cục bộ theo pha (mạnh ở Object).
- Cả hai đã có trong cùng code, nên chạy gộp chỉ là một cấu hình mới.
- Rủi ro: thành "cộng hai phương pháp", ít intuition mới.
- Chi phí: thấp (1 đợt × 4 bộ ≈ 5 giờ).

**D. Thay teacher viết tay bằng một cơ chế tổng quát: "bẻ quỹ đạo của chuyên gia theo pha".**
- Nhãn ở thế giới đã dời = chính quỹ đạo π0.5 đã đi (lấy từ rollout thành công), **bẻ cục bộ** về phía vật / nơi đặt mới
  rồi phát lại bằng bộ bám, thay cho approach / placer kịch bản.
- Lợi ích:
  - trả lời điểm yếu "teacher viết tay";
  - dùng được cho mọi kỹ năng có đích (mở ngăn kéo, đặt lên bếp);
  - khác ECT ở chỗ cục bộ theo pha tại state bất kỳ, không phải biến đổi toàn cảnh từ đầu.
- Hạ tầng đã có một phần (bộ phát lại của ECT, cổng, ba kiểm tra).
- Rủi ro: 1–2 ngày phát triển, chưa chắc tăng Goal / Long.

**E. Chứng minh tổng quát qua model khác (OpenVLA-OFT) và benchmark khác.**
- Đã có hạ tầng OFT. Nhánh phụ trước đây cho LIBERO-Plus 83.1 so với 68.7.
- Rủi ro: tốn compute; không giải quyết vấn đề novelty với ECT.

**Đề xuất:**
- Chờ kết quả ECT trong harness và đợt 19 (sáng nay giờ VN), rồi:
  - nếu ta thắng ECT ở Object và Spatial cùng ngân sách: chạy **B** ngay (rẻ, có cơ hội vượt SOTA), song song bắt đầu **D**
    làm phương pháp chính cho bản nháp;
  - nếu ECT trong harness ngang hoặc hơn ta: chuyển hẳn sang **D**, hoặc đổi trọng tâm bài.
- Dù chọn hướng nào cũng cần thêm seed (đợt 19 s8 / s9; ECT s8) trước khi viết.

**Câu hỏi cần trả lời:**
1. Chấp nhận định vị "theo pha + kiểm tra hợp lệ" (A / B), hay muốn một phương pháp tổng quát hơn (D) dù tốn thêm thời gian?
2. Có cần thắng ECT trên **cả 4 bộ**, hay chấp nhận thắng ở các bộ có lỗi theo pha và báo trung thực phần còn lại?
3. Có mở rộng sang OFT (E) cho bản nộp đầu không?

---

## 10. Các nhận định đã rút lại

- "Coshift +7 / xoay −5 trên LIBERO-Plus Robot": là nhiễu 60 tập.
- "Xoay học lối tắt tư thế tay": không xác nhận được.
- "Coshift chỉ chỗ trống đơn giản hơn mà tốt bằng": sai một phần; nó đổi ô position lấy ô swap.
- "Đổi màu vật gỡ phần lớn tradeoff ô object": không lặp lại với rr (§5.5).

## 11. Nhánh phụ (OpenVLA-OFT, LIBERO-Plus)

Tự distill dưới nhiễu thị giác render lại từ cùng state (A2): **83.1** so với đối chứng 68.7 (+14.4, 2 seed); cảnh chuẩn
97–98.
