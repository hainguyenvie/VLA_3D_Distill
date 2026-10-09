# Báo cáo phần mới (09/10/2026): mổ xẻ cơ chế π0.5, tái hiện đối thủ, chẩn đoán các ô kém

Cập nhật 09/10/2026, khoảng 17:00 giờ VN (10:00 UTC). Mọi số là của chúng tôi, đo trong cùng harness, trừ khi ghi "công bố".
Báo cáo tổng hợp trước đó: [week1_report.md](week1_report.md). Nhật ký chi tiết: [week1_status.md](week1_status.md).
Khảo sát tài liệu: [related_work.md](related_work.md) §11–12.

---

## 0. Tóm tắt một trang

**Vì sao chuyển hướng.** Phương pháp cũ (dời vật / dời giỏ trong simulator, "thầy" kịch bản sinh nhãn) về bản chất là
augmentation có gán nhãn lại. Nó gần với ECT và MimicGen / DemoGen, nên khó claim là ý tưởng mới. Góp ý nhận được:
augmentation chỉ nên là **công cụ**; đóng góp phải là **insight về chỗ hỏng bên trong model** và **can thiệp đúng chỗ đó**.

**Đã tìm ra gì:**
1. **π0.5 không mù.** Thông tin "vật / giỏ đã bị dời" có mặt bên trong phần sinh động tác (action expert), với cùng một mức ở
   base, ECT và model của ta.
2. **Nhưng ở khoảnh khắc quyết định, thông tin đó không ra tới động tác.**
   - Lúc tay còn xa vật: dời vật thì động tác chỉ bám theo khoảng 20%.
   - Lúc đang mang vật tới giỏ: dời giỏ thì bám theo 6–9%.
3. **Huấn luyện phản thực của ta không dạy model nhìn tốt hơn, mà dạy nó dùng điều nó đã biết.** Mức mã hoá như nhau ở ba
   model; mức bám theo thì của ta tăng mạnh (đổi chỗ: 0.03 → 0.53; dời giỏ: 0.06 → 0.69). ECT không làm được điều này
   trong setup của ta.
4. **Đã bác ba giả thuyết:**
   - lối tắt không đi qua trạng thái tay;
   - chỗ nghẽn không nằm ở lớp đọc ra cuối;
   - cổng thực thi không có ích (có hại ở Spatial).

**So với đối thủ, cùng setup:** hơn ECT rõ ở Object và Spatial (Spatial swap 65.5 so với 52.0; Object 54.0 so với 24.0),
hơn nhẹ ở Long. Goal và ô đổi màu kém, nhưng **đã khoanh được nguyên nhân về một vài task cụ thể**, và đang sửa.

**Đang chờ:**
- 6 tầng cuối của action expert có đủ để sửa không (chỗ cần can thiệp nằm ở đâu);
- 2606.27663 cài lại (đối thủ mạnh nhất);
- cấu hình v2 trên Goal và Long.

---

## 1. Công cụ đo: dựng thế giới phản thực để "hỏi" model

Lợi thế riêng của ta: tại **bất kỳ khoảnh khắc nào** của một episode, có thể dừng simulator, sửa đúng **một** thứ, chụp lại
ảnh, rồi xem model phản ứng thế nào. Ba phép đo:

| | Đo gì | Làm thế nào | Script |
|---|---|---|---|
| **E1** | Động tác có đi theo thứ bị thay đổi không, ở từng thời điểm | Tại các state của episode thành công, thay đổi một đầu vào: dời vật đích (3 / 6 / 12 / 20 cm), đổi chỗ vật đích với vật khác, dời giỏ khi đang mang, lệch trạng thái tay 5 cm, đổi tên vật trong câu lệnh. Đo động tác dự đoán dịch đi bao nhiêu theo hướng thay đổi. **1** = đi theo hoàn toàn, **0** = bỏ qua | `scripts/probe_mechanism.py` |
| **E2** | Thông tin vị trí có nằm bên trong model không, ở tầng nào | Ghi trạng thái ẩn của từng tầng (token ảnh, token câu lệnh, token cuối prompt, token động tác), probe tuyến tính. Bản chặt: so **cùng một state** trước và sau khi dời, hỏi độ chênh trạng thái ẩn có mã hoá đúng độ dời không | `scripts/probe_internals.py`, `scripts/analyze_internals.py` |
| **E3** | Token động tác "nhìn" vào đâu | Tỉ lệ attention từ token động tác sang từng nhóm: ảnh ngoài, ảnh cổ tay, câu lệnh, trạng thái tay, chính nó | `scripts/probe_internals.py` |

**Cấu trúc π0.5 cần biết:**

```
Ảnh ngoài + ảnh cổ tay → SigLIP → 512 token ảnh ─┐
"Task: <câu lệnh>, State: <~120 token chữ số>; Action:" ─┤→ PaliGemma (VLM, 18 tầng)
                                                          │  (mỗi tầng, token động tác nhìn sang đây)
50 token động tác (nhiễu + thời gian) → Action expert (18 tầng) → lớp đọc ra → động tác
```

Trạng thái tay được viết thẳng vào câu lệnh, dài khoảng 120 token, gấp khoảng 10 lần phần câu lệnh (13 token).

---

## 2. Kết quả mổ xẻ

### 2.1 Lối tắt **không** đi qua trạng thái tay

| | Trạng thái tay thật | Trạng thái tay hằng số (không mang thông tin) |
|---|---|---|
| π0.5 gốc, Object chuẩn (chỉ test) | 99 | **100** |
| π0.5 gốc, Object swap (chỉ test) | 18.5 | **14.0** |
| Distill bỏ trạng thái tay, Spatial swap | (base 40.0) | **40.5** |

→ π0.5 gần như không dùng trạng thái tay. Bỏ nó khi test hay khi train đều không cải thiện swap. Giả thuyết "cắt đường trạng
thái tay" **bị bác**.

### 2.2 Bản đồ độ nhạy (E1), π0.5 gốc

**Object** (trung vị):

| Thời điểm | Dời 3 cm | Dời 20 cm | Đổi chỗ | Đổi tên trong câu lệnh | Lệch trạng thái tay |
|---|---|---|---|---|---|
| Đầu pha tiếp cận (tay cách vật > 15 cm) | **0.18** | **0.22** | 0.22 | 0.25 | 0.0 |
| Giữa pha (5–15 cm) | 0.61 | 0.31 | **0.03** | **0.02** | 0.0 |
| Sát vật (< 5 cm) | 0.51 | 0.29 | 0.01 | −0.02 | 0.0 |
| **Đang mang (dời giỏ)** | **0.09** | **0.06** | — | — | 0.0 |

- **Spatial:** đầu pha 0.21–0.30, đổi chỗ 0.08–0.14, đang mang (dời đĩa) 0.17–0.30.
- **Goal:** đầu pha 0.23–0.33 (đổi chỗ 0.42), đang mang 0.48 → 0.14 theo cỡ dời.

**Đọc:**
1. **Ở khoảnh khắc quyết định, model gần như mù** (khoảng 20%).
2. **Khi đã tiến vào, model là bộ bám cục bộ:** dời ít bám nhiều hơn dời xa ("vùng bám" hẹp); lựa chọn đã khoá, nên đổi
   chỗ hay đổi tên đều ≈ 0.
3. **Động tác đặt vật vào giỏ thuộc lòng hoàn toàn** (Object 6–9%).
4. Trạng thái tay không có tác dụng ở mọi pha.

### 2.3 Bên trong model (E2, E3)

- **Probe trong cùng state** (π0.5 gốc, Object, R² theo tầng):

  | | Token ảnh (VLM) | Token động tác, tầng đầu | Token động tác, tầng sau |
  |---|---|---|---|
  | Dời vật ≥ 12 cm, trước khi kẹp | 0.7–0.8 | ~0.05 | **~0.58** |
  | Dời giỏ ≥ 12 cm, đang mang | ~0.9 | ~0.2 | **0.74–0.82** |

  Thông tin "đã bị dời" có mặt trong token ảnh ngay từ đầu, và được đưa vào token động tác qua các tầng.
- **Attention của token động tác gần như giống hệt nhau** giữa lúc mù (đầu pha, đang mang) và lúc bám (sát vật):
  - câu lệnh 0.6–0.8 ở vài tầng;
  - ảnh cổ tay 0.2–0.3;
  - ảnh ngoài 0.1–0.3 ở các tầng đầu;
  - các tầng cuối chủ yếu nhìn lẫn nhau.

  → Không phải lỗi "không nhìn vào ảnh".
- **Lưu ý:** R² cao không chứng minh tín hiệu lớn, vì probe chuẩn hoá có thể khuếch đại một tín hiệu nhỏ nhưng nhất quán.

### 2.4 So ba model: mã hoá như nhau, khác ở việc dùng

**Bám theo (E1, Object):**

| | Base | ECT (train) | **Ta – full (train)** |
|---|---|---|---|
| Đầu pha, dời vật | 0.13–0.22 | 0.14–0.23 | 0.20–0.30 |
| Giữa pha, dời 20 cm | 0.31 | 0.37 | **0.67** |
| Giữa pha, đổi chỗ | 0.03 | 0.07 | **0.53** |
| Giữa pha, đổi tên trong câu lệnh | 0.02 | 0.01 | **0.54** |
| Đang mang, dời giỏ 12–20 cm | 0.06 | 0.11 | **0.44–0.69** |

**Mã hoá (E2, token động tác tầng sau, R² trong cùng state):**

| | Base | ECT | Ta |
|---|---|---|---|
| Dời vật xa | 0.58 | 0.62 | 0.65 |
| Đổi chỗ | 0.55 | 0.56 | 0.70 |
| Dời giỏ xa | 0.80 | 0.86 | 0.85 |

→ **Insight chính:** cả ba model đều "biết". Huấn luyện phản thực của ta **dạy model dùng điều đã biết**; ECT (trong setup
của ta) thì không. Vẫn còn một chỗ mù chung: **đầu pha tiếp cận** (0.2–0.3 với cả ba). Khoảnh khắc quyết định đầu tiên chưa
ai sửa được.

### 2.5 Chỗ cần sửa nằm ở đâu? (ablation phạm vi huấn luyện) — **phía VLM**

Cùng dữ liệu phản thực như run "full" s7, chỉ khác phần được phép thay đổi (Object):

| Phần được train | Số tham số | Swap | Position | Chuẩn | Lan | Đổi màu |
|---|---|---|---|---|---|---|
| Không train (base) | 0 | 18.7 | 12.2 | 99.3 | 99.2 | 92.5 |
| Chỉ lớp đọc ra cuối | 34 nghìn | 23.5 | 20.5 | 100 | 96.0 | 92.5 |
| 6 tầng cuối action expert + đọc ra | 4.65 triệu | 34.5 | 30.5 | **82.5** | 81.0 | 80.5 |
| Toàn bộ action expert | 13.9 triệu | 33.0 | 32.5 | 94.5 | 91.5 | 84.5 |
| **Chỉ VLM** (action expert đóng băng) | 39.2 triệu | **50.0** | **35.5** | **96.0** | 91.5 | (đang đo) |
| Toàn bộ (LoRA) | lớn hơn | 53.0 | 39.0 | 96.0 | 96.5 | 80.0 |

- **Riêng phía VLM đã đủ** để lấy gần hết gain: khoảng 94% gain swap và 86% gain position, trong khi action expert giữ nguyên.
- Riêng action expert sửa được việc đặt vật (position), nhưng chỉ khoảng một nửa việc chọn vật (swap).
- Ép thay đổi vào cuối action expert thì phá hành vi gốc.

→ **Bức tranh cơ chế.** Action expert đọc prefix của VLM qua attention. Ở base, thông tin "vật đích ở đâu" có trong những gì
nó đọc, nhưng bị lấn át bởi thành phần "câu lệnh → quỹ đạo thuộc lòng". Huấn luyện phản thực **sửa cách VLM trình bày** thông
tin đó, để action expert (không đổi) đọc ra được. Khớp với ECT (lựa chọn task đọc ra từ KV của prefix).

**Đang thu hẹp tiếp:**
- chỉ phép chiếu K/V của VLM (2.65 triệu tham số; đây chính là thứ action expert đọc);
- chỉ 9 tầng sau của VLM (19.6 triệu);
- nhân bản "chỉ VLM" với seed 8.

**Tradeoff đi cùng nhau:** run nào không học nhìn (base, chỉ lớp đọc ra, ECT, 2606.27663) thì ô đổi màu giữ khoảng 92–93; run
nào học nhìn thật thì ô đổi màu tụt (74–85).

---|---|---|---|---|
| Không train (base) | 0 | 18.7 | 12.2 | 92.5 |
| **Chỉ lớp đọc ra cuối** | 34 nghìn | **23.5** | 20.5 | 92.5 |
| 6 tầng cuối action expert + lớp đọc ra | 4.65 triệu | 34.5 | (đang đo) | (đang đo); **chuẩn tụt 82.5** |
| Toàn bộ (LoRA) | lớn hơn nhiều | 53.0 | 39.0 | 80.0 |

→ Giả thuyết "nghẽn ở lớp đọc ra cuối" **bị bác**. Ép thay đổi vào nửa sau của action expert thì lấy được một phần gain
(34.5) nhưng **phá hành vi gốc** (Object chuẩn 82.5); chỉ train toàn bộ mới vừa có gain vừa giữ được hành vi (53.0 / 96.0).
→ Thay đổi cần thiết nằm ở **phần sớm hơn** (tầng đầu action expert hoặc phía VLM), khớp với ECT (lựa chọn task đọc ra được
từ KV của prefix phía VLM). Đang chạy phép chia đôi: chỉ train VLM (39.2 triệu tham số) và chỉ train toàn bộ action expert
(13.9 triệu), cùng dữ liệu.

**Tradeoff đi cùng nhau:** run nào không học nhìn (base, chỉ lớp đọc ra, ECT) thì ô đổi màu giữ 92.5; run nào học nhìn thật
thì ô đổi màu tụt (74–85).

---

## 3. So với đối thủ

### 3.1 Setup đã tương xứng chưa?

| Mặt | Tương xứng? | Chi tiết |
|---|---|---|
| Đánh giá | **có** | π0.5 gốc của ta khớp π0.5 chính thức mà ECT đo: Object 18.7 so với 18.2, Long 8.5 so với 9.4 (Spatial 40 so với 46.6, Goal 29 so với 34.2) |
| Huấn luyện | **không** | ta: LoRA, từng bộ, 20 × 1024 state. ECT: 4 bộ cùng lúc, 30k bước × batch 64. 2606.27663: 69 task, full fine-tune, 30k bước × batch 256 |
| Thông tin lúc test | **không** | 2606.27663 dùng vị trí oracle từ simulator **ngay cả lúc test** |

→ So công bằng = **cài lại đối thủ vào setup của ta.** Đối thủ có code (Spatial Forcing, ROCKET, GAM) thì theo tài liệu
không chữa swap; ba đối thủ liên quan nhất (2606.27663, ECT, QuoVLA) **chưa công bố code**.

### 3.2 Bảng so sánh trong cùng setup (seed 7; trong ngoặc là seed 8)

| Bộ / ô | Base | ECT 15% | ECT 50% | **Ta – v1** | **Ta – bỏ cổng** | Ta so với ECT |
|---|---|---|---|---|---|---|
| Object swap | 18.7 | 24.0 | (đang đo) | 44.5 (54.0) | 54.0 | ✅ hơn rất rõ |
| Object position | 12.2 | 16.5 | | 38.5 (38.0) | 36.0 | ✅ hơn rất rõ |
| Object đổi màu | 92.5 | 92.5 | | 84.0 (85.5) | 86.0 | ❌ kém khoảng 7 (tradeoff) |
| Spatial swap | 40.0 | 52.0 | 48.0 | 55.0 (53.5) | **65.5** | ✅ hơn rõ |
| Goal swap | 29.0 | 24.5 | | 28.5 | — | ≈ base |
| Goal chuẩn | 97.5 | 96.5 | | **89.5** | — | ❌ (task 3, xem §4) |
| Long swap | 8.5 | 14.0 | | 17.0 | — | ✅ hơn nhẹ |
| Long chuẩn | 95.0 | 84.0 | | 85.5 | — | ≈ (cả hai tụt) |

- ECT 50% cặp **không tốt hơn** ECT 15% (Spatial 48.0 so với 52.0), nên so ở 15% không làm yếu baseline.
- Cảnh chuẩn Object và Spatial của ta: 95.5–99.

### 3.3 2606.27663 (đối thủ mạnh nhất) — kết quả: không tăng gì trong setup của ta

| Swap | Base | **2606.27663 cài lại** (điểm 3D oracle, cả lúc test) | ECT | Ta |
|---|---|---|---|---|
| Object | 18.7 | **18.5** | 24.0 | 54.0 |
| Spatial | 40.0 | **40.0** | 52.0 | 65.5 |

- Cảnh chuẩn vẫn 98.5 / 99.0.
- **Kiểm tra kênh hình học:**
  - đầu ra MLP có độ lớn khoảng 1.6 nhưng gần như **không đổi theo d**;
  - d đổi 10 cm thì đầu ra chỉ đổi 0.05–0.08.

  Model học một độ lệch cố định và **bỏ qua vị trí đích**.
- **Giải thích:** trên dữ liệu bố cục cố định, vị trí đích luôn đi cùng quỹ đạo đã thuộc, nên không có áp lực phải dùng kênh
  mới. Bài gốc train 69 task với bố cục đa dạng. Bản 20 task của chính họ (với GR00T) cũng cho Spatial chỉ 2 → 15.
- **Hệ quả:** *có thông tin là chưa đủ, phải có dữ liệu buộc model dùng nó* — đúng vai trò của dữ liệu phản thực. Thí
  nghiệm tự nhiên tiếp theo: điểm 3D **cộng** dữ liệu phản thực của ta.
- **E1 trên model này:** giống hệt base (đầu pha 0.12–0.21, đang mang 0.06–0.09), dù kênh nhận đúng vị trí mới khi vật / giỏ
  bị dời. Ô đổi màu 93.0, bằng base. → Model **có** vị trí đích ngay trong đầu vào mà **vẫn không dùng**; đây là bằng chứng
  mạnh nhất cho insight.
- **Điểm 3D cộng dữ liệu phản thực:** lần đầu sụp ở vòng 2, vì dữ liệu phản thực tạo gradient mạnh qua kênh d nên MLP lớn
  nhanh và phá action expert đang bị đóng băng. Đang chạy lại với learning rate của MLP × 0.1.

**Phần cài đặt:**

- **Cài lại:** d = vị trí đích con − vị trí tay (3D, oracle). Đích là vật cho tới khi vật được nhấc lên quá 1 cm, sau đó là
  nơi đặt theo mục tiêu BDDL. d đi qua MLP 3 → 1024 → 1024 (lớp cuối khởi tạo 0), rồi cộng vào điều biến thời gian AdaLN ở
  mọi tầng action expert. Thầy π0.5 không thấy d.
- **Sự cố của mình:** lần đầu cho MLP learning rate gấp 10 lần. Object sụp về 0% ở vòng 2–4 vì đầu ra MLP lớn nhanh làm lệch
  phần action expert đang bị đóng băng. Đã dừng và chạy lại với learning rate chung như bài gốc; giờ ổn định (vòng 5–6, cảnh
  chuẩn 97–100%).
- **Sẽ chẩn đoán:**
  - nó có hết mù ở đầu pha không (E1);
  - nó có yếu ở Spatial khi train từng bộ không (bài chỉ báo GR00T: 15.2);
  - Goal / Long (bài không báo);
  - ô đổi màu.

---

## 4. Vì sao ta kém ở vài ô dù cùng một tư tưởng (phân tích theo task)

### 4.1 Goal: cả ba ô tụt đều do 2 task dùng chung một cái tủ

| Task Goal | Ô | Base | ECT | Ta |
|---|---|---|---|---|
| 3. Mở ngăn kéo trên rồi bỏ bát vào | chuẩn / lan / đổi màu / swap | 100 / 90 / 45 / 100 | 85 / 70 / 40 / 70 | **45 / 25 / 0 / 5** |
| 0. Mở ngăn kéo giữa (cùng tủ) | chuẩn / lan / đổi màu | 90 / 100 / 95 | 85 / 85 / 75 | **60 / 55 / 45** |
| 8. Bát lên đĩa | swap | 20 | 15 | **55** |
| 9. Chai lên giá | swap | 0 | 0 | **55** |

8 task còn lại bằng hoặc hơn base.

- **Nguyên nhân:** ở task 3 phải mở ngăn kéo trước, nhưng "thầy" retarget bảo "đi tới cái bát" ngay từ đầu. Kiểm tra đồng ý
  không lọc được vì ngăn kéo và cái bát nằm cùng hướng. Hỏng lan sang task 0 (cùng tủ).
- **Sửa (`--hindsight`):** dùng chính quỹ đạo của episode để chỉ đặt cặp "tiến tới vật" trong đoạn nó thật sự tiến tới vật
  đó, tức từ lần nhả gripper cuối trước khi kẹp vật cho tới lúc bắt đầu kẹp. Chạy thử: task 3 giữ 125 / 149 cặp (bỏ 24 cặp ở
  giai đoạn mở ngăn kéo); task 5 (đẩy đĩa, không nhấc) bỏ hết 78 cặp.
- **Nếu sửa được:** Goal chuẩn về khoảng 97; Goal swap ước tính khoảng 38 (giữ được phần tăng ở task 8 và 9), hơn cả base
  (29) lẫn ECT (24.5).

### 4.2 Long chuẩn: tụt gần như chỉ ở task 0, do **rơi vật giữa đường**

- Task 0 (bỏ súp chữ cái và sốt cà chua vào giỏ): base 95, ta 35 (chạy lại 20 tập: 65); các task khác 70–100.
- Kiểu lỗi (`diag_lg_uni_t0_steps`): **6 / 7 lần thất bại là rơi giữa đường.** Kẹp đúng súp chữ cái (lần kẹp đầu luôn đúng
  vật), nhấc, mang khoảng 40 cm rồi rơi trước khi tới giỏ (có lần nghiêng 40–70°). Vì vậy vật thứ hai không bao giờ được đụng
  tới. Base 0 lần rơi.
- **Nghi vấn:** nhãn relocate dạy "thả khi đã ở trên giỏ (đã bị dời)" → model học nhả sớm ở cảnh thật.
- **Đang thử (`--relocate_no_grip`):** che chiều gripper trong cặp relocate, chỉ dạy hướng di chuyển.

### 4.3 Ô đổi màu Object: tradeoff có cơ chế

- Tụt chủ yếu ở task 4 (tương cà: 100 → 55 / 65) và task 0 (35 → 15).
- Base và ECT "bền" với đổi màu **vì chúng không nhìn vật**. Model của ta đã học nhìn nhưng nhận vật **theo màu**.
- Đổi màu ngẫu nhiên khi train đã giảm được một phần (74 → 85).
- **Đề xuất:** không cố vượt base ở ô này; trình bày như insight "benchmark thưởng cho việc không nhìn".

### 4.4 Có nên cố vượt ở **tất cả** các ô?

| Loại | Ô | Nên làm gì |
|---|---|---|
| Lỗi sửa được (bug phân pha / teacher) | Goal chuẩn / lan / đổi màu; Long chuẩn | **Phải sửa** (tiêu chí không sụp cảnh chuẩn) — đang chạy v2 |
| Tradeoff có cơ chế | Ô đổi màu | Không cần vượt base; giải thích và giảm thiểu |
| Ngoài phạm vi | Ô task (đổi hẳn task), LIBERO-Plus | Báo trung thực |

---

## 5. Các ablation khác

| Ablation | Kết quả | Kết luận |
|---|---|---|
| **Bỏ cổng thực thi** (relocate trên mọi task) | Spatial swap **65.5** (có cổng 55.0 / 53.5); Object 54.0 / position 36.0 / đổi màu 86.0 (≈ có cổng) | Cổng ngưỡng 60% loại nhầm task có lợi nhất (Spatial task 7: bát trên bếp, swap 0 → 85–100 khi được dạy). **Bỏ cổng.** Vì quyết định sau khi thấy kết quả test, phải trình bày như ablation |
| ECT 50% cặp | Spatial swap 48.0 (15%: 52.0) | Baseline không bị làm yếu bởi tỉ lệ 15% |
| Distill bỏ trạng thái tay | Spatial swap 40.5 (= base) | Đóng hướng trạng thái tay |
| Chỉ train lớp đọc ra | Object swap 23.5 | Chỗ sửa không nằm ở lớp cuối |

---

## 6. Khảo sát tài liệu (khoảng 40 bài) — phần liên quan tới hướng mới

**Đã chật:**
- **Chẩn đoán "mã hoá được nhưng không điều khiển"** trên π0.5: *Encoded but Not in Control* (2610.06235), *Not All Features
  Are Created Equal* (2603.19233), VLA-Trace, ECT. **Phần chẩn đoán của ta trùng họ**; phần riêng là so ba model để chỉ ra
  training thay đổi việc dùng, không thay đổi mã hoá.
- **Đưa 3D dày đặc** (depth / VGGT) vào VLA: hơn 20 bài, không chữa swap ở nơi nào có đo. Khớp với việc model vốn đã "thấy".
- **2606.27663:** điểm 3D oracle vào AdaLN của action expert ở mọi bước. Rất mạnh trên Object / Spatial (train 69 task).

**Còn mở:**
- Định vị chỗ hỏng bên trong π0.5 theo tầng × thời điểm rồi can thiệp đúng chỗ đó.
- Chỉ đưa thông tin hình học vào ở những khoảnh khắc mù.
- Dùng chính phần mã hoá của model thay cho oracle / VLM ngoài.
- Kết quả mạnh trên cả 4 bộ với train chuẩn.
- Lỗi thực hiện của Goal.

---

## 7. Đang chạy (card 1 và 3; card 0 và 2 đã trả lại)

| Run | Nội dung | Tiến độ | Có kết quả |
|---|---|---|---|
| `p05_ocd_full_expert_late_s7` | chỉ train 6 tầng cuối + đọc ra | train xong, đang đánh giá | khoảng 1 giờ |
| `p05_geo1_s7`, `p05sp_geo1_s7` | 2606.27663 cài lại (Object, Spatial) | vòng 5–6 / 20, ổn định | khoảng 3–4 giờ |
| `p05gl_v2_s7` | cấu hình v2 trên Goal (hindsight, bỏ cổng) | vừa khởi động | khoảng 5 giờ |
| `p05lg_v2nogrip_s7` | cấu hình v2 trên Long + không dạy thời điểm thả | vòng 0 | khoảng 8 giờ |
| `p05_ect50_s7` | ECT 50% Object (ô swap) | đang đánh giá | khoảng 1 giờ |

---

## 8. Hướng tiếp và các quyết định

**Ngắn hạn (chờ kết quả đang chạy):**
1. Nếu v2 sửa được Goal và Long, cấu hình v2 trên cả 4 bộ, 2 seed, sẽ là kết quả chính cho phần "hơn ECT cùng setup".
2. So với 2606.27663 cài lại. Nếu nó mạnh hơn ta ở Object / Spatial (nhiều khả năng, vì có oracle lúc test), phải trình bày
   rõ khác biệt thông tin; và xem nó có yếu ở Goal / Long / ô đổi màu như dự đoán không.

**Phương pháp chính (chờ kết quả "6 tầng cuối"):**
- Ý tưởng **Decode-then-Act**: lấy vị trí đích ra từ chính bên trong model (đầu giải mã, giám sát bằng toạ độ simulator lúc
  train), nối thẳng vào các tầng sinh động tác, chỉ mở ở khoảnh khắc mù.
- Hai phát hiện mới điều chỉnh ý này:
  - chỗ cần sửa không phải lớp cuối, nên đường nối phải đi vào **nhiều tầng** (giống AdaLN của 2606.27663);
  - chỗ mù còn lại chung cho cả ba model là **đầu pha tiếp cận**, nên đó là chỗ can thiệp có giá trị nhất.
- Khác 2606.27663 ở ba điểm: không oracle lúc test, chỉ ở khoảnh khắc mù, và suy ra từ phân tích.

**Câu hỏi cần quyết định:**
1. Paper đứng trên insight **"VLA biết nhưng không dùng; huấn luyện phản thực dạy dùng, không dạy nhìn"** kèm can thiệp nhắm
   đúng khoảnh khắc mù, có đủ không?
2. Có cần chạy setup lớn (4 bộ, ngân sách lớn) để so với **số công bố** của ECT / 2606.27663, hay chấp nhận so trong cùng setup?
3. Benchmark thứ hai (RoboCasa / CALVIN) để chứng minh tổng quát: làm ngay hay sau khi chốt phương pháp?

---

## 9. Nhận định đã rút lại / sai lầm của chính mình

| Nhận định / quyết định | Thực tế |
|---|---|
| "Lối tắt đi qua token trạng thái tay" | Sai: bỏ trạng thái tay không đổi gì |
| "Chỗ nghẽn ở lớp đọc ra cuối" | Sai ở dạng đơn giản: chỉ train lớp cuối → 23.5 |
| Cổng thực thi ngưỡng 60% | Có hại ở Spatial (loại task có lợi nhất) |
| Learning rate MLP của 2606.27663 × 10 | Lỗi của mình, làm sụp run Object; đã sửa và chạy lại |
| "Đổi màu vật gỡ tradeoff ô đổi màu" | Chỉ một phần (74 → 85), không lặp lại ở mọi cấu hình |
| Dùng `obj_of_interest[0]` làm vật đích | Sai ở vài task Long (task 6 nhận đĩa làm vật đích); kiểm tra đồng ý đã loại được |
