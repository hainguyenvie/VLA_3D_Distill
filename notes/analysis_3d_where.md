# 3D cần ở đâu? Phân tích lại toàn bộ kết quả (06/10/2026)

Mục đích: từ số liệu của chính ta và các bài mới nhất, trả lời bốn câu: headroom còn ở đâu; vấn đề ở đó có phải vấn đề
3D không; vì sao các cách "thêm 3D" hiện có chưa lấy được phần đó; và nếu thêm 3D thì ở tầng nào, tác động ra sao.
Số của ta: [week1_status.md](week1_status.md), [week1_report.md](week1_report.md). Paper: [related_work.md](related_work.md).

## 1. Headroom còn ở đâu

LIBERO-Plus Object, cùng loại nhiễu, số tốt nhất đã biết (ta đo trừ khi ghi "công bố"):

| Loại nhiễu | OFT (SOTA zero-shot) | Spatial Forcing | Tốt nhất đã công bố | Teacher 1-camera của ta | Còn trống? |
|---|---|---|---|---|---|
| Camera | 48 | 57 | **87** (CVAC, flow-VLA, train trên cặp góc nhìn của demo), 93 (OFT train trên data nhiễu) | 13–20 | gần như hết: data hoặc render lại đã giải quyết |
| Light / Background / Language | 90 / 98 / 97 | 100 / 97 / 100 | 95 / 94 / 86 | 30 / 73–82 / 72–75 | hết |
| Sensor Noise | 65 | 42 | 89 (OFT train trên data nhiễu) | 40–43 | hết nếu có data; 3D feature alignment còn làm tệ đi |
| Layout | 67 | 70 | 78 (data nhiễu) | 58–62 | còn 20–30 điểm |
| **Robot-init** | 27 | 38 | **30 (data nhiễu không giúp)** | **45–55** | **còn 50–60 điểm, chưa ai lấy được** |

Hai nhận xét:
- Mọi "headroom thị giác" (camera, ánh sáng, nhiễu ảnh) đều được lấp bằng data hoặc bằng render lại demo (CVAC 87 ở
  Camera). Làn này đã đông, và kết quả V1 của ta (Camera 13 → 42) nằm trong làn đó với số thấp hơn họ.
- Headroom còn nguyên nằm ở **dịch chuyển trạng thái** (Robot-init, Layout): model mạnh nhất có data nhiễu cũng chỉ 30
  ở Robot-init; model 1-camera không proprio của ta lại đạt 45–55. Đây là chỗ đáng làm.

## 2. Vấn đề ở đó là gì (từ finding của ta)

1. **Hình học có trong feature nhưng action không dùng.** Probe trên feature đóng băng đọc được vị trí vật so với tay
   kẹp sai dưới 1 cm, kể cả ở các episode kẹp trượt 3–7 cm; cú kẹp trượt lệch theo quỹ đạo thuộc lòng (r = 0.79), không
   theo sai số nhận thức (r = 0.11). Tức lỗi nằm ở **tầng đọc ra action**, không ở tầng biểu diễn.
2. **SOTA cũng thế khi dịch chuyển trạng thái.** OFT dưới Robot-init: 31/43 failure là đưa tay tới sai chỗ (lệch 19 cm):
   đổi tư thế xuất phát thì policy vẫn thực hiện chuyển động quen. LIBERO-Plus cũng kết luận "replay of trajectory
   patterns tied to the training initialization".
3. **Proprio có thể là lối tắt.** Model có proprio (OFT, OFT train trên data nhiễu) đạt 27–30 ở Robot-init; model không
   proprio (teacher 1-camera, student đã distill) đạt 45–55. Giả thuyết: khi có proprio, action được "khoá" vào trạng
   thái thân robot (q → a) thay vì vào quan hệ 3D tay kẹp – vật; đổi tư thế xuất phát thì khoá sai. ThinkProprio
   (2602.06575) cũng ghi nhận proprio chỉ được đưa vào như tín hiệu điều kiện muộn. **Chưa kiểm chứng** — xem D2.
4. **Trong phân phối, loss depth lên feature không đổi gì** (B4 = B2 = 86.2; probe depth sai 3–5% ở cả episode thành
   công lẫn hỏng). Trên state của teacher thì có (73.2 → 85.0): depth làm regulariser cho distill lệch phân phối; khớp với
   2608.08904 (post-training action làm sụp depth ở các lớp cuối qua MLP writes).
5. **Dưới nhiễu thị giác thì việc đọc hình học hỏng thật** (probe 0.4 cm → 6–10 cm ở episode hỏng), và render lại state
   chữa được (V1). Nhưng nó không chạm Robot-init / Layout (V1: 45 / 45, bằng baseline).
6. **Teacher chỉ dạy được ở pha tiếp cận** (tiếp quản trước lần kẹp đầu: 95%; sau: 3%) → tín hiệu sửa lỗi có giá trị nằm
   ở quãng 40–50 bước đầu, nơi quyết định "đi tới đâu".

Kết luận: vấn đề ở phần headroom còn lại là **action không phụ thuộc vào quan hệ 3D tay kẹp – vật mà phụ thuộc vào thói
quen (quỹ đạo, proprio)**. Nó là vấn đề 3D theo nghĩa: thứ còn thiếu là *sự phụ thuộc của action vào hình học 3D*, chứ
không phải *thông tin* hình học.

## 3. Vì sao các cách thêm 3D hiện có chưa lấy được phần đó

| Cách | Tầng | Vì sao không chạm Robot-init / Layout |
|---|---|---|
| Căn feature theo VGGT / depth (Spatial Forcing, ROCKET, GaussianWAM, MVUCF) | biểu diễn | thêm hình học vào feature vốn đã có hình học (probe < 1 cm); không đổi việc action có dùng nó hay không; SF +1.7 tổng, Robot-init 38 |
| Token không gian vào action head (FALCON 2510.17439, GEAR-VLA) | đọc ra action | đúng tầng, nhưng vẫn học bằng BC trên demo: ở state demo, proprio / thói quen đã giải thích được action, không có gì ép action phụ thuộc hình học (đúng hiện tượng VA-OPD chỉ ra cho VLM) |
| Data nhiễu (OFT+ 20k demo) | dữ liệu | chữa mọi trục thị giác, Robot-init vẫn 30: demo từ tư thế mới vẫn dạy "từ tư thế này đi quỹ đạo này" |
| Render lại demo / state dưới góc nhìn khác (CVAC, InfiNoVA, V1 của ta) | quan sát | chỉ đổi ảnh, trạng thái 3D giữ nguyên → không dạy action thay đổi theo trạng thái |
| Chuẩn hoá góc nhìn lúc test (GS-VLA, AnyCamVLA) | quan sát | chỉ camera |

Điểm chung: không cách nào tạo ra **cặp trạng thái 3D khác nhau với cùng mọi thứ còn lại** để ép action thay đổi theo
hình học. Đó là thứ simulator (và 3DGS có sửa pose vật ngoài đời) làm được mà demo không làm được.

## 4. Thêm 3D ở tầng nào, tác động ra sao

| Tầng | Đã thử / dự đoán | Kết luận |
|---|---|---|
| Quan sát: render lại state dưới góc nhìn / ánh sáng khác | V1: Camera 13 → 42, Noise 22 → 53; Robot-init không đổi | chữa đúng trục thị giác; làn đã đông (CVAC 87) |
| Biểu diễn: loss depth lên token ảnh lớp 24 | B4 = B2; B3 > B2′ | chỉ là regulariser cho distill lệch phân phối; trần thấp |
| **Đọc ra action: distill trên cặp thế giới khác trạng thái 3D** | chưa chạy | giả thuyết chính: dời vật (hoặc tư thế robot) ở state on-policy, teacher gán nhãn cả hai, loss lên **hiệu action** giữa hai thế giới (CW-OPD cho VLA) → ép action theo hình học; đích Robot-init / Layout |

Vì sao đặt ở tầng đọc ra action và bằng cặp phản thực (xuất phát từ finding nào):
- Finding 1 nói thông tin đã có ở feature và còn ở ngay vị trí action; cái thiếu là *đạo hàm* của action theo vị trí vật.
  Cặp phản thực giám sát đúng đạo hàm đó, thứ mà loss endpoint (BC, distill thường) không giám sát (CW-OPD chứng minh
  bằng gradient cho VLM).
- Finding 2–3 nói lỗi của SOTA dưới Robot-init cũng là "không đổi hành vi theo trạng thái", nên cùng cơ chế.
- Finding 6 nói chỗ cần giám sát là pha tiếp cận on-policy, nên cặp phải dựng trên state student tự đi tới trước lần
  kẹp đầu, không phải trên demo.
- Teacher đáng tin cho thế giới "dời vật vài cm" (nằm trong phân phối của nó: 95%); với "dời tư thế robot" teacher chỉ
  45–55% nên phải lọc nhãn theo episode teacher thành công.

Điều kiện tiên quyết (đang đo, D1): teacher có thật sự "đi theo vật" khi vật bị dời không? Nếu teacher cũng phát lại
quỹ đạo (p ≈ 0) thì không có gì để distill, và supervisor phải là bộ điều khiển hình học đặc quyền cho pha tiếp cận
(phép thử oracle 24 episode: task "nhầm chỗ" 7/8, task cần điểm kẹp chính xác 0/8 → cần cả độ cao / điểm kẹp).

## 5. Thí nghiệm theo thứ tự

- **D1** `scripts/probe_counterfactual.py` (đang chạy, L40): dời vật ±3 / ±6 cm ở 231 state tiếp cận của student;
  p = mức chuyển động của chunk đi theo vật (1 = theo hết, 0 = không phản ứng). So student / teacher / student đã
  distill / OFT. Quyết định "distill hiệu từ teacher" hay "teacher hình học đặc quyền".
- **D2** OFT dưới Robot-init với proprio bị che / nhiễu lúc test (60 task, rẻ): nếu che proprio mà *tăng* thì giả thuyết
  lối tắt proprio đúng, và cách sửa gồm cả "proprio dropout" lúc distill.
- **M1** distill trên cặp thế giới dời vật (backbone token, lr cosine, cùng teacher / ngân sách với cos_b2): đo
  LIBERO-Plus Robot-init / Layout và một test trong phân phối với init state mới sinh từ simulator.
- **M2** bản OFT của M1 (nhánh A).
- Đối chứng cho mọi claim: cos_b2m (trộn người lái, không nhiễu), 2D-augment cùng nhãn, teacher gán nhãn trên ảnh nhiễu.

## 6. Hệ quả cho contribution

- Giữ "render lại state dưới góc nhìn khác" làm một **trường hợp** của cơ chế "cặp thế giới", không phải contribution.
- Contribution đề xuất: **on-policy distillation trên cặp thế giới khác trạng thái 3D** để ép action phụ thuộc hình học,
  xuất phát từ finding "hình học có mà không dùng"; 3D cần vì phải dựng và render được thế giới đối chứng; đích là
  phần headroom chưa ai lấy (Robot-init, Layout).
- Rủi ro: teacher không đi theo vật (D1 trả lời); cặp dời vật quá nhỏ không chuyển sang Robot-init (cần cả thế giới dời
  tư thế robot); thêm một họ phương pháp mới trong khi thời gian có hạn.
