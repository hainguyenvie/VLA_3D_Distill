# Báo cáo tuần 1 — On-policy 3D distillation cho VLA

Cập nhật 07/10/2026, 17:45 UTC. Mọi số đo là của chúng tôi trừ khi ghi "công bố". Chi tiết và lịch sử quyết định:
[week1_status.md](week1_status.md); số liệu từng run: [../results/week1_summary.csv](../results/week1_summary.csv);
bối cảnh paper: [related_work.md](related_work.md); phân tích "3D cần ở đâu": [analysis_3d_where.md](analysis_3d_where.md).

## 1. Tóm tắt

- **Baseline đã tái lập đúng.** Student 1-traj 51.4 (công bố 54.9), teacher full-SFT 95.8 (95.3), OpenVLA-OFT 96.8
  (98.4), π0.5 98.5–100 (LeRobot 99.0). Distill on-policy (bản VLA-OPD của ta) đạt **93.0** trên LIBERO-Object với lr
  giảm cosine (VLA-OPD công bố 93.8).
- **LIBERO chuẩn đã bão hoà; headroom nằm ở các benchmark nhiễu.** LIBERO-Plus Object: OFT 70.2, Spatial Forcing 71.9.
  LIBERO-PRO Object với π0.5 (backbone của các đối thủ mạnh nhất): đổi chỗ vật **18.5**, dời vị trí **10.5**, đổi
  mục tiêu **0**, trong khi đổi câu lệnh 100 và đổi màu vật 94.
- **Idea ban đầu (loss depth trên state on-policy) không hơn baseline** (86.2 so với 86.2; LIBERO-Plus 45.0 so với 48.1).
  Ở chế độ lr ổn định, nhánh state-teacher + depth còn thua baseline (83.6 / 41.2 so với 93.0 / 51.4).
- **Failure analysis cho intuition rõ ràng:** policy *biết* vật ở đâu nhưng action *không theo* vật.
  - Student 1-traj: feature đọc vị trí vật so với tay kẹp sai < 1 cm, nhưng dời vật 6 cm thì chunk không đổi (probe
    phản thực p = 0.00); cú kẹp trượt theo quỹ đạo thuộc lòng (r = 0.79).
  - π0.5: trên cảnh quen bám vật tốt (p = 0.91), nhưng khi vật đổi chỗ thì **91% lần hỏng là nhặt nhầm vật ở vị trí
    quen** (lệch 21 cm), và không còn bám vật được gọi tên (p = 0.00). Lỗi là **chọn đích theo vị trí đã thuộc**.
- **Render lại state on-policy là cơ chế có hiệu lực:** trên OpenVLA-OFT, tự distill dưới nhiễu thị giác (A2) đạt
  **82.1** trên LIBERO-Plus ở vòng 4 (OFT gốc 70.2, SF 71.9), không tụt trên cảnh chuẩn; trên student 1-camera (V1)
  hai seed cho 56.4 / 60.2 (baseline 51.4, teacher 50). Phần tăng nằm ở Camera / Noise / Light; Robot-init và Layout
  không đổi.
- **Phương pháp mới (đang chạy trên π0.5):** phản thực **đổi chỗ vật** trên state on-policy, nhãn chính xác. Số sớm
  (vòng 4/20) chưa tốt: ô swap 23.5 so với 22.0 của distill không phản thực và 18.5 của π0.5 gốc, đổi lại cảnh chuẩn
  tụt 100 → 84. Chưa có kết quả nào vượt SOTA.
- **Hạ tầng:** từ 06/10 toàn bộ chạy trên node 8×H200 (render GPU, eval nhanh gấp 3, 8 run song song); máy 1×H200 cũ đã
  được xoá sạch dữ liệu dự án.

## 2. Thiết lập

| Hạng mục | Lựa chọn |
|---|---|
| Nhánh token (VLA-OPD) | student = OpenVLA-OFT token rời rạc, 1 camera, SFT 1 demo/task; teacher full-SFT (Object), teacher RL SimpleVLA-RL (Long); reverse-KL trên 56 token; LoRA r=32; 20 vòng × 2048 state |
| Nhánh OFT | checkpoint OpenVLA-OFT chính thức (2 camera + proprio, head L1); teacher = chính nó đóng băng; loss L1 |
| Nhánh π0.5 | `lerobot/pi05_libero_finetuned` (PyTorch); LoRA r=32; flow matching trên chunk 50 bước; 20 vòng × 1024 state |
| Eval | greedy; LIBERO-Object 100–500 episode; LIBERO-Plus 60 task mỗi loại nhiễu (420); LIBERO-PRO Object 5 ô × 200 episode |
| Máy | node 8×H200 (từ 06/10), mỗi run một card, render EGL; trước đó 1×H200 (render CPU) và 2×L40 |

Mọi nhánh trong một so sánh chạy cùng máy, cùng renderer, cùng ngân sách; các run dở dang trên máy cũ đã được chạy
lại từ đầu trên máy mới thay vì ghép.

## 3. Trần và headroom

**LIBERO-Plus Object (420 task):**

| Model | Tổng | Camera | Robot-init | Noise | Layout | Light | BG | Lang |
|---|---|---|---|---|---|---|---|---|
| Student 1-traj | 21.2 | 3 | 18 | 7 | 33 | 13 | 35 | 38 |
| Teacher full-SFT (1 camera) | 50.0 | 13–20 | 45–55 | 40–43 | 58–62 | 30 | 73–82 | 72–75 |
| OpenVLA-OFT | 70.2–71.4 | 48 | 27–30 | 65–72 | 67 | 90–93 | 98 | 95–97 |
| Spatial Forcing | 71.9 | 57 | 38 | 42 | 70 | 100 | 97 | 100 |
| π0.5 (Robot / Layout) | — | — | **83** | — | **82** | — | — | — |
| Tốt nhất đã công bố (4 suite) | 90–92 | 82–98 | 78–88 | 90–98 | 85–97 | 97–99 | 96–99 | 83–95 |

**LIBERO-PRO Object, π0.5 (200 episode mỗi ô):** chuẩn 98.5 · đổi câu lệnh 100 · đổi màu vật 94 · **đổi chỗ 18.5** ·
**dời vị trí 10.5** · **đổi mục tiêu 0**. ECT (đối thủ gần nhất, cùng π0.5) công bố đổi chỗ 36 → 59 với protocol riêng.

Đọc: với backbone mạnh, mọi trục thị giác đã gần trần; headroom thật nằm ở **dịch chuyển trạng thái của vật**
(LIBERO-PRO), nơi π0.5 gần như sụp hoàn toàn.

## 4. Nhánh token: baseline và ma trận 2×2 (LIBERO-Object, 500 episode, seed 7)

| | Không 3D | + loss depth |
|---|---|---|
| State của student, lr cố định | 86.2 (B2) | 86.2 (B4) |
| State của teacher, lr cố định | 73.2 (B2′) | 85.0 (B3) |
| State của student, lr cosine | **93.0** (cos_b2) | — |
| State của teacher, lr cosine | đang chạy lại | 83.6 (cos_b3) |

- Distill on-policy hơn hẳn trên state của teacher: đúng tiền đề VLA-OPD.
- Loss depth không giúp trên state on-policy; "depth giúp trên state teacher" ở lr cố định phần lớn là ổn định hoá
  (ở lr cosine, cos_b3 thua cos_b2).
- Mọi run lr cố định sụp 1 task ở cuối; lr cosine sửa được (93.0, không task nào dưới 72).
- LIBERO-Long: teacher RL 85.5 (công bố 91.7); distill lr cố định 59.0 trên 200 episode (VLA-OPD 78.9); bản lr cosine
  đang chạy lại trên máy mới.

## 5. Failure analysis

1. **Student 1-traj hỏng vì lệch vài cm lúc gắp, quyết định rất sớm.** Near miss 46%, làm đổ vật 22%, nhầm chỗ 14%,
   nhầm vật 13%. Dung sai gắp khoảng 2.5 cm. Teacher tiếp quản trước lần kẹp đầu cứu được 95%, sau đó chỉ 3%.
2. **Không phải thiếu nhận thức 3D trong phân phối.** Probe depth sai 3–5%; probe vị trí vật so với tay kẹp sai 0.7–0.9 cm
   kể cả ở episode hỏng. Cú kẹp trượt tương quan r = 0.79 với độ lệch khỏi quỹ đạo trung bình, 0.11 với sai số probe.
3. **Probe phản thực (dời vật đích 3–6 cm ở state tiếp cận, đo phần chuyển động của chunk đi theo vật):**

   | | p trung vị | sát vật | xa vật |
   |---|---|---|---|
   | Student 1-traj | 0.00 | 0.03 | 0.00 |
   | Teacher full-SFT | 0.03 | 0.27 | 0.00 |
   | π0.5, cảnh chuẩn | **0.91** | 1.43 | 0.25 |
   | π0.5, cảnh đổi chỗ | **0.00** | 0.02 | −0.04 |

4. **Dưới nhiễu thị giác thì nhận thức hỏng thật** (probe 0.4 cm → 6–10 cm ở episode hỏng của Camera / Noise / Layout).
5. **OFT dưới Robot-init đưa tay tới sai chỗ** (31/43 failure, lệch 19 cm). Che proprio của OFT lúc test không giúp
   (Robot-init 27 → 17), nên giả thuyết "proprio là lối tắt" không được ủng hộ.
6. **π0.5 dưới đổi chỗ vật nhặt nhầm vật ở vị trí quen** (148/163 failure). Với dời vị trí: rơi / đặt sai 86, nhầm vật
   50, near miss 28.

## 6. Render lại state on-policy dưới nhiễu thị giác (V1, A2)

| | Chuẩn | LIBERO-Plus | Camera | Light | Noise | Robot | Layout |
|---|---|---|---|---|---|---|---|
| Teacher 1-camera | 95.2 | 50.0 | 13–20 | 30 | 40–43 | 45–55 | 58–62 |
| B2 lr cosine (baseline) | 93.0 | 51.4 | 5 | 63 | 25 | 48 | 52 |
| **V1 seed 7 / seed 8** | 87.8 / 77.2 | **56.4 / 60.2** | 42 / 40 | 75 / 75 | 53 / 53 | 45 / 52 | 45 / 55 |
| OpenVLA-OFT gốc | 96.8 | 70.2 | 48 | 90 | 65 | 27 | 67 |
| **A2 (OFT tự distill, vòng 4)** | 98 | **82.1** | **92** | **100** | **98** | 23 | 65 |

- Cơ chế: teacher gán nhãn trên ảnh chuẩn, student học trên ảnh render lại của cùng state mô phỏng dưới camera /
  ánh sáng / nhiễu ngẫu nhiên.
- A2 vòng 20 và đối chứng A1 (cùng vòng lặp, không nhiễu) đang chạy lại trên máy mới; cần A1 để chắc mức tăng không
  đến từ bản thân việc fine-tune.
- Hạn chế: dải nhiễu lúc train rộng cỡ LIBERO-Plus nên không còn là zero-shot theo loại nhiễu; làn "render lại góc
  nhìn" đã có người làm trên demo (Cross-View Action Consistency, Camera 79.8 → 87.2). Vì vậy đây là bằng chứng phụ,
  không phải contribution chính.

## 7. Phương pháp hiện tại: on-policy counterfactual distillation trên π0.5

Ở mỗi state trước khi kẹp mà student tự đi tới, simulator render thêm một **thế giới phản thực** (chỉ đổi pose khi
render, vật lý không đổi). Loss flow matching ghép cặp (chung noise và time cho hai thế giới) cộng một số hạng nhất
quán giữa hai thế giới. Ba nhánh cùng ngân sách:

- **swap:** vật đích đổi chỗ với một vật khác, câu lệnh đổi sang tên vật đang ở vị trí cũ; nhãn chính xác = chunk gốc.
- **shift:** vật đích dời 2–10 cm, π0.5 gốc gán nhãn thế giới đó.
- **base:** distill on-policy không phản thực (đối chứng).

Kết quả sớm (adapter vòng 4/20, ô đổi chỗ của LIBERO-PRO, 200 episode):

| | Đổi chỗ (PRO swap) | Object chuẩn (100 ep) |
|---|---|---|
| π0.5 gốc | 18.5 | 100 |
| base | 22.0 | 100 |
| swap | 23.5 | **84** |
| shift | đang chấm | 100 |

Chưa có tín hiệu tốt: swap chỉ ngang base trên ô đổi chỗ và làm tụt cảnh chuẩn. Loss phản thực đã rất thấp (0.65 → 0.02),
tức student học được cặp trong train nhưng chưa chuyển sang cảnh đổi chỗ thật. Giả thuyết đang kiểm: cặp được dựng
trên quỹ đạo đã hướng về vị trí cũ, còn quyết định "đi về đâu" nằm ở vài chunk đầu; lr 5e-5 có thể quá mạnh cho π0.5.
Eval ở vòng 8 và 12 đã xếp hàng; bước tiếp theo nếu xu hướng không đổi: lr thấp hơn, trọng số cao cho state đầu tập,
LoRA không đụng encoder ảnh.

## 8. Mục tiêu và trạng thái

| Mục tiêu | Trạng thái |
|---|---|
| Baseline tái lập trong 3 điểm quanh VLA-OPD | **Đạt trên Object** (93.0 so với 93.8); Long đang chạy lại |
| Student vượt teacher trên LIBERO-Plus | **Đạt** (V1: 56.4 / 60.2 so với 50.0, hai seed) |
| Vượt OFT / SF trên LIBERO-Plus cùng backbone | A2 vòng 4: 82.1 (OFT 70.2, SF 71.9); chờ A2 vòng 20 và đối chứng A1 |
| Chữa lỗi chọn đích theo vị trí của π0.5 (LIBERO-PRO, đối thủ ECT 36 → 59) | Đang chạy; vòng 4 chưa có tín hiệu |

## 9. Hạn chế và rủi ro

- Hầu hết kết quả là một seed; eval 100 episode có sai số 4–5 điểm, 60 task mỗi loại nhiễu 6 điểm, 200 episode khoảng 3.
- Chẩn đoán "biết mà không theo" đã được nhiều bài 2026 nêu (memory trap, shortcut priors, ECT); phần mới phải nằm ở
  phương pháp, chưa có.
- Phương pháp trên π0.5 có thể chỉ ngang ECT; phải chứng minh phần "on-policy" bằng đối chứng cặp dựng trên demo.
- Nhãn trung gian của máy mới trùng tên tài khoản đã lọt vào lịch sử git trước khi đổi tên; chờ quyết định có viết
  lại history không.

## 10. Việc đang chạy

Trên node 8×H200: π0.5 swap / shift / base (vòng 7–10/20, tự chấm Object + 5 ô LIBERO-PRO + LIBERO-Plus Robot / Layout
khi xong); A2 / A1 nhánh OFT (vòng 14/20); cos_b2 / cos_b2p (vòng 19/20); Long lr cosine (vòng 19/40).
