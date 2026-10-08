# Báo cáo tiến độ — thế giới phản thực 3D và teacher có đặc quyền cho VLA

Cập nhật 09/10/2026, 01:00 UTC. Mọi số là của chúng tôi (π0.5 bản LeRobot, cùng harness, 200 tập mỗi ô LIBERO-PRO, cùng
ngân sách 20 vòng × 1024 state) trừ khi ghi "công bố". Nhật ký chi tiết: [week1_status.md](week1_status.md); số liệu từng run:
[../results/week1_summary.csv](../results/week1_summary.csv); đối thủ: [related_work.md](related_work.md) (§9–10).

## 1. Một câu

VLA fine-tune trên bố cục cố định học "câu lệnh → vị trí đã thuộc" thay vì "nhìn vật ở đâu"; ta sửa bằng cách distill trong
**thế giới phản thực 3D** (dựng lại state mô phỏng, dời riêng vật đích hoặc nơi đặt để quan hệ tay–đích đổi thật) với
**teacher có đặc quyền theo pha** (biết vị trí thật), vì chính VLA không làm thầy được ở đó.

## 2. Chẩn đoán (π0.5, LIBERO / LIBERO-PRO)

- Cảnh chuẩn 94–99%, nhưng ô swap / position của LIBERO-PRO: Object 18.5 / 10.5, Spatial 42, Goal 26, Long 9.
- Lỗi tách theo pha, pha nào cũng **đi theo vị trí đã thuộc**:
  - **pha tiếp cận** (Object swap, một phần Goal): đi tới vật đang đứng ở ô quen của vật đích (20/20 tập ở các task hỏng);
  - **pha đặt** (Spatial swap, Object position, Goal swap): nhặt đúng vật rồi đặt xuống **cách chỗ nơi đặt quen 3 cm**, cách nơi
    đặt thật 25 cm.
- π0.5 nhớ **cả quỹ đạo**, không chỉ vị trí vật: hỏi nó trong thế giới đã đưa vật về chỗ quen nhưng tay ở chỗ lạ ("teacher chuẩn
  hoá") chỉ cho 22.0 trên Object swap (tự làm 18.5) → nó không làm teacher được ở bất kỳ thế giới lệch nào.
- **Các biến đổi có nhãn chính xác** (xoay cả cảnh, dời tay cùng vật, soi gương) **giữ nguyên quan hệ tay–đích**, nên lối tắt
  "đi đoạn đã thuộc" vẫn khớp nhãn; thực nghiệm: không tăng trên Spatial, tăng vừa trên Object.

## 3. Phương pháp

Distill π0.5 (LoRA) với flow matching trên các state của rollout; ở một phần state (≤ 15%), render thêm thế giới phản thực từ
**cùng state mô phỏng** và gán nhãn bằng teacher kịch bản có đặc quyền:

| Phản thực | Pha | Dời gì | Nhãn |
|---|---|---|---|
| **retarget** | tiếp cận | riêng vật đích (chỗ trống, hoặc chiếm chỗ vật khác) | tới 10 cm trên vật đích, gripper mở; mask sau khi tới |
| **relocate** | đặt | riêng nơi đặt (lấy từ mục tiêu BDDL: đĩa, giỏ, vùng trên bếp / nóc tủ) | mang tới trên nơi đặt, hạ, thả; mask sau khi thả |
| coshift (phụ) | tiếp cận | vật đích + tay (IK) | chunk gốc của π0.5 (nhãn chính xác) |

Các nhãn sinh trên mô hình động học khớp từ log (OSC: dịch chuyển ≈ g·a·5 cm / bước); mọi teacher qua **cổng kiểm tra trong mô
phỏng** trước khi dùng (thực thi nhãn theo chunk trong thế giới đã dời: đặt / tới đúng ≥ 80%), và chỉ dùng ở task qua cổng.

## 4. Kết quả chính (π0.5; base = distill cùng ngân sách không phản thực)

| Bộ / ô | Base | Aug 2D (openpi) | **Phương pháp** | Seed |
|---|---|---|---|---|
| Spatial — swap | 40.0 | — | **64.7** (relocate) | 3 |
| Object — swap | 18.7 | 24.2 | **53.0** (retarget); **63.5** (retarget + relocate) | 3; 1 (s8 chạy) |
| Object — position | 12.2 | 13.0 | **42.3** (coshift + relocate); 33 (relocate) | 2; 2 |
| Cảnh chuẩn Object / Spatial | 98–100 | 98–99 | 97–99.5 | |

Cận trên: tiếp cận oracle rồi π0.5 tự làm = 59.0 trên Object swap; retarget đạt 90% mức đó, retarget + relocate vượt.

**Ablation (Spatial swap, s7):** thế giới phản thực + nhãn π0.5 = 40.0; nhãn kịch bản ở thế giới gốc = 35.5; kết hợp = **64.7**
→ cần cả hai. **Nguồn state:** state của π0.5 gốc thay cho state on-policy cho kết quả như nhau (retarget 50.5 vs 51.8; coshift
22.5 vs 24.0) → không claim on-policy. Số hạng nhất quán theo cặp: không đóng góp ở vòng cuối.

**Phép thử lối tắt:** một vật phụ bắt đầu lệch chỗ ~20 cm → retarget không đi nhặt nó (0/100 nhầm vật) → gain swap không đến
từ "nhặt vật lệch chỗ".

## 5. Tradeoff và giới hạn

- **Vị trí ↔ ngoại hình.** Ô "object" của LIBERO-PRO đổi màu vật đích; base làm tốt (93) *vì* tìm vật theo vị trí. Retarget dạy
  tìm theo ngoại hình nên tụt (72.5 / 76.0; retarget + relocate 80.5). Đang thử ngẫu nhiên hoá màu vật khi train.
- **Goal / Long chưa tăng.** Goal swap: retarget 32.5, relocate (bản cũ, lọc task) 28.5, base 29.0; lỗi Goal chủ yếu là đặt lên
  nóc tủ — teacher mới (theo mục tiêu BDDL) vừa qua cổng một phần ở đó, đang chạy. Long: bản đầu làm tụt cảnh chuẩn; đang chạy
  lại với teacher đã sửa, chỉ ở task giỏ.
- **Teacher viết tay theo pha** (nhặt-đặt) — không tổng quát cho kỹ năng khác (mở ngăn, xoay núm); thử thay bằng chính π0.5 ở
  thế giới chuẩn hoá đã thất bại (§2).
- **Chưa so được SOTA.** ECT (công bố) Object swap 38.3 → 70.8 theo protocol khác (π0.5 gốc của họ 38.3, của ta 18.5). Về ý
  tưởng ECT gần nhất: cũng biến đổi hình học + bộ thực thi phát lại để gán nhãn, trên demo. Cần cài lại ECT trong harness.
- LIBERO-Plus Robot / Layout (60 tập) dao động ±8 giữa hai lần đo; không dùng làm kết luận.

## 6. Nhánh phụ (OFT, LIBERO-Plus)

Tự distill dưới nhiễu thị giác render lại từ cùng state (A2): **83.1** vs đối chứng 68.7 (+14.4, 2 seed), cảnh chuẩn 97–98.

## 7. Đang chạy / tiếp theo

- Đang chạy: retarget + relocate s8; chế độ `full` (retarget / coshift + relocate); retarget + màu ngẫu nhiên; Goal relocate theo
  mục tiêu BDDL; Long relocate (task giỏ).
- Tiếp: một cấu hình duy nhất chạy trên cả 4 bộ; cài lại ECT để so công bằng; thêm seed; viết bản nháp.
