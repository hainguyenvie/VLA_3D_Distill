# Báo cáo tuần 1 — On-policy 3D distillation cho VLA

Cập nhật 06/10/2026, 02:00 UTC. Mọi số đo là của chúng tôi trừ khi ghi "công bố". Chi tiết và lịch sử quyết định:
[week1_status.md](week1_status.md); số liệu từng run: [../results/week1_summary.csv](../results/week1_summary.csv);
bối cảnh paper: [related_work.md](related_work.md); nhật ký tái lập: [reproduction_log.md](reproduction_log.md).

## 1. Tóm tắt

- **Mốc tái lập được** (LIBERO-Object, 500 episode, greedy): student 1-traj 51.4 (công bố 54.9), teacher full-SFT 95.8
  (95.3), OpenVLA-OFT chuẩn 96.8 (98.4). Hai máy / hai renderer cho số khớp nhau trên Object.
- **LIBERO chuẩn đã bão hoà; headroom nằm ở LIBERO-Plus.** Trên 420 task LIBERO-Plus Object: OFT chuẩn 70.2, Spatial
  Forcing 71.9 (chỉ hơn 1.7 điểm, 91/420 task cả hai cùng hỏng), teacher 1-camera của ta 50.0, student 21.2.
- **Baseline distill (bản VLA-OPD của ta) đạt 93.0** trên LIBERO-Object với lr giảm cosine (VLA-OPD công bố 93.8) và
  86.2 với lr cố định; khoảng hụt của bản lr cố định là do quá trình distill dao động (sụp từng task ở vòng cuối).
  LIBERO-Plus của baseline: 48.1 (lr cố định), 51.4 (lr cosine) — ngang teacher (50.0), Camera vẫn 5%.
- **Idea ban đầu (loss depth trên state on-policy) không hơn baseline**: 86.2 so với 86.2 trên bản chuẩn, 45.0 so với
  48.1 trên LIBERO-Plus. Trên state của teacher thì depth có giúp (73.2 → 85.0; LIBERO-Plus 39.0 → 49.3), chưa chắc
  cỡ bao nhiêu (một seed).
- **Failure analysis chỉ ra vấn đề nằm ở đâu**: trong phân phối, feature của student đã đọc được vị trí vật so với
  tay kẹp chính xác dưới 1 cm kể cả lúc nó kẹp trượt 3–7 cm (lỗi ở điều khiển, không ở nhận thức); dưới nhiễu thị
  giác thì chính việc đọc hình học hỏng (sai 6–10 cm ở các episode hỏng). Teacher chỉ dạy được ở những gì nó nhìn thấy.
- **Bước sửa theo phân tích đó đã cho kết quả đầu tiên vượt teacher**: render lại chính các state huấn luyện dưới camera
  / ánh sáng / nhiễu ngẫu nhiên cho student, teacher gán nhãn từ ảnh chuẩn → **87.8 trên bản chuẩn, 56.4 trên
  LIBERO-Plus** (teacher 50.0; Camera 13 → 42, Noise 22 → 53, Light 62 → 75). Một seed; seed 2 và bản lr cosine đang chạy.
- Chưa có kết quả nào vượt SOTA tuyệt đối (OFT 70.2 / SF 71.9 trên LIBERO-Plus với backbone 2 camera + proprio).

## 2. Thiết lập

| Hạng mục | Lựa chọn |
|---|---|
| Setting | của VLA-OPD / SimpleVLA-RL: student = OpenVLA-OFT token rời rạc, 1 camera, không proprio, SFT 1 demo/task |
| Teacher | full-SFT 454 demo (`Haozhan72/…-trajall`), phân phối token chuyển sang bin của student; Long: teacher RL cùng dòng dõi |
| Distill | reverse-KL dạng đóng trên 56 token; LoRA r=32; AdamW lr 1e-4; 20 vòng × 2048 state; rollout lấy mẫu T=1.6 |
| Eval | greedy, 512 bước; LIBERO-Object 500 episode; LIBERO-Plus 60 task mỗi loại nhiễu (420), 1 trial |
| Máy | H200 (render CPU, nhiều job song song), 2×L40 (render GPU, hạn chế RAM). Hai máy không nối với nhau |

Sai khác so với VLA-OPD: LoRA thay vì full-parameter; khoảng 41k state so với khoảng 300k; teacher full-SFT chuyển
bin thay vì teacher RL cùng dòng dõi (Object không có teacher RL công khai dùng được).

## 3. Trần và headroom (LIBERO-Plus Object, 420 task)

| Model | Tổng | Camera | Robot-init | Noise | Layout | Light | Background | Language |
|---|---|---|---|---|---|---|---|---|
| Student 1-traj | 21.2 | 3 | 18 | 7 | 33 | 13 | 35 | 38 |
| Teacher full-SFT (1 camera) | 50.0 | 13–20 | 45–55 | 40–43 | 58–62 | 30 | 73–82 | 72–75 |
| OpenVLA-OFT chuẩn (2 camera + proprio) | 70.2 | 48 | 27 | 65 | 67 | 90 | 98 | 97 |
| Spatial Forcing (cùng backbone OFT) | 71.9 | 57 | 38 | 42 | 70 | 100 | 97 | 100 |
| OFT train trên data LIBERO-Plus (công bố, 4 suite) | 79.5 | 93 | 30 | 89 | 78 | 95 | 94 | 86 |

Đọc: Robot-init là loại nhiễu không ai chữa được, kể cả bằng data; Camera / Noise / Light chữa được bằng data hoặc
thích nghi; 3D feature alignment (SF) chỉ thêm 1.7 điểm và còn tụt ở Noise.

## 4. Baseline distill và ma trận 2×2 (LIBERO-Object chuẩn, 500 episode, seed 7)

| | Không 3D | + loss depth |
|---|---|---|
| State của student (on-policy) | **86.2** (B2) | 86.2 (B4) |
| State của teacher | 73.2 (B2′) | 85.0 (B3) |

- Distill trên state của chính student hơn hẳn trên state của teacher (86.2 so với 73.2): đúng tiền đề on-policy.
- Depth không giúp trên state on-policy; trên state của teacher thì giúp (+11.8, phần lớn do B2′ sụp task 8 ở vòng
  cuối; bỏ task sụp của mỗi bên vẫn hơn ở 6/8 task). Cách đọc tạm: depth neo feature nên những gì học trên khung hình
  của teacher chuyển sang khung hình của student tốt hơn; trên state on-policy không có khoảng cách phân phối đó.
- Mọi run ở lr cố định đều dao động và sụp 1 task ở cuối (B2 task 5: 60 → 64 → 46 → 30 qua các checkpoint). Bản lr
  cosine (1e-4 → 1e-5) của B2 đạt **93.0** trên 500 episode (không task nào dưới 72), tức mốc 1 (trong 3 điểm quanh
  VLA-OPD) đã đạt trên Object; ma trận 2×2 cần chạy lại ở chế độ ổn định này trước khi đọc hiệu ứng của depth.

LIBERO-Plus của các adapter cuối: B2 48.1, B2′ 39.0, B3 49.3, B4 45.0 (teacher 50.0).

## 5. Failure analysis

1. **Student hỏng vì lệch vài cm lúc gắp, và số phận được quyết định rất sớm.** 243 failure: near miss 46%, làm đổ vật
   22%, nhầm chỗ 14%, nhầm vật 13%, rơi sau khi nhấc 4%. Dung sai gắp khoảng 2.5 cm (SR 95% dưới 1.5 cm, 35% ở 2.5–4 cm,
   4% trên 4 cm). Teacher tiếp quản **trước** lần kẹp đầu: 95% thành công; **sau** lần kẹp hỏng đầu: 3%.
2. **Không phải thiếu nhận thức 3D.** Probe trên feature đóng băng: depth sai 3–5% (bằng nhau ở episode thành công
   và hỏng); vector tay kẹp → vật sai trung vị 0.7–0.9 cm ở student (teacher 0.4–0.6), vẫn dưới 1 cm ở hai query cuối
   trước khi kẹp của các episode hỏng. Cú kẹp trượt tương quan r = 0.79 với độ lệch khỏi quỹ đạo trung bình (phát lại
   quỹ đạo thuộc lòng), chỉ 0.11 với sai số của probe.
3. **Dưới nhiễu thị giác thì nhận thức hỏng thật.** Cùng probe (fit trên cảnh chuẩn) áp sang LIBERO-Plus: episode
   thành công 1–3 cm, episode hỏng ở Camera / Noise / Layout 6–8 cm. Teacher hỏng chủ yếu vì đưa tay tới sai chỗ
   (116/213, lệch 14–22 cm).
4. **OFT chuẩn (SOTA) dưới Robot-init cũng đưa tay tới sai chỗ** (31/43 failure, lệch 19 cm): policy phát lại chuyển
   động quen thay vì bám theo vật khi tư thế xuất phát bị dời.

## 6. Bước sửa: distill dưới nhiễu thị giác, teacher nhìn cảnh sạch

Giữ nguyên vòng lặp distill; mỗi episode huấn luyện rút một nhiễu (camera quay quanh điểm nhìn ±75°, nâng tới 15°, xa
tới 2 lần, lệch ngắm ±10°; ánh sáng ×0.3–1.7; nhiễu Gauss, blur). Student được cho ảnh nhiễu (và tự lái trên đó ở nửa
số episode), teacher gán nhãn và lái từ ảnh chuẩn của **cùng state mô phỏng**. Đây là chỗ cần cảnh 3D (simulator ở
đây; 3DGS ngoài đời).

| | Chuẩn (500 ep) | LIBERO-Plus | Camera | Light | Noise | Robot-init | Layout |
|---|---|---|---|---|---|---|---|
| Teacher | 95.2 | 50.0 | 13–20 | 30 | 40–43 | 45–55 | 58–62 |
| B2 (baseline distill, lr cố định) | 86.2 | 48.1 | 13 | 62 | 22 | 47 | 48 |
| B2 với lr cosine | 93.0 | 51.4 | 5 | 63 | 25 | 48 | 52 |
| **V1** (nhiễu thị giác) | **87.8** | **56.4** | **42** | **75** | **53** | 45 | 45 |
| V2 (V1 + depth trên ảnh nhiễu) | 77.0 | 50.2 | 23 | 75 | 43 | 47 | 37 |

- V1 là kết quả đầu tiên đạt mốc "student vượt chính teacher của nó trên LIBERO-Plus (≥ 55)", mức tăng nằm đúng ba
  loại nhiễu được render lại; Robot-init và Layout không đổi (đúng như dự đoán: chúng không phải nhiễu thị giác).
- V2 không hơn V1, nhưng checkpoint cuối của V2 là checkpoint sụp (eval 100 ep: 92 ở vòng 16–18, 76 ở vòng 20), nên
  chưa kết luận được gì về depth dưới nhiễu. Chạy lại với lr cosine.
- Cần nói rõ khi báo cáo: dải nhiễu lúc train rộng cỡ LIBERO-Plus, nên kết quả **không còn là zero-shot theo loại
  nhiễu**; không dùng demo mới và không dùng task / file nhiễu của LIBERO-Plus. Mốc so công bằng là teacher / student
  cùng backbone và OFT train trên data nhiễu (79.5).

## 7. Mục tiêu đã đặt và trạng thái

| Mục tiêu | Mốc | Trạng thái |
|---|---|---|
| 1. Baseline tái lập trong 3 điểm quanh VLA-OPD (Object 93.8), rồi 4 suite ≥ 90 | Object ≥ 91 | **93.0 với lr cosine (đạt)**; 86.2 với lr cố định. Long: teacher RL 85.5 (công bố 91.7); distill lr cố định 59.0 trên 200 ep (VLA-OPD 78.9); bản lr cosine xếp hàng |
| 2. Student vượt teacher trên LIBERO-Plus (≥ 55) | ≥ 55 | **56.4** (V1, một seed); seed 2 đang chạy |
| 3. OFT chuẩn trên LIBERO-Plus, Robot-init ≥ 50 | ≥ 75 tổng | **A2 vòng 4: 82.1 tổng** (OFT 70.2, SF 71.9; Camera 92, Noise 98) — nhưng Robot-init 23, Layout 65 không đổi; đối chứng A1 và vòng 20 đang chạy |
| 4. RoboTwin 2.0, 3 seed | — | chưa làm |

## 8. Hạn chế và rủi ro

- Hầu hết là **một seed**; eval 100 episode có sai số 4–5 điểm, 60 task mỗi loại nhiễu có sai số 6 điểm. Các run ở lr
  cố định sụp ngẫu nhiên 1 task ở cuối, nên so sánh adapter cuối giữa các nhánh còn nhiễu.
- Trên Long, student cho 27% (L40, render GPU) so với 16–17% (H200, render CPU) trên cùng 100 episode; chưa rõ nguyên
  nhân, mọi run Long đang ở H200 nên so sánh nội bộ vẫn cùng renderer.
- "Vượt SOTA" theo nghĩa các paper cùng nhánh dùng là vượt đối thủ trực tiếp trên cùng backbone / cùng setting, không
  phải bảng xếp hạng tuyệt đối (lớp π0.5 đạt 85–88 trên LIBERO-Plus).
- Bản 2×2 "3D trên state on-policy" không ủng hộ giả thuyết ban đầu; câu chuyện 3D hiện là "cầu nối giữa hai phân
  phối" (state teacher → state student; góc nhìn chuẩn → góc nhìn lạ), cần thêm bằng chứng.

## 9. Việc đang chạy và việc tiếp theo

Đang chạy: V1 seed 8 (H200); V1 + lr cosine (L40); baseline lr cosine (H200, chấm 500 ep); LIBERO-Long baseline
(vòng 34/40). Mỗi run tự chấm 500 episode chuẩn và LIBERO-Plus khi xong.

Tiếp theo, theo thứ tự: (1) chốt baseline với lr cosine (Object, rồi Long); (2) V1 với 3 seed và V2 (depth) với lr
cosine để trả lời "3D có thêm gì dưới nhiễu"; (3) seed 2 cho B2′ / B3 và đo probe trên feature của hai adapter đó;
(4) Robot-init: thử nhiễu trạng thái xuất phát của robot trong rollout (teacher gán nhãn) — đây là headroom mà cả
data lẫn 3D feature alignment đều chưa chữa được; (5) quyết định về backbone OFT chuẩn (mục tiêu 3) và RoboTwin 2.0.

## 10. Sai sót vận hành đã ghi nhận

Frame trễ sau khi tối ưu render (đã sửa, có gate); công cụ takeover thiếu trạng thái gripper / controller (đã sửa, có
đối chứng); dừng run reverse-KL quá sớm; một tham số giao diện làm nhánh B3 crash một giờ; tính tay bộ nhớ GPU gây OOM
cho run Long (đã thay bằng cổng bộ nhớ có khoá); hai lần tự giết phiên ssh bằng `pkill -f`. Chi tiết trong
[week1_status.md](week1_status.md).
