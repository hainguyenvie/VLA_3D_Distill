# Bản đồ related work và hệ quả cho thiết kế thí nghiệm

Tổng hợp 05/10/2026 từ việc đọc trực tiếp arXiv / GitHub / HuggingFace. Ký hiệu: **[V]** đã đọc tại nguồn,
**[S]** suy luận từ bằng chứng gián tiếp, **[?]** chưa kiểm được. Số trong bảng là số paper tự báo cáo.

## 1. Năm điều làm thay đổi kế hoạch tuần 1

1. **VLA-OPD không có code, không có checkpoint, không có phụ lục** [V]. Repo `IRPN-LAB/VLA-OPD` chỉ có trang
   project ("Code (Coming Soon)"). Learning rate, optimizer, LoRA hay full fine-tune, nhiệt độ rollout, số
   trial/seed đánh giá đều không công bố. → B2 của ta là **reimplementation**, phải ghi rõ mọi sai khác.
2. **VLA-OPD dựng trên SimpleVLA-RL** [S, bằng chứng mạnh]: teacher là SimpleVLA-RL, B=64/G=8 "following
   SimpleVLA-RL", dòng student-init 48.9 trùng từng số với Table 5 của SimpleVLA-RL, tên metric trong hình
   trùng veRL. Student vì thế là biến thể **OpenVLA-OFT token rời rạc** (1 ảnh third-person, không wrist,
   không proprio, chunk 8, 7×8 token, 256 bin), **không phải** checkpoint OFT chính thức (L1 head, 2 ảnh).
3. **Checkpoint dùng được ngay** [V]: student 1-traj và full-SFT của SimpleVLA-RL cho cả 4 suite
   (`Haozhan72/Openvla-oft-SFT-libero{10,-spatial,-object,-goal}-{traj1,trajall}`); teacher RL chỉ có cho
   Long (`Haozhan72/openvla-oft-libero10-traj1-rl`); **RLinf có teacher GRPO cho cả 4 suite**
   (`RLinf/RLinf-OpenVLAOFT-GRPO-LIBERO-{spatial,object,goal,long}`), huấn luyện từ chính các checkpoint
   traj1 đó.
4. **LIBERO chuẩn đã bão hoà** (97–99%). Headroom thật nằm ở (a) setting few-shot của VLA-OPD (student
   48.9 → distill 87.4 → teacher 93.9) và (b) **LIBERO-Plus zero-shot**: OpenVLA-OFT 69.6, riêng
   *Robot-init* 31.9 và *Camera* 56.4. Robot-init là dịch chuyển sang trạng thái ngoài expert **theo định
   nghĩa**, tức đúng chỗ giả thuyết của ta phải tạo ra khác biệt.
5. **Chưa thấy ai làm 3D supervision trên trạng thái on-policy cho VLA** (hai lượt tìm độc lập, khoảng 20
   truy vấn). Mọi paper geometry distillation hiện có đều giám sát trên frame demo.

## 2. Nhánh on-policy distillation / RL post-training

**VLA-OPD (2603.26666)** [V trừ khi ghi khác]
- Chỉ student hành động; teacher gán nhãn mọi trạng thái student đi qua, không thực thi; không trộn β, không
  gộp dữ liệu kiểu DAgger; mỗi vòng dùng rollout mới.
- Loss là policy gradient với reward reverse-KL một mẫu trên token student lấy mẫu:
  `r_t = −(log π_S(a_t|s_t) − log π_T(a_t|s_t))`, dùng thẳng làm advantage (không chuẩn hoá GRPO).
  Có giữ PPO clip hay không: không nói.
- Ablation: forward-KL sụp sớm (entropy nổ), hard-CE kẹt thấp (entropy sụp); chạy trên RoboTwin, không
  phải LIBERO.
- Ngân sách: B=64, G=8 → khoảng 512 trajectory mỗi bước [S]. Object đạt 93.8 sau **20 bước**, Long 78.9 sau
  **50 bước** (đọc từ hình, ±1).

| LIBERO (Table 2) | Spatial | Object | Goal | Long | Avg |
|---|---|---|---|---|---|
| Teacher | 94.2 | 96.1 | 94.6 | 90.7 | 93.9 |
| OpenVLA-OFT 1-traj (student init) | 63.6 | 54.9 | 59.6 | 17.3 | 48.9 |
| VLA-OPD (Distill) | 84.3 | 93.8 | 92.5 | 78.9 | 87.4 |
| VLA-OPD (Distill + GRPO) | 93.4 | 95.3 | 94.5 | 90.2 | 93.4 |

- Dòng student-init được **chép** từ SimpleVLA-RL; đường cong của chính họ bắt đầu ở khoảng 49.5 trên
  Object. Teacher 93.9 không khớp checkpoint công khai nào → có lẽ là lần chạy GRPO riêng của họ [S].

**SimpleVLA-RL (2509.09674, ICLR 2026)** [V]
- 1-traj SFT 63.6 / 54.9 / 59.6 / 17.3 (48.9); +RL 98.2 / 98.7 / 98.8 / 91.7; full SFT 91.6 / 95.3 / 90.6 /
  86.5; full + RL 99.1.
- Rollout lấy mẫu T=1.6, đánh giá greedy, 50 trial mỗi task. Train và val dùng **cùng** init state.
- Hạ tầng nặng: full-parameter, 8×A800 80 GB, khoảng 45 phút mỗi bước; có báo cáo OOM trên 8×48 GB và OOM
  RAM của Ray trên 8×L40. → không chạy nguyên bản trên 2×L40; ta viết vòng lặp gọn riêng.

**RLinf** [V]: teacher GRPO đạt Object 97.68 / Spatial 94.76 / Goal 93.96 / Long 90.93 (đánh giá có lấy mẫu
T=1.6). Cùng model card đo student traj1 ở Object **28.83**, Spatial 52.22, Goal 49.40, Long 14.92, thấp hơn
nhiều so với 54.9 của SimpleVLA-RL → **số student-init phụ thuộc mạnh vào protocol**; ta phải tự đo và cố
định protocol của mình.

**Khác** [V]: WAM-OPD (2608.22364; on-policy cho world-action model, RoboTwin, không có LIBERO); RIPT-VLA
(2505.17016; OFT 96.7→97.5; **LIBERO-Plus 68.4, không hơn OFT 69.6** → on-policy chỉ với tín hiệu action
không mua được robustness); PLD (2511.00091); πRL (2510.25889); ROAD-VLA (2606.25800).

## 3. Nhánh distill 3D / geometry vào VLA

| Phương pháp | Tín hiệu 3D | Giữ lại lúc inference | Backbone | LIBERO | LIBERO-Plus | Code |
|---|---|---|---|---|---|---|
| **Spatial Forcing** (2510.12276) | feature VGGT ↔ token ảnh lớp 24/32, cosine, α=0.5 | không | **OpenVLA-OFT** | 98.5 | không có số đáng tin | code + checkpoint công khai |
| **ROCKET** (2602.17951) | VGGT, 10 cặp lớp, projector dùng chung | không | **OpenVLA-OFT** | 98.5 | 81.7 (train trên data LIBERO-Plus, không zero-shot) | công khai, có script SF và baseline cùng ngân sách |
| GaussianWAM (2608.24714) | depth + semantic + alpha render từ trường Gaussian, head trên token lớp cuối, trọng số 0.01/0.01/0.005 | không | FastWAM / Cosmos | 97.6 / 98.6 | **52.05 → 71.29**; direct CLIP+VGGT 69.37 | repo chỉ có README |
| FOCAL-VLA (2609.21228) | feature VGGT pool theo vùng liên quan subtask | 8+8 query token | π0.5 | 97.9 | không có | không |
| MVUCF (2608.01826) | **depth GT + calibration** | không | GR00T-N1.6 | 98.9 | 42.8 → 65.2 | không |
| GaussianDream++ (2608.25659) | tái dựng 3DGS + world token | token | π0.5 | 98.6 | 87.8 (π0.5 gốc 85.5) | một phần |

Chi tiết cần nhớ:
- GaussianWAM: trường Gaussian chỉ thêm **+1.9** so với distill trực tiếp CLIP+VGGT (một lần chạy);
  biến thể bỏ alpha còn hơn bản đầy đủ ở Camera và Robot. Phần lớn lợi ích đến từ bản thân tín hiệu
  geometry/semantic, không phải cách tổ chức thành Gaussian.
- FOCAL-VLA: bản "chỉ supervision" (query token không nối vào action expert) 58.5 so với 63.4 trên RoboCasa →
  định hình representation thuần cho lợi ích khiêm tốn; và "chỉ distill vùng liên quan" đã có người làm
  theo nghĩa **vùng ảnh**, chưa ai làm theo nghĩa **vùng không gian trạng thái**.
- GLaD trên LIBERO-PRO: distill VGGT offline cải thiện nhiễu vật thể nhưng **nhiễu vị trí vẫn gần 0** →
  bằng chứng gián tiếp rằng 3D offline không chữa được dịch chuyển trạng thái.
- 2608.08904: fine-tune action làm **giảm** khả năng giải mã depth ở các lớp sau của VLA → động cơ cho việc
  giữ tín hiệu 3D trong lúc post-training.
- TRACE-GS (2608.10286): "on-policy distillation với privileged geometry" nhưng cho khôi phục 3DGS bằng
  diffusion, không phải robot → nên trích dẫn như họ hàng gần về lập luận.

## 4. Benchmark có headroom

**LIBERO-Plus (2510.13626)** [V]: 10.030 task nhiễu, chỉ để test, 7 chiều; thay thế trực tiếp LIBERO, đánh
giá 1 trial mỗi task.

| OpenVLA-OFT | Camera | Robot | Lang | Light | BG | Noise | Layout | Total |
|---|---|---|---|---|---|---|---|---|
| 4 suite | 56.4 | **31.9** | 79.5 | 88.7 | 93.3 | 75.8 | 74.2 | 69.6 |
| suite Object | 38.9 | 25.4 | 99.0 | 73.7 | 97.6 | 72.3 | 71.8 | 66.5 |
| suite Long | 38.7 | 38.2 | 87.0 | 89.4 | 86.8 | 63.5 | 76.9 | 66.4 |

- Bẫy đã biết: issue #64, `task.language` lấy từ tên file biến thể nên id nhiễu bị dính vào prompt ở 4
  chiều (sửa xong Camera tăng 6.5 điểm trên mẫu con 40%). Phải sửa và ghi rõ.
- Chưa có bảng zero-shot sạch nào cho {OFT, Spatial Forcing, ROCKET} trên cùng protocol; tự nó đã là đóng góp.
- Trần tuyệt đối khoảng 88 (lớp π0.5) ngoài tầm OFT; mọi paper nhánh này đều claim **mức tăng trên cùng
  backbone**.

**LIBERO-PRO (2510.03827)**: nhiễu vị trí gần 0 với hầu hết model; khoảng trống lớn nhất nhưng rủi ro cao.

## 5. Hệ quả cho thiết kế

1. **Backbone và protocol**: theo đúng dòng SimpleVLA-RL / VLA-OPD (OFT token rời rạc, 1 ảnh, chunk 8) để B0
   và B2 so được với số đã công bố. Bắt đầu ở LIBERO-Object (đường cong VLA-OPD ngắn nhất: 20 bước).
2. **B2 (OPD)**: tự viết vòng lặp gọn. Vì parallel decoding cho phân phối tích trên 56 token, reverse-KL
   **tính được dạng đóng** từ logits hai bên; gradient của nó bằng kỳ vọng của estimator một mẫu trong
   paper, phương sai thấp hơn. Cài cả hai, ghi rõ sai khác (LoRA thay vì full-parameter, ngân sách nhỏ hơn).
3. **Tín hiệu 3D mức A (oracle)**: depth + camera của simulator, head nhẹ trên token ảnh (kiểu GaussianWAM /
   MVUCF). Mức B: feature VGGT kiểu Spatial Forcing (cùng backbone OFT, có code tham chiếu).
4. **Thước đo chính ngoài success rate**: LIBERO-Plus zero-shot, đặc biệt Robot-init, Camera, Layout.
5. **Chẩn đoán rẻ trước khi huấn luyện**: probe tuyến tính depth trên feature của student, so trạng thái
   expert với trạng thái student tự đi tới, và tương quan sai số probe với KL teacher–student. Nếu khả năng
   giải mã 3D sụt đúng ở chỗ student bất đồng với teacher thì có intuition trước khi tốn GPU cho B3/B4.
6. **Rủi ro phải kiểm soát**: lợi ích của định hình representation thuần có thể nhỏ; so sánh B3 và B4 cần
   cùng ngân sách và nhiều seed; đánh giá trên cùng loại GPU với lúc huấn luyện (cảnh báo của OFT/SF).
