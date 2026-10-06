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
| **ROCKET** (2602.17951) | VGGT, 10 cặp lớp, projector dùng chung | không | **OpenVLA-OFT** | 98.5 | 81.7 so với baseline 80.0; paper không nói rõ zero-shot, baseline 80.0 trùng mức OFT train trên data LIBERO-Plus (79.5) [S] | công khai, có script SF và baseline cùng ngân sách |
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

- OFT fine-tune trên data LIBERO-Plus (hơn 20.000 trajectory có nhiễu, 100k bước): tổng **79.5**, Camera 92.8,
  Light 94.9, Noise 89.3, nhưng **Robot-init vẫn 30.3** và Layout 77.6 [V]. Nhiễu thị giác chữa được bằng data;
  dịch chuyển trạng thái của robot thì không.
- OFT chỉ có camera third-person đạt 16.8 ở Camera, so với 59.7 khi có thêm camera cổ tay [V] → khớp với 13–20
  của teacher 1-camera mà ta đo.
- 2512.02902 (CVPR 2026) [V]: nhiễu camera / ánh sáng / nền / noise được sửa bằng thích nghi one-shot vài nghìn
  tham số ở encoder ảnh (π0.5: 48.5 → 87–91 ở góc nhìn mới); không đụng tới Robot-init.
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

## 6. Các paper cùng nhánh dựng benchmark thế nào, và mục tiêu đặt cho paper này

Đọc lại tại nguồn 05/10/2026 (arXiv HTML; các mục "không nói" là paper không ghi).

**Nhánh on-policy distill / post-training**: không bài nào claim vượt SOTA tuyệt đối bằng distill trên benchmark
đã bão hoà. Họ tạo headroom bằng một student yếu, lấy teacher làm trần, và claim phần khoảng cách thu hẹp được
cùng chi phí.

| Paper | Cách tạo headroom | Claim chính | Benchmark thứ hai | Robustness | Seed / trial |
|---|---|---|---|---|---|
| VLA-OPD | student 1 demo/task trên 4 suite LIBERO (48.9) | 87.4 so với teacher 93.9; ít bước hơn GRPO 3 lần; ít quên task chưa thấy | RoboTwin 2.0, 4 task (45.2 → 71.1, teacher 74.0) | không có | không nói |
| SimpleVLA-RL | 1 demo/task (48.9 → 96.9) và full data (91 → 99.1) | SOTA LIBERO + hiệu quả khi thiếu data | RoboTwin 1.0 / 2.0, robot thật (sim-to-real) | task chưa thấy | 50 trial/task, chạy 3 lần |
| RIPT-VLA | 1 demo (3.5 → 97.2), hai backbone | QueST +10.9; OFT 96.7 → 97.5 | LIBERO-90, MetaWorld ML45 | nhiễu init state | 3 seed cho đường few-shot |
| WAM-OPD | student một bước, yếu (33.8) | 33.8 → 65.7, giữ tốc độ của student | robot thật | không có | 60 trial/task |

**Nhánh 3D**: mọi bài đều claim mức tăng trên **cùng backbone**, chỉ khác thành phần 3D; LIBERO chuẩn chỉ là
dòng kiểm tra (+1 đến +1.5), trọng lượng nằm ở benchmark robustness / benchmark khó hơn / hiệu quả train / robot
thật.

| Paper | Backbone | LIBERO chuẩn | Phần tạo khác biệt | Benchmark thứ hai | Robot thật |
|---|---|---|---|---|---|
| Spatial Forcing | OFT, π0 | 97.1 → 98.5 | hội tụ nhanh 3.8 lần; 5% data đạt 75.8; probe depth làm động cơ | RoboTwin 2.0 easy / hard | có |
| ROCKET | OFT, π0.5 | 98.5 với khoảng 4% compute của SF | LIBERO-Plus 80.0 → 81.7 | RoboTwin 2.0 | không |
| GaussianWAM | FastWAM, Cosmos Policy | 97.6 / 98.6 | LIBERO-Plus zero-shot 52.05 → 71.29 và 71.52 → 77.30 | RoboTwin 2.0 | có |
| MVUCF | GR00T-N1.6 | 97.4 → 98.9 (3 checkpoint) | LIBERO-Plus zero-shot +22.4 (1 seed) | RoboTwin 38.6 → 61.9 | có |
| FOCAL-VLA | π0.5 | 96.9 → 97.9 | RoboCasa 55.2 → 63.4 | — | có |
| GaussianDream++ | π0.5 | 98.6 | LIBERO-Plus 85.5 → 87.8 | — | — |

**Mục tiêu đề xuất** (số của ta đo trên suite Object, xem week1_status.md):

1. *Setting chính = setting của VLA-OPD* (student 1 demo/task, teacher làm trần), vì đó là nơi headroom có sẵn
   theo cấu trúc và đối thủ trực tiếp là VLA-OPD (Distill 87.4 trung bình; teacher 93.9). Khoảng cách distill →
   teacher của họ nằm ở **Long (78.9 so với 90.7) và Spatial (84.3 so với 94.2)**; Object chỉ còn 2 điểm
   (93.8 so với 96.1). Mốc: trung bình 4 suite ≥ 90 không dùng RL, trong đó Long ≥ 85; baseline OPD không-3D
   của chính ta phải tái lập được trong khoảng 3 điểm quanh số của họ trước đã.
2. *Headline = robustness trên cùng backbone*: LIBERO-Plus zero-shot, điều chưa paper distill nào báo cáo. Hiện
   tại OPD không-3D 42.9, teacher 49.8 / 50.0. Mốc: **student vượt chính teacher của nó** (≥ 55 trên Object, tức
   ≥ +10 so với OPD không-3D), và trong ma trận 2×2 thì 3D trên state của student hơn 3D trên state của teacher
   một khoảng lớn hơn nhiễu (đánh giá cuối trên toàn bộ 2.576 task Object, sai số chuẩn khoảng 1 điểm).
3. *Mở rộng sang backbone SOTA (phần thưởng, không phải điều kiện)*: OFT chuẩn trên LIBERO-Plus zero-shot, so
   với OFT 70.2 và Spatial Forcing 71.9 mà ta đã đo. Đích hợp lý nhất là **Robot-init** (OFT 27, SF 38, kể cả
   OFT train trên data nhiễu cũng chỉ 30.3): ≥ 50 ở Robot-init và ≥ 75 tổng, LIBERO chuẩn không tụt dưới 97.
4. *Để thành paper hoàn chỉnh*: thêm RoboTwin 2.0 (cả hai nhánh đều dùng làm benchmark thứ hai), 3 seed cho các
   so sánh chính, một hình chẩn đoán làm động cơ (như probe depth của Spatial Forcing; của ta là takeover +
   dung sai gắp).

"Vượt SOTA" theo nghĩa các paper này dùng là vượt đối thủ trực tiếp trong cùng setting và cùng backbone. Bảng xếp
hạng tuyệt đối của LIBERO-Plus (lớp π0.5, 85–88) nằm ngoài tầm của backbone OFT và không bài nào trong hai nhánh
lấy làm đích.

## 7. Bổ sung 05/10 (tối): bài gần với hướng "render lại state dưới góc nhìn khác" và với sự bất ổn của distill

- **S2VOPD (2608.14144)** [V]: on-policy distillation cho VLM, bất đối xứng dựng từ đầu vào: teacher (EMA) nhìn
  ảnh gốc, student nhìn ảnh bị giảm chất lượng (thu nhỏ 0.3–0.6×, nhiễu Gauss); Qwen 4B 70.7 → 77.4 trên các bench
  perception. Không có robot, không có đổi góc nhìn. Là họ hàng gần nhất về cơ chế "teacher nhìn sạch, student
  nhìn bẩn"; phần của ta khác ở chỗ nhiễu góc nhìn cần cảnh 3D và ở vòng điều khiển kín.
- **InfiNoVA (2609.27734)** [V, chỉ abstract]: dựng mỗi demo thành Gaussian 4D rồi render góc nhìn mới để tăng
  cường **demo** (offline, giữ nguyên action); robot thật, 5.4× so với không tăng cường dưới góc nhìn lạ. Không có
  teacher, không có state on-policy.
- **GS-VLA (2608.19066)** [V] và **AnyCamVLA (2603.05868)**: chuẩn hoá góc nhìn **lúc test** (render ảnh về góc
  camera lúc train rồi mới đưa cho policy đóng băng); cần depth đã hiệu chuẩn và pose camera lúc chạy. π0.5 trên
  LIBERO-Spatial khi dời camera 1 m: 42.6 → 86.8. Hướng của ta không cần gì thêm lúc test.
- **AnyViewDex (2609.20107)**: distill teacher–student cho thao tác khéo léo bất biến góc nhìn; distill giữ được
  61–72% năng lực teacher so với 33–49% nếu chỉ domain randomization.
- **2609.35259** (động học của distill on/off-policy, LLM) [V]: reverse-KL nhạy với policy sinh rollout và với
  learning rate, đôi khi sụp ở lr cao; **learning rate quyết định mức quên**; forward-KL bền hơn. Khớp với hiện
  tượng sụp theo task của B2 / B4 ở lr cố định 1e-4 → thử giảm lr theo cosine trước khi kết luận gì về khoảng
  cách 86 so với 93.8.
- Benchmark có thể chuyển sang nếu cần: **LIBERO-VPro (2609.24350)** (robustness thị giác vòng kín), **RoboRecover
  (2609.28952)** (phục hồi sau lệch thực thi; liên quan trực tiếp tới kết quả takeover "sau lần kẹp hỏng thì không
  cứu được"), RoboTwin 2.0 (benchmark thứ hai của hầu hết các bài cùng nhánh).

## 8. Bổ sung 06/10: các bài đã làm "render lại cùng state" và "distill phụ thuộc thị giác"

- **Cross-View Action Consistency (2608.06965)** [V, abstract]: reset MuJoCo về đúng state của **demo** LIBERO, render góc
  camera chuẩn và góc nhiễu, loss nhất quán giữa vận tốc flow của hai góc (VLA flow-based, không teacher). LIBERO-Plus
  nhánh Camera: 79.8 → **87.2** (3 seed); đối chứng ghép cặp ngẫu nhiên sụp còn 25.8. Robot thật: 53.3 → 74.4 ở camera
  lạ. Không đụng Robot-init / Layout. → "cùng state vật lý, render lại góc khác" đã có người làm trên demo.
- **Grounding Actions in Camera Space (2508.13103)**, **From Fixed to Free Cameras (2607.05396)**: render nhiều góc camera
  cho mỗi trajectory demo; VistaBot (2604.21914), VISTA (2409.03685; InfiNoVA báo VISTA gần như không phục hồi được).
- **VA-OPD (2605.21924)** và **CW-OPD (2609.38777)** [V, abstract], cho VLM: on-policy distillation chuẩn làm student
  khớp câu trả lời của teacher **mà không tăng mức phụ thuộc vào bằng chứng thị giác**; CW-OPD dựng cặp "thế giới" chỉ
  khác ở bằng chứng quyết định câu trả lời và distill **sự thay đổi** của teacher giữa hai thế giới. Chưa có bản cho
  VLA / action. Khớp với failure analysis của ta (hình học có trong feature nhưng action không dùng; phát lại quỹ đạo).
- **Distilling Realizable Students from Unrealizable Teachers (2505.09546)**, **Student-Informed Teacher Training
  (ICLR 2025)**: distill có bất đối xứng thông tin teacher–student là chủ đề đã chín trong robot learning; reviewer sẽ
  đọc "teacher nhìn sạch, student nhìn nhiễu" là privileged distillation + domain randomization.

## 9. Bổ sung 06/10 (chiều): vấn đề "phát lại quỹ đạo" đã được đặt tên và chữa một phần; bảng LIBERO-Plus thật

Vấn đề ta chẩn đoán (hình học có trong feature nhưng action đi theo thói quen / proprio) đã được nhiều bài 2026 nêu:
- **Memory Trap** (Affordance Field Intervention, CVPR 2026): VLA phát lại quỹ đạo đã thuộc thay vì thích nghi với cảnh
  mới; can thiệp lúc test bằng trường affordance.
- **Perturbot (2610.04616)** [V]: "shortcut priors" (salience capture, noun lock-in, motor inertia), đo bằng GroundFscore
  (cặp sửa null / sửa nhân quả); chữa bằng can thiệp dữ liệu (distractor, caption chi tiết, relabel đoạn hỏng); π0.5
  robot thật 42 → 84. Không dùng state mô phỏng, không on-policy.
- **When Instructions Retrieve Trajectories (2609.39971)** [V]: 69% failure là đi theo quỹ đạo quen; proprio/trạng thái
  nội bộ dẫn dắt việc chọn quỹ đạo. Chữa bằng **Equivariant Counterfactual Training (ECT)**: cặp phản thực
  "cùng câu lệnh, cảnh khác, action khác" dựng bằng biến đổi hình học demo (đối xứng, dịch) rồi replay trong simulator;
  loss BC trên cả hai. LIBERO-PRO Object swap 38.3 → 70.8; robot thật 3/40 → 35/40. **Đây gần như là ý "cặp thế giới"
  của ta, trên demo thay vì state on-policy, không teacher.**
- **Motion-Centric Action Frames (2605.11809)**: token proprio gây lối tắt state → action; thay bằng khung action theo
  chuyển động. **Direct Action-Head Injection of a Grounded 3D Point (2606.27663)**: tiêm một điểm 3D đã grounding vào
  action head, đảo chiều sụp đổ dưới nhiễu vị trí của LIBERO-PRO.

**Bảng LIBERO-Plus (4 suite) trong PatchWAM (2609.25961, Table 5)** [V]: π0 68.8 (Robot 21.1), OpenVLA*-Full 73.0
(49.6), π0.5 77.4 (41.7), π0.5 + AXIS 89.5 (78.2), CAC-VLA 90.1 (78.4), **QuoVLA 90.3 (Robot 87.6, zero-shot)**,
Anchor-Align 90.8 (59.1), Hermite-VLAReg 91.4 (85.4), PatchWAM 91.8 (71.0; train trên data nhiễu), ABot-M0.5 83.4
(87.4). Tức SOTA thật trên LIBERO-Plus là **90–92 tổng, 85–88 ở Robot-init**, không phải 70 của OFT. Các số này
không phân biệt rõ zero-shot / có data nhiễu cho mọi dòng; QuoVLA được ghi là zero-shot.

Hệ quả: (1) "Robot-init chưa ai lấy được" chỉ đúng trong họ OFT; ở các backbone flow / WAM đã lên 78–88. (2) Mọi claim
"vượt SOTA" trên LIBERO-Plus phải so với 90+, ngoài tầm của OFT 7B một camera / hai camera. (3) Chỗ chưa ai làm: cặp
phản thực dựng **trên state on-policy** (nơi failure được quyết định, theo takeover) với nhãn teacher, và cho post-training
của một checkpoint có sẵn; ECT làm trên demo, VLA-OPD on-policy nhưng không phản thực.
