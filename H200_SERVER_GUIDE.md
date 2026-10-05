# Máy chủ — hướng dẫn dùng cho project VLA 3D Distill (2×L40, H200 khi được nhường)

> **File duy nhất về máy chủ của project này** (On-Policy 3D / Gaussian Distillation for VLA). Viết lại
> 05/10/2026 từ bản của project `3dgs_in_refiner`, sau khi probe trực tiếp máy 2×L40. Các bài học chung
> của operator (§8–§12) được giữ nguyên vì vẫn đúng.
> Alias ssh, host, IP, port, tên tài khoản, đường dẫn tuyệt đối **không** nằm trong file này: chúng ở
> `infra.env` (gitignored, mẫu ở `infra.env.example`) và `~/.ssh/config` của operator. Repo có thể public.

**Mục lục** — [0 Hiện trạng](#0-hiện-trạng-0510) · [1 Truy cập](#1-truy-cập-và-đồng-bộ-code) ·
[2 Workspace](#2-workspace-trên-máy-2l40) · [3 Tài khoản dùng chung](#3-máy-2l40-là-tài-khoản-dùng-chung) ·
[4 Luật GPU](#4-luật-dùng-gpu) · [5 Hai GPU](#5-dùng-cả-hai-gpu) ·
[6 Môi trường](#6-môi-trường-python--mô-phỏng) · [7 Data và trọng số](#7-dataset-và-trọng-số) ·
[8 Chạy job](#8-chạy-job-và-theo-dõi) · [9 Bẫy](#9-bẫy-đã-dẫm-phải) · [10 Dọn dẹp](#10-dọn-dẹp) ·
[11 Kỷ luật](#11-kỷ-luật-thí-nghiệm) · [12 Backup](#12-backup) · [13 Checklist](#13-checklist-mở-đầu-mỗi-phiên)

## 0. Hiện trạng (05/10)

| | **Máy chính: 2×L40** | **Máy H200** |
|---|---|---|
| Biến trong `infra.env` | `L40_SSH`, `L40_REMOTE_ROOT` | `H200_SSH`, `H200_REMOTE_ROOT` (để trống) |
| Trạng thái với project này | **dùng hằng ngày** | **chưa được dùng**: đang thuộc project khác của operator |
| GPU | 2 × L40 46 GB (driver 580), hai socket, hai NUMA | 1 × H200 141 GB |
| CPU / RAM | 96 core, 125 GB; người khác dùng khoảng một nửa (load ≈ 24) | 22 core, 235 GB |
| Đĩa | 838 GB, còn **189 GB** (05/10); workspace project khác của operator chiếm 169 GB | 2,9 TB |
| Tài khoản | **dùng chung** với nhiều người (home có thư mục của họ) | root |
| Shell / giờ | **zsh**, **UTC+7** | bash, UTC |
| Có sẵn | git, tmux, rsync, wget, curl, docker, miniconda ở home; EGL của NVIDIA; **không có ffmpeg** | — |

**Quy tắc về H200:** mặc định project này chỉ chạy trên 2×L40. Khi một thí nghiệm không vừa VRAM
(46 GB mỗi card) hoặc cần chạy lớn, **báo operator** kèm ước lượng VRAM và thời gian; operator sẽ bảo
project kia nhường card. Không tự ý ssh lên máy H200 để chạy.

## 1. Truy cập và đồng bộ code

```bash
. ./infra.env
ssh $L40_SSH
scripts/sync.sh          # rsync repo local -> $L40_REMOTE_ROOT/repo (có --delete, bỏ qua mọi thứ trong .gitignore)
```

- **Đăng nhập bằng key** (alias trong `~/.ssh/config`). Mật khẩu từng được gửi qua chat thì **không lưu
  ở đâu cả**, không ghi vào file, log hay memory.
- **Tuyệt đối không** chép nội dung `~/.ssh/config`, IP, port, tên host, tên tài khoản, đường dẫn tuyệt
  đối của home, token hay SSH key vào repo, commit message hay output. Quét **trước** khi push (git
  history không xoá được):

```bash
git grep -niE "([0-9]{1,3}\.){3}[0-9]{1,3}|BEGIN .*PRIVATE|hf_[A-Za-z0-9]{20,}|/home/[a-z]+/|/root/projects"
```

- HF token chỉ nằm trong HF cache của máy cần nó; không copy sang máy khác, không đưa vào log.
- **Local là nguồn sự thật, server chỉ là nơi chạy.** Sửa code ở local rồi `scripts/sync.sh`; không sửa
  code trực tiếp trong `repo/` trên server (lần sync sau sẽ ghi đè).
- `L40_REMOTE_ROOT` ghi **tương đối so với home trên server** (không có `~`): `~` trong `infra.env` bị
  shell local nội suy thành home của laptop và rsync sẽ ghi nhầm chỗ.
- Kết quả nhẹ (json, csv, bảng) kéo về local bằng `rsync`/`scp` sau mỗi phiên rồi commit.

## 2. Workspace trên máy 2×L40

Một thư mục riêng cho project, **ngang hàng** với thư mục của project khác của operator:

```text
$L40_REMOTE_ROOT/
├── repo/          bản sao rsync của repo git này (code, configs, scripts) — chỉ ghi qua scripts/sync.sh
├── envs/          conda env (envs/oft)
├── checkpoints/   hf/ (HF_HOME), torch/
├── data/          dataset (RLDS, hdf5), tfds
├── third_party/   repo baseline clone nguyên bản (openvla-oft, LIBERO, SimpleVLA-RL, ...)
├── outputs/       checkpoint huấn luyện, rollout, video, kết quả thô
├── logs/          log của từng job
├── scratch/       script nháp, thử nghiệm một lần
├── .cache/ .libero/   cache pip / config LIBERO (không ghi ra home dùng chung)
```

Mọi script job trên server bắt đầu bằng `. "$(dirname "$0")/env.sh"` (`scripts/server/env.sh`): nó đặt
`W` (workspace), `REPO`, `PY`, `HF_HOME`, `LIBERO_CONFIG_PATH`, `MUJOCO_GL=egl`, `PYTHONPATH`. Nhờ vậy
không có đường dẫn tuyệt đối nào trong repo.

## 3. Máy 2×L40 là tài khoản dùng chung

- **Làm việc trong đúng thư mục** `$L40_REMOTE_ROOT`. Không đọc hay sửa thư mục, key, docker container
  của người khác, kể cả workspace project khác của operator. Home có private key quyền rộng của người
  khác: không đụng vào.
- **Người khác chạy job GPU trên cùng máy** (05/10: một job PyTorch của người khác giữ khoảng 14,6 GB
  trên card 1 đã hơn một ngày; từng thấy job tăng từ 7 lên 13 GB trong lúc mình chạy). Tài khoản có
  `sudo`, nhưng **không kill job của người khác** trừ khi operator cho phép rõ ràng *cho lần đó*. Khi cần
  dọn card, đưa lệnh để operator tự chạy.
- **zsh**: glob không khớp sẽ **dừng cả lệnh** (`zsh: no matches found`). Bọc glob trong `ls … 2>/dev/null`
  hoặc nháy, đặc biệt trong chuỗi lệnh `ssh '…'`.
- Cấu hình mặc định của nhiều thư viện ghi vào home (`~/.libero`, `~/.cache`, `~/.keras`): home là của
  chung, nên luôn chuyển hướng vào workspace bằng biến môi trường (`env.sh` đã làm).
- CPU 96 core nhưng load nền khoảng 24: rollout mô phỏng song song nên giới hạn ở khoảng 16–24 tiến trình.

## 4. Luật dùng GPU

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader   # PID nào giữ card
ps -o pid,user,etime,args -p <pid>                                        # của ai
```

- Chọn card bằng `CUDA_VISIBLE_DEVICES=<i>`, trong code dùng `cuda:0`; `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  (đã đặt trong `env.sh`). MuJoCo render bằng EGL trên GPU: đặt `MUJOCO_EGL_DEVICE_ID` trùng với card đã
  chọn.
- Chừa chỗ cho job của người khác tăng bộ nhớ: kiểm `--query-compute-apps` ngay trước khi chạy, chừa
  ≥ 5 GB trên card đang có người dùng.
- Job dài phải chịu được OOM giữa chừng: ghi kết quả từng episode (jsonl), chạy lại thì **resume** từ id
  đã xong; training lưu checkpoint định kỳ và tự resume.
- Cổng chờ VRAM cho job dài:

```bash
G=0; vgate() { until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)" -le 1000 ]; do sleep 120; done; }
```

- **Khi nào xin H200:** batch huấn luyện không vừa 46 GB dù đã giảm batch + gradient accumulation; cần
  chạy nhiều seed song song cho bảng kết quả cuối; hoặc card 1 bị người khác giữ lâu làm nghẽn tiến độ.

## 5. Dùng cả hai GPU

**P2P giữa hai GPU của máy này hỏng âm thầm** (đo ở project trước: `x.to("cuda:1")` với tensor bf16 cỡ
hidden state cho kết quả sai 20/20 lần; model chia bằng `device_map` ra logits toàn 0 hoặc NaN mà không
báo lỗi). Hệ quả cho project này:

- Ưu tiên **mỗi card một job độc lập** (ví dụ card 0 huấn luyện, card 1 rollout/eval; hoặc mỗi card một
  seed). Đây là cách dùng hai card an toàn nhất.
- Không chia một model qua hai card bằng `device_map`. Nếu buộc phải chia: mọi lần chuyển GPU↔GPU đi qua
  CPU, và kiểm logits hữu hạn, khác 0, khớp bản một GPU trên một input nhỏ.
- DDP hai card (NCCL) **chưa được kiểm trên máy này**. Trước khi tin kết quả DDP: đặt
  `NCCL_P2P_DISABLE=1`, rồi so loss vài bước đầu với bản một GPU cùng seed và cùng batch hiệu dụng.

## 6. Môi trường Python + mô phỏng

- Một env chính `envs/oft` (conda, python 3.10, đặt trong workspace), dựng bằng
  `scripts/server/setup_env.sh` (chạy lại được, bước đã xong thì bỏ qua). Gọi thẳng `$PY`
  (`envs/oft/bin/python`), **không `activate`** trong script.
- Phiên bản ghim theo OpenVLA-OFT: torch 2.2.0 (cu121, tự kèm CUDA runtime), transformers bản fork của
  OpenVLA-OFT, robosuite 1.4.1 + LIBERO. Baseline nào xung đột phụ thuộc thì tạo env riêng trong `envs/`.
- LIBERO hỏi tương tác ở lần import đầu nếu chưa có config: `setup_env.sh` tạo sẵn
  `$LIBERO_CONFIG_PATH/config.yaml` trong workspace.
- Render headless: `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` (máy có `libEGL_nvidia`). Không có màn hình
  và không có ffmpeg hệ thống: ghi video bằng `imageio[ffmpeg]` trong env.
- Thiếu gói lặt vặt khi chạy baseline mới → cài vào env của project, không cài global, không `sudo apt`.

## 7. Dataset và trọng số

- Chỉ dùng **bản public chính thức**, kiểm được bằng byte hoặc sha256. Ghi model ID + revision vào
  `notes/reproduction_log.md` ngay khi tải.
- `HF_HOME=<workspace>/checkpoints/hf`. HF mới dùng xet: thư mục model chỉ chứa symlink sang `hub/blobs`,
  nên `du` thư mục model ra rất nhỏ. Kiểm bằng `df` hoặc `du -L`.
- **Ngân sách đĩa:** máy còn 189 GB và dùng chung. Một checkpoint OpenVLA-OFT khoảng 15 GB → không tải
  cả bộ checkpoint mọi suite cùng lúc; tải theo suite đang chạy, xoá (qua thùng rác, §10) khi xong.
  Checkpoint huấn luyện chỉ giữ LoRA/adaptor + head khi có thể. Kiểm `df -h ~` trước mỗi lô lớn.
- **Repo gated** trả 401 khi không có token: tìm mirror chính thức, tải thẳng trên máy đích, rồi so
  sha256 với metadata LFS của HF trước khi dùng.
- Tốc độ tải trên máy này: HF khoảng 20 MB/s. Copy file GB giữa hai server qua laptop rất chậm
  (từng đo 144 KB/s) → tải thẳng trên máy đích.

## 8. Chạy job và theo dõi

```bash
mkdir -p logs
setsid nohup bash repo/scripts/server/<job>.sh > logs/<job>.log 2>&1 < /dev/null &
```

1. **`setsid … < /dev/null &`**, không phải `nohup … & disown`: ssh đóng thì job con chết theo.
2. **Phiên ssh dài hay bị rớt (exit 255).** Đừng chờ job trong một lệnh ssh dài. Cách bền: job chạy
   tách hẳn trên server, còn phía local dùng **vòng lặp nhiều phiên ssh ngắn**:

```bash
until ssh -n -o ConnectTimeout=20 $L40_SSH "grep -q JOB_DONE $L40_REMOTE_ROOT/logs/job.log" 2>/dev/null; do sleep 60; done
```

3. **Viết script ở local rồi sync lên**, đừng dựng script bằng heredoc trong chuỗi `ssh '…'`. Nháy lồng
   nhau và f-string Python có `\"` trong `python -c` đều từng gây SyntaxError; chuỗi lệnh dài còn dễ tự
   khớp với `pgrep` (§9).
4. **Không sửa script bash đang chạy**: bash đọc file dần dần, và `scripts/sync.sh` ghi đè `repo/`. Muốn
   đổi thì viết script mới (tên khác), dừng script cũ theo PID. Job python con vẫn sống sau khi bash cha
   bị kill, vì chúng được init nhận làm con.
5. **Chuỗi job trên GPU**: mỗi card một hàng đợi tuần tự; bước sau chờ bước trước bằng
   `while kill -0 <PID> 2>/dev/null; do sleep 30; done`.
6. **Guard tái chạy trên file cuối**, cổng rẻ trước bước đắt (chạy thử vài episode / vài chục step rồi
   mới chạy đủ), `python -u`, in tiến độ định kỳ.

## 9. Bẫy đã dẫm phải

| Triệu chứng | Nguyên nhân | Cách xử |
|---|---|---|
| ssh của mình tự chết (exit 255) khi dừng job | `pkill -f`, `pgrep -f` hay `ps \| grep "[b]ash x"` khớp chính chuỗi lệnh ssh hoặc zsh của mình (đã xảy ra nhiều lần) | liệt kê rồi kill **theo PID cụ thể**, từng PID một |
| rsync ghi vào `/home/<local-user>/…` trên server | `~` trong `infra.env` bị shell local nội suy | ghi `L40_REMOTE_ROOT` tương đối so với home, không có `~` |
| Model chia 2 GPU ra logits 0 / NaN / token vô nghĩa | P2P GPU↔GPU hỏng trên máy 2×L40 | mỗi card một job; nếu phải chia thì chuyển qua CPU (§5) |
| OOM khi nạp model dù card trống lúc kiểm | job người khác tăng bộ nhớ, hoặc chính mình chạy 2 process cùng card | kiểm `--query-compute-apps` ngay trước khi chạy; chừa ≥ 5 GB |
| Một job "chạy" hàng giờ mà không tiến | NaN hoặc vòng lặp không hội tụ | kiểm vài dòng output đầu tiên trước khi bỏ đi |
| Job nền chết khi ssh đóng | không tách session | `setsid nohup … < /dev/null &` |
| `zsh: no matches found` làm hỏng cả chuỗi lệnh | glob rỗng trên zsh | `ls glob 2>/dev/null`, hoặc bỏ glob |
| Hai process ghi cùng một file jsonl | chạy lại một shard ở card khác trong khi hàng đợi cũ vẫn sẽ chạy nó | dừng hàng đợi cũ theo PID trước, hoặc ghi ra file khác rồi gộp theo id |
| File tạm bị ghi đè | nhiều process cùng ghi một tên file tạm | tên file tạm có `os.getpid()`, rồi `os.replace` |
| `uint32_t` undeclared khi build extension | gcc 13 | `CXXFLAGS="-include cstdint"` |
| Run hỏng tưởng hoàn tất | guard chỉ kiểm file trung gian | kiểm file cuối và số dòng hoặc id |
| `pip … \| tail` báo OK nhưng thiếu module | pipe nuốt lỗi build | không pipe output pip |
| `$VAR` rỗng trong script sinh bằng heredoc | nội suy sớm | heredoc `<<'EOF'` (hoặc viết local rồi sync) |
| `OverflowError` khi import gsplat / pycolmap | numpy 2 | ghim `numpy==1.26.4` trong env 3DGS |
| Số lệch khoảng 1e-4 dù tính tất định | TF32 | tắt TF32 khi so fp32 |
| Kết quả trên hai máy lệch nhau | code hai đầu khác phiên bản | đồng bộ bằng `scripts/sync.sh`, so md5 |

## 10. Dọn dẹp

**Đi qua thùng rác, đừng `rm -rf` thẳng:**

```bash
T=<workspace>/_trash_$(date +%Y%m%d); mkdir -p $T; mv <thứ-cần-bỏ> $T/    # kiểm vẫn chạy rồi mới rm -rf $T
```

| | dựng lại được | **không** dựng lại được |
|---|---|---|
| **nhẹ** | log | **code, plan, config, report, bảng kết quả theo từng episode** → git / local |
| **nặng** | checkpoint tải về, dataset, env, cache target 3D | checkpoint tự huấn luyện, rollout on-policy đã thu (giữ đến khi có report) |

Gom script nháp vào `scratch/` trong workspace, đừng để rải ở gốc.

## 11. Kỷ luật thí nghiệm

1. Đóng băng plan, gate và split **trước** khi chạy; không nới gate sau khi thấy số.
2. Xác nhận job xong bằng **artefact** (đủ số episode, đủ id), không bằng exit code.
3. Siêu tham số chọn trên **tập task / seed dev**, số báo cáo chạy trên seed đánh giá cố định; mọi
   phương pháp trong một bảng dùng cùng số episode và cùng init state.
4. Kiểm protocol bằng cách tái tạo số baseline đã công bố trước khi so (checkpoint gốc khớp số paper).
5. Nghi harness (chuẩn hoá action, thứ tự camera, xoay ảnh, quy ước gripper, seed) **trước** khi kết
   luận hiện tượng; success rate gần 0 với checkpoint công khai gần như luôn là lỗi tiền xử lý.
6. "CI chứa 0" ≠ "không có hiệu ứng"; 50 episode một task có sai số chuẩn khoảng 7 điểm ở mức 50%.
7. Mỗi kết quả vào `results/week1_summary.csv` kèm commit, config và seed; không để số chỉ nằm trong log.

## 12. Backup

Server là working copy. Code, plan, report, bảng kết quả nhẹ phải về local và lên git remote sau mỗi
phiên. Output lớn: thống kê kích thước rồi quyết, không tự ý xoá. So hai cây bằng
`find . -type f -printf "%s %P\n" | sort | md5sum`. Không xoá nguồn trong cùng lượt với backup.

## 13. Checklist mở đầu mỗi phiên

```bash
. ./infra.env
ssh -n -o ConnectTimeout=20 $L40_SSH 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; uptime; df -h ~ | tail -1'
```

Bốn câu hỏi trước lệnh đầu tiên: card nào **thực sự trống** (và của ai đang giữ); job này có vừa 46 GB
không (không vừa → báo operator xin H200); chạy 6 tiếng rồi chết thì **resume** có cứu được không;
output cuối là file nào để guard.
