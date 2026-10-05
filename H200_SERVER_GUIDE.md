# Máy chủ — hướng dẫn dùng cho project VLA 3D Distill (2×L40 + H200)

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
| Biến trong `infra.env` | `L40_SSH`, `L40_REMOTE_ROOT` | `H200_SSH`, `H200_REMOTE_ROOT` |
| Trạng thái với project này | **máy xác nhận**: eval mốc và eval cuối với render GPU đúng chuẩn upstream; một job mỗi lúc | **máy chính từ 05/10**: train + rollout (render CPU), nhiều job song song |
| GPU | 2 × L40 46 GB (driver 580), hai socket, hai NUMA | 1 × H200 141 GB |
| CPU / RAM | 96 core, 125 GB **không có swap**; người khác dùng 60–95 GB và dao động (load nền ≈ 24) | 22 core, 235 GB |
| Đĩa | 838 GB, còn **189 GB** (05/10); workspace project khác của operator chiếm 169 GB | 2,9 TB |
| Tài khoản | **dùng chung** với nhiều người (home có thư mục của họ) | root |
| Shell / giờ | **zsh**, **UTC+7** | bash, UTC |
| Có sẵn | git, tmux, rsync, wget, curl, docker, miniconda ở home; EGL của NVIDIA; **không có ffmpeg** | git, tmux, rsync, `uv` + `micromamba` ở `~/.local/bin`; mạng nhanh mọi nơi (PyPI 50 MB/s, HF 80 MB/s) |

**Quy tắc về H200:** operator cho dùng từ 05/10 ("util, VRAM hay RAM chưa dùng hết thì cứ đưa lên chạy").
Card có thể bị project kia của operator lấy lại: kiểm `nvidia-smi` trước mỗi lần chạy, không đụng thư mục
project kia. Workspace riêng nằm cạnh nó trong `~/projects/`.

**Render MuJoCo trên H200: GPU không dùng được, CPU thì được** (đo 05/10, driver 580.159 open kernel
module, GPU passthrough):
- NVIDIA EGL chỉ sống khi nó là context duy nhất trên GPU. Hai tiến trình render cùng lúc, hoặc một tiến
  trình render trong khi có bất kỳ context CUDA nào (kể cả model đang nghỉ), bị driver kill: `NVRM Xid 31`
  (MMU fault ở engine đồ hoạ) hoặc `Xid 109` (ctx switch timeout). Lock tuần tự hoá render không cứu được.
- Vì vậy H200 render bằng CPU: OSMesa (`mesalib=24` từ conda-forge, cài bởi `build_env.sh`), bật bằng file
  `<workspace>/machine.env` (ngoài git) đặt `MUJOCO_GL=osmesa`. Mỗi chunk chỉ render 1 frame nên chi phí nhỏ.
- **Renderer có ảnh hưởng nhưng nhỏ ở mức success rate**: trên cùng 100 episode (greedy), GPU (L40) so với
  CPU (H200): student 52% / 53%, full-SFT 98% / 93%, teacher RLinf 70% / 75%. Ở mức logits thì lệch rõ
  (KL ≈ 0.3 trên cùng state, so với KL ≈ 0.01 giữa hai máy trên cùng frame). Quy tắc: **mọi nhánh trong một
  bảng so sánh dùng cùng một renderer**; checkpoint cuối được xác nhận lại trên L40 (render GPU, chuẩn upstream).
- Bài học: con số "render CPU làm tụt 20–35 điểm" từng ghi ở đây là sai, do bug frame trễ của `EnvRunner`
  (chỉ lộ từ episode thứ hai của mỗi worker). Gate phải chạy nhiều episode liên tiếp
  (`scripts/check_env_stepping.py`), và một kết quả lạ phải được soi theo thứ tự episode trước khi quy cho
  môi trường.
- Hai server **không được nối trực tiếp với nhau** (quyết định của operator). Dữ liệu qua lại chỉ đi vòng
  laptop và phải gọn: L40 → laptop khoảng 0,8 MB/s, laptop → H200 khoảng 10 MB/s.

**Nút thắt thật của máy 2×L40 là RAM, không phải VRAM** (đo 05/10). Mỗi env LIBERO chiếm khoảng 1,7 GB
RAM, tiến trình chính giữ hai policy 7B chiếm khoảng 1,5 GB (đỉnh 5 GB lúc nạp) → một job 8 env cần khoảng
15 GB. Máy không có swap và một job của người khác dao động 36 GB trở lên, nên RAM trống thực tế chỉ
20–50 GB. Hệ quả: **mỗi lúc chỉ chạy một job có env**, tối đa 8 env; hai card không dùng song song cho
rollout được. Sự cố 05/10 (09:42–09:51): hai job eval × 10 env (35 GB) trùng lúc job kia phình to làm máy
cạn RAM, load lên trên 500 trong khoảng 8 phút; đã dừng job và thêm các chốt chặn ở §4.

## 1. Truy cập và đồng bộ code

```bash
. ./infra.env
ssh $L40_SSH             # hoặc $H200_SSH
scripts/sync.sh [l40|h200|all]   # rsync repo local -> <workspace>/repo (có --delete, bỏ qua mọi thứ trong .gitignore)
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
- CPU 96 core nhưng load nền khoảng 24, và RAM mới là giới hạn (§0): tối đa 8 env cho một job.

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
  chạy nhiều seed hoặc nhiều biến thể song song (RAM của máy này chỉ cho một job có env mỗi lúc); hoặc
  card 1 bị người khác giữ lâu làm nghẽn tiến độ.
- **Chốt chặn RAM và CPU (bắt buộc, đã cài sẵn trong script):**
  - `env.sh` giới hạn thread (`OMP/OPENBLAS/MKL/NUMBA/TF_*`): không có thì mỗi tiến trình mở khoảng 96
    thread BLAS/TF, 20 worker đẩy load average lên hàng trăm.
  - `run_py.sh` chờ đến khi `MemAvailable ≥ MIN_FREE_GB` (mặc định 24) mới chạy, và chạy với `nice`.
  - Collector tự dừng (MemoryError) khi `MemAvailable < MIN_FREE_GB_RUN` (mặc định 4); episode đã xong
    được lưu, chạy lại là resume.
  - Worker env không import torch (chỉ `libero.libero.envs`); init state do tiến trình chính gửi sang.
  - Trước khi chạy job mới: `free -g`, và đếm RAM job mình đang giữ
    (`ps -u $USER -o rss=,args= | awk '/<workspace>/ {s+=$1} END {print s/1e6 " GB"}'`).

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
- Checkpoint tải bằng `scripts/server/hf_fetch.py` (chỉ dùng stdlib, chạy trước cả khi có env): mỗi repo
  về một thư mục thường `checkpoints/<org>__<name>/`, kiểm sha256 từng file LFS, ghi revision vào
  `.hf_fetch.json`. Config trỏ thẳng tới thư mục đó.
- **Mạng của máy này không đều** (đo 05/10): HF 16–30 MB/s, conda-forge 30 MB/s,
  `download.pytorch.org` 11 MB/s; nhưng **PyPI và GitHub bị bóp theo từng kết nối, 30–100 KB/s**, hay
  rớt kết nối. Mirror PyPI Trung Quốc không dùng được. Cách xử trong `scripts/server/build_env.sh`:
  torch + thư viện CUDA lấy từ index pytorch; các gói còn lại resolve bằng `pip install --dry-run --report`
  rồi tải bằng nhiều range request song song (`pfetch.sh`, `fetch_report.py`) và cài offline từ thư mục
  wheel; repo GitHub lấy bằng `git clone --depth 1` hoặc file zip.
- Copy file GB giữa hai server qua laptop rất chậm (từng đo 144 KB/s) → tải thẳng trên máy đích.

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
| `pip install` đứng hàng chục phút | PyPI bị bóp 30–100 KB/s mỗi kết nối; pip 23 còn tải cả wheel chỉ để resolve | nâng pip trước, resolve bằng dry-run, tải wheel song song (§7) |
| Kill job nhưng job khác tự mọc lên | script hàng đợi gọi bằng đường dẫn tương đối nên không khớp bộ lọc theo tên workspace; nó thấy bước hiện tại chết và chạy bước kế | kill script hàng đợi **trước**, rồi mới tới các tiến trình con |
| ssh tự chết khi kill job trên máy H200 | shell đăng nhập ở đó là bash, nên bộ lọc `comm=="bash"` + tên script khớp chính lệnh ssh của mình | loại `$$` khỏi danh sách PID (`awk -v me=$$ '$1!=me && …'`) |
| Worker env chết không lời nào, `dmesg` có `NVRM: Xid 31/109` | render EGL trên H200 khi có context GPU khác (§0) | trên H200 dùng `MUJOCO_GL=osmesa` qua `machine.env` |
| Success rate tụt mạnh, lỗi xen kẽ theo từng lượt episode của worker | frame trễ/đen từ episode thứ hai: timer lấy mẫu camera của robosuite bị lệch khi tắt/bật camera qua reset | bật camera trước `env.reset()`; in chuỗi thành/bại theo thứ tự hoàn thành để soi |
| Hai lần render cùng state ra ảnh khác nhau | robosuite lấy mẫu camera ở substep 24/25 của control step, không phải state cuối | render đúng thời điểm đó (`EnvRunner`), kiểm bằng `scripts/check_env_stepping.py` |
| ssh `Connection timed out during banner exchange` | máy cạn RAM, đang thrash | thử lại bằng vòng lặp ssh ngắn; việc đầu tiên khi vào được là dừng job của mình |
| Hai `pip install` cùng ghi một env | kill script bash cha nhưng pip con vẫn sống, rồi chạy lại script | sau khi kill, liệt kê tiến trình con còn sống và kill theo PID trước khi chạy lại |
| `conda create python=3.10` xong không có pip | conda-forge không kéo pip theo python | thêm `pip` vào lệnh create |
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
