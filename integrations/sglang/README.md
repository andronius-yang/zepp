# Zepp in SGLang

Runs the Zepp layer as an MoE backend of SGLang v0.5.3 (`--moe-a2a-backend zepp`), for Qwen3-MoE models.

## Layout

```
patches/sglang-v0.5.3.patch   registers the zepp backend in SGLang v0.5.3
zepp_sglang                   adapter: layer, runtime, calibration, token check
launch                        server and benchmark scripts
```

## Install

Requirements: CUDA 12.9, NVSHMEM 3.2, gcc 12, Python 3.11 with PyTorch 2.8 (CUDA 12.9), SGLang 0.5.3.

```
export ZEPP_CONDA_ENV=/path/to/sglang/env
conda create -p $ZEPP_CONDA_ENV python=3.11
pip install --index-url https://download.pytorch.org/whl/cu129 torch==2.8.0 torchvision torchaudio==2.8.0
pip install sglang==0.5.3
git clone --branch v0.5.3 --depth 1 https://github.com/sgl-project/sglang.git && cd sglang
git apply <zepp>/integrations/sglang/patches/sglang-v0.5.3.patch && pip install --no-deps -e python
cd <zepp> && source env/setup_sglang.sh && ./build.sh && pip install --no-deps -e integrations/sglang/zepp_sglang
```

## Calibrate

Record routing with stock SGLang, then derive the placement and buffer sizes:

```
CALIBRATE=1 SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=$OUT/dumps launch/server.sh baseline $MODEL
python -m sglang.bench_serving --backend sglang --port 30000 --dataset-name sharegpt --num-prompts 512
curl -X POST localhost:30000/dump_expert_distribution_record
python -m zepp_sglang.calibrate --dumps $OUT/dumps --model $MODEL --ranks <ranks> --smax 2048 --out $OUT/calib
```

## Serve

```
ZEPP_CALIB_DIR=$OUT/calib launch/server.sh ours $MODEL
launch/serve_bench.sh http://<host>:30000 $OUT/bench
```

Start one `server.sh` per node.
