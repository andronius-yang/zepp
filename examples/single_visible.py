"""Run a torchrun worker with only its own GPU visible (as serving frameworks do), then the given
script: bench/launch.sh examples/single_visible.py examples/serving_check.py <args>"""
import os
import runpy
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("LOCAL_RANK", "0")
os.environ["LOCAL_RANK"] = "0"
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
