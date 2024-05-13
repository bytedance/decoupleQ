if [ $# -eq 0 ]; then
  echo "error: need model path!"
  exit
fi

pip3 install -r requirements.txt

python3 llama.py --model $1/llama-7b --dataset c4 --true-sequential --act-order --new-eval \
--wbits 2 \
--group-size -1 \
--nsamples 128 \
--max-iter-num 4 \
--iters-before-round 200 \
--inner-iters-for-round 5 \
--blockwise-minimize-epoch 4 \
--round-fn gptq \
--blockwise-minimize-lr 1.0e-5 \
--train-LN
