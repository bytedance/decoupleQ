pip3 install datasets==1.17.0

python3 llama.py PATH/llama-7b c4 --true-sequential --act-order --new-eval \
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
