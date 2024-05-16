if [ $# -eq 0 ]; then
  echo "error: need model path!"
  exit
fi

pip3 install -r requirements.txt

python3 llama.py --model $1 --inference --quant_pt $2
