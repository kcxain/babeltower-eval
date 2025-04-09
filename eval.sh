model='gpt-4o-2024-05-13'
python -m babeltower_eval.eval_passk \
    --model_to_eval $model \
    --dataset 'kcxain/BabelTower' \
    --source 'cpp' \
    --model_mode 'trans' \
    --sample_num 20 \
    --save_translated_code \