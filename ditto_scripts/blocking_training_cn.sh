python ./blocking_ditto/train_blocker.py \
  --train_fn "../ditto_prep/data/fuzzy/blocker_data_fuzzy_cn_NEW_train.txt" \
  --valid_fn "../ditto_prep/data/fuzzy/blocker_data_fuzzy_cn_NEW_val.txt" \
  --model_fn "../blocking_ditto/model_fuzzy_cn.pth" \
  --batch_size 64 \
  --n_epochs 40 \
  --lm bert 