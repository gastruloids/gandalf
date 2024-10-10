# Helper script for training models
for fold_no in 0 1 2 3 4; do
  echo "Training $model on fold $fold_no"
  poetry run python3 turner_lab/train.py --fold_no $fold_no --name "gamma-280" --num_images 280
done
