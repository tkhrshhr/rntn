for weight in 0.0001 0.0005 0.0007 0.0009; do
  for model in rt rd rc; do
    echo $model
    echo $weight
    python train.py -r 1 -m $model -w $weight -d 25 > log/$model$weight &
  done;
  for p in 1, 2, 3, 10; do
    echo rs
    echo $weight
    python train.py -r 1 -m rs -w $weight -d 25 -p $p -q $p > log/rs$p$weight &
  done;
    echo rn
    echo $weight
    python train.py -r 1 -m rn -w $weight -d 45 > log/rn$weight &
done;
