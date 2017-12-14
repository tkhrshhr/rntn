for d in 25 50 75; do
  echo $d
  for m in rn rt rd rc; do
    echo $m
    python train.py -r 1 -m $m -d $d -p 1 -q 1 -e 1 > log/1215_m$m\d$d &
  done
  m=rs
  for p in 1 2 4; do
    echo $m
    python train.py -r 1 -m $m -d $d -p $p -q $p -e 1 > log/1215_m$m\d$d\p$p &
  done
done
