i=5
echo $i
for w in 0.0001 0.0003 0.0005 0.0007 0.0009 0.001; do
  echo $w
  echo rn
    python train.py -r 1 -m rn -w $w -d 45 -i $i > log/rn\_d45w$w\i$i &
  for m in rt rd rc; do
    echo $m
      python train.py -r 1 -m $m -w $w -d 25 -i $i > log/$m\_d25w$w\i$i &
  done
done
