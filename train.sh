for m in rt rd rc ; do
  python train.py -r 1 -m $m -v 0 -w 0.0005 -d 25 > log/0121_no_matpro_$m\_w$w &
done
