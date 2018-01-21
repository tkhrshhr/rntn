for w in 0.0001 0.0003 0.0005 0.0007 0.0009 0.001; do
    echo $w
    for m in 1 2 4 8 16; do
	echo $m
	python train.py -r 1 -m rs -w $w -d 25 -p $m -q $m > log/rs_d25m$m\w$w &
    done;
done;
