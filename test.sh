for file in `\find trained_model/ -maxdepth 1 -type f`; do
    file=$(basename $file)
    python test.py -s $file
done
