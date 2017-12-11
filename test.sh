for file in `\find trained_model_12-9/ -maxdepth 1 -type f`; do
    file=$(basename $file)
    python test.py -s $file
done
