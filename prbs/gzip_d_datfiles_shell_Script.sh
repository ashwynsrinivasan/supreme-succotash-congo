END=32
for i in $(seq 4 $END); do gzip -d $i.dat.gz; done

