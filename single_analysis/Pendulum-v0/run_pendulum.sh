
# This is a small testbench that lets you define parameters,
# either in the scipt or by throwing them in the command line. 
# It will store the data and make a plot (live). 
# This scipt is not guaranteed to be thread-safe!
# Jan Scholten, 2018

> file.csv
KEYWORD="ppmp"
pushd .
cd ../..
python -u ppmp.py --header --max-episodes 150 --fb-amount 0.28 $@ | tee $(dirs -l +1)/file.csv &
PY_PROCID=$!
popd

# While python is busy, keep updating the pdf
sleep 10; # wait for some data
while kill -0 "$PY_PROCID" >/dev/null 2>&1; do
	sleep 2; python plotter.py
done
python plotter.py

args=$@ # as inline expansion fails
date=$(date +%s | tail -c 7)
cp -b live_view.pdf "$KEYWORD $args $date".pdf
cp -b file.csv "old/$KEYWORD $args $date".csv
cp -b ../../v7.py "old/$KEYWORD $args $date".py
cp -b run_pendulum.sh "old/$KEYWORD $args $date".sh
touch live_view.pdf file.csv run_pendulum.sh