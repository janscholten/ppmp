
# This is a small testbench that lets you define parameters,
# either in the scipt or by throwing them in the command line. 
# It will store the data and make a plot (live). 
# This scipt is not guaranteed to be thread-safe!
# Jan Scholten, 2018

> file.csv
KEYWORD="ppmp"
pushd .
cd ../..
python -u ppmp.py --header --env LunarLanderContinuous-v2 \
$@ | tee $(dirs -l +1)/file.csv &
PY_PROCID=$!
popd

# While python is busy, keep updating the pdf
while kill -0 "$PY_PROCID" >/dev/null 2>&1; do
	sleep 17; python plotter.py
done
python plotter.py

args=$@ # as inline expansion fails
cp -b lunarlive.pdf "alr=$ACTLR, hf=$HFVAR, in=$INVAR, tau=$TAU, $KEYWORD $args".pdf
cp -b file.csv "old/alr=$ACTLR, hf=$HFVAR, in=$INVAR, tau=$TAU, $KEYWORD $args".csv
touch lunarlive.pdf file.csv run_lunar.sh