# source activate ddpg
STUDY=$(basename $0 | head -c -4)
DATE=$(date +%d%m%y%H%M)
APPENDIX=""
FILE="analysis_batch/csv/$STUDY$APPENDIX.csv"

# Check if file is new
echo Saving results to $FILE
if [ -f $FILE ]; then echo "Beware for overwriting!"; exit 1; fi

while read seed; do
    for error in 0 0.1 0.2 0.3; do
    	env="Pendulum-v0"
        python ppmp.py --algorithm pmp --random-seed $seed --env $env --error $error --fb-amount 0.27 \
        >> analysis_batch/csv/subfiles/$STUDY.$DATE.$env.$seed.$error.csv 2>/dev/null &

    	env="MountainCarContinuous-v0"
        python ppmp.py --algorithm pmp --random-seed $seed --env $env --error $error --fb-amount 0.17 \
        >> analysis_batch/csv/subfiles/$STUDY.$DATE.$env.$seed.$error.csv 2>/dev/null &
    done
	wait
done < analysis_batch/scripts/seeds.txt

# Put header in place
python -u ppmp.py --header-only | grep --line-buffered Environment > $FILE 2>/dev/null

cat analysis_batch/csv/subfiles/$STUDY.$DATE* >> $FILE
chmod -w $FILE

# Backup current work
conda list --export > conda_env.txt
tar -cf .backup/$STUDY.tar.gz $FILE *.py studies/$STUDY.sh conda_env.txt
rm conda_env.txt 
git add -f .backup/$STUDY.tar.gz; git add analysis_batch/csv/*
git commit -m "Automatic commit for study '$STUDY', $(date)"
git pull
git push

echo Done
