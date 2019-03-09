if [ $# -lt 2 ]; then echo Provide tag and env, e.g. run_ppmp.sh js p && exit 1; fi

if [[ $2 == [Pp]* ]]; then 
	env='Pendulum-v0' 
	envtag=pd
fi
if [[ $2 == [Mm]* ]]; then 
	env='MountainCarContinuous-v0'
	envtag=mc
fi

pushd .
cd ..
ID=$1$(date +%N | cut -c -5)
python -u ppmp.py --header --env $env --algorithm ppmp_human --random-seed $ID | \
	tee human_analysis/csv/$envtag\_ppmp\_$ID.csv
popd