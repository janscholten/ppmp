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
cd ../../dcoach/

ID=$1$(date +%N | cut -c -5)
python -u main.py --header --env $env --human --random-seed $ID | \
	tee $(dirs -l +1)/csv/$envtag\_dcoach\_$ID.csv
popd