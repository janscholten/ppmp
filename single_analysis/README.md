This directory contains a subdirectory for each environment, in which single test runs of the myddpg.py algorithm may be executed.
For example, calling the `./run_pendulum.sh` script from the `Pendulum-v0/` directory will execute the algorithm in the background, whilst outputting to the screen and regulary updating a plot in `live_view.pdf`. Upon termination, this plot (and the raw csv) will be stored. Tags and parameters can be defined in the script, or passed as an argument (e.g. `./run_pendulum.sh --max-episodes 300`), and these will constitute the name of the saved pdf.

This testsuite allows early termination whilst preserving the results, by putting the script in background (append call with `&` or press `ctrl+z`) and calling 
`kill $(ps | grep python | egrep -o [0-9]\{3,6\})` to kill the actual algorithm but not the bash script.
Likewise, terminating the script e.g. by `ctrl+c` does not affect the python process running in background.
