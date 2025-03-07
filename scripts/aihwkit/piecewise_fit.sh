SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
FIT_SCRIPT_SOURCE="/scratch/msc24h18/msc_project/src/msc_project/utils/fit_piecewise.py"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <analyzed data directory>"
    exit 1
fi

data_directory="$1"

files=$(find ${data_directory} -type f -name "*_Summary.dat")

N_BATCH=10
for file in $files; do
    ((i=i%N_BATCH)); ((i++==0)) && wait

    out_dir=$(dirname $file)/aihwkit_piecewise_fit.png
    if ! python ${FIT_SCRIPT_SOURCE} $file --savefig $out_dir ; then
        echo -e "\e[31mError\e[0m analyzing file"
    else
        echo -e "\e[33m$file\e[0m"
    fi &
done
