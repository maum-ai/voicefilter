# copy this to root directory of data and ./normalize-resample.sh
# https://unix.stackexchange.com/questions/103920/parallelize-a-bash-for-loop

open_sem(){
    mkfifo pipe-$$
    exec 3<>pipe-$$
    rm pipe-$$
    local i=$1
    for((;i>0;i--)); do
        printf %s 000 >&3
    done
}
run_with_lock(){
    local x
    read -u 3 -n 3 x && ((0==x)) || exit $x
    (
     ( "$@"; )
    printf '%.3d' $? >&3
    )&
}

N=32 # set "N" as your CPU core number.
open_sem $N
for f in $(find . -name "*.flac"); do
    run_with_lock ffmpeg-normalize "$f" -ar 16000 -o "${f%.*}-norm.wav"
done
