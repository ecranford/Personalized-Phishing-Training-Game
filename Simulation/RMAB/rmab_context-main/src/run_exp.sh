B=( 0.1 0.2 0.3 )
SS=( 0 1 2 )
# t1=( 2000 1500 1000 500 )
# t2=( 1500 1000 500 500 500 500 500 )
# N=( 4 7 )
# T=( 't1[@]' 't2[@]' )

# for e in "${T[@]}"; do
#     echo "${!e}";
# done

# i=0
# for td in "${T[@]}"; do
for sts in "${SS[@]}"; do
    for bu in "${B[@]}"; do
        # echo "python simulator.py -pc -1 -d rmab_context_diffT -l 5000 -s 0 -ws 0 -ls 0 -g 0.95 -adm 3 -A 2 -n 5000 -lr 1 -N 20 -sts ${sts} -nt ${N[i]} -b ${bu} -td ${!td}"
        # python simulator.py -pc -1 -d rmab_context_diffT -l 5000 -s 0 -ws 0 -ls 0 -g 0.95 -adm 3 -A 2 -n 5000 -lr 1 -N 20 -sts ${sts} -nt ${N[i]} -b ${bu} -td ${!td}
        echo "python simulator.py -pc -1 -d rmab_context_diffT -l 5000 -s 0 -ws 0 -ls 0 -g 0.95 -adm 3 -A 2 -n 5000 -lr 1 -N 20 -sts ${sts} -b ${bu}"
        python simulator.py -pc -1 -d rmab_context_diffT -l 5000 -s 0 -ws 0 -ls 0 -g 0.95 -adm 3 -A 2 -n 5000 -lr 1 -N 20 -sts ${sts} -b ${bu}
    done
done
# i=$((i+1))
# done