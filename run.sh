# python main.py --multirun dataset.domain=t_task0,t_task1,t_task2,t_task3,v_task0,v_task1,v_task2,v_task3

# for few_shot_num in 4, 5, 6, 8, 10, 12, 15, 20
# run "python test_HScore.py  --multirun few_shot_num=$() dataset.domain=t_task0,t_task1,t_task2,t_task3,v_task0,v_task1,v_task2,v_task3 > logs/domain_all_8_task_fewshot_2.txt"

for f in 4 5 6 8 10 12 15 20 # 4, 5, 
do
    echo "few_shot_num: $f"
    # echo few_shot_num=$f 
    python test_HScore.py few_shot_num=$f dataset.domain=t_task0,t_task1,t_task2,t_task3,v_task0,v_task1,v_task2,v_task3  --multirun > logs/domain_all_8_task_fewshot_$f.txt
done


