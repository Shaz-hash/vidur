




## Task 1.1.1


Now we will be using the new cache of vidur developed from the task /home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur_vllm_real_testing/task.md for executing the alpha-go zero experiments over the ondemand servers to train the models like we did the experiment in the :
/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGoZero_EXP3_UNTRAINED_.98_2k_MCTS_3s_p95_DEPENDENCE_LEAF_RELATIVE_1ROLLOUT_DNN_MARKOV_1_THREAD_PER_GAME_PUCT_1p25_DIRICHLET12_OVER_N_SAMPLE15

First create the prefill-profile.csv using the new vidur's trained models from the GPU with the latest flash-infer over vllm and then have them all properly transfered to the on-demand servers with the name : flash-infer_prefill_profile.csv with the same prefill request size of 128 , 256 , ..... 4096 with their modelled batch execution time that we profiled here  :
/home/shazer/Desktop/Research/Vidur/vidur-classical-search/simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING


then with the following configs just to repeat :
- 2k MCTS iterations 
- 3 seconds rollout from the leaf like before
- 0.98 discounting 
- following GV3 game semantics
- 0.5 Exploration constant for both selfplay and eval
- 1 thread per rollout
- dirchlet noise epsilon : 0.25, same concentration of alpha = 12/N number of canon actions for the first 15 actions, then the sampling 
- same win ratio of threshold .57 i..e win games 82 out of 140
- full tree rollout, 5 seconds games
- the rest will be same as the experiment dir above which i mentioned

these exeperimetns and prefill profiles.csv and other output files from the experimetn will be here in the xl-exp3 :
/home/ubuntu/vidur-classical-search/simulator_output/GV3_Agent/AlphaGOZERO_new_vidur_flash_infer_models
this will be the dir for these new experiments

you may read the /home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur/AlphaGoZero/readMe.md , get the the overall idea (not the hyperparams values cuz they are different now)
So you have to set up this cache in all of the ondemand servers so that they can train new controller and adversary models like we normally did before.
Rest of the GV3 semantics will remain same as before.
Make sure native implementation correctly uses the new vidur models for the batch execution. 
So complete this on the ondemand experiments and let me know when the experiment has correctly started


## Task 1.1.2

The function of this task is to run actual traces, by passing actual prompts of certain lengths over the vllm on mew1's gpu and find out how our model works from the local machine.

The overall pipeline will be create request and send it to the vllm , there's is going to be a simulator that will create a snapshot state out of the existing current real state of the system and then provide it to the virtual simulator/environment. then our alphago zero will use that state and the current time (this does not include the time for mcts to create action, its just essentially the summation of past batch executions so far which is infact the time pass calculated the same way we used to find actual gpu compute time for llm inference/forward when profiling and comparing against vidur). So the models use that to find the best action just like how it works on vidur simulator.
Essentially we will not include the time for mcts to make this decision but rather only consider the time it took for gpu computation and time will progress it by that only on every decision.

you also need to consider the fast-forward/ decode only mechanism which runs whenever there's no prefill/request remaninng in the system to the next adversary tick by running actual decodes batches to progress time via batch execution over gpu. otherwise just shift the time to next adversary turn.

you also need to make sure that when converting actual system state into simulator state on whcih our mcts work is correct, should consider edge cases where multiple requessts come into the system at same time so the state should cover this and other edge cases aswell correctly.

you will have to implement this in vllm's scheduler layer which creates snapshot state and manages request slos and time correctly and then passes the state into vidur's alpha go zero where the alpha go zero uses the models and state and time correctly to tell the decision that will take place over actual gpu and therefore will likely increase time due to batch execution time on the gpu


Now that previous task has been comppleted , we need to extend /home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur_vllm_real_testing so that we have a another config file called : test_traces_on_GPU_with_AlphaGOZERO_models_config.py


this will allow us to define the following :
- model type 
- kernel versions (same as what we did for Task 1.1.1 )
- GPU config, for now its only TP = 1 and PP = 1
- controller & adversary's model path in mew1 
- static trace length to be tested 
- GPU number on mew1 , by default 2 
- trace type : in_distribution i.e. the actual distribution on whcih the model is trained on for example : requests will be of sizes 128..... 4096 the ones which the model experiences. out_distribution means request can be of any size of from 128 to 4096 but when making the simulator it will be rounded to nearest higher chunk for model inference to be better for example : request of 157 will be translated as 256 to the model.
- rest of MCTS /alpha go zero params which are the same as what we defined for the Task 1.1.1 above for now


document what you did in the .md file as future context. name it task 1.1.2_implementation.md 


## Task 1.1.3

Now your goal is to actually run test the pipeline first. For this, from the on-demand workers currently running on the servers first bring the current prompted models from the on-demand and have them used for testing and then seeing how controller performs on 20s trace. but before we need to do some testing.
Here are the overall steps :
- first bring the models shipped so we can use the pipeline in the task 1.1.2
- then we do a small 2 second trace test where you see how much the batch execution time takes for a particular batch and see how much it varies from vidur batch execution time, based on the steps/actions taken by mcts with same config of alpha go zero as described in task 1.1.2
- whatever the batch action is , note the total prefill size, decode requests number, prefill request numbers, prefill_request_stats : this will be a list which shows for each request id in the total prefill size, completed prefill size, remaining prefill , decode request stats which is similarly the list where each decode request id will have the context . 
using these columns you can construct the exact batch and see how much simulation time takes on the vidur models created from this config file. and then this csv will have finally columns : recorded_batch_time_from_trace, recorded_batch_time_from_vidur
- ideally the difference should not be more than 10% but we need to see this. So in the /home/shazer/Desktop/Research/Vidur/vidur-classical-search/simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING/comparison_test.csv local machine have these values noted 
- create this test file in the /home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur_vllm_real_testing/tests with the name : test_batch_execution_error.py , make sure you use the new cache that we trained i.e. /home/shazer/Desktop/Research/Vidur/vidur-classical-search/simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING/vidur_predictor_cache/
- and ideally the time on this pipeline should also advance by this time , the batch execution time, aswell.










