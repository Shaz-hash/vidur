
# Task 1.1 Todo:

(i deleted your previous VLLM dir in the mew1 in /home/shaz because of incorrect experiment)
For this task you need to follow this readMe.md as the task required to do. 
High-level goal is to set up the latest version of the vllm with the kernels that vidur/sarathi serve can
support that is flash infer latest versions etc
Once we have this set up complete then we see how much in-accuracy is between the vidur's new profiled models and the actual set up vllm with sarahi served recommended kernels which it tells in its .md files and so for vidur's in it .md files. 


## Task 1.1.1 : Setting up VLLM on mew 1 that matches the requirement of the Vidur/Sarathi Serve
- Set up the vllm and Sarathi Serve such that it follows what vidur guides on the adding a new model from profiling from actual GPU as guided on the /home/shazer/Desktop/Research/Vidur/vidur-classical-search/docs/profiling.md. 
- We will set this up in the mew1 dir : /home/shaz/
- Then we will model flash-infer latests kernels that vidur/Sarathi Serve is based on over vllm.
- after profiling is done then we will train the vidur's profiling models for attention, mlp etc for the simulator with the same config as our initial model config which we trained the batch executor in our previous experiments :
 "--replica_config_model_name", "meta-llama/Meta-Llama-3-8B",
        "--replica_config_device", "a100",
        "--replica_config_network_device", "a100_dgx",
        "--cluster_config_num_replicas", "1",
        "--replica_config_tensor_parallel_size", "1",
        "--replica_config_num_pipeline_stages", "1",
        "--global_scheduler_config_type", "round_robin",
        "--replica_scheduler_config_type", "vllm_v1",
        "--vllm_v1_scheduler_config_batch_size_cap", "512",
        "--execution_time_predictor_config_type", "random_forest",
        "--random_forest_execution_time_predictor_config_prediction_max_tokens_per_request", "8192",
        "--random_forest_execution_time_predictor_config_prediction_max_batch_size", "256",
        "--random_forest_execution_time_predictor_config_prediction_max_prefill_chunk_size", "4096",
        "--random_forest_execution_time_predictor_config_cache_dir", cache_dir,
        "--random_forest_execution_time_predictor_config_cache_mode", "require_cache",
        "--random_forest_execution_time_predictor_config_num_training_job_threads", "1",
        "--no-snapshot_rng_state",


## Task 1.1.2 : Comparison between the batch execution of the Vidur's new profiled and actual Vllm on the mew1
- once the set up is complete, then for the prefill profile.csv for example here /home/shazer/Desktop/Research/Vidur/vidur-classical-search/simulator_output/prefill_profile.csv, we will generate the batch execution of single request of this prefill size on the actual gpu , and then for 2 requests. (we should have a script for this that can allow me to later run this for batches of same prefill size but multiple requests)
- then on the local machine which has vidur's new model should do it here /home/shazer/Desktop/Research/Vidur/vidur-classical-search/vidur_vllm_real_testing for profiling similarly 
- then it should make a comparison and generate csv in the /home/shazer/Desktop/Research/Vidur/vidur-classical-search/simulator_output/VLLM_NEW_MODEL_PROFILING_TESTING for me to see how much is it deviating.
- Remember we need to profile the GPU batch execution time, so ideally you should only find the profile time of the inference request time that vidur's actually try to learn the same components that are considered in the vidur's profiling explicitly/implicitly.
- ideally the mismatch should not be big, but we should see the authentic picture regardless to see how closely vidur can model this new vllm based on its own recommended kernels that it required for sarathi-serve, but note the kernels should be the latest versions because that's what we are aiming to detect the mis-match of. 
- on mew1 make sure this part of the process of the task.md is properly seperated in the dir so we can have clean codebase there too, and in the local machine there should be config.py that we can later use to profile and install certain kernel versions and model options, profiling options through which we can make this whole task automated.

Let me know when this is done 






















