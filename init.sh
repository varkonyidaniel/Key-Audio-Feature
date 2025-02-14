  #!/bin/bash
  #
  #echo "permission - 644 -> chmod +x run_fl.sh"

  #exit

  #build:
  #sudo singularity build fl.simg exp.recipe
  #SICMD="singularity exec fl.simg"
  #SCRIPT= "/home/peter.kiss/SimpleFederatedLearning-master/simulation.py"

  SICMD="singularity exec --bind /$HOME/Key-Audio-Feature:/mnt /singularity/21_Peter/kaf.simg"
  SCRIPT="python3 1_preprocessimg_batch.py"
  set -x
  i=0
  #singularity exec --bind /$HOME/federated_simulations:/mnt /singularity/21_Peter/exp.simg conda run -n FL_trial python3 exp.py
  test=False

  for hive_id in 22
  do
          i=$((i+1))
          fn="$HOME/Key-Audio-Feature/run_scripts/run_prepoc_${i}.sh"
          touch $fn
          echo "#!/bin/bash
          ${SICMD} ${SCRIPT} brood $hive_id 2020.06.01 10:00:00 3 128 128 all all 1 3999"   >  $fn
          ./exp_runner.sh $fn

  done