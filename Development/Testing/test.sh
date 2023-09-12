
system="test_diffusionscaling"
prefix="test_diffusionscaling"
Temperature="188.0"
RelaxationTime="1000.0"
DiffusionTime="10000.0"
DiffusionCutoff="1.0"
ChangeThreshold="0.1"
Criteria_Slope="0.2"
Criteria_Cycles="20.0"
Criteria_RxnSelectionCount="100.0"
Window_Slope="20.0"
Window_Mean="nan"
Window_Pause="15.0"
Window_RxnSelection="15.0"
Scaling_Adjuster="0.1"
Scaling_Minimum="1e-10"

rm ${prefix}.concentration
rm ${prefix}.scale

# Run RMD script
python3 ~/bin/hybrid_mdmc/Development/hybridmdmc.py ${prefix}.end.data\
  -prefix ${prefix}\

  -temp ${Temperature}\
  -relax ${RelaxationTime}\
  -diffusion ${DiffusionTime}\

  -num_voxels '2 2 2'\
  # -x_bounds ''\
  # -y_bounds ''\
  # -z_bounds ''\

  -change_threshold ${ChangeThreshold}\
  -diffusion_cutoff ${DiffusionCutoff}\
  -scalerates 'cumulative'\
  -scalingcriteria_concentration_slope ${Criteria_Slope}\
  -scalingcriteria_concentration_cycles ${Criteria_Cycles}\
  -scalingcriteria_rxnselection_count ${Criteria_RxnSelectionCount}\
  -windowsize_slope ${Window_Slope}\
  -windowsize_scalingpause ${Window_Pause}\
  -windowsize_rxnselection ${Window_RxnSelection}\
  -scalingfactor_adjuster ${Scaling_Adjuster}\
  -scalingfactor_minimum ${Scaling_Minimum}\
   
  --debug --log
