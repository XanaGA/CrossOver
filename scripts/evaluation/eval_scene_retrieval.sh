export PYTHONWARNINGS="ignore"

# Scene Retrieval Inference
python run_evaluation.py --config-path "$(pwd)/configs/evaluation" --config-name eval_scene.yaml \
task.InferenceSceneRetrieval.val=['Scannet'] \
task.InferenceSceneRetrieval.ckpt_path=/home/xavi/CrossOver/checkpoints/model.safetensors \
hydra.run.dir=. hydra.output_subdir=null 